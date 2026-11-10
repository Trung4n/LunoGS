#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.pixel_index import decode_pixel_index, pixel_index_image
from gaussian_splatting.utils.sh_utils import RGB2SH, SH2RGB
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    """The map: a set of N 3D Gaussians, stored as flat per-Gaussian tensors.

    Mostly the original 3DGS GaussianModel. MonoGS changes:
      - the map starts EMPTY and grows incrementally, one keyframe at a time
        (create_pcd_from_image / extend_from_pcd_seq) instead of being built
        once from a COLMAP point cloud;
      - two SLAM bookkeeping tensors per Gaussian: unique_kfIDs (which
        keyframe created it) and n_obs (how many window keyframes see it),
        used by the backend's covisibility pruning;
      - the xyz learning-rate schedule avoids a closure so the object can be
        pickled across processes (see training_setup).

    Every learnable attribute is stored BEFORE its activation (e.g. log-scale,
    opacity logit, unnormalized quaternion) so the optimizer can move it
    freely in R^n; the get_* properties apply the activation. Always read the
    map through get_*.

    Every structural change (adding, cloning, splitting, pruning Gaussians)
    must also resize the Adam state in self.optimizer, hence the
    *_optimizer helpers at the bottom.
    """

    def __init__(self, sh_degree: int, config=None):
        # active_sh_degree: SH degree actually used for rendering. Only
        # oneupSHdegree() raises it, and nothing calls that in this codebase,
        # so it stays 0 (flat color) even if max_sh_degree is 3.
        self.active_sh_degree = 0
        # max_sh_degree: 0 or 3 (config Training.spherical_harmonics); sets how
        # many SH coefficients are allocated per Gaussian.
        self.max_sh_degree = sh_degree

        # --- learnable per-Gaussian parameters (pre-activation) ---
        # _xyz: (N,3) centers in world coordinates (no activation).
        self._xyz = torch.empty(0, device="cuda")
        # _features_dc: (N,1,3) degree-0 SH coefficient per RGB channel, i.e.
        # the base color (see RGB2SH).
        self._features_dc = torch.empty(0, device="cuda")
        # _features_rest: (N,(max_sh_degree+1)^2-1,3) higher-order SH
        # coefficients (view-dependent color). Empty second dim when degree 0.
        self._features_rest = torch.empty(0, device="cuda")
        # _scaling: (N,3) log of the standard deviation along each local axis.
        self._scaling = torch.empty(0, device="cuda")
        # _rotation: (N,4) quaternion (w,x,y,z), normalized in get_rotation.
        self._rotation = torch.empty(0, device="cuda")
        # _opacity: (N,1) opacity logit, sigmoid in get_opacity.
        self._opacity = torch.empty(0, device="cuda")

        # --- densification statistics (not learnable) ---
        # max_radii2D: (N,) largest on-screen radius (pixels) seen since the
        # last reset of the stats; used to prune huge Gaussians.
        self.max_radii2D = torch.empty(0, device="cuda")
        # xyz_gradient_accum: (N,1) sum over views of |screen-space position
        # gradient|; divided by denom (set in training_setup) to get the mean.
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        # --- SLAM bookkeeping (CPU int tensors) ---
        # unique_kfIDs: (N,) frame index of the keyframe that created each
        # Gaussian (inherited by clones/splits). Used by the backend's "slam"
        # pruning and by the GUI to color Gaussians by keyframe.
        self.unique_kfIDs = torch.empty(0).int()
        # n_obs: (N,) number of window keyframes that see each Gaussian;
        # recomputed by the backend right before pruning.
        self.n_obs = torch.empty(0).int()

        # Extra per-Gaussian parameters declared by the light model
        # (declare_extras, docs/DECISIONS.md D121), e.g. the ambient residual:
        # name -> (N,k) tensor, one Adam group each, carried through every
        # append / clone / split / prune like the attributes above. Empty
        # without a light model that needs them (original MonoGS).
        self._extras = {}
        self._extra_lr = {}

        # Adam over the six parameter tensors (created in training_setup).
        self.optimizer = None

        # Activations: storage space -> actual value.
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        # Last point cloud built by create_pcd_from_image_and_depth. Stored
        # but not read anywhere in this codebase.
        self.ply_input = None

        # isotropic: if True, new Gaussians get a single scale value instead of
        # 3. Always False here (anisotropy is only discouraged softly by the
        # backend's isotropic loss).
        self.isotropic = False

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        # 3D covariance Sigma = R S S^T R^T, returned as its 6 upper-triangle
        # entries. Only used when pipeline_params.compute_cov3D_python is True
        # (never in the configs); otherwise the CUDA rasterizer builds it.
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        # (N,3) standard deviations (> 0).
        frozen = self.__dict__.get("_frozen")
        if frozen is not None:
            return frozen["scaling"]
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # (N,4) unit quaternions.
        frozen = self.__dict__.get("_frozen")
        if frozen is not None:
            return frozen["rotation"]
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # (N,(max_sh_degree+1)^2,3) all SH coefficients, DC first.
        frozen = self.__dict__.get("_frozen")
        if frozen is not None:
            return frozen["features"]
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_albedo(self):
        # (N,3) the color the rasterizer actually splats at SH degree 0:
        # max(C0 * dc + 0.5, 0) (forward.cu, computeColorFromSH). With a light
        # model this is the LINEAR albedo (docs/DECISIONS.md D1), deliberately
        # not clamped above 1.
        return torch.clamp_min(SH2RGB(self._features_dc[:, 0, :]), 0.0)

    def get_extra(self, name):
        if name not in self._extras:
            raise KeyError(f"per-Gaussian parameter {name!r} was not declared (declare_extras)")
        return self._extras[name]

    def declare_extras(self, specs):
        """specs: {name: (values per Gaussian, Adam lr)} from
        shader.per_gaussian_params(). Called once, on the empty map, before
        training_setup."""
        if self.optimizer is not None or self.get_xyz.shape[0] != 0:
            raise RuntimeError("declare_extras must be called on the empty map, before training_setup")
        for name, (k, lr) in specs.items():
            self._extras[name] = torch.empty(0, k, device="cuda")
            self._extra_lr[name] = lr

    def _new_extras(self, n):
        # New Gaussians from a keyframe: every extra starts at 0 (D121).
        return {name: torch.zeros(n, t.shape[1], device="cuda") for name, t in self._extras.items()}

    def _select_extras(self, mask, repeat=1):
        return {name: t[mask].repeat(repeat, 1) for name, t in self._extras.items()}

    def _take_extras(self, optimizable_tensors):
        for name in self._extras:
            self._extras[name] = optimizable_tensors[name]

    @property
    def get_opacity(self):
        # (N,1) opacity in (0,1).
        frozen = self.__dict__.get("_frozen")
        if frozen is not None:
            return frozen["opacity"]
        return self.opacity_activation(self._opacity)

    def freeze_activations(self):
        """For a map snapshot that will not change (the frontend's, between
        two backend messages): scaling, rotation, opacity and features are
        computed once instead of at every render (docs/DECISIONS.md D130).
        Same values: the activations of the same tensors."""
        self._frozen = None
        frozen = {"scaling": self.get_scaling, "rotation": self.get_rotation,
                  "opacity": self.get_opacity, "features": self.get_features}
        self._frozen = frozen
        return self

    def get_covariance(self, scaling_modifier=1):
        # Uses the raw quaternion; build_rotation normalizes it internally.
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        # In 3DGS the SH degree is raised progressively during training. Not
        # called anywhere in MonoGS: with spherical_harmonics=True the higher
        # coefficients are allocated and saved but never used for rendering.
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(
        self, cam_info, init=False, scale=2.0, depthmap=None, *, albedo_map
    ):
        # Builds the parameters of the new Gaussians for one keyframe (it does
        # NOT add them to the map, see extend_from_pcd_seq).
        #
        # albedo_map: None (original MonoGS) -> colors come from the
        # keyframe's GT image after its exposure correction, so new Gaussians
        # have colors in the same reference exposure as the rest of the map.
        # Otherwise a (3,H,W) LINEAR albedo map from
        # light_models/albedo_init.py; each new Gaussian takes the value at
        # its own pixel (see create_pcd_from_image_and_depth). The 8-bit
        # "color" given to Open3D is then the PIXEL INDEX of every pixel
        # (gaussian_splatting/utils/pixel_index.py), so the points come back
        # labelled with the pixel they were back-projected from, exactly
        # (docs/DECISIONS.md D28).
        # Required keyword on purpose (docs/DECISIONS.md D10).
        cam = cam_info
        if albedo_map is None:
            image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
            image_ab = torch.clamp(image_ab, 0.0, 1.0)
            rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        else:
            _, h, w = albedo_map.shape
            rgb_raw = pixel_index_image(h, w)

        if depthmap is not None:
            # The path always taken in MonoGS: depth prepared by
            # FrontEnd.add_new_keyframe (sensor depth, or rendered/guessed
            # depth for monocular). 0 = no Gaussian at that pixel.
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            # Fallback when no depth map is passed; unused here because the
            # backend always passes one. For monocular it builds a noisy flat
            # plane at depth ~`scale`.
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(
            cam, rgb, depth, init, albedo_map=albedo_map
        )

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, *, albedo_map):
        # Back-projects an RGB-D pair into a world-space point cloud and turns
        # each point into an initial Gaussian. Returns
        # (xyz, features, scales, rots, opacities), all pre-activation.
        #
        # Keep 1 of every downsample_factor valid pixels: denser for the very
        # first frame (pcd_downsample_init, e.g. 32) than for later keyframes
        # (pcd_downsample, e.g. 64 or 128), since later keyframes mostly
        # overlap areas the map already covers.
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        # point_size: multiplier on the initial Gaussian size (see scales).
        # With adaptive_pointsize it is scaled by the median depth of the
        # frame (zeros included), capped at 0.05: farther scenes -> bigger
        # Gaussians.
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        # depth is already in meters (depth_scale=1); depths beyond 100 m are
        # dropped.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        # Open3D takes the world->camera extrinsic and outputs points in
        # WORLD coordinates, using the keyframe's current estimated pose.
        # Pixels with depth 0 are skipped (project_valid_depth_only).
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        if albedo_map is not None:
            # Light model: the "colors" are pixel indices (see
            # create_pcd_from_image); replace them by the float linear albedo
            # of each point's own pixel.
            idx = decode_pixel_index(new_rgb)
            _, h, w = albedo_map.shape
            values = albedo_map.detach().cpu().double().numpy().reshape(3, h * w)
            new_rgb = values[:, idx].T  # (N, 3)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        # Color: RGB -> degree-0 SH coefficient; higher-order coefficients 0.
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        # No-op kept from 3DGS (dim 1 has size 3, so `3:` is empty).
        features[:, 3:, 1:] = 0.0

        # Size: distCUDA2 gives each point's mean squared distance to its 3
        # nearest neighbours, computed among the NEW points only. The initial
        # std is sqrt(point_size * that distance), the same on all 3 axes, so
        # new Gaussians start as spheres roughly as big as the point spacing.
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        # Identity rotation (w=1) and opacity 0.5.
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities


    def init_lr(self, spatial_lr_scale):
        # spatial_lr_scale: multiplies the xyz and scaling learning rates, so
        # they are relative to scene size. slam.py sets it to 6.0 (fixed).
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        # Appends new Gaussians to the map (and to the optimizer), tagged with
        # the keyframe that created them. n_obs starts at 0.
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_extras=self._new_extras(new_xyz.shape[0]),
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, *, albedo_map
    ):
        # Entry point used by the backend (BackEnd.add_next_kf): create the
        # Gaussians for one keyframe, then add them to the map. albedo_map:
        # see create_pcd_from_image (None = original MonoGS colors).
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(
                cam_info, init, scale=scale, depthmap=depthmap, albedo_map=albedo_map
            )
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        # Creates the optimizer. Called once in slam.py while the map is still
        # EMPTY: every param group starts with a 0-size tensor and grows as
        # keyframes add Gaussians (cat_tensors_to_optimizer).
        #
        # percent_dense: size threshold (fraction of the scene extent) that
        # decides between cloning (small Gaussians) and splitting (large ones).
        self.percent_dense = training_args.percent_dense
        # denom: (N,1) number of views each Gaussian was visible in since the
        # last stats reset (denominator of the mean screen-space gradient).
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # One Adam param group per attribute, each with its own learning rate.
        # The group "name" is how the *_optimizer helpers find them.
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        for name, t in self._extras.items():
            l.append({"params": [t], "lr": self._extra_lr[name], "name": name})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # Not used: update_learning_rate calls `helper` directly with the
        # values stored below. In upstream 3DGS get_expon_lr_func returns a
        # closure; here it was rewritten to return a module-level function,
        # most likely because a closure cannot be pickled when the model is
        # sent to the backend/GUI processes.
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        # xyz learning-rate schedule: log-linear decay from lr_init to
        # lr_final over max_steps mapping iterations. lr_delay_mult has no
        # effect because lr_delay_steps is left at 0.
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        # Only the xyz learning rate is scheduled; the others stay constant.
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        # Column names of the standard 3DGS .ply format.
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        # Exports the map in the standard 3DGS .ply format (raw pre-activation
        # values, zero normals), so it opens in usual 3DGS viewers. Called by
        # eval_utils.save_gaussians. unique_kfIDs / n_obs are not saved.
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        # Sets EVERY opacity to 0.01 (and clears its Adam moments). Used once
        # during BackEnd.initialize_map: Gaussians the loss does not need stay
        # near-transparent and get pruned at the next densify_and_prune.
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        # MonoGS variant used by BackEnd.map(): Gaussians visible in any of
        # the given views keep their opacity; all others are set to 0.4.
        # (Adam moments are cleared for all of them.)
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # Loads a .ply written by save_ply. Not called anywhere in this
        # codebase; note it neither rebuilds the optimizer nor restores real
        # keyframe IDs (unique_kfIDs is set to float zeros).
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    # --- Optimizer surgery ---------------------------------------------------
    # Adam keeps per-element moments (exp_avg, exp_avg_sq) keyed by the
    # parameter tensor object. Whenever a parameter tensor is replaced or
    # resized, the moments must be replaced/resized the same way and re-keyed
    # to the new tensor, otherwise Adam would crash or mix up Gaussians.

    def replace_tensor_to_optimizer(self, tensor, name):
        # Replaces the whole parameter `name` by `tensor` (same size) and
        # resets its Adam moments to zero. Assumes Adam already has state for
        # it (i.e. at least one step was taken), otherwise stored_state is None.
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        # Keeps only the rows where `mask` is True, in every param group and
        # in the matching Adam moments. The new tensors are fresh leaves, so
        # any pending .grad is dropped.
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # Removes the Gaussians where `mask` is True: parameters, Adam state,
        # densification stats and SLAM bookkeeping (on CPU, hence .cpu()).
        # BackEnd.reset() uses it with an all-True mask to empty the map.
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._take_extras(optimizable_tensors)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # Appends new rows to every param group; the new rows get zero Adam
        # moments. tensors_dict maps group name -> tensor of new rows.
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        new_extras=None,
    ):
        # Common "append Gaussians" step, used by keyframe insertion
        # (extend_from_pcd), cloning and splitting.
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        new_extras = new_extras or {}
        if set(new_extras) != set(self._extras):
            raise ValueError(f"new Gaussians carry extras {sorted(new_extras)}, the map has {sorted(self._extras)}")
        d.update(new_extras)

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._take_extras(optimizable_tensors)

        # Resets the densification stats of ALL Gaussians, not just the new
        # ones (as in 3DGS). In MonoGS this also happens every time a keyframe
        # adds Gaussians, so the gradient statistics restart at each keyframe.
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # Over-reconstruction: a LARGE Gaussian (max scale >
        # percent_dense * scene_extent) with a high mean screen-space gradient
        # probably covers too much detail. Replace it by N=2 smaller Gaussians
        # sampled inside it (positions drawn from its own distribution,
        # scales divided by 0.8*N=1.6), then delete the original.
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # `grads` was computed before densify_and_clone added Gaussians, so
        # it is shorter; the clones get gradient 0 (never split right away).
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # Children inherit the parent's keyframe ID and observation count.
        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_extras=self._select_extras(selected_pts_mask, N),
        )

        # Delete the split originals (the appended children are kept).
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Under-reconstruction: a SMALL Gaussian (max scale <=
        # percent_dense * scene_extent) with a high mean screen-space gradient
        # sits in an area that needs more Gaussians. Duplicate it in place;
        # the next optimizer steps pull the two copies apart.
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_extras=self._select_extras(selected_pts_mask),
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # One densification round, called on a schedule by the backend:
        #   max_grad: mean screen-space gradient threshold for clone/split
        #   min_opacity: prune Gaussians more transparent than this
        #   extent: scene extent (clone/split size threshold, world-size prune)
        #   max_screen_size: prune Gaussians bigger than this on screen
        #     (pixels); None disables size pruning (initialize_map).
        #
        # Mean screen-space gradient per Gaussian since the last stats reset;
        # NaN (never visible, denom 0) -> 0.
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # NOTE: densify_and_clone above always calls densification_postfix
            # (even when nothing is cloned), which zeroes max_radii2D. So
            # big_points_vs is always False here and only the world-space
            # check has an effect. Same behaviour as upstream 3DGS.
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # Called after each backward pass, once per rendered view: for every
        # Gaussian visible in that view, add the norm of the gradient of its
        # projected 2D position (from render()'s dummy "viewspace_points"
        # tensor) and count one more view.
        # torch.where instead of boolean indexing (which synchronizes with the
        # GPU); same values (D131).
        keep = update_filter.view(-1, 1)
        norm = torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum += torch.where(keep, norm, torch.zeros_like(norm))
        self.denom += keep.to(self.denom.dtype)
