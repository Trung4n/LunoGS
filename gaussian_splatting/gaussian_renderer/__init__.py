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

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import RASTERIZER_PIXEL_OFFSET
from gaussian_splatting.utils.sh_utils import eval_sh
from utils.color_space import linear2sRGB

# This is the single "forward pass" of the whole SLAM system: both
# FrontEnd.tracking() (pose-only optimization) and BackEnd.map()/
# initialize_map()/color_refinement() (Gaussian + pose optimization) call
# render() every iteration and backward() through its output. Unlike vanilla
# 3DGS, `diff_gaussian_rasterization` here is MonoGS's own fork
# ("diff-gaussian-rasterization-w-pose"): its CUDA kernel additionally (a)
# accepts a camera-pose delta (theta/rho) and back-propagates gradients into
# it, and (b) returns a per-Gaussian `n_touched` pixel count used for
# keyframe/covisibility bookkeeping. Neither exists in the original
# graphdeco-inria renderer this file is based on.


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    *,
    shader,
    defer_shading=False,
    raster_capacity=0,
    raster_count=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Args:
        viewpoint_camera: a utils.camera_utils.Camera. Supplies the view/
            projection matrices AND the learnable cam_rot_delta/cam_trans_delta
            pose-delta parameters that this call's backward pass writes
            gradients into (see the theta/rho args passed to the rasterizer
            below).
        pc: the GaussianModel scene being rendered (read through its
            get_xyz/get_opacity/... activation properties, never its raw
            _xyz/_opacity/... tensors directly).
        pipe: pipeline_params from the config; toggles whether covariance/SH
            colors are computed on the Python/autograd side (slower, only
            used for debugging/ablation) or inside the CUDA rasterizer
            (default, faster - this is what actually runs during SLAM).
        scaling_modifier: uniform scale applied to every Gaussian's
            covariance; always 1.0 in this codebase (only relevant for
            interactive visualization elsewhere in 3DGS-derived projects).
        override_color: bypasses SH color evaluation entirely if given.
        mask: optional boolean index into the Gaussians to render only a
            subset. NOT USED ANYWHERE in this codebase (no caller passes it)
            and its branch below is actually broken if exercised: unlike the
            no-mask path it does not unpack `n_touched` from the rasterizer,
            yet the return dict below unconditionally includes "n_touched" -
            that would raise NameError. Left here only for API parity with
            upstream 3DGS.
        shader: REQUIRED keyword, no default on purpose: forgetting to pass
            it must fail loudly instead of silently rendering without the
            light model. Built once by light_models.build_shader(config).
            - None (Light.enabled: false): original MonoGS, the rasterized
              color IS "render" and nothing below changes.
            - a callable shader(gbuffer, viewpoint_camera) -> dict with
              "radiance_linear" (3,H,W) and optional auxiliary per-pixel maps
              (copied into the returned dict, see light_models/shaders.py).
              gbuffer = {"albedo", "depth", "opacity"} straight
              from the rasterizer. With the black background, albedo and
              depth are both opacity-weighted sums (sum_i w_i x_i). The
              shader keeps the albedo that way (correct with a black
              background and a multiplicative ambient term) but MUST divide
              depth by opacity before back-projecting, with the opacity
              threshold coming from the config (see docs/DECISIONS.md).

    Returns:
        None if the map is still empty (no Gaussians yet), otherwise a dict
        - see the bottom of this function for each field's meaning/consumer.
    """

    # Guard against rendering an empty map, e.g. before the very first
    # keyframe has been added, or right after a prune step removed
    # everything.
    if pc.get_xyz.shape[0] == 0:
        return None

    # `screenspace_points` is a dummy placeholder, not a meaningful value: its
    # only purpose is to be a differentiable stand-in for the Gaussians'
    # projected 2D (screen-space) positions ("means2D" below), so that after
    # loss.backward() we can read its .grad to know how much each Gaussian's
    # screen position wants to move — this drives the densification
    # statistics (see GaussianModel.add_densification_stats). The `+ 0` turns
    # it into a non-leaf tensor, which is why retain_grad() is needed for its
    # .grad to survive backward() (leaf tensors keep .grad automatically,
    # non-leaf ones don't unless told to).
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        # e.g. called under torch.no_grad() — fine, nothing downstream will
        # need .viewspace_points.grad in that case (rendering for the GUI,
        # or eval_rendering()).
        pass

    # Set up rasterization configuration
    # Half-angle tangents, the form the rasterizer's perspective projection
    # math expects (derived from Camera.FoVx/FoVy, in turn from fx/fy).
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        # The raw (view-independent) projection matrix, passed in ADDITION to
        # the combined view+proj matrix above. This MonoGS-specific field is
        # what lets the CUDA backward pass work out d(pixel)/d(pose) and
        # therefore compute gradients w.r.t. theta/rho further down.
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # Fixed binning capacity, no device-to-host copy of the instance
        # count; the true count is written to raster_count (CUDA graph replay
        # of the whole tracking iteration, docs/DECISIONS.md D133). 0 = the
        # original behaviour.
        capacity=raster_capacity,
        num_rendered_out=raster_count,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity  # per-Gaussian opacity (N,1) - input to rasterization

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # Slower path: build the 3D covariance in Python/autograd (useful for
        # debugging gradients); default config always has this False.
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        # Defensive branch: MonoGS's usual (non-isotropic) Gaussians already
        # store 3 scale components (see GaussianModel.isotropic /
        # create_pcd_from_image_and_depth), so this repeat() is normally a
        # no-op guard rather than something hit in practice.
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            # Same idea as compute_cov3D_python: reference/debug path that
            # evaluates spherical harmonics -> RGB in Python instead of
            # inside the CUDA kernel. Always False in this codebase's configs.
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # Default path: hand the raw SH coefficients to the rasterizer
            # and let the CUDA kernel evaluate view-dependent color itself.
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if mask is not None:
        # See the `mask` docstring note above: this path is unused and its
        # `n_touched` omission means the return statement below would crash
        # if it were ever taken.
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        # theta/rho: the camera's pose-delta parameters. They are pure
        # GRADIENT SINKS: the forward pass never reads their values (it always
        # renders at the camera's current R,T via viewmatrix/projmatrix). In
        # the backward pass the CUDA kernel computes dLoss/d(tau) for a small
        # pose perturbation tau = (rho, theta) evaluated at tau = 0, and
        # returns it as their gradient. This is what makes tracking possible:
        # after loss.backward(), `viewpoint_camera.cam_rot_delta.grad`/
        # `cam_trans_delta.grad` are what the pose optimizer steps on, and
        # pose_utils.update_pose() then applies the step to R,T and resets
        # the deltas to zero (so the "evaluated at 0" assumption holds again).
        #
        # NOTE: `opacity` here is REASSIGNED - it shadows the per-Gaussian
        # input `opacity` from above with a different thing: the rendered
        # per-pixel accumulated-alpha map (H, W). Same name, different shape
        # and meaning; only the post-call value is used below/by callers.
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    render_pkg = {
        "render": rendered_image,  # (3,H,W) rendered color image, sRGB-encoded (see below)
        "viewspace_points": screenspace_points,  # dummy tensor; .grad read for densification stats
        "visibility_filter": radii > 0,  # (N,) bool - which Gaussians actually landed on screen
        "radii": radii,  # (N,) on-screen radius per Gaussian, 0 if culled
        "depth": depth,  # (1,H,W) rendered depth map
        "opacity": opacity,  # (1,H,W) rendered accumulated-alpha map (NOT per-Gaussian opacity)
        "n_touched": n_touched,  # (N,) per-Gaussian pixel-touch count -> keyframe covisibility / pruning
    }

    # Light model hook. Contract: "render" is ALWAYS sRGB-encoded, the space
    # of the observed images, of evaluation and of the GUI, exactly as in the
    # baseline. So eval/GUI keep reading "render" unchanged. With a shader,
    # the rasterized color is the (opacity-weighted, linear) albedo, and the
    # linear radiance is exposed separately for losses computed in linear
    # space; the loss picks "render" or "radiance_linear" according to
    # LightTracking.loss_color_space.
    if shader is not None:
        gbuffer = {
            "albedo": rendered_image,
            "depth": depth,
            "opacity": opacity,
            # Rasterized pixel i looks along OpenCV ray i + 0.5 (D29).
            "pixel_offset": RASTERIZER_PIXEL_OFFSET,
        }
        # Per-Gaussian light quantities (e.g. the ambient residual, D121):
        # one more pass of the same rasterizer per map, with precomputed
        # colours instead of the SH, so gradients reach those per-Gaussian
        # values, the Gaussians' geometry and the pose as for the albedo.
        extra_rasters = shader.extra_rasters(pc)
        if mask is not None and extra_rasters:
            raise NotImplementedError("extra light rasters with a Gaussian mask")
        for key, colors in extra_rasters.items():
            if key in gbuffer:
                raise ValueError(f"extra raster {key!r} clashes with the G-buffer")
            gbuffer[key] = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=colors,
                opacities=pc.get_opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                theta=viewpoint_camera.cam_rot_delta,
                rho=viewpoint_camera.cam_trans_delta,
            )[0]
        if defer_shading:
            # The caller shades later with apply_shading (the tracking loop
            # replays shading + loss as one CUDA graph, D130).
            render_pkg["gbuffer"] = gbuffer
            return render_pkg
        apply_shading(render_pkg, gbuffer, shader, viewpoint_camera)
    return render_pkg


def apply_shading(render_pkg, gbuffer, shader, viewpoint_camera):
    """Second half of render() with a shader: shades the G-buffer and fills
    radiance_linear, the shader's maps, albedo and the sRGB render into
    render_pkg (in place, also returned)."""
    rendered_image = gbuffer["albedo"]
    shaded = shader(gbuffer, viewpoint_camera)
    radiance_linear = shaded["radiance_linear"]
    if radiance_linear.shape != rendered_image.shape:
        raise ValueError(
            f"shader returned shape {tuple(radiance_linear.shape)}, "
            f"expected {tuple(rendered_image.shape)}"
        )
    clash = (set(shaded) - {"radiance_linear"}) & (set(render_pkg) | {"albedo"})
    if clash:
        raise ValueError(f"shader output keys {sorted(clash)} clash with render_pkg")
    render_pkg.update(shaded)  # radiance_linear (3,H,W) linear + aux maps
    render_pkg["albedo"] = rendered_image  # (3,H,W) linear, opacity-weighted
    render_pkg["render"] = linear2sRGB(radiance_linear)
    return render_pkg
