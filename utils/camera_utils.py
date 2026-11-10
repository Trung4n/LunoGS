import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from light_models import read_min_gradient
from utils.slam_utils import image_gradient, image_gradient_mask


_BLOCKS = {}


def _block_layout(h, w, row, col, device):
    """For a (h, w) image cut into row x col blocks at round(r * h / row),
    round(c * w / col): [(block ids, flat pixel indices (n_blocks, block
    size)) per block size], and the (h, w) block id of every pixel."""
    key = (h, w, row, col, str(device))
    if key not in _BLOCKS:
        row_edges = [round(r * h / row) for r in range(row + 1)]
        col_edges = [round(c * w / col) for c in range(col + 1)]
        block_of_pixel = torch.empty(h, w, dtype=torch.long)
        groups = {}
        for r in range(row):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(col):
                c0, c1 = col_edges[c], col_edges[c + 1]
                b = r * col + c
                block_of_pixel[r0:r1, c0:c1] = b
                ys, xs = torch.meshgrid(torch.arange(r0, r1), torch.arange(c0, c1), indexing="ij")
                groups.setdefault((r1 - r0, c1 - c0), []).append((b, (ys * w + xs).flatten()))
        layout = [(torch.tensor([b for b, _ in g], device=device),
                   torch.stack([idx for _, idx in g]).to(device)) for g in groups.values()]
        _BLOCKS[key] = (layout, block_of_pixel.to(device))
    return _BLOCKS[key]


def _block_medians(image, row, col):
    """((row * col,) median of each block, (h, w) block id of each pixel) for a
    (h, w) image; torch.median of each block (the lower middle value)."""
    h, w = image.shape
    layout, block_of_pixel = _block_layout(h, w, row, col, image.device)
    flat = image.flatten()
    medians = torch.empty(row * col, dtype=image.dtype, device=image.device)
    for ids, idx in layout:
        medians[ids] = flat[idx].median(dim=1).values
    return medians, block_of_pixel


class Camera(nn.Module):
    """One video frame plus everything needed to render/optimize it.

    A Camera bundles: the GT image/depth for this frame, its intrinsics
    (shared by every frame from the same dataset) and its extrinsics (pose),
    which is the thing tracking actually estimates. It subclasses nn.Module
    only so its pose-delta/exposure tensors are proper nn.Parameter objects
    (get gradients, move with .to(device), etc.) — frontend/backend build
    their own torch.optim.Adam over a hand-picked subset of these parameters
    rather than calling camera.parameters(), so there is no forward().

    One Camera instance is created per dataset frame (see
    init_from_dataset, called from FrontEnd.run()) and lives in
    FrontEnd.cameras / BackEnd.viewpoints for the rest of the run; clean()
    releases its heavy tensors if the frame is not turned into a keyframe.
    """

    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        # uid: this frame's index in the dataset. Doubles as its identity key
        # everywhere else in the pipeline (FrontEnd.cameras dict, keyframe
        # window lists, occ_aware_visibility dict, optimizer param-group names
        # like f"rot_{uid}").
        self.uid = uid
        self.device = device

        # R, T: CURRENT ESTIMATED world-to-camera pose (rotation, translation),
        # i.e. what tracking/BA is solving for. Initialized to identity here;
        # tracking() immediately overwrites it via update_RT() with the
        # previous frame's pose before optimizing. Plain tensors (not
        # nn.Parameter) because they are updated manually in pose_utils.update_pose,
        # not through autograd.
        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        # R_gt, T_gt: GROUND-TRUTH pose, constant for this frame's lifetime.
        # Only used to (a) bootstrap frame 0 / the init window and (b) as the
        # reference trajectory for ATE evaluation — never fed into the
        # tracking/mapping loss itself.
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        # original_image: GT color tensor (CHW, float in [0,1]) for this frame.
        self.original_image = color
        # depth: GT depth map (numpy array) if the sensor provides one, else None
        # (monocular) — filled in by MonocularDataset.__getitem__.
        self.depth = depth
        # grad_mask: high-image-gradient pixel mask, filled in later by
        # compute_grad_mask(); restricts the monocular tracking loss to
        # textured/edge regions.
        self.grad_mask = None

        # Intrinsics, copied from the dataset (identical for every Camera
        # built from the same dataset instance).
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        # cam_rot_delta / cam_trans_delta: the actual OPTIMIZATION VARIABLES
        # for tracking. NOT the pose itself — a small se(3) tangent-space
        # increment (axis-angle rotation + translation) around the current
        # R,T. Each tracking/mapping optimizer step: render at the current
        # R,T (the rasterizer's forward pass ignores the delta's value) ->
        # backward, where the rasterizer writes dLoss/d(delta), evaluated at
        # delta = 0, into these tensors' .grad -> Adam.step() moves them ->
        # pose_utils.update_pose() folds the delta into R,T via SE3_exp() and
        # resets it to zero. A delta that is stepped but never folded has no
        # effect at all on rendering.
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # exposure_a, exposure_b: learnable per-frame affine brightness
        # correction, image_ab = exp(a) * image + b (see get_loss_tracking /
        # get_loss_mapping). Compensates auto-exposure/white-balance changes
        # between frames so the photometric loss compares like with like.
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # Perspective projection matrix (shared across all frames of a run,
        # built once in slam_frontend/slam_gui and passed in here).
        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        # Factory used by FrontEnd.run(): pulls (image, depth, gt_pose) from
        # the dataset and pairs it with the dataset's shared intrinsics to
        # build one Camera per loop iteration.
        return Camera.init_from_data(dataset, idx, projection_matrix, dataset[idx])

    @staticmethod
    def init_from_data(dataset, idx, projection_matrix, data):
        # Same as init_from_dataset, with dataset[idx] already read (the
        # frontend reads the next frame ahead, utils/perf.py, D128).
        gt_color, gt_depth, gt_pose = data
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        # Factory used by the GUI viewer to build a throwaway Camera for an
        # arbitrary user-controlled viewpoint (no GT image/depth attached —
        # this Camera is only ever used to render, never to track/optimize).
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    def _pose_matrices(self):
        # (world_view_transform, full_proj_transform, camera_center), cached
        # for the current R, T, projection_matrix (docs/DECISIONS.md D128):
        # a render reads them several times and tracking renders the same pose
        # until update_pose replaces R, T. The key holds the tensors
        # themselves (not their id(), which Python reuses) and their version
        # counters (in-place writes).
        key = (self.R, self.R._version, self.T, self.T._version, self.projection_matrix)
        cache = getattr(self, "_matrix_cache", None)
        if cache is None or any(a is not b if torch.is_tensor(a) else a != b
                                for a, b in zip(cache[0], key)):
            wv = getWorld2View2(self.R, self.T).transpose(0, 1)
            fp = wv.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
            # inverse(world_view)[3, :3] for a rigid pose: -R^T T, read from
            # wv (float32 like before, even when R, T are float64: the first
            # frame gets the dataset's float64 ground-truth pose).
            cc = -(wv[:3, :3] @ wv[3, :3])
            cache = (key, (wv, fp, cc))
            self._matrix_cache = cache
        return cache[1]

    @property
    def world_view_transform(self):
        # 4x4 view matrix (world -> camera), transposed to match the
        # rasterizer's expected (row-vector-on-the-left) matrix convention.
        return self._pose_matrices()[0]

    @property
    def full_proj_transform(self):
        # Combined view + projection matrix (world -> clip space), what the
        # rasterizer actually uses to place Gaussians on screen.
        return self._pose_matrices()[1]

    @property
    def camera_center(self):
        # Camera position in world space (inverse of the view matrix). Used by
        # the renderer for view-dependent SH color evaluation.
        return self._pose_matrices()[2]

    def update_RT(self, R, t):
        # Overwrite the current pose estimate outright (as opposed to the
        # incremental cam_rot_delta/cam_trans_delta above). Called to seed a
        # new frame with the previous frame's pose, to apply a converged
        # tracking delta (pose_utils.update_pose), and to apply poses synced
        # back from the backend after bundle adjustment.
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        # Build a mask that marks "informative" (high-texture/edge) pixels,
        # used to restrict the monocular photometric loss to regions where
        # gradients actually carry pose information (flat/textureless areas
        # are ambiguous and would just add noise). Called once per frame,
        # right after it's created — not recomputed during optimization.
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            # Replica-specific variant: threshold locally per 32x32 grid cell
            # (each cell's own median) instead of one global median, since
            # Replica renders have more uniform texture across the frame and
            # a single global threshold would miss weakly-textured regions.
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            # Rounded edges so the cells tile the whole image even when h, w
            # are not multiples of row, col.
            # Each pixel against its own block's median, as the per-block loop
            # did; blocks of the same size are reduced together (one median
            # per size instead of 1024 small ones, docs/DECISIONS.md D128).
            medians, block_of_pixel = _block_medians(img_grad_intensity[0], row, col)
            self.grad_mask = (img_grad_intensity > (medians * multiplier)[block_of_pixel][None])
        else:
            # Generic case: single global-median threshold over the whole image.
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        # Optional absolute floor on the gradient (LightTracking.min_gradient,
        # docs/DECISIONS.md D37): the relative thresholds above select noise
        # below one gray level in dark regions. Disabled = original MonoGS.
        min_gradient = read_min_gradient(config)
        if min_gradient.enabled:
            self.grad_mask = self.grad_mask & (
                img_grad_intensity * 255.0 >= min_gradient.threshold_8bit
            )

    def to_device(self, device):
        """Moves the frame's tensors to `device` (the backend on another GPU,
        slam.py --backend_gpu, docs/DECISIONS.md D128). The pose deltas and
        exposure stay leaf Parameters (their optimizers are built after)."""
        self.device = device
        for name in ("original_image", "grad_mask", "R", "T", "R_gt", "T_gt", "projection_matrix"):
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        for name in ("cam_rot_delta", "cam_trans_delta", "exposure_a", "exposure_b"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, nn.Parameter(value.detach().to(device), requires_grad=value.requires_grad))
        self.frame_cache = None
        self._matrix_cache = None
        return self

    def clean(self):
        # Explicitly drop references to this frame's heavy CUDA tensors
        # (image/depth/mask/learnable params). Called by FrontEnd.cleanup()
        # right after tracking, for every frame that is NOT made a keyframe
        # (only its R/T are still needed: to seed the next frame's tracking
        # and for evaluation). Frames stay in FrontEnd.cameras for the whole
        # run, so without this their GPU memory would never be freed.
        # Keyframes are never cleaned on the frontend side.
        self.original_image = None
        self.depth = None
        self.grad_mask = None
        self.frame_cache = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None


def build_projection_matrix(fx, fy, cx, cy, width, height, device):
    # The (transposed) projection matrix shared by every Camera of a run.
    # Moved out of FrontEnd.run() unchanged so tools/verify/ builds cameras
    # exactly like SLAM does. znear/zfar are MonoGS's original hard-coded
    # clipping planes (not light-model values).
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        W=width,
        H=height,
    ).transpose(0, 1)
    return projection_matrix.to(device=device)
