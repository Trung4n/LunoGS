"""The whole tracking iteration replayed as one CUDA graph
(Training.cuda_graph_iteration, docs/DECISIONS.md D133).

A tracking iteration is: rasterize, shade, loss, backward, Adam step, pose
update. Run eagerly, the CPU launches its ~1000 kernels slower than the GPU
runs them, and it waits for the GPU twice per iteration (the rasterizer's
instance count, the convergence test). Here one frame runs:

1. one eager iteration: fills the frame's constants (observation, sensor
   normals, ... in static buffers, utils/perf.frame_constant), the light
   parameters on the GPU (light_models/device.on_device) and gives the
   rasterizer's exact instance count;
2. the next iterations as replays of a recorded graph of the same function,
   testing convergence every `check_every` replays (one wait each).

Same iterations as the eager loop:
- the pose update runs on the GPU in float64 (as update_pose_host) and is
  frozen once the step norm is below the threshold, so replays after
  convergence leave the pose as the eager loop would have left it;
- the rasterizer runs with a fixed binning capacity (no instance count on the
  host); a frame where the count exceeded it is tracked again with more room;
- the map is copied into fixed-size buffers, the padding entries culled by a
  negative opacity (submodules/diff-gaussian-rasterization, D133);
- the render package returned is rendered at the pose before the last update,
  as the eager loop returns its last iteration's render.

The graph is recorded again when what it was recorded for changes: the
shader's Python state (warm-ups), the binning capacity, the map buffer size.

Only with the light model and without the exposure affine (as
cuda_graph_tracking, utils/graphed_tracking.py)."""
import math

import torch

from gaussian_splatting.gaussian_renderer import render
from light_models import read_light_tracking
from utils.config_utils import require_key
from utils.pose_utils import CONVERGED_THRESHOLD, SE3_exp
from utils.slam_utils import get_loss_tracking

WHERE = "Training.cuda_graph_iteration"


def read_graph_iteration(config):
    """The options dict, or None when disabled."""
    block = require_key(config["Training"], "cuda_graph_iteration", "Training")
    if not isinstance(block, dict):
        raise ValueError(f"{WHERE} must be a mapping with 'enabled', got {block!r}")
    enabled = require_key(block, "enabled", WHERE)
    if not isinstance(enabled, bool):
        raise ValueError(f"{WHERE}.enabled must be true or false, got {enabled!r}")
    if not enabled:
        extra = set(block) - {"enabled"}
        if extra:
            raise ValueError(f"{WHERE}: keys {sorted(extra)} given while disabled")
        return None
    keys = {"enabled", "check_every", "raster_capacity_margin", "map_capacity_margin"}
    extra = set(block) - keys
    if extra:
        raise ValueError(f"{WHERE}: unexpected keys {sorted(extra)}")
    check_every = require_key(block, "check_every", WHERE)
    if isinstance(check_every, bool) or not isinstance(check_every, int) or check_every < 1:
        raise ValueError(f"{WHERE}.check_every must be an integer >= 1, got {check_every!r}")
    opts = {"check_every": check_every}
    for name in ("raster_capacity_margin", "map_capacity_margin"):
        v = require_key(block, name, WHERE)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 1:
            raise ValueError(f"{WHERE}.{name} must be a number > 1, got {v!r}")
        opts[name] = float(v)
    if require_key(config["Training"], "cuda_graph_tracking", "Training"):
        raise ValueError(f"{WHERE} replaces Training.cuda_graph_tracking: enable only one")
    return opts


class _StaticMap:
    """Fixed-size copies of the frozen map's activated tensors, what render()
    reads from a GaussianModel. Entries past the map are padding."""

    def __init__(self):
        self.size = 0
        self.n = 0
        self.get_xyz = None

    def load(self, gaussians, margin):
        """Copies the map in; True when the buffers were reallocated."""
        src = {
            "get_xyz": gaussians.get_xyz,
            "get_opacity": gaussians.get_opacity,
            "get_scaling": gaussians.get_scaling,
            "get_rotation": gaussians.get_rotation,
            "get_features": gaussians.get_features,
        }
        n = src["get_xyz"].shape[0]
        grown = (
            self.get_xyz is None
            or n > self.size
            or any(getattr(self, k).shape[1:] != v.shape[1:] or getattr(self, k).dtype != v.dtype
                   for k, v in src.items())
        )
        with torch.no_grad():
            if grown:
                self.size = math.ceil(n * margin)
                for k, v in src.items():
                    buf = torch.zeros((self.size,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                    setattr(self, k, buf)
                # Finite padding: unit scale, identity rotation.
                self.get_scaling.fill_(1.0)
                self.get_rotation[:, 0] = 1.0
            for k, v in src.items():
                getattr(self, k)[:n].copy_(v.detach())
            # Padding is skipped by the rasterizer (negative opacity).
            self.get_opacity[n:] = -1.0
        self.n = n
        self.active_sh_degree = gaussians.active_sh_degree
        self.max_sh_degree = gaussians.max_sh_degree
        return grown


class GraphedTrackingLoop:
    def __init__(self, config, shader, opts):
        if shader is None:
            raise ValueError(f"{WHERE} needs the light model (Light.enabled: true)")
        if read_light_tracking(config).exposure_affine:
            raise ValueError(f"{WHERE} does not support LightTracking.exposure_affine: true")
        self.config = config
        self.shader = shader
        self.opts = opts
        lr = config["Training"]["lr"]
        dev = torch.device("cuda", torch.cuda.current_device())
        f32 = dict(device=dev, dtype=torch.float32)
        self.R = torch.eye(3, **f32)
        self.T = torch.zeros(3, **f32)
        self.R_last = torch.eye(3, **f32)
        self.T_last = torch.zeros(3, **f32)
        # Parameters: they stand in for the Camera's (an nn.Module).
        self.rot = torch.nn.Parameter(torch.zeros(3, **f32))
        self.trans = torch.nn.Parameter(torch.zeros(3, **f32))
        self.params = [self.rot, self.trans]
        # The frontend's Adam (FrontEnd.tracking), state on the GPU.
        self.optimizer = torch.optim.Adam(
            [{"params": [self.rot], "lr": lr["cam_rot_delta"]},
             {"params": [self.trans], "lr": lr["cam_trans_delta"]}],
            fused=True, capturable=True)
        self.done = torch.zeros((), device=dev, dtype=torch.bool)
        self.iters = torch.zeros((), device=dev, dtype=torch.int32)
        self.count = torch.zeros(1, device=dev, dtype=torch.int32)
        self.count_max = torch.zeros(1, device=dev, dtype=torch.int32)
        self.eye4 = torch.eye(4, device=dev, dtype=torch.float64)
        self.map = _StaticMap()
        self.pool = {}  # static per-frame constants, see utils/perf.frame_constant
        self.capacity = 0
        self.graph = None
        self.key = None
        # Statistics for the timing log.
        self.captures = 0
        self.reruns = 0
        self.iterations = 0

    # ---- per frame ----

    def begin_frame(self, viewpoint):
        viewpoint.frame_cache = {}
        viewpoint.frame_pool = self.pool

    def end_frame(self, viewpoint):
        viewpoint.frame_cache = None
        viewpoint.frame_pool = None

    def _reset(self, R0, T0):
        with torch.no_grad():
            self.R.copy_(R0)
            self.T.copy_(T0)
            self.R_last.copy_(self.R)
            self.T_last.copy_(self.T)
            self.rot.zero_()
            self.trans.zero_()
            self.done.zero_()
            self.iters.zero_()
            self.count_max.zero_()
            # A fresh Adam per frame, as FrontEnd.tracking (in place: the
            # graph holds these tensors).
            for state in self.optimizer.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        v.zero_()

    def _iteration(self, cam, pipe, background, capacity):
        """One tracking iteration on the static state (eager or recorded)."""
        pkg = render(cam, self.map, pipe, background, shader=self.shader,
                     raster_capacity=capacity, raster_count=self.count)
        loss = get_loss_tracking(self.config, pkg, cam)
        grads = torch.autograd.grad(loss, self.params)
        for p, g in zip(self.params, grads):
            p.grad = g
        self.optimizer.step()
        with torch.no_grad():
            active = ~self.done
            # update_pose_host on the GPU: float64, then float32.
            tau = torch.cat([self.trans, self.rot]).double()
            w2c = self.eye4.clone()
            w2c[:3, :3] = self.R.double()
            w2c[:3, 3] = self.T.double()
            new = SE3_exp(tau) @ w2c
            self.R_last.copy_(torch.where(active, self.R, self.R_last))
            self.T_last.copy_(torch.where(active, self.T, self.T_last))
            self.R.copy_(torch.where(active, new[:3, :3].float(), self.R))
            self.T.copy_(torch.where(active, new[:3, 3].float(), self.T))
            self.iters.add_(active.int())
            self.done.logical_or_(tau.norm() < CONVERGED_THRESHOLD)
            torch.maximum(self.count_max, self.count, out=self.count_max)
            self.rot.zero_()
            self.trans.zero_()

    def _capture(self, cam, pipe, background):
        self.graph = None
        torch.cuda.synchronize()
        cam._matrix_cache = None  # the pose matrices must be computed in the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self._iteration(cam, pipe, background, self.capacity)
        cam._matrix_cache = None
        self.graph = graph
        self.captures += 1

    def track(self, viewpoint, gaussians, pipe, background, R0, T0, max_iters):
        """Tracks viewpoint from (R0, T0); sets its pose and returns the
        render package at the pose before the last update (see module doc)."""
        o = self.opts
        saved = (viewpoint.R, viewpoint.T, viewpoint.cam_rot_delta, viewpoint.cam_trans_delta)
        try:
            viewpoint.R, viewpoint.T = self.R, self.T
            viewpoint.cam_rot_delta, viewpoint.cam_trans_delta = self.rot, self.trans
            viewpoint._matrix_cache = None
            grown = self.map.load(gaussians, o["map_capacity_margin"])
            while True:
                self._reset(R0, T0)
                self._iteration(viewpoint, pipe, background, 0)  # eager, exact count
                done, count = self.done.item(), int(self.count.item())
                key = (self.shader.graph_state(), self.map.size, viewpoint.image_height, viewpoint.image_width)
                remaining = max_iters - 1
                if not done and remaining > 0:
                    if self.graph is None or grown or key != self.key or count > self.capacity:
                        if count > self.capacity:
                            self.capacity = math.ceil(count * o["raster_capacity_margin"])
                        self._capture(viewpoint, pipe, background)
                        self.key, grown = key, False
                    while remaining > 0:
                        k = min(o["check_every"], remaining)
                        for _ in range(k):
                            self.graph.replay()
                        remaining -= k
                        if self.done.item():
                            break
                count_max = int(self.count_max.item())
                if count_max <= max(self.capacity, count):
                    break
                # Some replay overflowed the binning capacity: dropped
                # instances, this frame is tracked again with more room.
                self.capacity = math.ceil(count_max * o["raster_capacity_margin"])
                self.graph = None
                self.reruns += 1
            self.iterations = int(self.iters.item())
            R, T = self.R.clone(), self.T.clone()
            R_last, T_last = self.R_last.clone(), self.T_last.clone()
        finally:
            viewpoint.R, viewpoint.T, viewpoint.cam_rot_delta, viewpoint.cam_trans_delta = saved
            viewpoint._matrix_cache = None
        # The eager loop's last render: at the pose before the last update.
        viewpoint.update_RT(R_last, T_last)
        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipe, background, shader=self.shader)
            loss = get_loss_tracking(self.config, render_pkg, viewpoint)
        viewpoint.update_RT(R, T)
        return loss, render_pkg
