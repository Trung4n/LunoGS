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
import time

import torch

from gaussian_splatting.gaussian_renderer import RASTERIZER_PIXEL_OFFSET, apply_shading, render
from light_models import read_light_tracking
from utils.config_utils import require_key
from utils.logging_utils import Log
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
    keys = {"enabled", "check_every", "raster_capacity_margin", "map_capacity_margin", "compile_shading"}
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
    compile_shading = require_key(block, "compile_shading", WHERE)
    if not isinstance(compile_shading, bool):
        raise ValueError(f"{WHERE}.compile_shading must be true or false, got {compile_shading!r}")
    opts["compile_shading"] = compile_shading
    if require_key(config["Training"], "cuda_graph_tracking", "Training"):
        raise ValueError(f"{WHERE} replaces Training.cuda_graph_tracking: enable only one")
    return opts


GBUFFER_KEYS = {"albedo", "depth", "opacity", "pixel_offset"}


def lift_constants(gm):
    """Turns the CUDA tensors a traced graph read from outside (get_attr
    constants) into extra inputs, in place. Returns them in input order: the
    caller passes the same tensor objects at every call, so the graph reads
    their current values (what a CUDA graph replay does too)."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    last = placeholders[-1]
    consts = []
    lifted = {}  # attribute name -> placeholder
    for node in list(gm.graph.nodes):
        if node.op != "get_attr":
            continue
        value = getattr(gm, node.target)
        if not (torch.is_tensor(value) and value.is_cuda):
            continue  # literals made inside the traced code stay constants
        if node.target not in lifted:
            with gm.graph.inserting_after(last):
                ph = gm.graph.placeholder(f"lifted_{len(consts)}")
            ph.meta = dict(node.meta)
            lifted[node.target] = ph
            consts.append(value)
            last = ph
        node.replace_all_uses_with(lifted[node.target])
        gm.graph.erase_node(node)
    for name in lifted:
        delattr(gm, name)
    gm.graph.lint()
    gm.recompile()
    return consts


class CompiledShadeLoss:
    """Shading + tracking loss of the frame being tracked, as a few fused
    kernels (Training.cuda_graph_iteration.compile_shading, D134).

    Eagerly, shading and loss are hundreds of small kernels, each reading and
    writing whole images: memory traffic, not arithmetic, sets their time.
    The eager code is traced once into an aten graph (make_fx). The tensors it
    reads from outside (the frame's constants in static buffers, the light
    parameters on the GPU, R and T) sit at fixed addresses, as the CUDA graph
    requires anyway, and become inputs; the graph is compiled by Inductor,
    forward and backward. Python decisions made while tracing (component
    forms, warm-ups) are fixed in the trace, exactly as in a recorded CUDA
    graph: it is traced again when the shader's graph_state changes."""

    def __init__(self, config, shader, camera, albedo, depth, opacity, loss=None):
        """loss(render_pkg, camera) -> scalar; None = the tracking loss. The
        colour refinement passes its own (utils/graphed_refinement.py, D136)."""
        from torch.fx.experimental.proxy_tensor import make_fx

        if loss is None:
            def loss(pkg, cam):
                return get_loss_tracking(config, pkg, cam)

        def shade_loss(a, d, o):
            gbuffer = {"albedo": a, "depth": d, "opacity": o, "pixel_offset": RASTERIZER_PIXEL_OFFSET}
            pkg = {"depth": d, "opacity": o}
            apply_shading(pkg, gbuffer, shader, camera)
            return loss(pkg, camera)

        example = [t.detach() for t in (albedo, depth, opacity)]
        gm = make_fx(shade_loss)(*example)
        self.consts = lift_constants(gm)
        self.fn = torch.compile(gm, dynamic=False)
        # Compile forward and backward now, outside any CUDA graph capture.
        leaves = [t.clone().requires_grad_() for t in example]
        torch.autograd.grad(self(*leaves), leaves)

    def __call__(self, albedo, depth, opacity):
        return self.fn(albedo, depth, opacity, *self.consts)


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
        self.shade = None  # CompiledShadeLoss (compile_shading)
        self.shade_key = None
        # Statistics for the timing log.
        self.captures = 0
        self.compiles = 0
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

    def _iteration(self, cam, pipe, background, capacity, shade=None):
        """One tracking iteration on the static state (eager or recorded).
        shade: CompiledShadeLoss for shading + loss, None = the eager code."""
        # Only the pose is optimized: the rasterizer's backward skips the
        # per-Gaussian colour and opacity sums (D134; view-independent
        # colours only, the light model's case).
        pkg = render(cam, self.map, pipe, background, shader=self.shader,
                     defer_shading=shade is not None,
                     raster_capacity=capacity, raster_count=self.count,
                     raster_pose_only=self.map.active_sh_degree == 0)
        if shade is None:
            loss = get_loss_tracking(self.config, pkg, cam)
        else:
            gbuffer = pkg["gbuffer"]
            if set(gbuffer) != GBUFFER_KEYS:
                raise NotImplementedError(
                    f"{WHERE}.compile_shading: extra G-buffer rasters {sorted(set(gbuffer) - GBUFFER_KEYS)}")
            loss = shade(gbuffer["albedo"], gbuffer["depth"], gbuffer["opacity"])
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

    def _compile_shading(self, cam, pipe, background):
        """CompiledShadeLoss traced on the current G-buffer (after the frame's
        eager iteration filled the frame's constants)."""
        self.shade = None
        with torch.no_grad():
            pkg = render(cam, self.map, pipe, background, shader=self.shader, defer_shading=True,
                         raster_capacity=0, raster_count=self.count)
        g = pkg["gbuffer"]
        if set(g) != GBUFFER_KEYS:
            raise NotImplementedError(
                f"{WHERE}.compile_shading: extra G-buffer rasters {sorted(set(g) - GBUFFER_KEYS)}")
        start = time.perf_counter()
        self.shade = CompiledShadeLoss(self.config, self.shader, cam, g["albedo"], g["depth"], g["opacity"])
        self.compiles += 1
        Log(f"shading + tracking loss compiled ({self.compiles}): {time.perf_counter() - start:.1f} s",
            tag="Light")

    def _capture(self, cam, pipe, background):
        self.graph = None
        torch.cuda.synchronize()
        cam._matrix_cache = None  # the pose matrices must be computed in the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self._iteration(cam, pipe, background, self.capacity, self.shade)
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
                        shade_key = key[:1] + key[2:]  # not the map buffer size
                        if o["compile_shading"] and (self.shade is None or shade_key != self.shade_key):
                            self._compile_shading(viewpoint, pipe, background)
                            self.shade_key = shade_key
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
