"""The offline colour refinement as one CUDA graph per iteration
(Results.color_refinement_graph, docs/DECISIONS.md D136).

BackEnd.color_refinement runs 26000 iterations: render one random keyframe,
L1 + D-SSIM, backward, Adam. Eagerly the CPU launches the kernels slower than
the GPU runs them. Here the iteration is recorded once on a "slot" camera and
replayed; before each replay the chosen keyframe is copied into the slot:
its pose and the constants of its observation (utils/perf.frame_constant:
observed image in the loss space, masks, sensor normals, ...), all at fixed
addresses. The keyframe is chosen as before (Python random).

Only when the refinement leaves the geometry alone (colour groups f_dc,
f_rest; the light model refines f_dc, D89): then each keyframe renders the
same number of Gaussian instances at every iteration, measured once at the
start, and the rasterizer runs with a fixed capacity above all of them.

Differences to the eager loop, all at the float rounding level: fused Adam
(state carried over from mapping), and with compile_shading the shading +
loss compiled into fused kernels (CompiledShadeLoss, D134)."""
import copy
import math

import torch
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from utils.config_utils import require_key
from utils.graphed_iteration import GBUFFER_KEYS, CompiledShadeLoss
from utils.logging_utils import Log
from utils.slam_utils import get_loss_refinement

WHERE = "Results.color_refinement_graph"
COLOUR_GROUPS = {"f_dc", "f_rest"}


def read_refinement_graph(results_cfg):
    """The options dict, or None when disabled."""
    block = require_key(results_cfg, "color_refinement_graph", "Results")
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
    extra = set(block) - {"enabled", "raster_capacity_margin", "compile_shading"}
    if extra:
        raise ValueError(f"{WHERE}: unexpected keys {sorted(extra)}")
    margin = require_key(block, "raster_capacity_margin", WHERE)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not margin >= 1:
        raise ValueError(f"{WHERE}.raster_capacity_margin must be a number >= 1, got {margin!r}")
    compile_shading = require_key(block, "compile_shading", WHERE)
    if not isinstance(compile_shading, bool):
        raise ValueError(f"{WHERE}.compile_shading must be true or false, got {compile_shading!r}")
    return {"raster_capacity_margin": float(margin), "compile_shading": compile_shading}


def _copy_into(dst, src, key):
    """Copies a frame constant (tensor, or tuple of them) into the slot's."""
    if torch.is_tensor(dst):
        dst.copy_(src)
    elif isinstance(dst, tuple):
        for i, (d, s) in enumerate(zip(dst, src)):
            _copy_into(d, s, (key, i))
    elif dst != src:
        raise ValueError(f"{WHERE}: frame constant {key!r} differs between keyframes: {dst!r} / {src!r}")


def _clone(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    return value


class GraphedRefinement:
    def __init__(self, config, gaussians, shader, pipe, background, viewpoints, refined, lambda_dssim, opts):
        if not set(refined) <= COLOUR_GROUPS:
            raise ValueError(
                f"{WHERE} needs Results.color_refinement_params within {sorted(COLOUR_GROUPS)} "
                f"(the geometry fixed), got {refined}")
        if opts["compile_shading"] and shader is None:
            raise ValueError(f"{WHERE}.compile_shading needs the light model")
        self.config, self.gaussians, self.shader = config, gaussians, shader
        self.pipe, self.background, self.lambda_dssim = pipe, background, lambda_dssim
        self.opts = opts
        self.keys = list(viewpoints.keys())
        self.viewpoints = viewpoints
        groups = [g for g in gaussians.optimizer.param_groups if g["name"] in refined]
        self.params = [p for g in groups for p in g["params"]]
        # The model's Adam, as a fused capturable one with its state.
        self.optimizer = torch.optim.Adam(
            [{"params": g["params"], "lr": g["lr"], "eps": g["eps"], "betas": g["betas"]} for g in groups],
            fused=True, capturable=True)
        for p in self.params:
            old = gaussians.optimizer.state.get(p)
            if old:
                self.optimizer.state[p] = {
                    "step": torch.as_tensor(old["step"], dtype=torch.float32, device=p.device).clone(),
                    "exp_avg": old["exp_avg"].clone(), "exp_avg_sq": old["exp_avg_sq"].clone()}
        self.count = torch.zeros(1, dtype=torch.int32, device=gaussians.get_xyz.device)

    # ---- the slot camera ----

    def _loss(self, pkg, cam):
        return get_loss_refinement(self.config, pkg, cam, self.lambda_dssim)

    def _prepare(self):
        """Renders every keyframe once: fills its frame constants and gives
        its instance count; builds the slot from the first keyframe."""
        counts = []
        with torch.no_grad():
            for k in self.keys:
                cam = self.viewpoints[k]
                if cam.frame_cache is None:
                    cam.frame_cache = {}
                pkg = render(cam, self.gaussians, self.pipe, self.background, shader=self.shader,
                             raster_count=self.count)
                self._loss(pkg, cam)
                counts.append(int(self.count.item()))
        first = self.viewpoints[self.keys[0]]
        keys = set(first.frame_cache)
        for k in self.keys:
            if set(self.viewpoints[k].frame_cache) != keys:
                raise ValueError(f"{WHERE}: keyframes {self.keys[0]} and {k} have different frame constants")
        slot = copy.copy(first)  # shares the module internals; R, T and the constants are its own
        slot.R, slot.T = first.R.detach().clone(), first.T.detach().clone()
        slot.original_image = first.original_image.clone()
        slot.frame_cache = {k: _clone(v) for k, v in first.frame_cache.items()}
        slot.frame_pool = None
        slot._matrix_cache = None
        self.slot = slot
        self.capacity = math.ceil(max(counts) * self.opts["raster_capacity_margin"])
        return counts

    def _load(self, cam):
        slot = self.slot
        with torch.no_grad():
            slot.R.copy_(cam.R)
            slot.T.copy_(cam.T)
            slot.original_image.copy_(cam.original_image)
            for k, v in cam.frame_cache.items():
                _copy_into(slot.frame_cache[k], v, k)

    # ---- one iteration ----

    def _iteration(self, shade):
        g = self.gaussians
        pkg = render(self.slot, g, self.pipe, self.background, shader=self.shader,
                     defer_shading=shade is not None,
                     raster_capacity=self.capacity, raster_count=self.count)
        if shade is None:
            loss = self._loss(pkg, self.slot)
        else:
            gb = pkg["gbuffer"]
            if set(gb) != GBUFFER_KEYS:
                raise NotImplementedError(f"{WHERE}.compile_shading: extra G-buffer rasters")
            loss = shade(gb["albedo"], gb["depth"], gb["opacity"])
        grads = torch.autograd.grad(loss, self.params)
        for p, grad in zip(self.params, grads):
            p.grad = grad
        self.optimizer.step()
        with torch.no_grad():
            # Updated but unused, as in the eager loop.
            vis, radii = pkg["visibility_filter"], pkg["radii"]
            g.max_radii2D.copy_(torch.where(vis, torch.maximum(g.max_radii2D, radii), g.max_radii2D))
            torch.maximum(self.count_max, self.count, out=self.count_max)

    def run(self, iterations, choose):
        """iterations refinement steps; choose() -> index into the keyframe
        list (the eager loop's random choice)."""
        counts = self._prepare()
        self.count_max = torch.zeros_like(self.count)
        shade = None
        it = iter(tqdm(range(1, iterations + 1)))
        # First iteration eager: Adam's lazy state, kernel selection.
        next(it)
        self._load(self.viewpoints[self.keys[choose()]])
        self._iteration(None)
        if self.opts["compile_shading"]:
            with torch.no_grad():
                pkg = render(self.slot, self.gaussians, self.pipe, self.background, shader=self.shader,
                             defer_shading=True, raster_count=self.count)
            gb = pkg["gbuffer"]
            shade = CompiledShadeLoss(self.config, self.shader, self.slot, gb["albedo"], gb["depth"],
                                      gb["opacity"], loss=self._loss)
            del pkg, gb
        torch.cuda.synchronize()
        self.slot._matrix_cache = None  # the pose matrices are computed in the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self._iteration(shade)
        self.slot._matrix_cache = None
        for _ in it:
            self._load(self.viewpoints[self.keys[choose()]])
            graph.replay()
        count_max = int(self.count_max.item())
        if count_max > self.capacity:
            raise RuntimeError(f"{WHERE}: {count_max} Gaussian instances above the capacity {self.capacity}")
        Log(f"color refinement as a CUDA graph: {len(self.keys)} keyframes, "
            f"instances {min(counts)}-{max(counts)}, capacity {self.capacity}")
