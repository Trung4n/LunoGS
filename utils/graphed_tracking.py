"""Shading + tracking loss of one tracking iteration replayed as CUDA graphs
(Training.cuda_graph_tracking, docs/DECISIONS.md D130).

After the rasterizer, a tracking iteration runs the shader and the loss: a few
hundred small kernels whose launches, not their work, set the time (the CPU
launches them slower than the GPU runs them). Their forward and backward are
recorded once and then replayed with one call each. Same kernels, same values.

What a recorded graph reads must stay at the same address:
- per iteration: the G-buffer maps, R and T: copied into static inputs;
- per frame: the observation's constants (frame_constant) and the sensor
  normals go to static buffers (camera.frame_pool), filled by the frame's first
  iteration, which runs eagerly;
- the light parameters on the GPU: updated in place (light_models/device.py).
Python decisions inside the shader depend on the keyframe count (warm-ups,
D43, D121): the graphs are recorded again whenever it changes.

Not supported, raising on use: extra G-buffer rasters (per-Gaussian ambient,
D121) and the exposure affine (per-frame parameters inside the loss)."""
import torch

from gaussian_splatting.gaussian_renderer import RASTERIZER_PIXEL_OFFSET, apply_shading
from light_models import read_light_tracking
from utils.slam_utils import get_loss_tracking

GBUFFER_KEYS = {"albedo", "depth", "opacity", "pixel_offset"}


class GraphedShadeLoss:
    def __init__(self, config, shader):
        if shader is None:
            raise ValueError("Training.cuda_graph_tracking needs the light model (Light.enabled: true)")
        if read_light_tracking(config).exposure_affine:
            raise ValueError("Training.cuda_graph_tracking does not support LightTracking.exposure_affine: true")
        self.config = config
        self.shader = shader
        self.pool = {}  # static per-frame constants, see utils/perf.frame_constant
        self.key = None
        self.replay = None
        self.out_keys = None
        self.captures = 0

    def begin_frame(self, viewpoint):
        viewpoint.frame_cache = {}
        viewpoint.frame_pool = self.pool

    def end_frame(self, viewpoint):
        viewpoint.frame_cache = None
        viewpoint.frame_pool = None

    def _shade_loss(self, camera, albedo, depth, opacity):
        gbuffer = {"albedo": albedo, "depth": depth, "opacity": opacity,
                   "pixel_offset": RASTERIZER_PIXEL_OFFSET}
        pkg = {"depth": depth, "opacity": opacity}
        apply_shading(pkg, gbuffer, self.shader, camera)
        return get_loss_tracking(self.config, pkg, camera), pkg

    def step(self, render_pkg, viewpoint, key, eager):
        """Shades render_pkg (from render(..., defer_shading=True)) and returns
        (tracking loss, completed render_pkg). eager: run the ordinary code
        (a frame's first iteration fills the frame's static buffers)."""
        gbuffer = render_pkg.pop("gbuffer")
        if set(gbuffer) != GBUFFER_KEYS:
            raise NotImplementedError(
                f"Training.cuda_graph_tracking: extra G-buffer rasters {sorted(set(gbuffer) - GBUFFER_KEYS)}")
        albedo, depth, opacity = gbuffer["albedo"], gbuffer["depth"], gbuffer["opacity"]
        if eager:
            loss, pkg = self._shade_loss(viewpoint, albedo, depth, opacity)
            render_pkg.update({k: v for k, v in pkg.items() if k not in ("depth", "opacity")})
            return loss, render_pkg
        rt = (viewpoint.R.float().contiguous(), viewpoint.T.float().contiguous())
        if self.replay is None or self.key != key:
            self._capture(viewpoint, (albedo, depth, opacity) + rt)
            self.key = key
        outs = self.replay(albedo, depth, opacity, *rt)
        render_pkg.update(dict(zip(self.out_keys, outs[1:])))
        render_pkg["albedo"] = albedo
        return outs[0], render_pkg

    def _capture(self, camera, inputs):
        """Records forward and backward of shading + loss for these inputs."""
        self.replay = None
        torch.cuda.synchronize()
        static_in = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
        grad_idx = [i for i, x in enumerate(static_in) if x.requires_grad]
        out_keys = []

        def fn(albedo, depth, opacity, r, t):
            saved = camera.R, camera.T
            camera.R, camera.T = r, t
            try:
                loss, pkg = self._shade_loss(camera, albedo, depth, opacity)
            finally:
                camera.R, camera.T = saved
            keys = [k for k in pkg if k not in ("depth", "opacity", "albedo") and torch.is_tensor(pkg[k])]
            out_keys[:] = keys
            return (loss,) + tuple(pkg[k].detach() for k in keys)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm-up: lazy allocations, kernel selection
                outs = fn(*static_in)
                torch.autograd.grad(outs[0], [static_in[i] for i in grad_idx])
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        fwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd, capture_error_mode="thread_local"):
            static_out = fn(*static_in)
        static_grad_loss = torch.empty_like(static_out[0])
        bwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bwd, pool=fwd.pool(), capture_error_mode="thread_local"):
            static_grads = torch.autograd.grad(static_out[0], [static_in[i] for i in grad_idx],
                                               grad_outputs=static_grad_loss)
        n_in = len(static_in)

        class Replay(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                for s, x in zip(static_in, args):
                    if s.data_ptr() != x.data_ptr():
                        s.copy_(x)
                fwd.replay()
                outs = tuple(o.detach() for o in static_out)
                ctx.mark_non_differentiable(*outs[1:])
                return outs

            @staticmethod
            def backward(ctx, grad_loss, *unused):
                static_grad_loss.copy_(grad_loss)
                bwd.replay()
                result = [None] * n_in
                for i, g in zip(grad_idx, static_grads):
                    result[i] = g.clone()
                return tuple(result)

        self.replay = Replay.apply
        self.out_keys = list(out_keys)
        self.captures += 1
