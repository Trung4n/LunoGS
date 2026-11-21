# The co-located light shader (CLAUDE.md section 4.1): a lamp rigidly
# mounted on the camera, all lighting geometry in the camera frame.
#
#   x_C      = surface point from the G-buffer (gbuffer.py)
#   v        = x_C - t_CL,  d_l = |v|,  l = -v / d_l      (point -> light)
#   cos th   = (v / d_l) . a_L,  a_L = R_CL e_z           (light axis)
#   direct   = Intensity * Angular(cos th) * Falloff(d_l) * Cosine(n, l)
#   I_hat    = Ambient.combine(A, direct)
#
# Every form and number comes from the Light config and the per-scene params
# file (ParamSpec); nothing about the lamp is written here. The pose
# dependence is only through the rasterized maps, so gradients reach the
# camera pose through the rasterizer as before; the light parameters are
# never optimized in tracking (they are fixed CPU tensors here).

import math

import torch

# Imported for their @register side effect: the component forms must be in
# the registry before build_component looks them up.
import light_models.ambient  # noqa: F401
import light_models.angular  # noqa: F401
import light_models.cosine  # noqa: F401
import light_models.falloff  # noqa: F401
import light_models.normals  # noqa: F401
from light_models.device import on_device
from light_models.gbuffer import surface_points
from light_models.learning import make_learnable, read_schedule
from light_models.params import resolve_param
from light_models.registry import build_component
from utils.config_utils import require_key


def _vector(spec, shape):
    t = spec.tensor()
    if tuple(t.shape) != shape:
        raise ValueError(f"{spec.name} must have shape {shape}, got {tuple(t.shape)}")
    return t


class ColocatedShader:
    def __init__(self, light_cfg, params_data):
        where = "Light"
        intensity = resolve_param(
            require_key(light_cfg, "intensity", where), "Light.intensity", params_data
        )
        k = intensity.tensor()
        if k.dim() == 1 and k.shape[0] == 3:
            k = k.view(3, 1, 1)
        elif k.dim() != 0:
            raise ValueError(f"Light.intensity must be a number or 3 numbers, got {intensity.value!r}")
        self.intensity = k

        pose = require_key(light_cfg, "pose", where)
        t_spec = resolve_param(require_key(pose, "t_CL", "Light.pose"), "Light.pose.t_CL", params_data)
        self.t_CL = _vector(t_spec, (3,))
        r_cl_spec = resolve_param(require_key(pose, "R_CL", "Light.pose"), "Light.pose.R_CL", params_data)
        r_cl = _vector(r_cl_spec, (3, 3))
        # a_L = R_CL e_z, normalized (R_CL from a fit is only ~orthonormal).
        axis = r_cl.detach()[:, 2]
        self.axis = axis / axis.norm()
        # Learnable R_CL (D60) = a 2-DoF tilt of the light axis, rotation
        # vector (tilt_x, tilt_y, 0) in DEGREES about the camera x and y axes,
        # starting at 0: the cone is symmetric about its axis, so a rotation
        # about the axis itself is unobservable and not learned.
        self.tilt_deg = None
        if r_cl_spec.learnable:
            self.tilt_deg = torch.zeros(2, dtype=torch.float32, requires_grad=True)

        self.falloff = build_component(
            "falloff", require_key(light_cfg, "falloff", where), "Light.falloff", params_data
        )
        self.angular = build_component(
            "angular", require_key(light_cfg, "angular", where), "Light.angular", params_data
        )
        if getattr(self.angular, "scale", None) == "free" and intensity.learnable:
            # The knots of a free-scale spline carry K x Phi (D140): a learned
            # K would bring back the K / Phi scale they remove.
            raise ValueError("Light.angular.scale: free needs a fixed Light.intensity (learnable: false)")
        self.cosine = build_component(
            "cosine", require_key(light_cfg, "cosine", where), "Light.cosine", params_data
        )
        # Checked against Dataset.sensor_type by light_models.build_shader.
        self.requires_sensor_depth = self.cosine.requires_sensor_depth
        self.ambient = build_component(
            "ambient", require_key(light_cfg, "ambient", where), "Light.ambient", params_data
        )
        # Ambient forms with per-Gaussian values read rasterized maps from
        # the G-buffer and return auxiliary outputs (D121).
        self._ambient_maps = hasattr(self.ambient, "extra_rasters")
        # Ambient forms that are a field over the scene read the surface
        # points in the world frame (D126).
        self._ambient_field = getattr(self.ambient, "world_field", False)

        gb = require_key(light_cfg, "gbuffer", where)
        thr = require_key(gb, "opacity_thr", "Light.gbuffer")
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not 0 < thr <= 1:
            raise ValueError(f"Light.gbuffer.opacity_thr must be in (0, 1], got {thr!r}")
        extra = set(gb) - {"opacity_thr"}
        if extra:
            raise ValueError(f"Light.gbuffer: unexpected keys {sorted(extra)}")
        self.opacity_thr = float(thr)

        # Learnable light parameters (light_models/learning.py, D60).
        learnable = {}

        def add(path, tensor, spec, check=None, project=None):
            learnable[path] = make_learnable(path, tensor, spec, check, project)

        if intensity.learnable:
            add("intensity", self.intensity, intensity)
        if t_spec.learnable:
            add("pose.t_CL", self.t_CL, t_spec)
        if self.tilt_deg is not None:
            add("pose.R_CL", self.tilt_deg, r_cl_spec)
        for prefix, comp in (("falloff", self.falloff), ("angular", self.angular),
                             ("ambient", self.ambient)):
            for name, spec in comp._param_specs.items():
                if spec.learnable:
                    add(f"{prefix}.{name}", getattr(comp, name), spec,
                        getattr(comp, "check_values", None),
                        getattr(comp, f"project_{name}", None))
            # Learnable tensors that are not ParamSpecs (e.g. the spline's
            # knot values, D67): the component builds their LearnableParam.
            extra = getattr(comp, "learnable_extra", None)
            if extra is not None:
                learnable.update(extra(prefix))
        read_schedule(light_cfg, learnable)
        self._learnable = learnable

    # ---- learnable parameters (D60) ----

    def learnable_params(self):
        """dict path -> LearnableParam (empty when nothing is learnable)."""
        return self._learnable

    def set_learning(self, on):
        """Gradients on the learnable parameters: on in the backend (mapping),
        off in the frontend (tracking never optimizes the light)."""
        for p in self._learnable.values():
            p.tensor.requires_grad_(on)

    def own_learnable_storage(self):
        """Gives the learnable tensors private storage, same tensor objects.
        torch.multiprocessing moves CPU tensors handed to a spawned process
        into SHARED memory, so the backend's and the frontend's copies would
        otherwise be one buffer: the frontend loading a (slightly older)
        received state would then roll back what the backend just learned.
        Called by the backend before it starts learning (D60)."""
        with torch.no_grad():
            for p in self._learnable.values():
                p.tensor.data = p.tensor.data.clone()
        # Same for the non-learned state sent with the values (vertex keys,
        # D143): the frontend loading a received state must not overwrite
        # the backend's newer allocation.
        own = getattr(self.ambient, "own_storage", None)
        if own is not None:
            own()

    def _aux_state(self):
        """{path: value} of the components' non-learned state that goes with
        the learned values (the vertex keys of position_grid_sparse, D143)."""
        get = getattr(self.ambient, "aux_state", None)
        return {f"ambient.{k}": v for k, v in get().items()} if get is not None else {}

    def learnable_state(self):
        """Current values, plain lists / floats (sent backend -> frontend)."""
        state = {path: p.tensor.detach().tolist() for path, p in self._learnable.items()}
        state.update(self._aux_state())
        return state

    def learned_for_log(self):
        """(state for light_learned.csv, {path: array} of the parameters with a
        summary, and the components' aux state, for light_learned_large.npz,
        D126, D143)."""
        import numpy as np

        state, large = {}, {}
        for path, p in self._learnable.items():
            if p.summary is None:
                state[path] = p.tensor.detach().tolist()
            else:
                large[path] = p.tensor.detach().cpu().numpy()
                state.update({f"{path}.{k}": v for k, v in p.summary().items()})
        large.update({path: np.asarray(v, dtype=np.int64) for path, v in self._aux_state().items()})
        return state, large

    def load_learnable_state(self, state):
        aux = set(self._aux_state())
        if set(state) != set(self._learnable) | aux:
            raise ValueError(
                f"learned light state has {sorted(state)}, shader expects "
                f"{sorted(set(self._learnable) | aux)}"
            )
        if aux:
            self.ambient.load_aux_state({path[len("ambient."):]: state[path] for path in aux})
        with torch.no_grad():
            for path in self._learnable:
                t = self._learnable[path].tensor
                t.copy_(torch.tensor(state[path], dtype=t.dtype))

    def allocates_field(self):
        """True when the ambient field allocates its vertices from the
        surface seen (position_grid_sparse, D143)."""
        return hasattr(self.ambient, "allocate")

    def observe_surface(self, gbuffer, viewpoint_camera, companions=()):
        """Allocates the ambient field's vertices around the shading points of
        a rasterized G-buffer (backend, one keyframe): every pixel the map
        covers (opacity > 0), the trusted ones at their depth and the others
        at the fallback depth, i.e. every point where shading reads the field
        with a nonzero albedo, hence a gradient. Uncovered pixels (albedo 0)
        are left out. With only the trusted pixels, 22% of the vertices the
        dense grid learns were missing (ambvar_hard, D143). companions: the
        optimizer state of the field's values, moved with them when the
        layout changes (position_grid_growing, D144). Returns the number of
        new vertices."""
        points, _ = surface_points(gbuffer, viewpoint_camera, self.opacity_thr)
        if points is None:
            return 0
        covered = gbuffer["opacity"][0] > 0
        world = self._to_world(points, viewpoint_camera)
        return self.ambient.allocate(world[:, covered], companions)

    @staticmethod
    def _to_world(points, camera):
        # x_C = R x_W + T (Camera.R, Camera.T: world to camera).
        dev, dt = points.device, points.dtype
        rot = camera.R.to(dev, dt)
        trans = camera.T.to(dev, dt).view(3, 1, 1)
        return (rot.T @ (points - trans).flatten(1)).view_as(points)

    def _light_axis(self, dev, dt):
        axis = on_device(self.axis, dev, dt)
        if self.tilt_deg is None:
            return axis
        if self.tilt_deg.requires_grad and torch.is_grad_enabled():
            return self._tilted_axis(axis, dev, dt)  # mapping: differentiable
        # Without gradient (tracking) the tilted axis only changes when the
        # backend sends new values: kept in one buffer, computed again when
        # the tilt changes (its version counter). Same values; matrix_exp
        # waits for the GPU and cannot run inside a CUDA graph (D135).
        key = (self.tilt_deg._version, str(dev), dt)
        cached = getattr(self, "_axis_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        with torch.no_grad():
            value = self._tilted_axis(axis, dev, dt)
        if cached is not None and cached[0][1:] == key[1:]:
            cached[1].copy_(value)
            value = cached[1]
        self._axis_cache = (key, value)
        return value

    def _tilted_axis(self, axis, dev, dt):
        # on_device: the GPU copy stays at one address and follows the values
        # the backend sends (D135).
        tilt = on_device(self.tilt_deg, dev, dt) * (math.pi / 180.0)
        zero = torch.zeros((), device=dev, dtype=dt)
        # exp of the skew matrix of (tilt_x, tilt_y, 0): smooth at 0, unlike
        # the sin(theta)/theta form of Rodrigues.
        skew = torch.stack([
            torch.stack([zero, zero, tilt[1]]),
            torch.stack([zero, zero, -tilt[0]]),
            torch.stack([-tilt[1], tilt[0], zero]),
        ])
        return torch.linalg.matrix_exp(skew) @ axis

    def graph_state(self):
        """Every Python value that decides which kernels shading runs: a
        recorded CUDA graph stays valid while this tuple is unchanged
        (utils/graphed_tracking.py, D131). Components without such state
        contribute None."""
        state = []
        for comp in (self.falloff, self.angular, self.cosine, self.ambient):
            get = getattr(comp, "graph_state", None)
            state.append(get() if get is not None else None)
        return tuple(state)

    def set_keyframe_count(self, n):
        # Keyframes currently in the map, reported by FrontEnd / BackEnd
        # before each step; only components with a warm-up use it (D43, and
        # the per-Gaussian ambient residual, D121).
        self.cosine.set_keyframe_count(n)
        if self._ambient_maps:
            self.ambient.set_keyframe_count(n)

    # ---- per-Gaussian light parameters stored in the map (D121) ----

    def per_gaussian_params(self):
        """{name: (values per Gaussian, lr)} the GaussianModel must hold
        (GaussianModel.declare_extras), empty for the global forms."""
        return self.ambient.per_gaussian_params() if self._ambient_maps else {}

    def extra_raster_keys(self):
        """G-buffer keys the shader reads besides albedo, depth, opacity."""
        return self.ambient.extra_raster_keys() if self._ambient_maps else ()

    def extra_rasters(self, gaussians):
        """{gbuffer key: (N,3) per-Gaussian colours} that render() rasterizes
        in extra passes before calling the shader."""
        return self.ambient.extra_rasters(gaussians) if self._ambient_maps else {}

    # ---- normals rendered from the map (shortest_axis, D149) ----

    def _gaussian_normals(self):
        normals = getattr(self.cosine, "normals", None)
        return normals if hasattr(normals, "frame_rasters") else None

    def uses_gaussian_normals(self):
        """True when the shading reads normals rendered from the Gaussians'
        shapes (the backend then requires flat Gaussians)."""
        return self._gaussian_normals() is not None

    def tracking_refresh_every(self):
        """Tracking iterations between re-renders of the frame rasters at the
        current pose (shortest_axis.tracking_refresh_every), or None: rendered
        once per frame, at the initial pose."""
        normals = self._gaussian_normals()
        return None if normals is None else normals.tracking_refresh_every

    def frame_raster_gradient(self):
        """'all' or 'rotation' (shortest_axis.raster_gradient): what the
        frame rasters pass gradient to besides their colours."""
        return self._gaussian_normals().raster_gradient

    def frame_rasters(self, gaussians, camera):
        """{gbuffer key: function -> (N,3) colours} that render() rasterizes
        at every render, except while the map is frozen for the camera (the
        frame being tracked, the keyframes of the graphed refinement;
        utils/perf.map_frozen), where it rasterizes them once per frame
        without gradient (utils/perf.frame_constant). Empty during
        the Lambert warm-up, which reads no normal."""
        normals = self._gaussian_normals()
        if normals is None or not getattr(self.cosine, "active", True):
            return {}
        return normals.frame_rasters(gaussians, camera)

    def initial_orientation(self, camera, depth):
        """(normals (3,H,W) camera frame, valid (H,W), flatness) for the new
        Gaussians of a keyframe placed at `depth` (H,W tensor), or None when
        the shading does not read the Gaussians' shapes."""
        normals = self._gaussian_normals()
        if normals is None:
            return None
        n, ok = normals.initial_orientation(camera, depth)
        return n, ok, normals.init_flatness

    def __call__(self, gbuffer, viewpoint_camera):
        albedo = gbuffer["albedo"]
        points, valid = surface_points(gbuffer, viewpoint_camera, self.opacity_thr)
        if points is None:
            # Empty map: nothing to shade (albedo is all zero there anyway).
            zeros = torch.zeros_like(valid, dtype=albedo.dtype)
            return {"radiance_linear": albedo * 0.0, "light_valid": valid[None],
                    "light_direct": zeros[None]}
        dev, dt = points.device, points.dtype
        t = on_device(self.t_CL, dev, dt).view(3, 1, 1)
        axis = self._light_axis(dev, dt).view(3, 1, 1)

        v = points - t
        dist = v.norm(dim=0)  # > 0: points lie in front of the camera
        to_light = -v / dist[None]
        cos_theta = (v * axis).sum(0) / dist

        phi = self.angular(cos_theta)
        fall = self.falloff(dist)
        cos_term, aux = self.cosine(points, valid, to_light, gbuffer, viewpoint_camera)
        k = on_device(self.intensity, dev, dt)
        direct = k * (phi * fall * cos_term)
        if self._ambient_field:
            world = self._to_world(points, viewpoint_camera)
            radiance, amb = self.ambient(albedo, direct, world)
        elif self._ambient_maps:
            radiance, amb = self.ambient(albedo, direct, gbuffer)
        else:
            radiance, amb = self.ambient(albedo, direct), {}

        out = {
            "radiance_linear": radiance,
            "light_valid": valid[None],
            "light_direct": (direct if direct.dim() == 3 else direct[None]),
        }
        # Scalar regularizers, added by get_loss_mapping only (never in
        # tracking or the offline refinement): the ambient's L1 (D121) and the
        # supervision of normals rendered from the map (D149).
        penalties = [d["map_penalty"] for d in (aux, amb) if "map_penalty" in d]
        for name, value in aux.items():
            if name != "map_penalty":
                out["light_" + name] = value[None]
        if "ambient_residual" in amb:
            out["light_ambient_residual"] = amb["ambient_residual"]  # (3,H,W)
        if penalties:
            out["light_map_penalty"] = penalties[0] if len(penalties) == 1 else penalties[0] + penalties[1]
        return out
