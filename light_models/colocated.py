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
        self.cosine = build_component(
            "cosine", require_key(light_cfg, "cosine", where), "Light.cosine", params_data
        )
        # Checked against Dataset.sensor_type by light_models.build_shader.
        self.requires_sensor_depth = self.cosine.requires_sensor_depth
        self.ambient = build_component(
            "ambient", require_key(light_cfg, "ambient", where), "Light.ambient", params_data
        )

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

    def learnable_state(self):
        """Current values, plain lists / floats (sent backend -> frontend)."""
        return {path: p.tensor.detach().tolist() for path, p in self._learnable.items()}

    def load_learnable_state(self, state):
        if set(state) != set(self._learnable):
            raise ValueError(
                f"learned light state has {sorted(state)}, shader expects {sorted(self._learnable)}"
            )
        with torch.no_grad():
            for path, value in state.items():
                t = self._learnable[path].tensor
                t.copy_(torch.tensor(value, dtype=t.dtype))

    def _light_axis(self, dev, dt):
        axis = self.axis.to(dev, dt)
        if self.tilt_deg is None:
            return axis
        tilt = self.tilt_deg.to(dev, dt) * (math.pi / 180.0)
        zero = torch.zeros((), device=dev, dtype=dt)
        # exp of the skew matrix of (tilt_x, tilt_y, 0): smooth at 0, unlike
        # the sin(theta)/theta form of Rodrigues.
        skew = torch.stack([
            torch.stack([zero, zero, tilt[1]]),
            torch.stack([zero, zero, -tilt[0]]),
            torch.stack([-tilt[1], tilt[0], zero]),
        ])
        return torch.linalg.matrix_exp(skew) @ axis

    def set_keyframe_count(self, n):
        # Keyframes currently in the map, reported by FrontEnd / BackEnd
        # before each step; only components with a warm-up use it (D43).
        self.cosine.set_keyframe_count(n)

    def __call__(self, gbuffer, viewpoint_camera):
        albedo = gbuffer["albedo"]
        points, valid = surface_points(gbuffer, viewpoint_camera, self.opacity_thr)
        if points is None:
            # Empty map: nothing to shade (albedo is all zero there anyway).
            zeros = torch.zeros_like(valid, dtype=albedo.dtype)
            return {"radiance_linear": albedo * 0.0, "light_valid": valid[None],
                    "light_direct": zeros[None]}
        dev, dt = points.device, points.dtype
        t = self.t_CL.to(dev, dt).view(3, 1, 1)
        axis = self._light_axis(dev, dt).view(3, 1, 1)

        v = points - t
        dist = v.norm(dim=0)  # > 0: points lie in front of the camera
        to_light = -v / dist[None]
        cos_theta = (v * axis).sum(0) / dist

        phi = self.angular(cos_theta)
        fall = self.falloff(dist)
        cos_term, aux = self.cosine(points, valid, to_light, gbuffer, viewpoint_camera)
        k = self.intensity.to(dev, dt)
        direct = k * (phi * fall * cos_term)
        radiance = self.ambient(albedo, direct)

        out = {
            "radiance_linear": radiance,
            "light_valid": valid[None],
            "light_direct": (direct if direct.dim() == 3 else direct[None]),
        }
        for name, value in aux.items():
            out["light_" + name] = value[None]
        return out
