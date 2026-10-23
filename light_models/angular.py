# Angular(theta) = Phi, the emission profile of the lamp, as a function of
# cos(theta), theta = angle between (light -> point) and the light axis.

import math

import torch

from light_models.registry import declare_planned, register


@register("angular", "uniform")
class Uniform:
    """Phi = 1 everywhere (no cone)."""

    def __call__(self, cos_theta):
        return torch.ones_like(cos_theta)


@register("angular", "blender_spot", numeric=("half_angle_deg", "blend"))
class BlenderSpot:
    """Blender/EEVEE spot light (docs/DECISIONS.md D14):
        Phi = smoothstep((cos theta - cos a) / ((1 - cos a) * blend)),
        smoothstep(x) = x^2 (3 - 2x) with x clamped to [0, 1],
    a = half cone angle. Smoothstep in COS theta, not in theta. Renderer
    specific: for simulated data only; real hardware uses the generic forms.
    Same formula as Replica/scripts/physics_check/common.py (spot_phi)."""

    def __init__(self, half_angle_deg, blend):
        a, b = half_angle_deg.value, blend.value
        if not isinstance(a, (int, float)) or not 0 < a < 90:
            raise ValueError(f"{half_angle_deg.name} must be in (0, 90), got {a!r}")
        if not isinstance(b, (int, float)) or not 0 < b <= 1:
            raise ValueError(f"{blend.name} must be in (0, 1], got {b!r}")
        self.half_angle_deg = half_angle_deg.tensor()
        self.blend = blend.tensor()

    def check_values(self):
        # Domain of the form; re-checked after every update of a learned
        # parameter (light_models/learning.py, D60): leaving it is an error,
        # not something to clamp silently.
        a, b = float(self.half_angle_deg.detach()), float(self.blend.detach())
        if not 0 < a < 90:
            raise ValueError(f"Light.angular.half_angle_deg left (0, 90): {a!r}")
        if not 0 < b <= 1:
            raise ValueError(f"Light.angular.blend left (0, 1]: {b!r}")

    def __call__(self, cos_theta):
        a = self.half_angle_deg.to(cos_theta.device, cos_theta.dtype) * (math.pi / 180.0)
        blend = self.blend.to(cos_theta.device, cos_theta.dtype)
        cos_a = torch.cos(a)
        x = ((cos_theta - cos_a) / ((1.0 - cos_a) * blend)).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)


# smoothstep: generic profile in THETA with theta_in / theta_out (CLAUDE.md
# section 4.2); spline / mlp: learned profiles for real hardware.
declare_planned("angular", "smoothstep", "spline", "mlp")
