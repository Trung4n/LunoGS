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


def _nonincreasing_projection(values):
    """Euclidean projection of a sequence onto non-increasing sequences
    (pool adjacent violators): adjacent values that increase are merged into
    their mean until none does."""
    blocks = []  # [sum, count]
    for v in values:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return out


@register("angular", "spline",
          options=("knots_deg", "interpolation", "learn", "smoothness_weight"),
          nested={"init": "angular"})
class Spline:
    """Explicit, learnable emission profile (docs/DECISIONS.md D67): Phi given
    by its values at knots, for real lamps whose profile has no known form.

      knots_deg: angles theta of the knots, degrees; strictly increasing,
                 the first is 0, all below 90.
      interpolation: "linear_cos", linear in cos(theta) between knots (no
                 acos: its derivative is infinite on the light axis). Beyond
                 the last knot Phi keeps the last value.
      init:      another angular block (e.g. blender_spot with parameters
                 from the params file, perturbed or not), sampled at the
                 knots and divided by its value at theta = 0. No number is
                 copied into the YAML.
      learn:     {learnable: false} or {learnable: true, lr: <>, prior_weight: <>}.
      smoothness_weight: >= 0, penalty weight * sum of squared second
                 differences of the knot values (uniform knot spacing
                 assumed by this penalty).

    Constraints: Phi(0) = 1 fixed (the scale of Phi is not observable apart
    from K and the albedo, D65); the other knot values are learned and, after
    every step, PROJECTED onto non-increasing values in [0, 1] (projected
    gradient; pool adjacent violators, then the box), part of the model's
    definition, not a silent clamp. Learned path: <prefix>.knot_values.
    """

    INTERPOLATIONS = ("linear_cos",)

    def __init__(self, knots_deg, interpolation, learn, smoothness_weight, init):
        where = "Light.angular"
        if (not isinstance(knots_deg, list) or len(knots_deg) < 2
                or any(isinstance(k, bool) or not isinstance(k, (int, float)) for k in knots_deg)):
            raise ValueError(f"{where}.knots_deg must be a list of >= 2 numbers, got {knots_deg!r}")
        if knots_deg[0] != 0:
            raise ValueError(f"{where}.knots_deg must start at 0 (Phi(0) = 1), got {knots_deg[0]!r}")
        if any(b <= a for a, b in zip(knots_deg, knots_deg[1:])) or knots_deg[-1] >= 90:
            raise ValueError(f"{where}.knots_deg must be strictly increasing and below 90, got {knots_deg!r}")
        if interpolation not in self.INTERPOLATIONS:
            raise ValueError(f"{where}.interpolation must be one of {self.INTERPOLATIONS}, got {interpolation!r}")
        w = smoothness_weight
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ValueError(f"{where}.smoothness_weight must be a number >= 0, got {w!r}")
        if isinstance(init, Spline):
            raise ValueError(f"{where}.init must be another angular form, not a spline")
        if not isinstance(learn, dict) or "learnable" not in learn or not isinstance(learn["learnable"], bool):
            raise ValueError(f"{where}.learn must be a block with learnable: true|false, got {learn!r}")
        keys = {"learnable", "lr", "prior_weight"} if learn["learnable"] else {"learnable"}
        if set(learn) != keys:
            raise ValueError(f"{where}.learn must have exactly the keys {sorted(keys)}, got {sorted(learn)}")
        if learn["learnable"]:
            for name in ("lr", "prior_weight"):
                v = learn[name]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(f"{where}.learn.{name} must be a number >= 0, got {v!r}")

        cos_knots = torch.cos(torch.tensor(knots_deg, dtype=torch.float64) * (math.pi / 180.0))
        with torch.no_grad():
            v = init(cos_knots).to(torch.float64)
        if not float(v[0]) > 0:
            raise ValueError(f"{where}.init must be > 0 at theta = 0, got {float(v[0])!r}")
        v = (v / v[0]).tolist()
        if any(b > a for a, b in zip(v, v[1:])) or min(v) < 0:
            raise ValueError(f"{where}.init must be non-increasing and >= 0 at the knots, got {v!r}")
        self.knots_deg = list(knots_deg)
        self.cos_knots = cos_knots.to(torch.float32)
        self.knot_values = torch.tensor(v[1:], dtype=torch.float32, requires_grad=learn["learnable"])
        self.smoothness_weight = float(w)
        self._learn = dict(learn)

    # ---- learning hooks (light_models/learning.py, colocated.py) ----

    def learnable_extra(self, prefix):
        from light_models.learning import LearnableParam

        if not self._learn["learnable"]:
            return {}
        path = f"{prefix}.knot_values"
        return {path: LearnableParam(
            path, self.knot_values, self.knot_values.detach().clone(),
            float(self._learn["lr"]), float(self._learn["prior_weight"]),
            check=self.check_values, project=self.project,
            regularizer=self.regularizer if self.smoothness_weight > 0 else None,
        )}

    def project(self):
        v = _nonincreasing_projection(self.knot_values.detach().tolist())
        v = [min(1.0, max(0.0, x)) for x in v]
        self.knot_values.data.copy_(torch.tensor(v, dtype=self.knot_values.dtype))

    def check_values(self):
        v = [1.0] + self.knot_values.detach().tolist()
        if any(b > a for a, b in zip(v, v[1:])) or min(v) < 0:
            raise ValueError(f"Light.angular.knot_values left the non-increasing [0, 1] set: {v!r}")

    def regularizer(self):
        v = torch.cat([torch.ones(1, dtype=self.knot_values.dtype), self.knot_values])
        second = v[2:] - 2.0 * v[1:-1] + v[:-2]
        return self.smoothness_weight * (second ** 2).sum()

    def values(self):
        """(N,) Phi at the knots, Phi(0) = 1 first."""
        return torch.cat([torch.ones(1, dtype=self.knot_values.dtype), self.knot_values])

    def __call__(self, cos_theta):
        dev, dt = cos_theta.device, cos_theta.dtype
        # Ascending cos (descending theta) for bucketize.
        a = self.cos_knots.to(dev, dt).flip(0)
        va = self.values().to(dev, dt).flip(0)
        c = cos_theta.clamp(min=float(a[0]), max=float(a[-1]))
        hi = torch.bucketize(c.detach(), a).clamp(1, a.numel() - 1)
        lo = hi - 1
        t = (c - a[lo]) / (a[hi] - a[lo])
        return va[lo] + t * (va[hi] - va[lo])


# smoothstep: generic profile in THETA with theta_in / theta_out (CLAUDE.md
# section 4.2); mlp: not planned for this work (explicit models only, D65).
declare_planned("angular", "smoothstep", "mlp")
