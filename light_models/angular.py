# Angular(theta) = Phi, the emission profile of the lamp, as a function of
# cos(theta), theta = angle between (light -> point) and the light axis.

import math

import torch

from light_models.device import on_device
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
        a = on_device(self.half_angle_deg, cos_theta.device, cos_theta.dtype) * (math.pi / 180.0)
        blend = on_device(self.blend, cos_theta.device, cos_theta.dtype)
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
          options=("knots_deg", "interpolation", "learn", "smoothness_weight", "scale"),
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
                 knots and normalised as the scale says. No number is copied
                 into the YAML.
      learn:     {learnable: false} or {learnable: true, lr: <>, prior_weight: <>}.
      smoothness_weight: >= 0, penalty weight * sum of squared second
                 differences of the knot values (uniform knot spacing
                 assumed by this penalty).
      scale:     how the scale of Phi is fixed, a block (D140, D141); the
                 scale of Phi is not observable apart from K and the albedo
                 (D65, D73):
                 {type: unit_at_zero}: Phi(0) = 1 fixed, the other knot
                   values learned, in [0, 1]. Only the pixels of the first
                   segment pin the scale, a weak anchor when that segment is
                   short (D106); the 8-deg first segment that fixes it assumes
                   the core is flat up to 8 deg.
                 {type: free}: every knot value learned, Phi(0) included, no
                   upper bound: the knots carry K x Phi and K must be FIXED
                   (colocated.py refuses a learned K). For the
                   pre-calibration only, where the target albedo is fixed too
                   (fit_light.py exports K x Phi(0) and Phi / Phi(0)); in SLAM
                   it would trade with the learned albedo (D73).
                 {type: region_mean, anchor_deg: [lo, hi]}: every knot value
                   learned; Phi = v / mean(v over the knots with lo <= theta
                   <= hi). Scaling every v leaves Phi unchanged, so the scale
                   is fixed by construction and pinned by all the pixels of
                   that band, not by one point. The band only sets the UNIT of
                   Phi (mean 1 over it), it does not constrain the shape: a
                   band reaching past the core gives a core above 1, not a
                   wrong profile. For the self-calibration in SLAM (D141).

    Constraints: the learned knot values are, after every step, PROJECTED
    onto non-increasing values >= 0 (projected gradient; pool adjacent
    violators, then the bounds), and <= 1 for unit_at_zero; for region_mean
    they are then rescaled to mean 1 over the band (Phi unchanged), so the
    stored values ARE Phi. Part of the model's definition, not a silent
    clamp. Learned path: <prefix>.knot_values: the values at knots_deg[1:] for
    unit_at_zero, at every knot otherwise.
    """

    INTERPOLATIONS = ("linear_cos",)
    SCALES = ("unit_at_zero", "free", "region_mean")

    def __init__(self, knots_deg, interpolation, learn, smoothness_weight, init, scale):
        where = "Light.angular"
        if (not isinstance(knots_deg, list) or len(knots_deg) < 2
                or any(isinstance(k, bool) or not isinstance(k, (int, float)) for k in knots_deg)):
            raise ValueError(f"{where}.knots_deg must be a list of >= 2 numbers, got {knots_deg!r}")
        if knots_deg[0] != 0:
            raise ValueError(f"{where}.knots_deg must start at 0, got {knots_deg[0]!r}")
        if any(b <= a for a, b in zip(knots_deg, knots_deg[1:])) or knots_deg[-1] >= 90:
            raise ValueError(f"{where}.knots_deg must be strictly increasing and below 90, got {knots_deg!r}")
        if interpolation not in self.INTERPOLATIONS:
            raise ValueError(f"{where}.interpolation must be one of {self.INTERPOLATIONS}, got {interpolation!r}")
        self.scale, anchor = self._read_scale(scale, knots_deg, where)
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

        self.knots_deg = list(knots_deg)
        cos_knots = torch.cos(torch.tensor(knots_deg, dtype=torch.float64) * (math.pi / 180.0))
        self.cos_knots = cos_knots.to(torch.float32)
        # region_mean: weights of the mean over the band (1/n on its knots).
        self.anchor_weights = None
        if anchor is not None:
            self.anchor_weights = torch.tensor([1.0 / sum(anchor) if a else 0.0 for a in anchor],
                                               dtype=torch.float32)
        with torch.no_grad():
            v = init(cos_knots).to(torch.float64)
        # free: the init, normalised to 1 at theta = 0, is the starting K x Phi
        # divided by the (fixed) K, i.e. the starting guess is K's.
        stored = self._normalised(v, f"{where}.init")
        self.knot_values = torch.tensor(stored, dtype=torch.float32, requires_grad=learn["learnable"])
        self.smoothness_weight = float(w)
        self._learn = dict(learn)

    @classmethod
    def _read_scale(cls, scale, knots_deg, where):
        """(type, anchor mask or None) of the scale block."""
        if not isinstance(scale, dict) or scale.get("type") not in cls.SCALES:
            raise ValueError(f"{where}.scale must be a block with type one of {cls.SCALES}, got {scale!r}")
        kind = scale["type"]
        keys = {"type", "anchor_deg"} if kind == "region_mean" else {"type"}
        if set(scale) != keys:
            raise ValueError(f"{where}.scale ({kind}) must have exactly the keys {sorted(keys)}, got {sorted(scale)}")
        if kind != "region_mean":
            return kind, None
        band = scale["anchor_deg"]
        if (not isinstance(band, list) or len(band) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in band) or band[0] > band[1]):
            raise ValueError(f"{where}.scale.anchor_deg must be [lo, hi] with lo <= hi, got {band!r}")
        mask = [band[0] <= k <= band[1] for k in knots_deg]
        if not any(mask):
            raise ValueError(f"{where}.scale.anchor_deg {band!r} contains no knot of {knots_deg!r}")
        return kind, mask

    def _anchor_mean(self, values):
        return sum(w * x for w, x in zip(self.anchor_weights.tolist(), values))

    def _normalised(self, v, what):
        """Stored knot values of a profile v (float64, at every knot): checked
        non-increasing and >= 0, scaled as the scale type says."""
        if not float(v[0]) > 0:
            raise ValueError(f"{what} must be > 0 at theta = 0, got {float(v[0])!r}")
        v = v.tolist()
        if any(b > a for a, b in zip(v, v[1:])) or min(v) < 0:
            raise ValueError(f"{what} must be non-increasing and >= 0 at the knots, got {v!r}")
        if self.scale == "region_mean":
            m = self._anchor_mean(v)
            if not m > 0:
                raise ValueError(f"{what} is 0 over the whole anchor band, cannot normalise")
            return [x / m for x in v]
        v = [x / v[0] for x in v]
        return v if self.scale == "free" else v[1:]

    def restart_from(self, profile):
        """Sets the knot values from a callable profile(cos_theta), sampled at
        the knots and normalised as `init`, with its checks; the tensor object
        is kept (the optimizer holds it). Used by the first-frame estimate
        (light_models/profile_bootstrap.py, D112)."""
        if self.scale == "free":
            raise ValueError("Light.angular: restarting from a profile needs scale unit_at_zero or region_mean, "
                             "not free")
        with torch.no_grad():
            v = profile(self.cos_knots.to(torch.float64)).to(torch.float64)
        stored = self._normalised(v, "Light.angular: profile")
        self.knot_values.data.copy_(torch.tensor(stored, dtype=self.knot_values.dtype))

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
        upper = 1.0 if self.scale == "unit_at_zero" else math.inf
        v = [min(upper, max(0.0, x)) for x in v]
        if self.scale == "region_mean":
            m = self._anchor_mean(v)
            if not m > 0:
                raise ValueError("Light.angular.knot_values: 0 over the whole anchor band")
            v = [x / m for x in v]  # Phi unchanged, the stored values are Phi
        self.knot_values.data.copy_(torch.tensor(v, dtype=self.knot_values.dtype))

    def check_values(self):
        raw = self.knot_values.detach().tolist()
        v = [1.0] + raw if self.scale == "unit_at_zero" else raw
        if any(b > a for a, b in zip(v, v[1:])) or min(v) < 0:
            bounds = "[0, 1]" if self.scale == "unit_at_zero" else ">= 0"
            raise ValueError(f"Light.angular.knot_values left the non-increasing {bounds} set: {v!r}")
        if self.scale == "region_mean" and not self._anchor_mean(raw) > 0:
            raise ValueError(f"Light.angular.knot_values: 0 over the whole anchor band: {raw!r}")

    def regularizer(self):
        v = self.values()
        second = v[2:] - 2.0 * v[1:-1] + v[:-2]
        return self.smoothness_weight * (second ** 2).sum()

    def values(self):
        """(N,) Phi at the knots, theta = 0 first (1 for unit_at_zero)."""
        if self.scale == "free":
            return self.knot_values
        if self.scale == "region_mean":
            return self.knot_values / (self.knot_values * self.anchor_weights).sum()
        return torch.cat([torch.ones(1, dtype=self.knot_values.dtype), self.knot_values])

    def __call__(self, cos_theta):
        # values() built on the device from the persistent GPU copy of the
        # knot values (on_device), not a new CPU tensor at every call: same
        # values, one address for CUDA graph replay (D135).
        dev, dt = cos_theta.device, cos_theta.dtype
        knots = on_device(self.knot_values, dev, dt)
        if self.scale == "free":
            values = knots
        elif self.scale == "region_mean":
            # Normalised in the forward pass too, so the gradient has no
            # component along the (unobservable) scale of the knots.
            values = knots / (knots * on_device(self.anchor_weights, dev, dt)).sum()
        else:
            values = torch.cat([torch.ones(1, device=dev, dtype=dt), knots])
        return _linear_in_cos(self.cos_knots, values, cos_theta)


def _linear_in_cos(cos_knots, values, cos_theta):
    """Phi at cos_theta, linear in cos(theta) between the knots (cos_knots in
    knot order, i.e. decreasing); beyond the last knot the last value."""
    dev, dt = cos_theta.device, cos_theta.dtype
    # Ascending cos (descending theta) for bucketize.
    a = on_device(cos_knots, dev, dt).flip(0)
    va = on_device(values, dev, dt).flip(0)
    # Bounds as tensors on the device, the knots being strictly decreasing:
    # a[0] = min, a[-1] = max. No host value (D128), and traceable (D134).
    c = cos_theta.clamp(min=a[0], max=a[-1])
    hi = torch.bucketize(c.detach(), a).clamp(1, a.numel() - 1)
    lo = hi - 1
    t = (c - a[lo]) / (a[hi] - a[lo])
    return va[lo] + t * (va[hi] - va[lo])


@register("angular", "tabulated", numeric=("knots_deg", "values"), options=("interpolation",))
class Tabulated:
    """Fixed emission profile given as a TABLE read from the params file, e.g.
    the profile measured by the pre-calibration (tools/calibrate/fit_light.py,
    docs/DECISIONS.md D106): Phi(knots_deg[i]) = values[i], interpolated as
    the spline (`interpolation`: linear_cos), the last value beyond the last
    knot. knots_deg: start at 0, strictly increasing, below 90; values >= 0,
    as stored (no normalisation: the scale K x Phi is what the file holds).
    Not learnable itself: to refine a measured profile in mapping, use a
    spline whose `init` is this block."""

    def __init__(self, knots_deg, values, interpolation):
        where = "Light.angular"
        for spec in (knots_deg, values):
            if spec.learnable:
                raise NotImplementedError(
                    f"{spec.name}: a tabulated profile is fixed; learn it as a spline with init: tabulated")
        k, v = knots_deg.value, values.value
        if (not isinstance(k, list) or len(k) < 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in k)):
            raise ValueError(f"{where}.knots_deg must be a list of >= 2 numbers, got {k!r}")
        if k[0] != 0 or any(b <= a for a, b in zip(k, k[1:])) or k[-1] >= 90:
            raise ValueError(f"{where}.knots_deg must start at 0, increase strictly and stay below 90, got {k!r}")
        if (not isinstance(v, list) or len(v) != len(k)
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 for x in v)):
            raise ValueError(f"{where}.values must be {len(k)} numbers >= 0 (one per knot), got {v!r}")
        if interpolation not in Spline.INTERPOLATIONS:
            raise ValueError(f"{where}.interpolation must be one of {Spline.INTERPOLATIONS}, got {interpolation!r}")
        self.knots_deg = knots_deg.tensor()
        self.values = values.tensor()
        self.cos_knots = torch.cos(self.knots_deg * (math.pi / 180.0))

    def __call__(self, cos_theta):
        return _linear_in_cos(self.cos_knots, self.values, cos_theta)


@register("angular", "smoothstep", numeric=("theta_in", "theta_out"))
class Smoothstep:
    """Generic spot profile IN THE ANGLE (CLAUDE.md section 4.2), for lamps
    whose profile has no known formula: Phi = 1 up to theta_in, 0 from
    theta_out on, S((theta_out - theta) / (theta_out - theta_in)) between,
    S(x) = x^2 (3 - 2x). Not blender_spot, which is smooth in cos(theta) and
    specific to that renderer (D14)."""

    def __init__(self, theta_in, theta_out):
        a, b = theta_in.value, theta_out.value
        for spec in (theta_in, theta_out):
            if isinstance(spec.value, bool) or not isinstance(spec.value, (int, float)):
                raise ValueError(f"{spec.name} must be a number, got {spec.value!r}")
        if not 0 <= a < b < 90:
            raise ValueError(f"Light.angular needs 0 <= theta_in < theta_out < 90, got {a!r}, {b!r}")
        self.theta_in = theta_in.tensor()
        self.theta_out = theta_out.tensor()

    def check_values(self):
        a, b = float(self.theta_in.detach()), float(self.theta_out.detach())
        if not 0 <= a < b < 90:
            raise ValueError(f"Light.angular left 0 <= theta_in < theta_out < 90: {a!r}, {b!r}")

    def __call__(self, cos_theta):
        a = on_device(self.theta_in, cos_theta.device, cos_theta.dtype) * (math.pi / 180.0)
        b = on_device(self.theta_out, cos_theta.device, cos_theta.dtype) * (math.pi / 180.0)
        # acos only strictly inside the ramp: its derivative is infinite on
        # the light axis, where Phi is constant anyway.
        ramp = (cos_theta < torch.cos(a)) & (cos_theta > torch.cos(b))
        safe = torch.where(ramp, cos_theta, torch.cos((a + b) / 2.0))
        x = (b - torch.acos(safe)) / (b - a)
        s = x * x * (3.0 - 2.0 * x)
        return torch.where(ramp, s, (cos_theta >= torch.cos(a)).to(cos_theta.dtype))


# mlp: not planned for this work (explicit models only, D65).
declare_planned("angular", "mlp")
