# Ambient.combine(A, direct): how the rasterized albedo A (3,H,W, linear,
# opacity-weighted) and the direct light term (H,W or 3,H,W) make the
# predicted linear radiance. Both constant forms exist so they can be
# compared (CLAUDE.md section 4.3); the data verified the multiplicative one.

from light_models.registry import declare_planned, register


def _const(spec):
    v = spec.value
    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    ok = ok or (isinstance(v, list) and len(v) == 3)
    if not ok:
        raise ValueError(f"{spec.name} must be a number or 3 numbers (RGB), got {v!r}")
    t = spec.tensor()
    return t.view(-1, 1, 1) if t.dim() == 1 else t


def _check_nonnegative(name, t):
    # Ambient light cannot be negative; re-checked after every update of a
    # learned c (light_models/learning.py, D60, D65): leaving the domain is an
    # error, unless the ParamSpec asks for the projection below (D70).
    if bool((t.detach() < 0).any()):
        raise ValueError(f"{name} left [0, inf): {t.detach().tolist()!r}")


def _project_nonnegative(t):
    # ParamSpec at_bound: project (D70): back onto c >= 0 after a learning
    # step; True when a value was moved (counted in light_learned.csv). The
    # bound 0 is the definition of the domain, not a tunable threshold.
    moved = bool((t.detach() < 0).any())
    if moved:
        t.data.clamp_(min=0.0)
    return moved


def _direct3(direct):
    return direct if direct.dim() == 3 else direct[None]


@register("ambient", "none")
class NoAmbient:
    """I = A * direct."""

    def __call__(self, albedo, direct):
        return albedo * _direct3(direct)


@register("ambient", "multiplicative_const", numeric=("c",))
class MultiplicativeConst:
    """I = A * (direct + c): ambient light reflected by the albedo, as the
    World background of Blender does on diffuse surfaces."""

    def __init__(self, c):
        self.c = _const(c)
        self.check_values()

    def check_values(self):
        _check_nonnegative("Light.ambient.c", self.c)

    def project_c(self):
        return _project_nonnegative(self.c)

    def __call__(self, albedo, direct):
        c = self.c.to(albedo.device, albedo.dtype)
        return albedo * (_direct3(direct) + c)


@register("ambient", "additive_const", numeric=("c",))
class AdditiveConst:
    """I = A * direct + c: a constant offset independent of the albedo."""

    def __init__(self, c):
        self.c = _const(c)
        self.check_values()

    def check_values(self):
        _check_nonnegative("Light.ambient.c", self.c)

    def project_c(self):
        return _project_nonnegative(self.c)

    def __call__(self, albedo, direct):
        c = self.c.to(albedo.device, albedo.dtype)
        return albedo * _direct3(direct) + c


@register("ambient", "per_gaussian", numeric=("c",),
          options=("channels", "residual_lr", "residual_after_keyframes", "l1_weight"))
class PerGaussian(MultiplicativeConst):
    """Ambient light that varies over the scene (docs/DECISIONS.md D121):
    the irradiance of the static room lights at Gaussian i is E_i = c + r_i,
    a global level c (a ParamSpec, learned as for multiplicative_const) plus
    a per-Gaussian residual r_i stored in the map (GaussianModel extra
    "ambient_residual", `channels` = 1 or 3 values per Gaussian, optimized
    by the mapping Adam with `residual_lr`). For diffuse surfaces E does not
    depend on the camera, so it is a property of the surface, like the
    albedo. The residual is rasterized in a second pass with the colours
    A_i * (max(c + r_i, 0) - c), so that the radiance is
        I = A * (direct + c) + M,   M = rasterized A_i * (E_i - c),
    exactly sum_i w_i A_i (direct + E_i) where one Gaussian dominates a pixel.
    E_i >= 0 is the definition of the domain: the clamp keeps the shading
    physical; a residual clamped there gets no data gradient and the
    penalty below pulls it back.

    The residual is off (M = 0, no second pass) until the map has MORE than
    `residual_after_keyframes` keyframes, so c settles first. Mapping adds
    l1_weight * mean |M| per rendered keyframe (CLAUDE.md 4.3: without it
    the albedo goes to 0 and the colour moves into the ambient term, i.e.
    radiance is baked again). The penalty is on the rasterized M, in the
    units of the photometric loss, so l1_weight compares to Training.alpha
    whatever the albedo scale of the scene."""

    EXTRA = "ambient_residual"

    def __init__(self, c, channels, residual_lr, residual_after_keyframes, l1_weight):
        super().__init__(c)
        if channels not in (1, 3) or isinstance(channels, bool):
            raise ValueError(f"Light.ambient.channels must be 1 or 3, got {channels!r}")
        for name, v in (("residual_lr", residual_lr), ("l1_weight", l1_weight)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                raise ValueError(f"Light.ambient.{name} must be a positive number, got {v!r}")
        n = residual_after_keyframes
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"Light.ambient.residual_after_keyframes must be an integer >= 0, got {n!r}")
        self.channels = channels
        self.residual_lr = float(residual_lr)
        self.residual_after_keyframes = n
        self.l1_weight = float(l1_weight)
        # Runtime state, overwritten before every SLAM step (as Lambert's
        # warm-up, D43); None until then (verification scripts, GUI) = the
        # full model, residual on.
        self._n_keyframes = None

    def set_keyframe_count(self, n):
        self._n_keyframes = n

    def residual_active(self):
        return self._n_keyframes is None or self._n_keyframes > self.residual_after_keyframes

    def per_gaussian_params(self):
        """{extra name: (values per Gaussian, Adam learning rate)}; new
        Gaussians start at 0, i.e. at the global level c."""
        return {self.EXTRA: (self.channels, self.residual_lr)}

    def extra_raster_keys(self):
        return (self.EXTRA,) if self.residual_active() else ()

    def extra_rasters(self, gaussians):
        """{gbuffer key: (N,3) colours to rasterize}, empty while off."""
        if not self.residual_active():
            return {}
        r = gaussians.get_extra(self.EXTRA)
        albedo = gaussians.get_albedo
        c = self.c.to(albedo.device, albedo.dtype).view(1, -1)
        excess = (c + r).clamp_min(0.0) - c
        return {self.EXTRA: albedo * excess}

    def __call__(self, albedo, direct, gbuffer):
        radiance = super().__call__(albedo, direct)
        if not self.residual_active():
            return radiance, {}
        m = gbuffer[self.EXTRA]
        return radiance + m, {"ambient_residual": m,
                              "map_penalty": self.l1_weight * m.abs().mean()}


@register("ambient", "per_gaussian_gated", numeric=("c",),
          options=("channels", "residual_lr", "residual_after_keyframes", "l1_weight", "residual_direct_max"))
class PerGaussianGated(PerGaussian):
    """per_gaussian whose residual learns only where the lamp is weak
    (docs/DECISIONS.md D124). With an L1 loss every pixel votes on r_i with the
    sign of its residual. Under the lamp that residual is mostly the error of
    the direct term (normals, profile), much larger than any ambient signal,
    while r_i barely changes the image there; the wrong E then shows outside
    the cone and the albedo pays for it. So on pixels with direct above
    `residual_direct_max`, M and c still shade the image but give no gradient
    to the ambient term, and M no penalty. c needs the same gate: outside the
    cone r follows any change of c, so nothing would hold c against the votes
    from under the lamp, and it collapsed to 0 in the runs of D124. The albedo
    keeps its gradient through A * c. The penalty keeps the mean over all
    pixels, as the photometric loss, so l1_weight means the same as in
    per_gaussian. While the residual is off, c learns from every pixel, as in
    multiplicative_const."""

    def __init__(self, c, channels, residual_lr, residual_after_keyframes, l1_weight, residual_direct_max):
        super().__init__(c, channels, residual_lr, residual_after_keyframes, l1_weight)
        v = residual_direct_max
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"Light.ambient.residual_direct_max must be a number >= 0, got {v!r}")
        self.residual_direct_max = float(v)

    def __call__(self, albedo, direct, gbuffer):
        if not self.residual_active():
            return MultiplicativeConst.__call__(self, albedo, direct), {}
        m = gbuffer[self.EXTRA]
        d = _direct3(direct)
        free = d.amax(0, keepdim=True) <= self.residual_direct_max
        c = self.c.to(albedo.device, albedo.dtype)
        radiance = albedo * (d + c.where(free, c.detach()))
        return radiance + m.where(free, m.detach()), {
            "ambient_residual": m,
            "map_penalty": self.l1_weight * (m * free).abs().mean(),
        }


declare_planned("ambient", "sh")
