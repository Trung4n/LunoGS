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


# Per-Gaussian / SH ambient need an L1 penalty towards 0 (CLAUDE.md 4.3).
declare_planned("ambient", "per_gaussian", "sh")
