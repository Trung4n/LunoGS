"""angular 'smoothstep': the generic spot profile in the angle (CLAUDE.md
section 4.2, docs/DECISIONS.md D110).

    python -m pytest tests/light/test_smoothstep.py
"""
import math

import pytest

torch = pytest.importorskip("torch")

import light_models.angular  # noqa: E402,F401  (registers the forms)
from light_models.registry import build_component  # noqa: E402


def block(a=20.0, b=40.0, learnable=False):
    spec = lambda v: ({"source": "value", "value": v, "learnable": True, "lr": 0.1, "prior_weight": 0.0}  # noqa: E731
                      if learnable else {"source": "value", "value": v, "learnable": False})
    return {"type": "smoothstep", "theta_in": spec(a), "theta_out": spec(b)}


def cos_deg(*deg):
    return torch.cos(torch.tensor(deg, dtype=torch.float64) * math.pi / 180.0)


def test_flat_core_zero_outside_and_smooth_in_the_angle():
    phi = build_component("angular", block(), "Light.angular", None)
    torch.testing.assert_close(phi(cos_deg(0, 10, 20, 40, 50)),
                               torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-6, rtol=0)
    # Half way IN THE ANGLE: S(1/2) = 1/2; S(1/4) = 5/32.
    torch.testing.assert_close(phi(cos_deg(30, 35)), torch.tensor([0.5, 5.0 / 32.0], dtype=torch.float64),
                               atol=1e-5, rtol=0)


def test_gradients_are_finite_on_the_axis():
    phi = build_component("angular", block(0.0, 40.0, learnable=True), "Light.angular", None)
    c = torch.tensor([1.0, math.cos(math.radians(20)), 0.0], dtype=torch.float64, requires_grad=True)
    phi(c).sum().backward()
    assert torch.isfinite(c.grad).all()
    for p in (phi.theta_in, phi.theta_out):
        assert torch.isfinite(p.grad).all()


def test_spline_can_start_from_it():
    spline = build_component("angular", {
        "type": "spline", "knots_deg": [0, 10, 30, 50], "interpolation": "linear_cos",
        "learn": {"learnable": True, "lr": 0.01, "prior_weight": 0.0}, "smoothness_weight": 0.0,
        "init": block()}, "Light.angular", None)
    torch.testing.assert_close(spline.values().double(), torch.tensor([1.0, 1.0, 0.5, 0.0], dtype=torch.float64),
                               atol=1e-5, rtol=0)


@pytest.mark.parametrize("a, b", [(40.0, 20.0), (20.0, 20.0), (-1.0, 20.0), (20.0, 90.0)])
def test_angles_validated(a, b):
    with pytest.raises(ValueError, match="theta_in"):
        build_component("angular", block(a, b), "Light.angular", None)


def test_keys_are_required():
    b = block()
    del b["theta_out"]
    with pytest.raises(KeyError):
        build_component("angular", b, "Light.angular", None)
