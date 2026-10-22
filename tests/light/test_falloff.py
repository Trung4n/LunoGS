"""light_models/falloff.py: the forms of the distance falloff."""
import pytest

torch = pytest.importorskip("torch")

import light_models.falloff  # noqa: E402,F401  (registers the forms)
from light_models.params import resolve_param  # noqa: E402
from light_models.registry import build_component  # noqa: E402


def _build(block):
    return build_component("falloff", block, "Light.falloff", None)


def test_inverse_square():
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(_build({"type": "inverse_square"})(d), 1.0 / d ** 2)


def test_constant_is_the_inverse_square_at_the_reference_distance():
    f = _build({"type": "constant",
                "distance": {"source": "value", "value": 2.0, "learnable": False}})
    d = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=torch.float64)
    out = f(d)
    assert out.shape == d.shape
    torch.testing.assert_close(out, torch.full_like(d, 0.25))


@pytest.mark.parametrize("value, error", [(0, ValueError), (-1.0, ValueError),
                                          (True, TypeError), ("2", TypeError)])
def test_constant_distance_validated(value, error):
    # Non-numbers are refused by the ParamSpec itself, non-positive numbers
    # by the falloff.
    with pytest.raises(error, match="distance"):
        _build({"type": "constant",
                "distance": {"source": "value", "value": value, "learnable": False}})


def test_constant_requires_its_distance():
    with pytest.raises(KeyError, match="distance"):
        _build({"type": "constant"})


def test_resolve_param_is_the_source_of_the_distance():
    spec = resolve_param({"source": "value", "value": 2.36, "learnable": False},
                         "Light.falloff.distance", None)
    assert spec.value == 2.36
