"""angular 'tabulated' (docs/DECISIONS.md D106): a fixed emission profile read
as a table from the params file, e.g. the result of the pre-calibration.

    python -m pytest tests/light/test_tabulated.py
"""
import math

import pytest

torch = pytest.importorskip("torch")

import light_models.angular  # noqa: E402,F401  (registers the forms)
from light_models.registry import build_component  # noqa: E402

PARAMS = {"light": {"angular": {"knots_deg": [0, 10, 20, 30], "knot_values": [1.0, 0.9, 0.5, 0.0]}}}


def table(**over):
    block = {"type": "tabulated",
             "knots_deg": {"source": "params_file", "key": "light.angular.knots_deg", "learnable": False},
             "values": {"source": "params_file", "key": "light.angular.knot_values", "learnable": False},
             "interpolation": "linear_cos"}
    block.update(over)
    return block


def cos_deg(*deg):
    return torch.cos(torch.tensor(deg, dtype=torch.float64) * math.pi / 180.0)


def test_values_at_the_knots_and_linear_in_cos_between():
    phi = build_component("angular", table(), "Light.angular", PARAMS)
    torch.testing.assert_close(phi(cos_deg(0, 10, 20, 30, 45)),
                               torch.tensor([1.0, 0.9, 0.5, 0.0, 0.0], dtype=torch.float64), atol=1e-6, rtol=0)
    c = (cos_deg(10) + cos_deg(20)) / 2  # half way in cos between two knots
    torch.testing.assert_close(phi(c), torch.tensor([0.7], dtype=torch.float64), atol=1e-6, rtol=0)


def test_spline_initialised_from_the_table_reproduces_it():
    spline = build_component("angular", {
        "type": "spline", "knots_deg": [0, 10, 20, 30], "interpolation": "linear_cos",
        "learn": {"learnable": True, "lr": 0.01, "prior_weight": 0.0}, "smoothness_weight": 0.0,
        "init": table()}, "Light.angular", PARAMS)
    torch.testing.assert_close(spline.values().double(), torch.tensor([1.0, 0.9, 0.5, 0.0], dtype=torch.float64),
                               atol=1e-6, rtol=0)  # table stored as float32


@pytest.mark.parametrize("params, error", [
    ({"knots_deg": [5, 10], "knot_values": [1.0, 0.0]}, "start at 0"),
    ({"knots_deg": [0, 20, 10], "knot_values": [1.0, 0.5, 0.0]}, "increase strictly"),
    ({"knots_deg": [0, 10, 20], "knot_values": [1.0, 0.5]}, "one per knot"),
    ({"knots_deg": [0, 10], "knot_values": [1.0, -0.1]}, "one per knot"),
])
def test_table_validated(params, error):
    with pytest.raises(ValueError, match=error):
        build_component("angular", table(), "Light.angular", {"light": {"angular": params}})


def test_table_is_not_learnable_and_keys_are_strict():
    learn = {"source": "params_file", "key": "light.angular.knot_values", "learnable": True,
             "lr": 0.01, "prior_weight": 0.0}
    with pytest.raises(NotImplementedError, match="spline with init"):
        build_component("angular", table(values=learn), "Light.angular", PARAMS)
    with pytest.raises(ValueError, match="interpolation"):
        build_component("angular", table(interpolation="cubic"), "Light.angular", PARAMS)
    block = table()
    del block["interpolation"]
    with pytest.raises(KeyError):
        build_component("angular", block, "Light.angular", PARAMS)
