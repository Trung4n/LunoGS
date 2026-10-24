"""light_models/angular.py Spline (docs/DECISIONS.md D67): explicit learnable
emission profile, its constraints, its learning hooks, and self-calibration
on a synthetic wall (CPU).

    python -m pytest tests/light/test_spline.py
"""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import light_models.angular  # noqa: E402,F401  (registers the forms)
from light_models import build_shader  # noqa: E402
from light_models.angular import _nonincreasing_projection  # noqa: E402
from light_models.learning import LightLearner  # noqa: E402
from light_models.registry import build_component  # noqa: E402


def _spot(half, blend=0.35):
    return {"type": "blender_spot",
            "half_angle_deg": {"source": "value", "value": half, "learnable": False},
            "blend": {"source": "value", "value": blend, "learnable": False}}


def _spline_block(knots, init, learn=None, smoothness=0.0):
    return {"type": "spline", "knots_deg": knots, "interpolation": "linear_cos",
            "learn": learn or {"learnable": False}, "smoothness_weight": smoothness, "init": init}


def _cos(deg):
    return torch.cos(torch.as_tensor(deg, dtype=torch.float64) * (math.pi / 180.0))


# ---- representation ------------------------------------------------------------

def test_knots_reproduce_the_init_profile_and_phi0_is_one():
    knots = list(range(0, 49, 4))
    sp = build_component("angular", _spline_block(knots, _spot(30.0)), "Light.angular", None)
    spot = build_component("angular", _spot(30.0), "Light.angular", None)
    torch.testing.assert_close(sp(_cos(knots)), spot(_cos(knots)), atol=1e-6, rtol=0)
    assert float(sp(_cos(0.0))) == 1.0
    # Beyond the last knot: the last value (0 for a 30 deg cone).
    assert float(sp(_cos(60.0))) == pytest.approx(0.0)


def test_linear_in_cos_between_knots():
    sp = build_component("angular", _spline_block([0, 10, 20], {"type": "uniform"}), "a", None)
    with torch.no_grad():
        sp.knot_values.copy_(torch.tensor([0.6, 0.2]))
    c10, c20 = _cos(10.0), _cos(20.0)
    mid = (c10 + c20) / 2
    assert float(sp(mid)) == pytest.approx(0.4, abs=1e-6)


@pytest.mark.parametrize("step, max_err", [(4, 0.17), (2, 0.065), (1, 0.021)])
def test_knot_spacing_sets_the_error_at_the_cone_edge(step, max_err):
    # D67: the Blender cone edge is only ~6 deg wide; 4 deg knots cannot
    # follow it, which biased the learning (core pulled down).
    knots = list(range(0, 49, step))
    sp = build_component("angular", _spline_block(knots, _spot(30.0)), "a", None)
    spot = build_component("angular", _spot(30.0), "a", None)
    c = _cos(torch.linspace(0, 48, 2001, dtype=torch.float64))
    assert float((sp(c) - spot(c)).abs().max()) < max_err


def test_gradient_reaches_knots_and_cos_theta():
    sp = build_component("angular", _spline_block(list(range(0, 49, 4)), _spot(30.0),
                                                  {"learnable": True, "lr": 0.01, "prior_weight": 0.0}),
                         "a", None)
    c = _cos(torch.tensor([10.0, 29.0, 31.0])).float().requires_grad_(True)
    sp(c).sum().backward()
    assert sp.knot_values.grad is not None and float(sp.knot_values.grad.abs().sum()) > 0
    assert torch.isfinite(c.grad).all()


# ---- configuration -------------------------------------------------------------------

@pytest.mark.parametrize("block, error", [
    (_spline_block([5, 10], _spot(30.0)), ValueError),  # must start at 0
    (_spline_block([0, 10, 10], _spot(30.0)), ValueError),  # strictly increasing
    (_spline_block([0, 95], _spot(30.0)), ValueError),  # below 90
    (_spline_block([0], _spot(30.0)), ValueError),  # >= 2 knots
    ({**_spline_block([0, 10], _spot(30.0)), "interpolation": "cubic"}, ValueError),
    (_spline_block([0, 10], _spot(30.0), smoothness=-1.0), ValueError),
    (_spline_block([0, 10], _spot(30.0), learn={"learnable": True}), ValueError),  # lr missing
    (_spline_block([0, 10], _spot(30.0), learn={"learnable": False, "lr": 0.1}), ValueError),
    (_spline_block([0, 10], _spline_block([0, 10], _spot(30.0))), ValueError),  # init spline
])
def test_spline_config_validated(block, error):
    with pytest.raises(error):
        build_component("angular", block, "Light.angular", None)


def test_init_reads_the_params_file_and_perturb():
    # The nested init block is built with the params file (no number copied
    # into the YAML), and may be perturbed.
    init = {"type": "blender_spot",
            "half_angle_deg": {"source": "params_file", "key": "a", "learnable": False,
                               "perturb": {"add": 4.0}},
            "blend": {"source": "params_file", "key": "b", "learnable": False}}
    sp = build_component("angular", _spline_block(list(range(0, 49, 2)), init), "a", {"a": 30.0, "b": 0.35})
    wide = build_component("angular", _spot(34.0), "a", None)
    knots = list(range(0, 49, 2))
    torch.testing.assert_close(sp(_cos(knots)), wide(_cos(knots)), atol=1e-6, rtol=0)


# ---- constraints and learning hooks ------------------------------------------------

def test_nonincreasing_projection():
    assert _nonincreasing_projection([1.0, 0.5, 0.7, 0.2]) == pytest.approx([1.0, 0.6, 0.6, 0.2])
    assert _nonincreasing_projection([0.2, 0.4, 0.6]) == pytest.approx([0.4, 0.4, 0.4])
    assert _nonincreasing_projection([0.9, 0.5, 0.1]) == pytest.approx([0.9, 0.5, 0.1])


def test_projection_restores_monotone_values_in_unit_range():
    sp = build_component("angular", _spline_block([0, 10, 20, 30], _spot(25.0),
                                                  {"learnable": True, "lr": 0.01, "prior_weight": 0.0}),
                         "a", None)
    with torch.no_grad():
        sp.knot_values.copy_(torch.tensor([1.3, 0.2, 0.5]))
    sp.project()
    v = sp.values().tolist()
    assert v[0] == 1.0 and all(b <= a for a, b in zip(v, v[1:])) and min(v) >= 0 and max(v) <= 1
    sp.check_values()


def test_learnable_path_schedule_and_state():
    cfg = {"Training": {"spherical_harmonics": False}, "Light": {
        "enabled": True, "shader": {"type": "colocated"},
        "intensity": {"source": "value", "value": 30.0, "learnable": False},
        "pose": {"t_CL": {"source": "value", "value": [0.0, -0.05, 0.0], "learnable": False},
                 "R_CL": {"source": "value", "value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "learnable": False}},
        "falloff": {"type": "inverse_square"},
        "angular": _spline_block(list(range(0, 49, 4)), _spot(30.0),
                                 {"learnable": True, "lr": 0.01, "prior_weight": 0.0}, smoothness=0.1),
        "cosine": {"type": "none"},
        "ambient": {"type": "multiplicative_const", "c": {"source": "value", "value": 0.12, "learnable": False}},
        "gbuffer": {"opacity_thr": 0.5},
        "schedule": [{"after_keyframes": 0, "unfreeze": ["angular.knot_values"]}]}}
    sh = build_shader(cfg, {})
    assert sorted(sh.learnable_params()) == ["angular.knot_values"]
    learner = LightLearner(sh.learnable_params())
    assert learner.prior_loss(1) is not None  # the smoothness regularizer
    state = sh.learnable_state()
    assert len(state["angular.knot_values"]) == 12
    del cfg["Light"]["schedule"]
    with pytest.raises(KeyError, match="schedule"):
        build_shader(cfg, {})


# ---- self-calibration on a synthetic wall --------------------------------------------

H, W, F, Z = 34, 60, 30.0, 2.0
CAM = SimpleNamespace(fx=F, fy=F, cx=W / 2, cy=H / 2)


def _config(angular, schedule=None):
    v = lambda x: {"source": "value", "value": x, "learnable": False}  # noqa: E731
    light = {"enabled": True, "shader": {"type": "colocated"}, "intensity": v(30.0),
             "pose": {"t_CL": v([0.0, -0.05, 0.0]), "R_CL": v([[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
             "falloff": {"type": "inverse_square"}, "angular": angular, "cosine": {"type": "none"},
             "ambient": {"type": "multiplicative_const", "c": v(0.12)}, "gbuffer": {"opacity_thr": 0.5}}
    if schedule:
        light["schedule"] = schedule
    return {"Training": {"spherical_harmonics": False}, "Light": light}


@pytest.mark.parametrize("init", [_spot(34.0), {"type": "uniform"}], ids=["wider_cone", "flat"])
def test_self_calibration_learns_the_blender_profile_from_a_wider_start(init):
    # D67: from a cone 4 deg too WIDE, or from a FLAT profile (no knowledge
    # of the cone at all), 1 deg knots, 2D camera motion that carries surface
    # points across the cone edge: the learned knots match the Blender
    # profile. (From a NARROWER start the core collapses towards a wrong
    # albedo scale instead, a local minimum: see D67.)
    torch.manual_seed(0)
    knots = list(range(0, 49, 1))
    offsets = [(20 * i, 20 * j) for i in range(3) for j in range(3)]
    world = 0.2 + 0.6 * torch.rand(3, H + 40, W + 40)

    def crop(a, o):
        return a[:, o[1]:o[1] + H, o[0]:o[0] + W]

    def gb(alb):
        return {"albedo": alb, "depth": torch.full((1, H, W), Z),
                "opacity": torch.ones(1, H, W), "pixel_offset": 0.0}

    truth = build_shader(_config(_spot(30.0)), {})
    obs = [truth(gb(crop(world, o)), CAM)["radiance_linear"].detach() for o in offsets]
    sh = build_shader(_config(_spline_block(knots, init,
                                            {"learnable": True, "lr": 0.01, "prior_weight": 0.0}),
                              [{"after_keyframes": 0, "unfreeze": ["angular.knot_values"]}]), {})
    learner = LightLearner(sh.learnable_params())
    with torch.no_grad():
        alb = torch.zeros_like(world)
        cnt = torch.zeros_like(world)
        s = sh(gb(torch.ones(3, H, W)), CAM)["radiance_linear"]
        for o, ob in zip(offsets, obs):
            crop(alb, o).add_(ob / s)
            crop(cnt, o).add_(1)
        alb = (alb / cnt.clamp_min(1)).requires_grad_(True)
    opt = torch.optim.Adam([alb], lr=0.01)
    for _ in range(400):
        opt.zero_grad()
        learner.zero_grad()
        loss = sum((sh(gb(crop(alb, o)), CAM)["radiance_linear"] - ob).abs().mean()
                   for o, ob in zip(offsets, obs))
        loss.backward()
        opt.step()
        learner.step(1)
    target = build_component("angular", _spot(30.0), "a", None)(_cos(knots)).float()
    assert float((sh.angular.values() - target).abs().max()) < 0.05
