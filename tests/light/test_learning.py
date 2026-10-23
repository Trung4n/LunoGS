"""light_models/learning.py + the learnable parameters of the colocated
shader (docs/DECISIONS.md D60): schedule parsing, state round trip, prior,
and self-calibration on a synthetic multi-view wall (CPU).

    python -m pytest tests/light/test_learning.py
"""
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader  # noqa: E402
from light_models.learning import LightLearner  # noqa: E402

# Camera with the proportions of the data (600x340, fx 300) at 1/10.
H, W, F, Z = 34, 60, 30.0, 2.0
PARAMS = {"K": 30.0, "t": [0.0, -0.05, 0.0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          "a": 30.0, "b": 0.35, "c": 0.12}
CAM = SimpleNamespace(fx=F, fy=F, cx=W / 2, cy=H / 2)


def _p(key, **kw):
    return {"source": "params_file", "key": key, "learnable": False, **kw}


def _learn(**perturb):
    block = {"learnable": True, "lr": 0.05, "prior_weight": 0.0}
    if perturb:
        block["perturb"] = perturb
    return block


def _config(half=None, rcl=None, schedule=None):
    light = {
        "enabled": True, "shader": {"type": "colocated"}, "intensity": _p("K"),
        "pose": {"t_CL": _p("t"), "R_CL": _p("R", **(rcl or {}))},
        "falloff": {"type": "inverse_square"},
        "angular": {"type": "blender_spot", "half_angle_deg": _p("a", **(half or {})), "blend": _p("b")},
        "cosine": {"type": "none"},
        "ambient": {"type": "multiplicative_const", "c": _p("c")},
        "gbuffer": {"opacity_thr": 0.5},
    }
    if schedule is not None:
        light["schedule"] = schedule
    return {"Training": {"spherical_harmonics": False}, "Light": light}


def _gbuffer(albedo):
    return {"albedo": albedo, "depth": torch.full((1, H, W), Z),
            "opacity": torch.ones(1, H, W), "pixel_offset": 0.0}


# ---- configuration ------------------------------------------------------------

def test_no_learnable_no_schedule():
    sh = build_shader(_config(), PARAMS)
    assert sh.learnable_params() == {} and sh.learnable_state() == {}


def test_schedule_without_learnable_is_an_error():
    with pytest.raises(ValueError, match="no light parameter is learnable"):
        build_shader(_config(schedule=[{"after_keyframes": 0, "unfreeze": ["intensity"]}]), PARAMS)


def test_learnable_requires_a_schedule_entry():
    with pytest.raises(KeyError, match="schedule"):
        build_shader(_config(half=_learn()), PARAMS)
    with pytest.raises(KeyError, match="never unfrozen"):
        build_shader(_config(half=_learn(), rcl=_learn(),
                             schedule=[{"after_keyframes": 0, "unfreeze": ["angular.half_angle_deg"]}]),
                     PARAMS)


@pytest.mark.parametrize("entry, error", [
    ({"after_keyframes": 0, "unfreeze": ["angular.blend"]}, ValueError),  # not learnable
    ({"after_keyframes": -1, "unfreeze": ["angular.half_angle_deg"]}, ValueError),
    ({"after_keyframes": 0, "unfreeze": []}, TypeError),
    ({"after_keyframes": 0, "unfreeze": ["angular.half_angle_deg"], "lr": 1}, ValueError),
])
def test_schedule_validated(entry, error):
    with pytest.raises(error):
        build_shader(_config(half=_learn(), schedule=[entry]), PARAMS)


def test_unfrozen_twice_is_an_error():
    e = {"after_keyframes": 0, "unfreeze": ["angular.half_angle_deg"]}
    with pytest.raises(ValueError, match="twice"):
        build_shader(_config(half=_learn(), schedule=[e, e]), PARAMS)


def test_rgb_intensity_cannot_learn():
    params = dict(PARAMS, K=[30.0, 30.0, 30.0])
    cfg = _config(schedule=[{"after_keyframes": 0, "unfreeze": ["intensity"]}])
    cfg["Light"]["intensity"] = _p("K", **_learn())
    with pytest.raises(NotImplementedError, match="intensity"):
        build_shader(cfg, params)


# ---- state, freezing, schedule, prior ----------------------------------------

def _both():
    return build_shader(_config(
        half=_learn(add=2.0), rcl=_learn(rotate_deg=5.0, axis=[1.0, 0.0, 0.0]),
        schedule=[{"after_keyframes": 3, "unfreeze": ["angular.half_angle_deg"]},
                  {"after_keyframes": 5, "unfreeze": ["pose.R_CL"]}]), PARAMS)


def test_state_round_trip_and_pickling():
    a, b = _both(), _both()
    with torch.no_grad():
        a.learnable_params()["angular.half_angle_deg"].tensor.fill_(31.5)
        a.learnable_params()["pose.R_CL"].tensor.copy_(torch.tensor([-1.0, 0.5]))
    b.load_learnable_state(a.learnable_state())
    assert b.learnable_state() == pytest.approx(a.learnable_state())
    c = pickle.loads(pickle.dumps(a))  # backend / GUI processes get a pickled copy
    assert c.learnable_state() == pytest.approx(a.learnable_state())
    with pytest.raises(ValueError, match="expects"):
        b.load_learnable_state({"angular.half_angle_deg": 30.0})


def test_frontend_copy_does_not_track_gradients():
    sh = _both()
    sh.set_learning(False)
    out = sh(_gbuffer(torch.full((3, H, W), 0.5)), CAM)["radiance_linear"]
    assert not out.requires_grad


def test_schedule_gates_the_updates():
    sh = _both()
    learner = LightLearner(sh.learnable_params())
    assert learner.active(3) == [] and learner.active(4) == ["angular.half_angle_deg"]
    assert sorted(learner.active(6)) == ["angular.half_angle_deg", "pose.R_CL"]
    start = sh.learnable_state()
    out = sh(_gbuffer(torch.full((3, H, W), 0.5)), CAM)["radiance_linear"]
    out.sum().backward()
    learner.step(4)  # only the half-angle may move
    now = sh.learnable_state()
    assert now["angular.half_angle_deg"] != start["angular.half_angle_deg"]
    assert now["pose.R_CL"] == start["pose.R_CL"]


def test_prior_pulls_to_the_starting_value():
    cfg = _config(half={"learnable": True, "lr": 0.05, "prior_weight": 2.0},
                  schedule=[{"after_keyframes": 0, "unfreeze": ["angular.half_angle_deg"]}])
    sh = build_shader(cfg, PARAMS)
    learner = LightLearner(sh.learnable_params())
    assert float(learner.prior_loss(1).detach()) == 0.0
    with torch.no_grad():
        sh.learnable_params()["angular.half_angle_deg"].tensor.fill_(31.0)
    assert float(learner.prior_loss(1).detach()) == pytest.approx(2.0)
    assert learner.prior_loss(0) is None  # not active yet


def test_leaving_the_domain_raises():
    sh = build_shader(_config(half=_learn(), schedule=[
        {"after_keyframes": 0, "unfreeze": ["angular.half_angle_deg"]}]), PARAMS)
    learner = LightLearner(sh.learnable_params())
    t = sh.learnable_params()["angular.half_angle_deg"].tensor
    with torch.no_grad():
        t.fill_(89.99)
    t.grad = torch.tensor(-1.0)  # Adam step of +lr: crosses 90
    with pytest.raises(ValueError, match="half_angle_deg"):
        learner.step(1)


def test_tilt_matches_the_perturb_rotation():
    # A learned tilt of -g about x undoes a perturb of +g about x.
    sh = build_shader(_config(rcl=_learn(rotate_deg=4.0, axis=[1.0, 0.0, 0.0]), schedule=[
        {"after_keyframes": 0, "unfreeze": ["pose.R_CL"]}]), PARAMS)
    nominal = build_shader(_config(), PARAMS)
    with torch.no_grad():
        sh.learnable_params()["pose.R_CL"].tensor.copy_(torch.tensor([-4.0, 0.0]))
    torch.testing.assert_close(sh._light_axis("cpu", torch.float32),
                               nominal._light_axis("cpu", torch.float32), atol=1e-6, rtol=0)


# ---- self-calibration on a synthetic wall ----------------------------------------

def _self_calibrate(cfg, offsets, steps, alb_lr):
    """Observations rendered with the TRUE parameters from several views of a
    textured wall (view = crop at an offset); the model starts from the
    perturbed parameters and learns them JOINTLY with the albedo (shared by
    all views), as mapping does."""
    torch.manual_seed(0)
    mx = max(o[0] for o in offsets)
    my = max(o[1] for o in offsets)
    world = 0.2 + 0.6 * torch.rand(3, H + my, W + mx)

    def crop(a, o):
        return a[:, o[1]:o[1] + H, o[0]:o[0] + W]

    truth = build_shader(_config(), PARAMS)
    obs = [truth(_gbuffer(crop(world, o)), CAM)["radiance_linear"].detach() for o in offsets]
    sh = build_shader(cfg, PARAMS)
    learner = LightLearner(sh.learnable_params())
    with torch.no_grad():  # albedo init: de-shaded at the WRONG parameters
        alb = torch.zeros_like(world)
        cnt = torch.zeros_like(world)
        s = sh(_gbuffer(torch.ones(3, H, W)), CAM)["radiance_linear"]
        for o, ob in zip(offsets, obs):
            crop(alb, o).add_(ob / s)
            crop(cnt, o).add_(1)
        alb = (alb / cnt.clamp_min(1)).requires_grad_(True)
    opt = torch.optim.Adam([alb], lr=alb_lr)
    for _ in range(steps):
        opt.zero_grad()
        learner.zero_grad()
        loss = sum((sh(_gbuffer(crop(alb, o)), CAM)["radiance_linear"] - ob).abs().mean()
                   for o, ob in zip(offsets, obs))
        loss.backward()
        opt.step()
        learner.step(1)
    return sh.learnable_state()


@pytest.mark.parametrize("error", [2.0, -2.0])
def test_self_calibration_recovers_the_half_angle(error):
    cfg = _config(half=_learn(add=error),
                  schedule=[{"after_keyframes": 0, "unfreeze": ["angular.half_angle_deg"]}])
    state = _self_calibrate(cfg, [(4 * k, 0) for k in range(5)], steps=300, alb_lr=0.01)
    assert state["angular.half_angle_deg"] == pytest.approx(30.0, abs=0.05)


def test_self_calibration_recovers_the_light_axis_with_2d_motion():
    # Needs the cone edge to sweep the surface in 2D (D60): with views along
    # one line the albedo absorbs part of the tilt.
    cfg = _config(rcl=_learn(rotate_deg=5.0, axis=[1.0, 0.0, 0.0]),
                  schedule=[{"after_keyframes": 0, "unfreeze": ["pose.R_CL"]}])
    grid = [(8 * i, 8 * j) for i in range(3) for j in range(3)]
    state = _self_calibrate(cfg, grid, steps=600, alb_lr=0.01)
    assert state["pose.R_CL"] == pytest.approx([-5.0, 0.0], abs=0.1)


def test_own_storage_keeps_objects_and_values_but_unshares_memory():
    sh = _both()
    params = sh.learnable_params()
    before = {k: (p.tensor, p.tensor.data_ptr()) for k, p in params.items()}
    state = sh.learnable_state()
    sh.own_learnable_storage()
    for k, p in params.items():
        assert p.tensor is before[k][0]  # same object: components still see it
        assert p.tensor.data_ptr() != before[k][1]  # new buffer
        assert p.tensor.requires_grad and p.tensor.is_leaf
    assert sh.learnable_state() == pytest.approx(state)
