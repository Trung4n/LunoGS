"""Ambient light as a field over the scene (docs/DECISIONS.md D126). The oracle:
trilinear interpolation reproduces a linear function exactly, so a grid holding
r(x) = g . x must shade every pixel with exactly that value at its world point."""
import importlib.util
import os
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader  # noqa: E402
from light_models.diagnostics import LEARNED_LARGE_NAME, log_learned_light  # noqa: E402

PARAMS = {
    "t_CL_m": {"config_opencv": [0.0, -0.05, 0.0]},
    "R_CL": {"config": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    "intensity": {"K_theory": 3.0},
    "cone": {"half_angle_deg_config": 30.0, "blend_config": 0.35},
}
GRID = {"channels": 1, "cell_m": 0.5, "half_extent_m": 2.0, "residual_lr": 0.001, "l1_weight": 0.05}
SCHEDULE = [{"after_keyframes": 2, "unfreeze": ["ambient.residual"]}]


def _p(key):
    return {"source": "params_file", "key": key, "learnable": False}


def _config(ambient, schedule=SCHEDULE):
    light = {
        "enabled": True,
        "shader": {"type": "colocated"},
        "intensity": _p("intensity.K_theory"),
        "pose": {"t_CL": _p("t_CL_m.config_opencv"), "R_CL": _p("R_CL.config")},
        "falloff": {"type": "inverse_square"},
        "angular": {"type": "blender_spot", "half_angle_deg": _p("cone.half_angle_deg_config"),
                    "blend": _p("cone.blend_config")},
        "cosine": {"type": "none"},
        "ambient": ambient,
        "gbuffer": {"opacity_thr": 0.5},
    }
    if schedule is not None:
        light["schedule"] = schedule
    return {"Training": {"spherical_harmonics": False}, "Light": light}


def _grid(c=0.1, **over):
    block = {"type": "position_grid", "c": {"source": "value", "value": c, "learnable": False}}
    block.update(GRID)
    block.update(over)
    return build_shader(_config(block), PARAMS)


def _set_linear(shader, g, offset=0.0):
    """grid r(x) = g . x + offset; axes of the tensor: (channel, z, y, x)."""
    amb = shader.ambient
    n = amb.residual.shape[-1]
    v = torch.linspace(-amb.half_extent_m, amb.half_extent_m, n)
    z, y, x = torch.meshgrid(v, v, v, indexing="ij")
    with torch.no_grad():
        amb.residual[0] = g[0] * x + g[1] * y + g[2] * z + offset


def _view(h=4, w=5, z=1.5):
    rot = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    trans = torch.tensor([0.1, -0.2, 0.3])
    cam = SimpleNamespace(fx=20.0, fy=20.0, cx=w / 2, cy=h / 2, R=rot, T=trans)
    gbuffer = {"albedo": torch.full((3, h, w), 0.5), "depth": torch.full((1, h, w), z),
               "opacity": torch.ones(1, h, w), "pixel_offset": 0.5}
    return cam, gbuffer


def _world_points(cam, h, w, z):
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    xc = torch.stack([(u + 0.5 - cam.cx) / cam.fx * z, (v + 0.5 - cam.cy) / cam.fy * z, torch.full_like(u, z)])
    return (cam.R.T @ (xc - cam.T.view(3, 1, 1)).flatten(1)).view_as(xc)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("key", sorted(GRID))
def test_every_option_is_required(key):
    block = {"type": "position_grid", "c": {"source": "value", "value": 0.1, "learnable": False}}
    block.update({k: v for k, v in GRID.items() if k != key})
    with pytest.raises(KeyError, match=key):
        build_shader(_config(block), PARAMS)


@pytest.mark.parametrize("over", [{"channels": 2}, {"channels": True}, {"cell_m": 0}, {"half_extent_m": -1.0},
                                  {"residual_lr": 0}, {"l1_weight": -0.1}, {"cell_m": 0.3}])
def test_bad_options_raise(over):
    with pytest.raises(ValueError):
        _grid(**over)


def test_the_grid_must_be_in_the_schedule():
    block = {"type": "position_grid", "c": {"source": "value", "value": 0.1, "learnable": False}}
    block.update(GRID)
    with pytest.raises(KeyError):
        build_shader(_config(block, schedule=None), PARAMS)


def test_grid_size_and_learning_parameters():
    shader = _grid()
    assert tuple(shader.ambient.residual.shape) == (1, 9, 9, 9)
    p = shader.learnable_params()["ambient.residual"]
    assert p.lr == 0.001 and p.after_keyframes == 2 and p.prior_weight == 0.0
    assert tuple(_grid(channels=3).ambient.residual.shape) == (3, 9, 9, 9)


# ---------------------------------------------------------------- shading


def test_zero_field_is_the_constant_ambient():
    cam, gbuffer = _view()
    plain = build_shader(_config({"type": "multiplicative_const", "c": {"source": "value", "value": 0.1, "learnable": False}},
                                 schedule=None), PARAMS)
    expected = plain(gbuffer, cam)["radiance_linear"]
    out = _grid()(gbuffer, cam)
    torch.testing.assert_close(out["radiance_linear"], expected)
    assert float(out["light_map_penalty"]) == 0.0


def test_field_is_read_at_the_world_point_of_each_pixel():
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    plain = build_shader(_config({"type": "multiplicative_const", "c": {"source": "value", "value": 0.1, "learnable": False}},
                                 schedule=None), PARAMS)
    g = torch.tensor([0.02, -0.03, 0.05])
    shader = _grid()
    _set_linear(shader, g, offset=0.2)
    world = _world_points(cam, h, w, z)
    expected_r = (g.view(3, 1, 1) * world).sum(0) + 0.2
    out = shader(gbuffer, cam)
    torch.testing.assert_close(out["radiance_linear"], plain(gbuffer, cam)["radiance_linear"] + 0.5 * expected_r, atol=1e-5, rtol=0)
    torch.testing.assert_close(out["light_ambient_residual"], (0.5 * expected_r).expand(3, h, w), atol=1e-5, rtol=0)


def test_channels_three_shades_each_channel():
    shader = _grid(channels=3)
    with torch.no_grad():
        shader.ambient.residual[1] = 0.2
    radiance, _ = shader.ambient(torch.full((3, 1, 1), 0.5), torch.zeros(1, 1), torch.zeros(3, 1, 1))
    torch.testing.assert_close(radiance[:, 0, 0], torch.tensor([0.05, 0.15, 0.05]))


def test_irradiance_is_clamped_at_zero():
    shader = _grid(c=0.1)
    with torch.no_grad():
        shader.ambient.residual[:] = -0.4
    radiance, aux = shader.ambient(torch.ones(3, 1, 1), torch.zeros(1, 1), torch.zeros(3, 1, 1))
    assert radiance.eq(0).all()
    torch.testing.assert_close(aux["ambient_residual"], torch.full((3, 1, 1), -0.1))


# ---------------------------------------------------------------- gradients


def test_gradient_reaches_the_grid_and_c_but_not_the_position():
    shader = _grid()
    amb = shader.ambient
    amb.c = amb.c.clone().requires_grad_()
    points = torch.tensor([0.25, 0.0, 0.0]).view(3, 1, 1).requires_grad_()
    radiance, _ = amb(torch.ones(3, 1, 1), torch.zeros(1, 1), points)
    radiance.sum().backward()
    assert points.grad is None
    assert float(amb.c.grad) == 3.0
    g = amb.residual.grad[0]
    assert g.nonzero().shape[0] == 2  # halfway between two vertices along x
    assert float(g.sum()) == pytest.approx(3.0)


def test_penalty_acts_on_the_grid_only():
    shader = _grid(c=0.1)
    amb = shader.ambient
    amb.c = amb.c.clone().requires_grad_()
    with torch.no_grad():
        amb.residual[:] = 0.2
    albedo = torch.full((3, 1, 1), 0.5, requires_grad=True)
    _, aux = amb(albedo, torch.zeros(1, 1), torch.zeros(3, 1, 1))
    torch.testing.assert_close(aux["map_penalty"], torch.tensor(0.05 * 0.5 * 0.2))
    aux["map_penalty"].backward()
    assert albedo.grad is None and amb.c.grad is None
    assert float(amb.residual.grad.sum()) == pytest.approx(0.05 * 0.5)


def test_points_outside_take_the_border_and_are_counted():
    shader = _grid()
    amb = shader.ambient
    with torch.no_grad():
        amb.residual[..., -1] = 0.3  # the vertices at x = +half_extent
    points = torch.tensor([[[5.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
    radiance, _ = amb(torch.ones(3, 1, 2), torch.zeros(1, 2), points)
    assert float(radiance[0, 0, 0]) == pytest.approx(0.4)
    assert float(radiance[0, 0, 1]) == pytest.approx(0.1)
    assert amb.residual_summary()["outside_frac"] == 0.5


# ---------------------------------------------------------------- state and logs


def test_state_round_trip_learning_switch_and_pickle():
    shader = _grid()
    _set_linear(shader, torch.tensor([0.1, 0.0, 0.0]))
    other = pickle.loads(pickle.dumps(_grid()))
    other.load_learnable_state(shader.learnable_state())
    torch.testing.assert_close(other.ambient.residual, shader.ambient.residual)
    other.set_learning(False)
    assert not other.ambient.residual.requires_grad


def test_learning_updates_the_grid_after_its_schedule_only():
    from light_models.learning import LightLearner

    shader = _grid()
    learner = LightLearner(shader.learnable_params())
    radiance, _ = shader.ambient(torch.ones(3, 1, 1), torch.zeros(1, 1), torch.zeros(3, 1, 1))
    radiance.sum().backward()
    learner.step(2)
    assert shader.ambient.residual.eq(0).all()
    radiance, _ = shader.ambient(torch.ones(3, 1, 1), torch.zeros(1, 1), torch.zeros(3, 1, 1))
    radiance.sum().backward()
    learner.step(3)
    assert shader.ambient.residual.ne(0).any()


def test_log_keeps_a_summary_in_the_csv_and_the_values_for_the_evaluation(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "eval_map", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "verify", "eval_map.py"))
    eval_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_map)

    block = {"type": "position_grid", "c": {"source": "value", "value": 0.1, "learnable": True, "lr": 0.001, "prior_weight": 0.0}}
    block.update(GRID)
    schedule = [{"after_keyframes": 1, "unfreeze": ["ambient.c"]}, {"after_keyframes": 2, "unfreeze": ["ambient.residual"]}]
    shader = build_shader(_config(block, schedule), PARAMS)
    _set_linear(shader, torch.tensor([0.1, 0.0, 0.0]))
    state, large = shader.learned_for_log()
    assert set(large) == {"ambient.residual"} and "ambient.residual" not in state
    assert state["ambient.residual.vertices_moved"] > 0
    log_learned_light(str(tmp_path), 5, 3, state, {}, large)
    assert os.path.isfile(tmp_path / LEARNED_LARGE_NAME)
    header = (tmp_path / "light_learned.csv").read_text().splitlines()[0].split(",")
    assert len(header) < 10
    other = build_shader(_config(block, schedule), PARAMS)
    other.load_learnable_state(eval_map.learned_state(str(tmp_path / "light_learned.csv")))
    torch.testing.assert_close(other.ambient.residual, shader.ambient.residual)
    assert float(other.ambient.c) == pytest.approx(float(shader.ambient.c))


def test_the_run_zip_carries_the_grid_values(tmp_path):
    import zipfile

    from utils.record_zip import zip_run

    run = tmp_path / "experiment" / "time"
    run.mkdir(parents=True)
    log_learned_light(str(run), 5, 3, {"ambient.residual.min": 0.0}, {}, {"ambient.residual": torch.zeros(1, 2, 2, 2).numpy()})
    with zipfile.ZipFile(zip_run(str(run), str(tmp_path / "zips"))) as z:
        assert f"experiment__time/{LEARNED_LARGE_NAME}" in z.namelist()


def test_small_parameters_still_log_without_a_large_file(tmp_path):
    log_learned_light(str(tmp_path), 5, 3, {"ambient.c": 0.1}, {})
    assert not os.path.isfile(tmp_path / LEARNED_LARGE_NAME)
