"""Ambient field on a dense box that grows with the scene seen
(position_grid_growing, docs/DECISIONS.md D144). The oracle: inside its box it
shades exactly as the fixed dense position_grid holding the same values; a
point outside reads 0; growing keeps every learned value (and the optimizer
state) at its world vertex."""
import importlib.util
import os
import pickle

import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader  # noqa: E402
from light_models.diagnostics import LEARNED_LARGE_NAME, log_learned_light  # noqa: E402

from test_ambient_position_grid import PARAMS, SCHEDULE, _config, _grid, _set_linear, _view, _world_points  # noqa: E402

GROWING = {"channels": 1, "cell_m": 0.5, "capacity": 4096, "margin_m": 0.5, "residual_lr": 0.001, "l1_weight": 0.05}


def _growing(c=0.1, schedule=SCHEDULE, learn_c=False, **over):
    c_spec = {"source": "value", "value": c, "learnable": learn_c}
    if learn_c:
        c_spec.update(lr=0.001, prior_weight=0.0)
    block = {"type": "position_grid_growing", "c": c_spec}
    block.update(GROWING)
    block.update(over)
    return build_shader(_config(block, schedule), PARAMS)


def _vertex_positions(amb):
    """(3, n) world positions of the box's vertices, in buffer order."""
    x0, y0, z0, nx, ny, nz = amb.box.tolist()
    i = torch.arange(nx * ny * nz)
    idx = torch.stack([i % nx + x0, (i // nx) % ny + y0, i // (nx * ny) + z0])
    return idx.to(torch.float32) * amb.cell_m


def _set_linear_growing(shader, g, offset=0.0):
    amb = shader.ambient
    pos = _vertex_positions(amb)
    with torch.no_grad():
        amb.residual[0, :amb.n_vertices] = (g.view(3, 1) * pos).sum(0) + offset


def _point(x, y, z):
    return torch.tensor([[x], [y], [z]])


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("key", sorted(GROWING))
def test_every_option_is_required(key):
    block = {"type": "position_grid_growing", "c": {"source": "value", "value": 0.1, "learnable": False}}
    block.update({k: v for k, v in GROWING.items() if k != key})
    with pytest.raises(KeyError, match=key):
        build_shader(_config(block), PARAMS)


@pytest.mark.parametrize("over", [{"channels": 2}, {"cell_m": 0}, {"capacity": 4}, {"capacity": 1.0e4},
                                  {"margin_m": -0.1}, {"margin_m": True}, {"residual_lr": 0}, {"l1_weight": -1.0}])
def test_bad_options_raise(over):
    with pytest.raises(ValueError):
        _growing(**over)


def test_starts_empty_and_learns_like_the_other_grids():
    shader = _growing()
    amb = shader.ambient
    assert amb.n_vertices == 0 and tuple(amb.residual.shape) == (1, 4096)
    p = shader.learnable_params()["ambient.residual"]
    assert p.lr == 0.001 and p.after_keyframes == 2 and p.prior_weight == 0.0
    assert shader.allocates_field()


# ---------------------------------------------------------------- growth


def test_first_box_holds_the_cells_plus_the_margin():
    amb = _growing().ambient
    # Cells [0, 0.5) and [0.5, 1) along x: vertices 0..2 on x, 0..1 on y, z;
    # margin 0.5 m = 1 cell on every side.
    assert amb.allocate(torch.tensor([[0.1, 0.6], [0.1, 0.1], [0.1, 0.1]])) == 5 * 4 * 4
    assert amb.box.tolist() == [-1, -1, -1, 5, 4, 4]


def test_points_inside_do_not_grow_the_box():
    amb = _growing().ambient
    amb.allocate(_point(0.1, 0.1, 0.1))
    box = amb.box.tolist()
    # Within the margin, not outside the box: no growth.
    assert amb.allocate(_point(0.7, -0.3, 0.9)) == 0
    assert amb.box.tolist() == box and amb.growths == 0


def test_growth_only_on_the_side_that_needs_it():
    amb = _growing().ambient
    amb.allocate(_point(0.1, 0.1, 0.1))  # box x: -1..2
    amb.allocate(_point(2.2, 0.1, 0.1))  # cell 4: needs vertex 5, + 1 cell margin
    x0, y0, z0, nx, ny, nz = amb.box.tolist()
    assert (x0, x0 + nx - 1) == (-1, 6)
    assert (y0, ny, z0, nz) == (-1, 4, -1, 4)
    assert amb.growths == 1


def test_growth_keeps_every_value_at_its_world_vertex():
    shader = _growing()
    amb = shader.ambient
    amb.allocate(torch.tensor([[0.1, 0.9], [0.1, 0.4], [0.1, 0.1]]))
    g = torch.tensor([0.3, -0.2, 0.1])
    _set_linear_growing(shader, g, offset=0.05)
    probe = torch.tensor([[0.3, 0.7], [0.2, 0.1], [0.4, 0.05]]).view(3, 1, 2)
    before = amb.field(probe).clone()
    companion = amb.residual.detach().clone() * 2  # e.g. an Adam moment
    # Grow towards negative x, y and positive z: every old vertex moves.
    amb.allocate(torch.tensor([[-2.3], [-1.1], [3.2]]), companions=(companion,))
    torch.testing.assert_close(amb.field(probe), before)
    n = amb.n_vertices
    pos = _vertex_positions(amb)
    old = ((pos[0] >= -0.5) & (pos[0] <= 1.5) & (pos[1] >= -0.5) & (pos[1] <= 1.0)
           & (pos[2] >= -0.5) & (pos[2] <= 1.0))
    torch.testing.assert_close(amb.residual[0, :n][old], (g.view(3, 1) * pos[:, old]).sum(0) + 0.05)
    assert amb.residual[0, :n][~old].eq(0).all()
    torch.testing.assert_close(companion[0, :n], 2 * amb.residual.detach()[0, :n])


def test_too_large_a_box_is_an_error():
    amb = _growing(capacity=100).ambient
    with pytest.raises(RuntimeError, match="capacity"):
        amb.allocate(torch.tensor([[0.0, 20.0], [0.0, 0.0], [0.0, 0.0]]))
    assert amb.n_vertices == 0


# ---------------------------------------------------------------- shading


def test_empty_box_is_the_constant_ambient():
    cam, gbuffer = _view()
    plain = build_shader(_config({"type": "multiplicative_const", "c": {"source": "value", "value": 0.1, "learnable": False}},
                                 schedule=None), PARAMS)
    out = _growing()(gbuffer, cam)
    torch.testing.assert_close(out["radiance_linear"], plain(gbuffer, cam)["radiance_linear"])
    assert float(out["light_map_penalty"]) == 0.0


def test_same_shading_as_the_dense_grid():
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    g = torch.tensor([0.02, -0.03, 0.05])
    dense = _grid()
    _set_linear(dense, g, offset=0.2)
    growing = _growing()
    growing.observe_surface(gbuffer, cam)
    _set_linear_growing(growing, g, offset=0.2)
    want, got = dense(gbuffer, cam), growing(gbuffer, cam)
    torch.testing.assert_close(got["radiance_linear"], want["radiance_linear"], atol=1e-6, rtol=0)
    torch.testing.assert_close(got["light_ambient_residual"], want["light_ambient_residual"], atol=1e-6, rtol=0)
    torch.testing.assert_close(got["light_map_penalty"], want["light_map_penalty"], atol=1e-7, rtol=0)


def test_outside_the_box_reads_zero():
    amb = _growing(margin_m=0.0).ambient
    amb.allocate(_point(0.1, 0.1, 0.1))  # box = the cell [0, 0.5]^3
    with torch.no_grad():
        amb.residual[0, :8] = 1.0
    pts = torch.tensor([[[0.25, 0.75, 5.0]], [[0.25, 0.25, 0.0]], [[0.25, 0.25, 0.0]]])
    torch.testing.assert_close(amb.field(pts)[0, 0], torch.tensor([1.0, 0.5, 0.0]))


def test_channels_three():
    amb = _growing(channels=3, margin_m=0.0).ambient
    amb.allocate(_point(0.1, 0.1, 0.1))
    with torch.no_grad():
        amb.residual[1, :8] = 0.2
    radiance, _ = amb(torch.full((3, 1, 1), 0.5), torch.zeros(1, 1), torch.tensor([0.1, 0.1, 0.1]).view(3, 1, 1))
    torch.testing.assert_close(radiance[:, 0, 0], torch.tensor([0.05, 0.15, 0.05]))


# ---------------------------------------------------------------- gradients and the optimizer


def test_gradient_reaches_the_used_vertices_and_c_not_the_position():
    shader = _growing(learn_c=True, schedule=[{"after_keyframes": 1, "unfreeze": ["ambient.c"]},
                                              {"after_keyframes": 2, "unfreeze": ["ambient.residual"]}])
    amb = shader.ambient
    amb.allocate(torch.tensor([[0.1, 2.1], [0.1, 0.1], [0.1, 0.1]]))
    points = torch.tensor([0.25, 0.0, 0.0]).view(3, 1, 1).requires_grad_()
    radiance, _ = amb(torch.ones(3, 1, 1), torch.zeros(1, 1), points)
    radiance.sum().backward()
    assert points.grad is None
    assert float(amb.c.grad) == 3.0
    g = amb.residual.grad[0]
    assert g.nonzero().shape[0] == 2
    assert float(g.sum()) == pytest.approx(3.0)


def test_adam_state_moves_with_the_values():
    from light_models.learning import LightLearner

    shader = _growing()
    amb = shader.ambient
    amb.allocate(_point(0.1, 0.1, 0.1))
    learner = LightLearner(shader.learnable_params())
    probe = torch.tensor([0.3, 0.2, 0.1]).view(3, 1, 1)
    for _ in range(3):
        radiance, _ = amb(torch.ones(3, 1, 1), torch.zeros(1, 1), probe)
        radiance.sum().backward()
        learner.step(3)
    state = learner.optimizer.state[amb.residual]
    before = {k: v.clone() for k, v in state.items() if torch.is_tensor(v) and v.shape == amb.residual.shape}
    value_before = amb.field(probe).clone()
    moved_before = {k: v[v != 0].sort().values for k, v in before.items()}
    amb.allocate(_point(-3.1, 0.1, 0.1), companions=learner.state_tensors("ambient.residual"))
    assert amb.growths == 1
    torch.testing.assert_close(amb.field(probe), value_before)
    for k, v in state.items():
        if k in moved_before:
            # The same nonzero moments, now at the vertices' new places.
            torch.testing.assert_close(v[v != 0].sort().values, moved_before[k])
    # A step after the growth moves the same vertices as one before it would.
    radiance, _ = amb(torch.ones(3, 1, 1), torch.zeros(1, 1), probe)
    radiance.sum().backward()
    grad_vertices = amb.residual.grad[0].nonzero().flatten()
    learner.step(3)
    for k in moved_before:
        assert state[k][0, grad_vertices].ne(0).all()


# ---------------------------------------------------------------- state, sync, logs


def test_state_carries_the_box_to_the_frontend():
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    backend = _growing()
    backend.observe_surface(gbuffer, cam)
    _set_linear_growing(backend, torch.tensor([0.1, 0.0, 0.0]))
    state = backend.learnable_state()
    assert set(state) == {"ambient.residual", "ambient.residual_box"}
    frontend = pickle.loads(pickle.dumps(_growing()))
    frontend.set_learning(False)
    frontend.load_learnable_state(state)
    assert frontend.ambient.box.tolist() == backend.ambient.box.tolist()
    torch.testing.assert_close(frontend(gbuffer, cam)["radiance_linear"], backend(gbuffer, cam)["radiance_linear"])
    with pytest.raises(ValueError):
        frontend.load_learnable_state({"ambient.residual": state["ambient.residual"]})
    with pytest.raises(ValueError, match="capacity"):
        frontend.load_learnable_state({**state, "ambient.residual_box": [0, 0, 0, 100, 100, 100]})


def test_buffers_are_updated_in_place():
    shader = _growing()
    amb = shader.ambient
    ids = [id(t) for t in (amb.box, amb.residual)]
    amb.allocate(_point(0.1, 0.1, 0.1))
    amb.allocate(_point(-2.1, 0.1, 0.1))
    shader.load_learnable_state(shader.learnable_state())
    assert [id(t) for t in (amb.box, amb.residual)] == ids


def test_own_storage_separates_the_backend_box():
    shader = _growing()
    amb = shader.ambient
    shared = amb.box
    shader.own_learnable_storage()
    amb.allocate(_point(0.1, 0.1, 0.1))
    assert shared.eq(0).all()


def test_log_round_trip_for_the_evaluation(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "eval_map", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "verify", "eval_map.py"))
    eval_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_map)

    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    shader = _growing()
    shader.observe_surface(gbuffer, cam)
    _set_linear_growing(shader, torch.tensor([0.1, 0.0, 0.0]), offset=0.05)
    state, large = shader.learned_for_log()
    assert set(large) == {"ambient.residual", "ambient.residual_box"}
    assert state["ambient.residual.vertices_in_box"] == shader.ambient.n_vertices
    log_learned_light(str(tmp_path), 5, 3, state, {}, large)
    assert os.path.isfile(tmp_path / LEARNED_LARGE_NAME)
    other = _growing()
    other.load_learnable_state(eval_map.learned_state(str(tmp_path / "light_learned.csv")))
    torch.testing.assert_close(other(gbuffer, cam)["radiance_linear"], shader(gbuffer, cam)["radiance_linear"])


# ---------------------------------------------------------------- GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_cpu():
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    shader = _growing()
    amb = shader.ambient
    shader.observe_surface(gbuffer, cam)
    _set_linear_growing(shader, torch.tensor([0.1, -0.2, 0.05]))
    pts = _world_points(cam, h, w, z)
    with torch.no_grad():
        torch.testing.assert_close(amb.field(pts.cuda()).cpu(), amb.field(pts), atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_lookup_reads_a_grown_box():
    # The fused tracking path (D134): traced with make_fx, GPU buffers lifted
    # to inputs, compiled by Inductor. Growing the box afterwards must reach
    # the compiled graph through the same buffers (refreshed by one eager
    # call, as the frame's eager iteration does), with no recompilation.
    pytest.importorskip("triton")
    from torch.fx.experimental.proxy_tensor import make_fx

    from utils.graphed_iteration import lift_constants

    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    shader = _growing(margin_m=0.0)
    amb = shader.ambient
    amb.allocate(_point(0.1, 0.1, 0.1))
    pts = _world_points(cam, h, w, z).cuda()
    with torch.no_grad():
        amb.field(pts)
        gm = make_fx(lambda p: amb.field(p))(pts)
        consts = lift_constants(gm)
        fn = torch.compile(gm, dynamic=False)
        shader.observe_surface(gbuffer, cam)
        _set_linear_growing(shader, torch.tensor([0.1, -0.2, 0.05]), offset=0.3)
        want = amb.field(pts)
        got = fn(pts, *consts)
    if isinstance(got, (tuple, list)):
        got = got[0]
    torch.testing.assert_close(got, want, atol=1e-6, rtol=0)
    assert want.abs().max() > 0.1
