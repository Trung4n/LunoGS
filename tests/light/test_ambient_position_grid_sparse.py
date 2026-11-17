"""Ambient field with vertices allocated where the map is seen
(position_grid_sparse, docs/DECISIONS.md D143). The oracle: with every vertex
the surface touches allocated, it shades exactly as the dense position_grid
holding the same values; a vertex not allocated reads 0, as an untouched
vertex of the dense grid."""
import importlib.util
import os
import pickle

import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader  # noqa: E402
from light_models.ambient import _KEY_BITS, _KEY_OFFSET  # noqa: E402
from light_models.diagnostics import LEARNED_LARGE_NAME, log_learned_light  # noqa: E402

from test_ambient_position_grid import PARAMS, SCHEDULE, _config, _grid, _set_linear, _view, _world_points  # noqa: E402

SPARSE = {"channels": 1, "cell_m": 0.5, "capacity": 4096, "residual_lr": 0.001, "l1_weight": 0.05}


def _sparse(c=0.1, schedule=SCHEDULE, learn_c=False, **over):
    c_spec = {"source": "value", "value": c, "learnable": learn_c}
    if learn_c:
        c_spec.update(lr=0.001, prior_weight=0.0)
    block = {"type": "position_grid_sparse", "c": c_spec}
    block.update(SPARSE)
    block.update(over)
    return build_shader(_config(block, schedule), PARAMS)


def _vertex_positions(amb):
    """(3, n_allocated) world positions of the allocated vertices, slot order."""
    keys = amb.slot_keys[:amb.n_allocated]
    mask = (1 << _KEY_BITS) - 1
    idx = torch.stack([(keys >> (2 * _KEY_BITS)) & mask, (keys >> _KEY_BITS) & mask, keys & mask]) - _KEY_OFFSET
    return idx.to(torch.float32) * amb.cell_m


def _set_linear_sparse(shader, g, offset=0.0):
    amb = shader.ambient
    pos = _vertex_positions(amb)
    with torch.no_grad():
        amb.residual[0, :amb.n_allocated] = (g.view(3, 1) * pos).sum(0) + offset


def _observe(shader, cam, gbuffer):
    return shader.observe_surface(gbuffer, cam)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("key", sorted(SPARSE))
def test_every_option_is_required(key):
    block = {"type": "position_grid_sparse", "c": {"source": "value", "value": 0.1, "learnable": False}}
    block.update({k: v for k, v in SPARSE.items() if k != key})
    with pytest.raises(KeyError, match=key):
        build_shader(_config(block), PARAMS)


@pytest.mark.parametrize("over", [{"channels": 2}, {"cell_m": 0}, {"capacity": 4}, {"capacity": 100.0},
                                  {"capacity": True}, {"residual_lr": 0}, {"l1_weight": -0.1}])
def test_bad_options_raise(over):
    with pytest.raises(ValueError):
        _sparse(**over)


def test_no_bounding_cube_option():
    with pytest.raises(ValueError, match="half_extent_m"):
        _sparse(half_extent_m=8.0)


def test_pool_and_learning_parameters():
    shader = _sparse()
    amb = shader.ambient
    assert tuple(amb.residual.shape) == (1, 4096) and amb.n_allocated == 0
    p = shader.learnable_params()["ambient.residual"]
    assert p.lr == 0.001 and p.after_keyframes == 2 and p.prior_weight == 0.0
    assert shader.allocates_field()
    assert not _grid().allocates_field()


# ---------------------------------------------------------------- allocation


def test_allocation_takes_the_eight_corners_of_each_cell_once():
    amb = _sparse().ambient
    # Two points in one cell, one in the next cell along x: 8 + 4 vertices.
    pts = torch.tensor([[0.1, 0.2, 0.6], [0.1, 0.3, 0.1], [0.1, 0.1, 0.1]])
    assert amb.allocate(pts) == 12
    assert amb.allocate(pts) == 0
    assert amb.n_allocated == 12
    assert bool((amb.keys_sorted[:12][1:] > amb.keys_sorted[:12][:-1]).all())
    # Negative coordinates have their own cells; the vertex at the origin is
    # shared with the cells above, so 11 are new.
    assert amb.allocate(-pts) == 11


def test_slots_keep_their_values_when_more_vertices_come():
    amb = _sparse().ambient
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    with torch.no_grad():
        amb.residual[0, :8] = torch.arange(8.0)
    before = amb.field(torch.tensor([0.1, 0.1, 0.1]).view(3, 1, 1)).clone()
    amb.allocate(torch.tensor([[-3.1, 5.2], [2.1, -0.4], [0.1, 9.9]]))
    torch.testing.assert_close(amb.residual[0, :8], torch.arange(8.0))
    torch.testing.assert_close(amb.field(torch.tensor([0.1, 0.1, 0.1]).view(3, 1, 1)), before)


def test_full_pool_is_an_error():
    amb = _sparse(capacity=10).ambient
    with pytest.raises(RuntimeError, match="capacity"):
        amb.allocate(torch.tensor([[0.1, 0.6], [0.1, 0.1], [0.1, 0.1]]))


def test_points_too_far_to_key_are_an_error():
    amb = _sparse().ambient
    with pytest.raises(ValueError, match="keyed"):
        amb.allocate(torch.tensor([[1.0e6], [0.0], [0.0]]))


def test_observe_surface_uses_every_covered_pixel():
    # Covered but untrusted pixels are shaded at the fallback depth (the
    # median trusted depth, here z): they read the field, so they allocate.
    # Uncovered pixels (albedo 0, no gradient) do not.
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    gbuffer["opacity"][0, :, 0] = 0.0  # not covered
    gbuffer["depth"][0, :, 0] = 0.0
    gbuffer["opacity"][0, :, 1] = 0.1  # covered, below opacity_thr
    gbuffer["depth"][0, :, 1] = 0.3 * 0.1  # its own depth would be 0.3 m
    shader = _sparse()
    _observe(shader, cam, gbuffer)
    covered = _world_points(cam, h, w, z)[:, :, 1:].reshape(3, -1)
    ref = _sparse().ambient
    ref.allocate(covered)
    assert torch.equal(shader.ambient.keys_sorted, ref.keys_sorted)


# ---------------------------------------------------------------- shading


def test_nothing_allocated_is_the_constant_ambient():
    cam, gbuffer = _view()
    plain = build_shader(_config({"type": "multiplicative_const", "c": {"source": "value", "value": 0.1, "learnable": False}},
                                 schedule=None), PARAMS)
    out = _sparse()(gbuffer, cam)
    torch.testing.assert_close(out["radiance_linear"], plain(gbuffer, cam)["radiance_linear"])
    assert float(out["light_map_penalty"]) == 0.0


def test_same_shading_as_the_dense_grid():
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    g = torch.tensor([0.02, -0.03, 0.05])
    dense = _grid()
    _set_linear(dense, g, offset=0.2)
    sparse = _sparse()
    _observe(sparse, cam, gbuffer)
    _set_linear_sparse(sparse, g, offset=0.2)
    want, got = dense(gbuffer, cam), sparse(gbuffer, cam)
    torch.testing.assert_close(got["radiance_linear"], want["radiance_linear"], atol=1e-6, rtol=0)
    torch.testing.assert_close(got["light_ambient_residual"], want["light_ambient_residual"], atol=1e-6, rtol=0)
    torch.testing.assert_close(got["light_map_penalty"], want["light_map_penalty"], atol=1e-7, rtol=0)


def test_missing_corners_read_zero():
    amb = _sparse().ambient
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))  # the cell [0, 0.5)^3
    with torch.no_grad():
        amb.residual[0, :8] = 1.0
    # Inside the cell: all 8 corners present. In the next cell along x at
    # x = 0.75: only its 4 corners at x = 0.5 exist, weight 0.5 in total.
    pts = torch.tensor([[[0.25, 0.75]], [[0.25, 0.25]], [[0.25, 0.25]]])
    r = amb.field(pts)
    torch.testing.assert_close(r[0, 0], torch.tensor([1.0, 0.5]))


def test_channels_three():
    amb = _sparse(channels=3).ambient
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    with torch.no_grad():
        amb.residual[1, :8] = 0.2
    radiance, _ = amb(torch.full((3, 1, 1), 0.5), torch.zeros(1, 1), torch.tensor([0.1, 0.1, 0.1]).view(3, 1, 1))
    torch.testing.assert_close(radiance[:, 0, 0], torch.tensor([0.05, 0.15, 0.05]))


# ---------------------------------------------------------------- gradients


def test_gradient_reaches_the_used_slots_and_c_not_the_position():
    shader = _sparse(learn_c=True, schedule=[{"after_keyframes": 1, "unfreeze": ["ambient.c"]},
                                             {"after_keyframes": 2, "unfreeze": ["ambient.residual"]}])
    amb = shader.ambient
    amb.allocate(torch.tensor([[0.1, 2.1], [0.1, 0.1], [0.1, 0.1]]))
    points = torch.tensor([0.25, 0.0, 0.0]).view(3, 1, 1).requires_grad_()
    radiance, _ = amb(torch.ones(3, 1, 1), torch.zeros(1, 1), points)
    radiance.sum().backward()
    assert points.grad is None
    assert float(amb.c.grad) == 3.0
    g = amb.residual.grad[0]
    assert g.nonzero().shape[0] == 2  # halfway between two vertices along x
    assert float(g.sum()) == pytest.approx(3.0)
    assert g[amb.n_allocated:].eq(0).all()


def test_learning_updates_only_after_the_schedule():
    from light_models.learning import LightLearner

    shader = _sparse()
    amb = shader.ambient
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    learner = LightLearner(shader.learnable_params())
    for n_kf, moves in ((2, False), (3, True)):
        radiance, _ = amb(torch.ones(3, 1, 1), torch.zeros(1, 1), torch.tensor([0.1, 0.1, 0.1]).view(3, 1, 1))
        radiance.sum().backward()
        learner.step(n_kf)
        assert amb.residual.ne(0).any() == moves
    assert amb.residual[:, 8:].eq(0).all()


# ---------------------------------------------------------------- state, sync, logs


def test_state_carries_the_keys_to_the_frontend():
    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    backend = _sparse()
    _observe(backend, cam, gbuffer)
    _set_linear_sparse(backend, torch.tensor([0.1, 0.0, 0.0]))
    state = backend.learnable_state()
    assert set(state) == {"ambient.residual", "ambient.residual_keys"}
    frontend = pickle.loads(pickle.dumps(_sparse()))
    frontend.set_learning(False)
    frontend.load_learnable_state(state)
    assert frontend.ambient.n_allocated == backend.ambient.n_allocated
    torch.testing.assert_close(frontend(gbuffer, cam)["radiance_linear"], backend(gbuffer, cam)["radiance_linear"])
    with pytest.raises(ValueError):
        frontend.load_learnable_state({"ambient.residual": state["ambient.residual"]})


def test_key_buffers_are_updated_in_place():
    # The GPU copies of tracking (on_device) and its CUDA graphs keep the
    # same buffers (D130): allocation and sync never replace them.
    shader = _sparse()
    amb = shader.ambient
    ids = [id(t) for t in (amb.keys_sorted, amb.slot_of_sorted, amb.residual)]
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    shader.load_learnable_state(shader.learnable_state())
    assert [id(t) for t in (amb.keys_sorted, amb.slot_of_sorted, amb.residual)] == ids


def test_own_storage_separates_the_backend_keys():
    shader = _sparse()
    amb = shader.ambient
    shared = amb.slot_keys
    shader.own_learnable_storage()
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    assert shared.eq(torch.iinfo(torch.int64).max).all()


def test_log_round_trip_for_the_evaluation(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "eval_map", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "verify", "eval_map.py"))
    eval_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_map)

    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    shader = _sparse()
    _observe(shader, cam, gbuffer)
    _set_linear_sparse(shader, torch.tensor([0.1, 0.0, 0.0]), offset=0.05)
    state, large = shader.learned_for_log()
    assert set(large) == {"ambient.residual", "ambient.residual_keys"}
    assert state["ambient.residual.vertices_allocated"] == shader.ambient.n_allocated
    log_learned_light(str(tmp_path), 5, 3, state, {}, large)
    assert os.path.isfile(tmp_path / LEARNED_LARGE_NAME)
    other = _sparse()
    other.load_learnable_state(eval_map.learned_state(str(tmp_path / "light_learned.csv")))
    torch.testing.assert_close(other(gbuffer, cam)["radiance_linear"], shader(gbuffer, cam)["radiance_linear"])


# ---------------------------------------------------------------- GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_cpu_and_follows_new_allocations_in_place():
    from light_models.device import on_device

    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    shader = _sparse()
    amb = shader.ambient
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    pts = _world_points(cam, h, w, z)
    dev = pts.cuda().device  # "cuda:0", as the shading passes it
    gpu_keys = on_device(amb.keys_sorted, dev, torch.int64)
    _observe(shader, cam, gbuffer)
    _set_linear_sparse(shader, torch.tensor([0.1, -0.2, 0.05]))
    with torch.no_grad():
        want = amb.field(pts)
        got = amb.field(pts.cuda())
    torch.testing.assert_close(got.cpu(), want, atol=1e-6, rtol=0)
    assert on_device(amb.keys_sorted, dev, torch.int64) is gpu_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_lookup_reads_new_allocations():
    # The fused tracking path (D134): the shading is traced (make_fx), the
    # GPU buffers it reads become inputs (lift_constants), Inductor compiles
    # it. A later allocation must reach the compiled graph through the same
    # buffers, refreshed by one eager call as the frame's eager iteration does.
    pytest.importorskip("triton")
    from torch.fx.experimental.proxy_tensor import make_fx

    from utils.graphed_iteration import lift_constants

    h, w, z = 4, 5, 1.5
    cam, gbuffer = _view(h, w, z)
    shader = _sparse()
    amb = shader.ambient
    amb.allocate(torch.tensor([[0.1], [0.1], [0.1]]))
    pts = _world_points(cam, h, w, z).cuda()
    with torch.no_grad():
        amb.field(pts)
        gm = make_fx(lambda p: amb.field(p))(pts)
        consts = lift_constants(gm)
        fn = torch.compile(gm, dynamic=False)
        _observe(shader, cam, gbuffer)
        _set_linear_sparse(shader, torch.tensor([0.1, -0.2, 0.05]), offset=0.3)
        want = amb.field(pts)  # eager: refreshes the GPU buffers in place
        got = fn(pts, *consts)
    if isinstance(got, (tuple, list)):
        got = got[0]
    torch.testing.assert_close(got, want, atol=1e-6, rtol=0)
    assert want.abs().max() > 0.1
