"""Normals from the Gaussians' shortest axis (normal source shortest_axis)
and the shape regularizer (Training.scale_regularizer), docs/DECISIONS.md D149."""
import math
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from light_models.normals import ShortestAxis, quaternion_axes, quaternion_from_z  # noqa: E402
from light_models.registry import build_component  # noqa: E402
from utils.scale_regularizer import read_scale_regularizer  # noqa: E402

H, W = 4, 5
INIT = {"type": "depth_fd", "stencil_px": 1}


def _source(**over):
    block = {"type": "shortest_axis", "min_agreement": 0.5, "init_flatness": 0.1, "init": INIT,
             "supervision": None, "tracking_refresh_every": None,
             "raster_gradient": "all"}
    block.update(over)
    return build_component("normals", block, "Light.cosine.normal_source", None)


def _rand_quat(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=1, keepdim=True)


# ---- quaternion helpers ----

def test_axes_are_a_rotation():
    R = quaternion_axes(_rand_quat(50))
    eye = torch.eye(3, dtype=R.dtype).expand(50, 3, 3)
    torch.testing.assert_close(R @ R.transpose(1, 2), eye)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(50, dtype=R.dtype))


def test_axes_match_build_rotation_formula():
    # 90 deg about z: local x -> world y.
    q = torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]], dtype=torch.float64)
    R = quaternion_axes(q)[0]
    torch.testing.assert_close(R[:, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))


def test_quaternion_from_z_takes_z_to_the_normal_up_to_sign():
    g = torch.Generator().manual_seed(1)
    n = torch.randn(200, 3, generator=g, dtype=torch.float64)
    n = n / n.norm(dim=1, keepdim=True)
    n[0] = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)  # the half-turn case
    z = quaternion_axes(quaternion_from_z(n))[:, :, 2]
    torch.testing.assert_close((z * n).sum(1).abs(), torch.ones(200, dtype=torch.float64))


# ---- the normal source ----

class _Map:
    def __init__(self, xyz, q, s):
        self.get_xyz, self.get_rotation, self.get_scaling = xyz, q, s


def _camera(R=None, T=None):
    return SimpleNamespace(R=torch.eye(3) if R is None else R, T=torch.zeros(3) if T is None else T,
                           frame_cache=None)


def test_colours_are_the_shortest_axis_facing_the_camera():
    # Discs whose normal axis is local y (scale index 1), rotated about x by
    # 90 deg: local y -> world z. Two discs, one in front and one with the
    # normal pointing away from the camera, which must be flipped.
    q = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]] * 2)
    s = torch.tensor([[0.1, 0.001, 0.1]] * 2)
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.3, 0.0, 1.0]])
    colours = _source().frame_rasters(_Map(xyz, q, s), _camera())["gaussian_normal"]()
    torch.testing.assert_close(colours, torch.tensor([[0.0, 0.0, -1.0]] * 2), atol=1e-6, rtol=0)


def test_colours_are_in_the_camera_frame():
    # Camera rotated 90 deg about y (world -> camera); a disc facing world -x
    # in front of it.
    c, s_ = 0.0, 1.0
    R = torch.tensor([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])  # x_c = R x_w
    n_w = torch.tensor([1.0, 0.0, 0.0])
    q = quaternion_from_z(n_w[None]).float()
    xyz = (R.T @ torch.tensor([0.0, 0.0, 2.0]))[None]  # in front of the camera
    sc = torch.tensor([[0.1, 0.1, 0.001]])
    col = _source().frame_rasters(_Map(xyz, q, sc), _camera(R=R))["gaussian_normal"]()
    expect = R @ n_w
    expect = -expect if (expect @ (R @ xyz[0])) > 0 else expect
    torch.testing.assert_close(col[0], expect, atol=1e-6, rtol=0)


def _gbuffer(raw, opacity=1.0):
    return {"gaussian_normal": raw, "opacity": torch.full((1, H, W), opacity), "pixel_offset": 0.5}


def _points():
    p = torch.zeros(3, H, W)
    p[2] = 2.0
    return p


def test_blended_normal_is_normalized_and_validity_follows_agreement():
    raw = torch.zeros(3, H, W)
    raw[2] = -0.9  # agreeing Gaussians: length 0.9 of opacity 1
    raw[:, 0, 0] = torch.tensor([0.0, 0.0, -0.3])  # disagreeing: 0.3 < 0.5
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[1, 1] = False  # not trusted by the G-buffer
    n, ok = _source()(_points(), valid, _gbuffer(raw), _camera())
    assert not ok[0, 0] and not ok[1, 1]
    assert ok.sum() == H * W - 2
    torch.testing.assert_close(n[2][ok], torch.full((H * W - 2,), -1.0))
    assert n[:, ~ok].eq(0).all()


def test_tracking_reads_the_frame_constant_not_the_gbuffer():
    raw = torch.zeros(3, H, W)
    raw[2] = -1.0
    cam = _camera()
    cam.frame_cache = {("frame_raster", "gaussian_normal"): raw}
    cam.map_frozen = True
    g = _gbuffer(torch.zeros(3, H, W))  # would give no normal anywhere
    n, ok = _source()(_points(), torch.ones(H, W, dtype=torch.bool), g, cam)
    assert ok.all()
    # Mapping keyframes hold a frame cache too, but read the G-buffer.
    cam.map_frozen = False
    n, ok = _source()(_points(), torch.ones(H, W, dtype=torch.bool), g, cam)
    assert not ok.any()


def test_frozen_map_without_the_raster_raises():
    cam = _camera()
    cam.frame_cache = {}
    cam.map_frozen = True
    with pytest.raises(KeyError, match="not rendered"):
        _source()(_points(), torch.ones(H, W, dtype=torch.bool), _gbuffer(torch.zeros(3, H, W)), cam)


def test_placement_uses_the_init_source():
    # A fronto-parallel plane: depth_fd gives -z.
    g = {"depth": torch.full((1, H, W), 2.0), "opacity": torch.ones(1, H, W), "pixel_offset": 0.0,
         "placement": True}
    cam = SimpleNamespace(fx=100.0, fy=100.0, cx=2.0, cy=1.5)
    src = _source()
    n, ok = src.initial_orientation(cam, torch.full((H, W), 2.0))
    assert ok[1:-1, 1:-1].all()
    torch.testing.assert_close(n[2][ok], torch.full((int(ok.sum()),), -1.0))
    from light_models.gbuffer import pixel_rays

    plane = pixel_rays(cam, H, W, "cpu", torch.float32, 0.0) * 2.0
    n2, ok2 = src(plane, torch.ones(H, W, dtype=torch.bool), g, cam)
    torch.testing.assert_close(ok2, ok)
    torch.testing.assert_close(n2, n)


def test_missing_raster_outside_tracking_raises():
    with pytest.raises(KeyError, match="gaussian_normal"):
        _source()(_points(), torch.ones(H, W, dtype=torch.bool), {"opacity": torch.ones(1, H, W)}, _camera())


@pytest.mark.parametrize("over, err", [
    ({"min_agreement": 0}, ValueError),
    ({"init_flatness": 1.5}, ValueError),
    ({"init": {"type": "shortest_axis"}}, ValueError),
    ({"init": "depth_fd"}, TypeError),
])
def test_bad_options(over, err):
    with pytest.raises(err):
        _source(**over)


def test_pickles():
    pickle.loads(pickle.dumps(_source()))


# ---- the shape regularizer ----

def _cfg(block):
    return {"Training": {"scale_regularizer": block}}


def test_isotropic_is_monogs_term():
    s = torch.tensor([[0.01, 0.02, 0.06], [0.03, 0.03, 0.03]])
    reg = read_scale_regularizer(_cfg({"type": "isotropic", "weight": 10}))
    expect = 10 * torch.abs(s - s.mean(dim=1).view(-1, 1)).mean()
    torch.testing.assert_close(reg(s), expect)


def test_disc_term_and_its_gradient():
    s = torch.tensor([[0.002, 0.03, 0.01]], requires_grad=True)  # unsorted on purpose
    reg = read_scale_regularizer(_cfg({"type": "disc", "in_plane_weight": 2.0, "flatness_weight": 0.5}))
    loss = reg(s)
    torch.testing.assert_close(loss, torch.tensor(2.0 * 0.02 + 0.5 * 0.002 / 0.01))
    loss.backward()
    # Only the smallest scale is pushed by the flatness term (mid detached).
    torch.testing.assert_close(s.grad, torch.tensor([[0.5 / 0.01, 2.0, -2.0]]))


def test_regularizers_pickle():
    pickle.loads(pickle.dumps(read_scale_regularizer(_cfg({"type": "disc", "in_plane_weight": 1, "flatness_weight": 1}))))


@pytest.mark.parametrize("block, err", [
    ({"type": "isotropic"}, KeyError),
    ({"type": "disc", "in_plane_weight": 1}, KeyError),
    ({"type": "cube", "weight": 1}, ValueError),
    ({"type": "isotropic", "weight": 1, "flatness_weight": 1}, ValueError),
    ({"type": "isotropic", "weight": -1}, ValueError),
])
def test_regularizer_errors(block, err):
    with pytest.raises(err):
        read_scale_regularizer(_cfg(block))


def test_missing_regularizer_raises():
    with pytest.raises(KeyError):
        read_scale_regularizer({"Training": {}})


# ---- supervision by another normal source (mapping penalty) ----

def test_supervision_penalty_value_and_when_it_is_off():
    sup = {"weight": 2.0, "source": {"type": "gbuffer"}}
    src = _source(supervision=sup)
    n = torch.zeros(3, H, W)
    n[2] = -1.0
    ok = torch.ones(H, W, dtype=torch.bool)
    ref = torch.zeros(3, H, W)
    ref[1] = -math.sin(math.radians(60))
    ref[2] = -math.cos(math.radians(60))
    ref_ok = torch.ones(H, W, dtype=torch.bool)
    ref_ok[0] = False  # ignored rows
    g = {"normal": ref, "normal_valid": ref_ok, "opacity": torch.ones(1, H, W)}
    cam = _camera()
    p = src.penalty(n, ok, _points(), torch.ones(H, W, dtype=torch.bool), g, cam)
    torch.testing.assert_close(p, torch.tensor(2.0 * (1 - 0.5)))
    # Off: frozen map (tracking), placement shading, no supervision.
    cam.frame_cache, cam.map_frozen = {}, True
    assert src.penalty(n, ok, _points(), ok, g, cam) is None
    assert src.penalty(n, ok, _points(), ok, dict(g, placement=True), _camera()) is None
    assert _source().penalty(n, ok, _points(), ok, g, _camera()) is None


def test_supervision_reaches_the_lambert_aux_and_has_gradient():
    from light_models.cosine import Lambert  # noqa: F401  (registers lambert)

    lam = build_component("cosine", {
        "type": "lambert", "warmup_keyframes": 0,
        "normal_source": {"type": "shortest_axis", "min_agreement": 0.5, "init_flatness": 0.1,
                          "init": INIT, "supervision": {"weight": 1.0, "source": {"type": "gbuffer"}},
                          "tracking_refresh_every": None, "raster_gradient": "all"},
    }, "Light.cosine", {})
    raw = torch.zeros(3, H, W)
    raw[1] = -0.3
    raw[2] = -0.9
    raw.requires_grad_()
    ref = torch.zeros(3, H, W)
    ref[2] = -1.0
    g = {"gaussian_normal": raw, "opacity": torch.ones(1, H, W), "normal": ref,
         "normal_valid": torch.ones(H, W, dtype=torch.bool)}
    to_light = torch.zeros(3, H, W)
    to_light[2] = -1.0
    _, aux = lam(_points(), torch.ones(H, W, dtype=torch.bool), to_light, g, _camera())
    aux["map_penalty"].backward()
    # y = -0.3: descent (raw -= lr * grad) raises it towards 0, the reference.
    assert raw.grad is not None and raw.grad[1].lt(0).all()


@pytest.mark.parametrize("sup, err", [
    ({"weight": 0, "source": {"type": "gbuffer"}}, ValueError),
    ({"weight": 1.0}, ValueError),
    ({"weight": 1.0, "source": {"type": "shortest_axis"}}, ValueError),
    ({"weight": 1.0, "source": "gbuffer"}, TypeError),
])
def test_bad_supervision(sup, err):
    with pytest.raises(err):
        _source(supervision=sup)


# ---- re-rendering the map normals during tracking (tracking_refresh_every) ----

@pytest.mark.parametrize("value", [0, -1, True, 2.5, "10"])
def test_bad_tracking_refresh_every(value):
    with pytest.raises(ValueError):
        _source(tracking_refresh_every=value)


def test_tracking_refresh_every_reaches_the_shader_hook():
    from light_models.colocated import ColocatedShader

    assert _source().tracking_refresh_every is None
    assert _source(tracking_refresh_every=10).tracking_refresh_every == 10
    shader = ColocatedShader.__new__(ColocatedShader)  # only the hook under test
    shader.cosine = SimpleNamespace(normals=_source(tracking_refresh_every=10))
    assert shader.tracking_refresh_every() == 10
    shader.cosine = SimpleNamespace(normals=None)
    assert shader.tracking_refresh_every() is None


def test_drop_frame_rasters_rebuilds_into_the_same_pooled_buffer():
    from utils.perf import FRAME_RASTER, drop_frame_rasters, frame_constant

    cam = SimpleNamespace(frame_cache={}, frame_pool={})
    key = (FRAME_RASTER, "gaussian_normal")
    first = frame_constant(cam, key, lambda: torch.zeros(3, H, W))
    obs = frame_constant(cam, ("observation",), lambda: torch.full((2,), 7.0))
    drop_frame_rasters(cam)
    assert key not in cam.frame_cache and ("observation",) in cam.frame_cache
    again = frame_constant(cam, key, lambda: torch.ones(3, H, W))
    # Same memory (what a recorded CUDA graph reads), new values.
    assert again is first and bool((first == 1).all())
    assert frame_constant(cam, ("observation",), lambda: torch.zeros(2)) is obs


@pytest.mark.parametrize("value", ["none", None, True])
def test_bad_raster_gradient(value):
    with pytest.raises(ValueError):
        _source(raster_gradient=value)


def test_raster_gradient_reaches_the_shader_hook():
    from light_models.colocated import ColocatedShader

    shader = ColocatedShader.__new__(ColocatedShader)
    shader.cosine = SimpleNamespace(normals=_source(raster_gradient="rotation"))
    assert shader.frame_raster_gradient() == "rotation"
