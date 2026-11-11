"""Whole tracking iteration as one CUDA graph (docs/DECISIONS.md D133): the
options, the fixed-size map copy, and the rasterizer's capacity mode and
padding entries, each checked against the path they replace."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from utils.graphed_iteration import _StaticMap, read_graph_iteration  # noqa: E402


def _config(block, shading=False):
    return {"Training": {"cuda_graph_iteration": block, "cuda_graph_tracking": shading}}


ON = {"enabled": True, "check_every": 5, "raster_capacity_margin": 1.3, "map_capacity_margin": 1.2,
      "compile_shading": False}


def test_disabled_gives_none():
    assert read_graph_iteration(_config({"enabled": False})) is None


def test_enabled_reads_every_option():
    assert read_graph_iteration(_config(ON)) == {
        "check_every": 5, "raster_capacity_margin": 1.3, "map_capacity_margin": 1.2,
        "compile_shading": False}


@pytest.mark.parametrize("drop", ["check_every", "raster_capacity_margin", "map_capacity_margin", "compile_shading"])
def test_missing_option_raises(drop):
    block = {k: v for k, v in ON.items() if k != drop}
    with pytest.raises(KeyError):
        read_graph_iteration(_config(block))


@pytest.mark.parametrize("key,value", [("check_every", 0), ("check_every", 2.5), ("check_every", True),
                                       ("raster_capacity_margin", 1.0), ("map_capacity_margin", 0.5)])
def test_bad_option_raises(key, value):
    with pytest.raises(ValueError):
        read_graph_iteration(_config(dict(ON, **{key: value})))


def test_missing_block_raises():
    with pytest.raises(KeyError):
        read_graph_iteration({"Training": {"cuda_graph_tracking": False}})


def test_options_while_disabled_raise():
    with pytest.raises(ValueError):
        read_graph_iteration(_config({"enabled": False, "check_every": 5}))


def test_both_graph_modes_raise():
    with pytest.raises(ValueError):
        read_graph_iteration(_config(ON, shading=True))


def _model(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        get_xyz=torch.randn(n, 3, generator=g),
        get_opacity=torch.rand(n, 1, generator=g),
        get_scaling=torch.rand(n, 3, generator=g),
        get_rotation=torch.randn(n, 4, generator=g),
        get_features=torch.randn(n, 1, 3, generator=g),
        active_sh_degree=0, max_sh_degree=0)


def test_static_map_copies_and_pads():
    m = _StaticMap()
    src = _model(10)
    assert m.load(src, 1.5)
    assert m.size == 15 and m.n == 10
    for k in ("get_xyz", "get_opacity", "get_scaling", "get_rotation", "get_features"):
        torch.testing.assert_close(getattr(m, k)[:10], getattr(src, k), rtol=0, atol=0)
    assert bool((m.get_opacity[10:] < 0).all())
    assert torch.isfinite(m.get_scaling).all() and torch.isfinite(m.get_rotation).all()


def test_static_map_keeps_its_buffers_while_the_map_fits():
    m = _StaticMap()
    m.load(_model(10), 1.5)
    xyz = m.get_xyz
    assert not m.load(_model(14, seed=1), 1.5)  # fits: same buffers
    assert m.get_xyz is xyz and m.n == 14
    assert not m.load(_model(6, seed=2), 1.5)  # smaller: the rest is padding again
    assert bool((m.get_opacity[6:] < 0).all()) and bool((m.get_opacity[:6] >= 0).all())
    assert m.load(_model(16, seed=3), 1.5)  # does not fit: new buffers
    assert m.size == 24 and m.get_xyz is not xyz


# ------------------------------------------------------- CUDA: rasterizer

def _rasterizer():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dgr = pytest.importorskip("diff_gaussian_rasterization")
    if "capacity" not in dgr.GaussianRasterizationSettings._fields:
        pytest.skip("rasterizer built before D133")
    return dgr


def _scene(n=400, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xyz = torch.rand(n, 3, generator=g, device="cuda") * torch.tensor([2.0, 1.4, 1.0], device="cuda") \
        - torch.tensor([1.0, 0.7, -1.5], device="cuda")
    return {
        "means3D": xyz,
        "opacities": torch.rand(n, 1, generator=g, device="cuda") * 0.9 + 0.05,
        "scales": torch.rand(n, 3, generator=g, device="cuda") * 0.08 + 0.01,
        "rotations": torch.nn.functional.normalize(torch.randn(n, 4, generator=g, device="cuda"), dim=1),
        "shs": torch.rand(n, 1, 3, generator=g, device="cuda"),
    }


def _run(dgr, scene, capacity=0, count=None, pose_only=False):
    from utils.camera_utils import Camera

    h, w, f = 96, 128, 100.0
    cam = Camera.init_from_gui(0, torch.eye(4, device="cuda"), 2 * math.atan(w / (2 * f)),
                               2 * math.atan(h / (2 * f)), f, f, w / 2, h / 2, h, w)
    cam.update_RT(torch.eye(3, device="cuda"), torch.zeros(3, device="cuda"))
    settings = dgr.GaussianRasterizationSettings(
        image_height=h, image_width=w, tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0, viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform, projmatrix_raw=cam.projection_matrix, sh_degree=0,
        campos=cam.camera_center, prefiltered=False, debug=False, capacity=capacity, num_rendered_out=count,
        **({"pose_only": True} if pose_only else {}))
    leaves = {k: v.clone().requires_grad_() for k, v in scene.items()}
    means2d = torch.zeros_like(leaves["means3D"], requires_grad=True)
    img, radii, depth, opacity, _ = dgr.GaussianRasterizer(settings)(
        means2D=means2d, colors_precomp=None, cov3D_precomp=None,
        theta=cam.cam_rot_delta, rho=cam.cam_trans_delta, **leaves)
    num = img.grad_fn.num_rendered
    wgt = torch.linspace(0.5, 1.5, img.numel(), device="cuda").view_as(img)
    ((img * wgt).sum() + depth.sum() + opacity.sum()).backward()
    out = {"img": img.detach(), "depth": depth.detach(), "opacity": opacity.detach(), "radii": radii,
           "rho": cam.cam_trans_delta.grad.clone(), "theta": cam.cam_rot_delta.grad.clone()}
    out.update({"d_" + k: v.grad for k, v in leaves.items()})
    return out, num


def _close(a, b, keys=None):
    for k in keys or a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-5, msg=k)


def test_capacity_mode_renders_like_the_default():
    dgr = _rasterizer()
    scene = _scene()
    ref, num = _run(dgr, scene)
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    for cap in (num, num + 1000):
        out, returned = _run(dgr, scene, cap, count)
        assert returned == cap and int(count) == num
        _close(ref, out)


def test_pose_only_backward_gives_the_same_pose_gradient():
    dgr = _rasterizer()
    if "pose_only" not in dgr.GaussianRasterizationSettings._fields:
        pytest.skip("rasterizer built before D134")
    scene = _scene()
    ref, _ = _run(dgr, scene)
    out, _ = _run(dgr, scene, pose_only=True)
    _close(ref, out, ["img", "depth", "opacity", "rho", "theta", "d_means3D", "d_scales", "d_rotations"])
    assert float(out["d_opacities"].abs().max()) == 0.0 and float(out["d_shs"].abs().max()) == 0.0
    assert float(ref["d_opacities"].abs().max()) > 0.0


def test_default_mode_also_writes_the_count():
    dgr = _rasterizer()
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    _, num = _run(dgr, _scene(), 0, count)
    assert int(count) == num


def test_capacity_overflow_is_visible_in_the_count():
    dgr = _rasterizer()
    scene = _scene()
    _, num = _run(dgr, scene)
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    out, _ = _run(dgr, scene, num // 2, count)
    assert int(count) == num > num // 2
    assert torch.isfinite(out["img"]).all()


def test_negative_opacity_entries_are_skipped():
    dgr = _rasterizer()
    scene = _scene()
    ref, num = _run(dgr, scene)
    pad = _scene(150, seed=1)
    pad["opacities"] = torch.full_like(pad["opacities"], -1.0)
    both = {k: torch.cat([scene[k], pad[k]]) for k in scene}
    out, num_both = _run(dgr, both)
    n = scene["means3D"].shape[0]
    assert num_both == num
    assert bool((out["radii"][n:] == 0).all())
    _close(ref, out, ["img", "depth", "opacity", "rho", "theta"])
    torch.testing.assert_close(out["d_means3D"][:n], ref["d_means3D"], rtol=1e-4, atol=1e-5)
    assert float(out["d_means3D"][n:].abs().max()) == 0.0


# ------------------------------------------------------- traced shading (D134)

def test_lifted_constants_are_read_at_each_call():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from torch.fx.experimental.proxy_tensor import make_fx

    from utils.graphed_iteration import lift_constants

    scale = torch.tensor([2.0, 3.0], device="cuda")  # a static buffer, as in the loop
    offset = torch.tensor(0.5, device="cuda")

    def fn(x):
        return ((x * scale + offset) ** 2).sum() + torch.tensor(1.0)  # literal stays inside

    x = torch.ones(2, device="cuda")
    gm = make_fx(fn)(x)
    consts = lift_constants(gm)
    assert any(c is scale for c in consts) and any(c is offset for c in consts)
    assert not any(n.op == "get_attr" and torch.is_tensor(getattr(gm, n.target)) and getattr(gm, n.target).is_cuda
                   for n in gm.graph.nodes)
    torch.testing.assert_close(gm(x, *consts), fn(x))
    with torch.no_grad():  # new values in the same memory, as the loop writes them
        scale.mul_(-1.5)
        offset.fill_(0.25)
    torch.testing.assert_close(gm(x, *consts), fn(x))
    xg = x.clone().requires_grad_()
    g1, = torch.autograd.grad(gm(xg, *consts), xg)
    g2, = torch.autograd.grad(fn(xg), xg)
    torch.testing.assert_close(g1, g2)


# ------------------------------------------------------- colour refinement (D136)

def test_refinement_graph_options():
    from utils.graphed_refinement import read_refinement_graph

    assert read_refinement_graph({"color_refinement_graph": {"enabled": False}}) is None
    on = {"enabled": True, "raster_capacity_margin": 1.1, "compile_shading": True}
    assert read_refinement_graph({"color_refinement_graph": on}) == {
        "raster_capacity_margin": 1.1, "compile_shading": True}
    with pytest.raises(KeyError):
        read_refinement_graph({})
    with pytest.raises(KeyError):
        read_refinement_graph({"color_refinement_graph": {"enabled": True, "compile_shading": True}})
    with pytest.raises(ValueError):
        read_refinement_graph({"color_refinement_graph": dict(on, raster_capacity_margin=0.9)})
    with pytest.raises(ValueError):
        read_refinement_graph({"color_refinement_graph": {"enabled": False, "compile_shading": True}})


def test_refinement_graph_refuses_geometry_groups():
    from utils.graphed_refinement import GraphedRefinement

    with pytest.raises(ValueError, match="geometry"):
        GraphedRefinement({}, None, None, None, None, {}, ["f_dc", "xyz"], 0.2,
                          {"raster_capacity_margin": 1.1, "compile_shading": False})
