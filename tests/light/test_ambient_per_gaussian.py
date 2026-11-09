"""Ambient light per Gaussian (docs/DECISIONS.md D121): E_i = c + r_i.

CPU tests check the component on its own; the CUDA tests check the map
plumbing (GaussianModel extras through append / clone / split / prune) and
the extra rasterizer pass, with an exact oracle: a residual equal to delta
on every Gaussian rasterizes to delta * (albedo map), so the result must
equal multiplicative_const with c + delta."""
import math
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader  # noqa: E402

PARAMS = {
    "t_CL_m": {"config_opencv": [0.0, -0.05, 0.0]},
    "R_CL": {"config": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    "intensity": {"K_theory": 3.0},
    "cone": {"half_angle_deg_config": 30.0, "blend_config": 0.35},
}
AMB = {"channels": 1, "residual_lr": 0.001, "residual_after_keyframes": 2, "l1_weight": 0.05}


def _p(key):
    return {"source": "params_file", "key": key, "learnable": False}


def _config(ambient):
    return {
        "Training": {"spherical_harmonics": False},
        "Light": {
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
        },
    }


def _per_gaussian(c=0.1, **over):
    block = {"type": "per_gaussian", "c": {"source": "value", "value": c, "learnable": False}}
    block.update(AMB)
    block.update(over)
    return build_shader(_config(block), PARAMS)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("key", sorted(AMB))
def test_every_option_is_required(key):
    block = {"type": "per_gaussian", "c": {"source": "value", "value": 0.1, "learnable": False}}
    block.update({k: v for k, v in AMB.items() if k != key})
    with pytest.raises(KeyError, match=key):
        build_shader(_config(block), PARAMS)


@pytest.mark.parametrize("over", [{"channels": 2}, {"channels": True}, {"residual_lr": 0},
                                  {"l1_weight": -1.0}, {"residual_after_keyframes": -1},
                                  {"residual_after_keyframes": 1.5}])
def test_bad_options_raise(over):
    with pytest.raises(ValueError):
        _per_gaussian(**over)


def test_global_forms_declare_no_per_gaussian_values():
    block = {"type": "multiplicative_const", "c": {"source": "value", "value": 0.1, "learnable": False}}
    shader = build_shader(_config(block), PARAMS)
    assert shader.per_gaussian_params() == {}
    assert shader.extra_raster_keys() == ()
    assert shader.extra_rasters(None) == {}


def test_declares_its_map_value_and_survives_pickling():
    shader = pickle.loads(pickle.dumps(_per_gaussian(channels=3)))
    assert shader.per_gaussian_params() == {"ambient_residual": (3, 0.001)}


# ---------------------------------------------------------------- CPU behaviour


class _Map:
    def __init__(self, albedo, residual):
        self.get_albedo = albedo
        self._r = residual

    def get_extra(self, name):
        assert name == "ambient_residual"
        return self._r


def test_residual_off_until_more_than_n_keyframes():
    shader = _per_gaussian()
    gmap = _Map(torch.full((4, 3), 0.5), torch.full((4, 1), 0.2))
    assert shader.extra_raster_keys() == ("ambient_residual",)  # not reported yet: full model
    for n, on in ((0, False), (2, False), (3, True)):
        shader.set_keyframe_count(n)
        assert (shader.extra_raster_keys() == ("ambient_residual",)) is on
        assert bool(shader.extra_rasters(gmap)) is on


def test_raster_colours_are_albedo_times_clamped_excess():
    shader = _per_gaussian(c=0.1)
    albedo = torch.tensor([[0.5, 0.2, 1.0], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0]])
    r = torch.tensor([[0.3], [-0.05], [-0.4]])  # E = 0.4, 0.05, max(-0.3, 0) = 0
    colours = shader.extra_rasters(_Map(albedo, r))["ambient_residual"]
    excess = torch.tensor([[0.3], [-0.05], [-0.1]])
    torch.testing.assert_close(colours, albedo * excess)


def test_shading_and_penalty():
    shader = _per_gaussian(c=0.1)
    h, w = 2, 3
    albedo = torch.rand(3, h, w)
    direct = torch.rand(h, w)
    m = torch.randn(3, h, w) * 0.1
    radiance, aux = shader.ambient(albedo, direct, {"ambient_residual": m})
    torch.testing.assert_close(radiance, albedo * (direct[None] + 0.1) + m)
    torch.testing.assert_close(aux["map_penalty"], 0.05 * m.abs().mean())
    shader.set_keyframe_count(0)
    radiance, aux = shader.ambient(albedo, direct, {})
    torch.testing.assert_close(radiance, albedo * (direct[None] + 0.1))
    assert aux == {}


def _gated(**over):
    block = {"type": "per_gaussian_gated", "c": {"source": "value", "value": 0.1, "learnable": False},
             "residual_direct_max": 0.5}
    block.update(AMB)
    block.update(over)
    return build_shader(_config(block), PARAMS)


def test_gate_threshold_is_required():
    block = {"type": "per_gaussian_gated", "c": {"source": "value", "value": 0.1, "learnable": False}}
    block.update(AMB)
    with pytest.raises(KeyError, match="residual_direct_max"):
        build_shader(_config(block), PARAMS)


@pytest.mark.parametrize("bad", [-0.1, True, "0.5"])
def test_bad_gate_threshold_raises(bad):
    with pytest.raises(ValueError, match="residual_direct_max"):
        _gated(residual_direct_max=bad)


def test_gate_keeps_the_image_and_cuts_gradient_and_penalty_under_the_lamp():
    shader = _gated()
    albedo = torch.full((3, 1, 3), 0.5)
    direct = torch.tensor([[0.0, 0.5, 2.0]])  # the threshold itself is still free
    m = torch.tensor(0.2).repeat(3, 1, 3).requires_grad_()
    radiance, aux = shader.ambient(albedo, direct, {"ambient_residual": m})
    torch.testing.assert_close(radiance.detach(), albedo * (direct[None] + 0.1) + m.detach())
    torch.testing.assert_close(aux["ambient_residual"], m)
    radiance.sum().backward()
    assert m.grad[:, 0, :2].eq(1).all() and m.grad[:, 0, 2].eq(0).all()
    m.grad = None
    aux["map_penalty"].backward()
    expected = 0.05 / 9  # mean over all 9 values, of which 6 count
    assert m.grad[:, 0, :2].sub(expected).abs().max() < 1e-7 and m.grad[:, 0, 2].eq(0).all()


def test_gate_cuts_the_gradient_of_c_under_the_lamp_but_not_of_the_albedo():
    shader = _gated()
    amb = shader.ambient
    amb.c = amb.c.clone().requires_grad_()
    albedo = torch.full((3, 1, 2), 0.5, requires_grad=True)
    direct = torch.tensor([[0.0, 2.0]])
    radiance, _ = amb(albedo, direct, {"ambient_residual": torch.zeros(3, 1, 2)})
    torch.testing.assert_close(radiance[:, 0, 1].detach(), torch.full((3,), 0.5 * 2.1))
    amb_grad, alb_grad = torch.autograd.grad(radiance[:, 0, 1].sum(), (amb.c, albedo), retain_graph=True)
    assert amb_grad.eq(0).all()
    assert alb_grad[:, 0, 1].sub(2.1).abs().max() < 1e-6
    amb_grad = torch.autograd.grad(radiance[:, 0, 0].sum(), amb.c)[0]
    assert amb_grad.sub(1.5).abs().max() < 1e-6  # 3 channels of albedo 0.5


def test_c_learns_from_every_pixel_while_the_residual_is_off():
    shader = _gated()
    shader.set_keyframe_count(0)
    amb = shader.ambient
    amb.c = amb.c.clone().requires_grad_()
    radiance, _ = amb(torch.full((3, 1, 1), 0.5), torch.tensor([[2.0]]), {})
    assert torch.autograd.grad(radiance.sum(), amb.c)[0].sub(1.5).abs().max() < 1e-6


def test_gate_above_every_direct_matches_per_gaussian():
    albedo = torch.rand(3, 2, 3)
    direct = torch.rand(2, 3)
    m = torch.randn(3, 2, 3) * 0.1
    gated = _gated(residual_direct_max=10.0).ambient(albedo, direct, {"ambient_residual": m})
    plain = _per_gaussian().ambient(albedo, direct, {"ambient_residual": m})
    torch.testing.assert_close(gated[0], plain[0])
    torch.testing.assert_close(gated[1]["map_penalty"], plain[1]["map_penalty"])


def test_gate_uses_the_strongest_channel_and_stays_off_until_the_residual_opens():
    shader = _gated()
    direct = torch.zeros(3, 1, 1)
    direct[1] = 2.0
    m = torch.tensor(0.2).repeat(3, 1, 1).requires_grad_()
    radiance, _ = shader.ambient(torch.full((3, 1, 1), 0.5), direct, {"ambient_residual": m})
    radiance.sum().backward()
    assert m.grad.eq(0).all()
    shader.set_keyframe_count(0)
    assert shader.ambient(torch.ones(3, 1, 1), torch.zeros(1, 1), {})[1] == {}


def test_mapping_loss_adds_the_penalty_only_when_rendered(monkeypatch):
    from utils import slam_utils

    monkeypatch.setattr(slam_utils, "get_loss_images", lambda c, p, v: (None, None))
    monkeypatch.setattr(slam_utils, "apply_exposure_affine", lambda c, image, v: image)
    monkeypatch.setattr(slam_utils, "get_loss_weight", lambda *a: None)
    monkeypatch.setattr(slam_utils, "get_loss_mapping_rgbd", lambda *a: torch.tensor(1.0))
    config = {"Training": {"monocular": False}}
    pkg = {"depth": None}
    assert float(slam_utils.get_loss_mapping(config, pkg, None)) == 1.0
    pkg["light_map_penalty"] = torch.tensor(0.25)
    assert float(slam_utils.get_loss_mapping(config, pkg, None)) == 1.25


def test_clone_obj_detaches_dicts_of_tensors():
    from utils.multiprocessing_utils import clone_obj

    obj = SimpleNamespace(_extras={"a": torch.nn.Parameter(torch.ones(2, 1))}, other={"k": 1})
    out = clone_obj(obj)
    assert not out._extras["a"].requires_grad
    assert out.other == {"k": 1}


# ---------------------------------------------------------------- CUDA: map and renderer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _gaussians(n=200, extras=None):
    from munch import munchify

    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.sh_utils import RGB2SH

    g = GaussianModel(0, config={"Training": {}})
    g.init_lr(6.0)
    if extras is not None:
        g.declare_extras(extras)
    g.training_setup(munchify(dict(percent_dense=0.01, position_lr_init=1e-4, position_lr_final=1e-6,
                                   position_lr_delay_mult=0.01, position_lr_max_steps=100,
                                   feature_lr=1e-3, opacity_lr=0.05, scaling_lr=1e-3, rotation_lr=1e-3)))
    gen = torch.Generator().manual_seed(0)
    side = int(math.isqrt(n))
    v, u = torch.meshgrid(torch.linspace(-0.6, 0.6, side), torch.linspace(-1.0, 1.0, side), indexing="ij")
    xyz = torch.stack([u.flatten(), v.flatten(), torch.full((side * side,), 2.0)], 1).cuda()
    m = xyz.shape[0]
    rgb = torch.rand(m, 3, generator=gen).cuda() * 0.8 + 0.1
    feats = torch.zeros(m, 3, 1, device="cuda")
    feats[:, :, 0] = RGB2SH(rgb)
    scales = torch.log(torch.full((m, 3), 0.06, device="cuda"))
    rots = torch.zeros(m, 4, device="cuda")
    rots[:, 0] = 1
    opac = torch.full((m, 1), 4.0, device="cuda")  # sigmoid ~ 0.98
    g.extend_from_pcd(xyz, feats, scales, rots, opac, kf_id=0)
    return g


@cuda
def test_extras_follow_every_map_edit():
    g = _gaussians(extras={"ambient_residual": (1, 0.001)})
    r = g.get_extra("ambient_residual")
    assert r.shape == (g.get_xyz.shape[0], 1) and float(r.abs().max()) == 0.0
    assert "ambient_residual" in {grp["name"] for grp in g.optimizer.param_groups}
    with torch.no_grad():
        g._extras["ambient_residual"].copy_(g.get_xyz[:, :1])  # tag each row with its x
    n = g.get_xyz.shape[0]
    g.densify_and_clone(torch.ones(n, 1, device="cuda"), 0.5, 1e6)  # all small: cloned
    g.densify_and_split(torch.ones(2 * n, 1, device="cuda"), 0.5, 0.01)  # all large: split
    keep = torch.rand(g.get_xyz.shape[0], device="cuda") < 0.3
    g.prune_points(keep)
    assert g.get_extra("ambient_residual").shape[0] == g.get_xyz.shape[0]
    # Clones keep the parent's x exactly; split children are sampled around
    # it (scale 0.06), so the tag stays within a few sigma.
    err = (g.get_extra("ambient_residual")[:, 0] - g.get_xyz[:, 0]).abs()
    assert float(err.max()) < 0.5


@cuda
def test_declare_extras_only_on_the_empty_map():
    g = _gaussians()
    with pytest.raises(RuntimeError):
        g.declare_extras({"ambient_residual": (1, 0.001)})


def _render(g, shader):
    from gaussian_splatting.gaussian_renderer import render
    from utils.camera_utils import Camera

    h, w, f = 120, 160, 120.0
    cam = Camera.init_from_gui(0, torch.eye(4, device="cuda"), 2 * math.atan(w / (2 * f)),
                               2 * math.atan(h / (2 * f)), f, f, w / 2, h / 2, h, w)
    cam.update_RT(torch.eye(3, device="cuda"), torch.zeros(3, device="cuda"))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    return render(cam, g, pipe, torch.zeros(3, device="cuda"), shader=shader)


@cuda
def test_uniform_residual_equals_a_shifted_constant_ambient():
    delta = 0.07
    g = _gaussians(extras={"ambient_residual": (1, 0.001)})
    with torch.no_grad():
        g._extras["ambient_residual"].fill_(delta)
    pkg = _render(g, _per_gaussian(c=0.1))
    const = {"type": "multiplicative_const", "c": {"source": "value", "value": 0.1 + delta, "learnable": False}}
    ref = _render(g, build_shader(_config(const), PARAMS))
    torch.testing.assert_close(pkg["radiance_linear"], ref["radiance_linear"], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(pkg["light_ambient_residual"], delta * pkg["albedo"], atol=2e-6, rtol=1e-4)
    assert "light_map_penalty" in pkg


@cuda
def test_gradient_reaches_the_residual_and_the_pose():
    g = _gaussians(extras={"ambient_residual": (3, 0.001)})
    pkg = _render(g, _per_gaussian(c=0.1, channels=3))
    (pkg["radiance_linear"].sum() + pkg["light_map_penalty"]).backward()
    assert float(g.get_extra("ambient_residual").grad.abs().sum()) > 0


@cuda
def test_off_residual_renders_like_the_constant_form():
    g = _gaussians(extras={"ambient_residual": (1, 0.001)})
    with torch.no_grad():
        g._extras["ambient_residual"].fill_(0.3)
    shader = _per_gaussian(c=0.1)
    shader.set_keyframe_count(1)
    pkg = _render(g, shader)
    const = {"type": "multiplicative_const", "c": {"source": "value", "value": 0.1, "learnable": False}}
    ref = _render(g, build_shader(_config(const), PARAMS))
    torch.testing.assert_close(pkg["radiance_linear"], ref["radiance_linear"])
    assert "light_map_penalty" not in pkg and "light_ambient_residual" not in pkg
