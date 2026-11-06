"""ColocatedShader against the dataset repo's verified physics code.

Oracle: Replica/scripts/physics_check/common.py (light_geometry, spot_phi,
depth_normals), the functions the physics check validated on the rendered
data:  I_lin = A * (K * Phi * max(0, n.l) / d^2 + c). Needs torch and the
REPLICA_REPO environment variable (and OpenEXR, which common.py imports).
"""
import importlib.util
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader  # noqa: E402

PARAMS = {
    "t_CL_m": {"config_opencv": [0.0, -0.05, 0.0]},
    "R_CL": {"config": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    "intensity": {"K_theory": 30.396355092701334},
    "cone": {"half_angle_deg_config": 30.0, "blend_config": 0.35},
    "ambient": {"c_config": 0.12},
}


def _p(key):
    return {"source": "params_file", "key": key, "learnable": False}


def _config(cosine, ambient="multiplicative_const", angular="blender_spot"):
    ang = {"type": angular}
    if angular == "blender_spot":
        ang.update(half_angle_deg=_p("cone.half_angle_deg_config"), blend=_p("cone.blend_config"))
    amb = {"type": ambient}
    if ambient != "none":
        amb["c"] = _p("ambient.c_config")
    return {
        "Training": {"spherical_harmonics": False},
        "Light": {
            "enabled": True,
            "shader": {"type": "colocated"},
            "intensity": _p("intensity.K_theory"),
            "pose": {"t_CL": _p("t_CL_m.config_opencv"), "R_CL": _p("R_CL.config")},
            "falloff": {"type": "inverse_square"},
            "angular": ang,
            "cosine": cosine,
            "ambient": amb,
            "gbuffer": {"opacity_thr": 0.5},
        },
    }


def _common():
    root = os.environ.get("REPLICA_REPO")
    if not root:
        pytest.skip("REPLICA_REPO not set")
    pytest.importorskip("OpenEXR")
    path = os.path.join(root, "scripts", "physics_check", "common.py")
    spec = importlib.util.spec_from_file_location("replica_physics_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(common):
    # A tilted plane plus a bump, at the dataset's resolution/intrinsics.
    h, w = common.H, common.W
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 1.2 + 0.0015 * u + 0.0008 * v + 0.3 * np.exp(-((u - 380) ** 2 + (v - 150) ** 2) / 4000.0)
    valid = np.ones((h, w), bool)
    P = common.backproject(z)
    n = common.depth_normals(P, valid)
    rng = np.random.default_rng(0)
    albedo = rng.uniform(0.05, 0.9, (h, w, 3))
    cam = SimpleNamespace(fx=common.FX, fy=common.FY, cx=common.CX, cy=common.CY)
    return z, P, n, albedo, cam


def _oracle(common, P, n, albedo, cosine=True):
    t = PARAMS["t_CL_m"]["config_opencv"]
    d, cos_nl, cos_th = common.light_geometry(P, n, t)
    phi = common.spot_phi(cos_th, PARAMS["cone"]["half_angle_deg_config"], PARAMS["cone"]["blend_config"])
    k = PARAMS["intensity"]["K_theory"]
    geo = np.maximum(0.0, cos_nl) if cosine else 1.0
    return albedo * (k * phi * geo / d**2 + PARAMS["ambient"]["c_config"])[..., None]


def _gbuffer(z, albedo, normals=None):
    g = {
        "albedo": torch.from_numpy(albedo).permute(2, 0, 1).double(),
        "depth": torch.from_numpy(z)[None].double(),
        "opacity": torch.ones(1, *z.shape, dtype=torch.float64),
        "pixel_offset": 0.0,  # ground-truth-like maps: the oracle's convention
    }
    if normals is not None:
        ok = np.isfinite(normals).all(-1)
        g["normal"] = torch.from_numpy(np.where(ok[..., None], normals, 0.0)).permute(2, 0, 1)
        g["normal_valid"] = torch.from_numpy(ok)
    return g


def _interior(common):
    m = np.zeros((common.H, common.W), bool)
    m[2:-2, 2:-2] = True
    return m


def test_lambert_with_given_normals_matches_oracle():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "lambert", "normal_source": {"type": "gbuffer"}, "warmup_keyframes": 0}), PARAMS)
    out = shader(_gbuffer(z, albedo, n), cam)["radiance_linear"].permute(1, 2, 0).numpy()
    expected = _oracle(common, P, n, albedo)
    m = _interior(common)
    np.testing.assert_allclose(out[m], expected[m], rtol=1e-5, atol=1e-7)


def test_lambert_depth_fd_matches_oracle_normals():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "lambert", "normal_source": {"type": "depth_fd", "stencil_px": 1}, "warmup_keyframes": 0}), PARAMS)
    out = shader(_gbuffer(z, albedo), cam)
    expected = _oracle(common, P, n, albedo)
    m = _interior(common)
    got = out["radiance_linear"].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got[m], expected[m], rtol=1e-5, atol=1e-7)
    # Border pixels have no central difference: reported invalid.
    nv = out["light_normal_valid"][0].numpy()
    assert not nv[0].any() and not nv[:, -1].any() and nv[m].all()


def test_no_cosine_matches_oracle_without_nl():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "none"}), PARAMS)
    got = shader(_gbuffer(z, albedo), cam)["radiance_linear"].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, _oracle(common, P, n, albedo, cosine=False), rtol=1e-5, atol=1e-7)


def test_depth_is_divided_by_opacity():
    # Same scene rendered with opacity 0.8 everywhere: albedo and depth are
    # both opacity-weighted; the shading geometry must not change (D5).
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "none"}), PARAMS)
    full = shader(_gbuffer(z, albedo), cam)["radiance_linear"]
    g = _gbuffer(z, albedo)
    g = {"albedo": g["albedo"] * 0.8, "depth": g["depth"] * 0.8,
         "opacity": g["opacity"] * 0.8, "pixel_offset": 0.0}
    weighted = shader(g, cam)["radiance_linear"]
    torch.testing.assert_close(weighted, full * 0.8)


def test_low_opacity_pixels_use_median_depth_and_keep_gradient():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "none"}), PARAMS)
    g = _gbuffer(z, albedo)
    g["opacity"][0, :10, :] = 0.1
    g["depth"][0, :10, :] *= 0.1
    g["albedo"] = g["albedo"].clone().requires_grad_(True)
    out = shader(g, cam)
    assert not out["light_valid"][0, :10].any() and out["light_valid"][0, 10:].all()
    assert torch.isfinite(out["radiance_linear"]).all()
    out["radiance_linear"].sum().backward()
    assert (g["albedo"].grad[:, :10] != 0).all()


def test_additive_and_none_ambient_forms():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    c = PARAMS["ambient"]["c_config"]
    none = build_shader(_config({"type": "none"}, ambient="none"), PARAMS)
    add = build_shader(_config({"type": "none"}, ambient="additive_const"), PARAMS)
    mul = build_shader(_config({"type": "none"}), PARAMS)
    g = _gbuffer(z, albedo)
    base = none(g, cam)["radiance_linear"]
    torch.testing.assert_close(add(g, cam)["radiance_linear"], base + c)
    torch.testing.assert_close(mul(g, cam)["radiance_linear"], base + c * g["albedo"])


def test_gradient_reaches_depth_and_albedo():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "lambert", "normal_source": {"type": "depth_fd", "stencil_px": 1}, "warmup_keyframes": 0}), PARAMS)
    g = _gbuffer(z, albedo)
    g["depth"] = g["depth"].clone().requires_grad_(True)
    g["albedo"] = g["albedo"].clone().requires_grad_(True)
    shader(g, cam)["radiance_linear"].sum().backward()
    assert torch.isfinite(g["depth"].grad).all() and g["depth"].grad.abs().sum() > 0
    assert torch.isfinite(g["albedo"].grad).all()


def test_shader_pickles():
    shader = build_shader(_config({"type": "lambert", "normal_source": {"type": "depth_fd", "stencil_px": 1}, "warmup_keyframes": 0}), PARAMS)
    pickle.loads(pickle.dumps(shader))


@pytest.mark.parametrize(
    "light_patch, err",
    [
        ({"falloff": {"type": "learned"}}, NotImplementedError),
        ({"angular": {"type": "mlp"}}, NotImplementedError),
        ({"angular": {"type": "cone"}}, ValueError),
        ({"cosine": {"type": "lambert", "warmup_keyframes": 0}}, KeyError),  # normal_source missing
        ({"cosine": {"type": "lambert", "normal_source": {"type": "gbuffer"}}}, KeyError),  # warmup missing
        ({"cosine": {"type": "lambert", "normal_source": {"type": "gbuffer"}, "warmup_keyframes": -1}}, ValueError),
        ({"cosine": {"type": "lambert", "normal_source": {"type": "shortest_axis"}, "warmup_keyframes": 0}}, NotImplementedError),
        ({"ambient": {"type": "multiplicative_const"}}, KeyError),  # c missing
        ({"ambient": {"type": "none", "c": _p("ambient.c_config")}}, ValueError),
        ({"gbuffer": {"opacity_thr": 0}}, ValueError),
        ({"gbuffer": {}}, KeyError),
    ],
)
def test_config_errors(light_patch, err):
    cfg = _config({"type": "none"})
    cfg["Light"].update(light_patch)
    with pytest.raises(err):
        build_shader(cfg, PARAMS)


def test_rasterized_maps_are_shaded_along_shifted_rays():
    # A rasterized map (pixel_offset 0.5) of the same scene must equal the
    # oracle evaluated on the OpenCV grid shifted by half a pixel (D29).
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "none"}), PARAMS)
    g = _gbuffer(z, albedo)
    g["pixel_offset"] = 0.5
    got = shader(g, cam)["radiance_linear"].permute(1, 2, 0).numpy()
    v, u = np.mgrid[0:common.H, 0:common.W].astype(np.float64)
    xn, yn = (u + 0.5 - common.CX) / common.FX, (v + 0.5 - common.CY) / common.FY
    P_shift = np.stack([xn * z, yn * z, z], -1)
    np.testing.assert_allclose(got, _oracle(common, P_shift, n, albedo, cosine=False),
                               rtol=1e-5, atol=1e-7)


def test_pixel_offset_is_required():
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    shader = build_shader(_config({"type": "none"}), PARAMS)
    g = _gbuffer(z, albedo)
    del g["pixel_offset"]
    with pytest.raises(KeyError):
        shader(g, cam)


def test_lambert_warmup_behaves_like_no_cosine():
    # During the warm-up the lambert shader must equal the no-cosine one and
    # emit no n . l map (so pixel_weight does not apply); after it, n . l.
    common = _common()
    z, P, n, albedo, cam = _scene(common)
    lam_cfg = {"type": "lambert", "normal_source": {"type": "depth_fd", "stencil_px": 1},
               "warmup_keyframes": 3}
    lam = build_shader(_config(lam_cfg), PARAMS)
    none = build_shader(_config({"type": "none"}), PARAMS)
    g = _gbuffer(z, albedo)
    lam.set_keyframe_count(3)
    out = lam(g, cam)
    torch.testing.assert_close(out["radiance_linear"], none(g, cam)["radiance_linear"])
    assert "light_cos_nl" not in out and "light_cosine_warmup" in out
    lam.set_keyframe_count(4)
    out = lam(g, cam)
    assert "light_cos_nl" in out and "light_cosine_warmup" not in out
    m = _interior(common)
    got = out["radiance_linear"].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got[m], _oracle(common, P, n, albedo)[m], rtol=1e-5, atol=1e-7)
