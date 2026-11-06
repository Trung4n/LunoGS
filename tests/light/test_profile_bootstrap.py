"""light_models/profile_bootstrap.py (docs/DECISIONS.md D112): coarse emission
profile from the first frame. CPU, small synthetic frames made with the shader.

    python -m pytest tests/light/test_profile_bootstrap.py
"""
import math
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from light_models import build_shader, read_profile_bootstrap  # noqa: E402
from light_models.profile_bootstrap import (  # noqa: E402
    bootstrap_spline, check_shader_supports_bootstrap, estimate_cone, fit_cone)
from utils.color_space import linear2sRGB  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H, W, F = 170, 300, 150.0
PARAMS = dict(sat_8bit=250, min_cos_nl=0.3, bin_deg=1.0, min_pixels_per_bin=5, grid_deg=0.5,
              min_ramp_deg=2.0, min_contrast=2.0)
P = SimpleNamespace(**PARAMS)
KNOTS = [0, 8, *range(9, 51)]


def light(angular, cosine=None, profile_bootstrap=None):
    fixed = lambda v: {"source": "value", "value": v, "learnable": False}  # noqa: E731
    cfg = {"Training": {"spherical_harmonics": False}, "Dataset": {"sensor_type": "depth"}, "Light": {
        "enabled": True, "shader": {"type": "colocated"}, "intensity": fixed(1.0),
        "pose": {"t_CL": fixed([0.0, 0.0, 0.0]), "R_CL": fixed([[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
        "falloff": {"type": "inverse_square"}, "angular": angular,
        "cosine": cosine or {"type": "lambert", "warmup_keyframes": 0,
                             "normal_source": {"type": "sensor_depth_fd_edge", "stencil_px": 1,
                                               "trusted_pixels_only": False}},
        "ambient": {"type": "multiplicative_const", "c": fixed(0.004)},
        "gbuffer": {"opacity_thr": 0.5},
    }}
    if angular.get("learn", {}).get("learnable"):
        cfg["Light"]["schedule"] = [{"after_keyframes": 3, "unfreeze": ["angular.knot_values"]}]
    if profile_bootstrap is not None:
        cfg["Light"]["profile_bootstrap"] = profile_bootstrap
    return cfg


def spline_block():
    return {"type": "spline", "knots_deg": KNOTS, "interpolation": "linear_cos", "init": {"type": "uniform"},
            "learn": {"learnable": True, "lr": 0.01, "prior_weight": 0.0}, "smoothness_weight": 0.0}


def spot(half=30.0, blend=0.35):
    fixed = lambda v: {"source": "value", "value": v, "learnable": False}  # noqa: E731
    return {"type": "blender_spot", "half_angle_deg": fixed(half), "blend": fixed(blend)}


def wall_frame(albedo_seed=0, half=30.0, z=2.0, k_scale=0.4):
    """A slightly tilted wall with blotchy albedo lit by a true spot: (observed sRGB (3,H,W), depth, camera)."""
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    x, y = (u - cx) / F, (v - cy) / F
    # n . X = d with n tilted towards x: z(p) = d / (n . ray)
    n = torch.tensor([0.2, 0.1, -1.0]); n = n / n.norm()
    depth = float(n[2] * z) / (n[0] * x + n[1] * y + n[2])
    rng = np.random.default_rng(albedo_seed)
    coarse = torch.tensor(rng.uniform(0.5, 2.0, (9, 16)), dtype=torch.float32)[None, None]
    blotch = torch.nn.functional.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    # k_scale keeps the core of the cone unsaturated in this scene.
    albedo = (k_scale * blotch)[None].expand(3, H, W)
    camera = SimpleNamespace(fx=F, fy=F, cx=cx, cy=cy, depth=depth.numpy().astype(np.float64))
    true = build_shader(light(spot(half)), {})
    gbuffer = {"albedo": albedo, "depth": depth[None], "opacity": torch.ones(1, H, W), "pixel_offset": 0.0}
    with torch.no_grad():
        radiance = true(gbuffer, camera)["radiance_linear"]
    observed = torch.round(linear2sRGB(radiance.clamp(0.0, 1.0)).clamp(0.0, 1.0) * 255.0) / 255.0
    return observed, camera.depth, camera


# ---- configuration ------------------------------------------------------------

def test_absent_block_means_no_bootstrap():
    assert read_profile_bootstrap({}) is None


def test_every_key_is_required_and_extras_rejected():
    assert read_profile_bootstrap({"profile_bootstrap": dict(PARAMS)}).grid_deg == 0.5
    for key in PARAMS:
        block = {k: v for k, v in PARAMS.items() if k != key}
        with pytest.raises(KeyError, match=key):
            read_profile_bootstrap({"profile_bootstrap": block})
    with pytest.raises(ValueError, match="unexpected"):
        read_profile_bootstrap({"profile_bootstrap": dict(PARAMS, extra=1)})


@pytest.mark.parametrize("key, value", [("sat_8bit", 0), ("sat_8bit", 2.5), ("min_cos_nl", 1.0), ("bin_deg", 0),
                                        ("min_pixels_per_bin", 0), ("grid_deg", -1), ("min_ramp_deg", 0),
                                        ("min_contrast", 1.0), ("min_contrast", True)])
def test_values_validated(key, value):
    with pytest.raises(ValueError, match=key):
        read_profile_bootstrap({"profile_bootstrap": dict(PARAMS, **{key: value})})


def test_shader_must_be_able_to_use_it():
    block = dict(PARAMS)
    assert build_shader(light(spline_block(), profile_bootstrap=block), {}) is not None
    with pytest.raises(ValueError, match="spline"):
        build_shader(light(spot(), profile_bootstrap=block), {})
    fixed_spline = dict(spline_block(), learn={"learnable": False})
    with pytest.raises(ValueError, match="learnable spline"):
        build_shader(light(fixed_spline, profile_bootstrap=block), {})
    with pytest.raises(ValueError, match="sensor-depth"):
        build_shader(light(spline_block(), cosine={"type": "none"}, profile_bootstrap=block), {})


def test_refused_for_monocular():
    cfg = light(spline_block(), profile_bootstrap=dict(PARAMS))
    cfg["Dataset"]["sensor_type"] = "monocular"
    with pytest.raises(ValueError, match="monocular"):
        build_shader(cfg, {})


# ---- the fit -------------------------------------------------------------------

def curve(theta_in, theta_out, core=100.0, out=2.0, bumps=None):
    centers = np.arange(0.5, 50.0, 1.0)
    x = np.clip((theta_out - centers) / (theta_out - theta_in), 0, 1)
    medians = out + (core - out) * x * x * (3 - 2 * x)
    return centers, medians * (1.0 if bumps is None else bumps(centers))


def test_fit_recovers_a_clean_cone():
    fit = fit_cone(*curve(24.0, 30.0), P)
    assert fit.theta_in == pytest.approx(24.0, abs=0.5)
    assert fit.theta_out == pytest.approx(30.0, abs=0.5)
    assert fit.level_core / fit.level_out == pytest.approx(50.0, rel=0.05)


def test_fit_keeps_the_edge_when_the_albedo_changes_along_the_core():
    # A factor 2 across the core (surfaces of different albedo): the edge stays.
    bumps = lambda c: np.where(c < 15.0, 0.5, 1.0)  # noqa: E731
    fit = fit_cone(*curve(24.0, 30.0, bumps=bumps), P)
    assert fit.theta_out == pytest.approx(30.0, abs=1.0)


def test_fit_raises_without_contrast():
    centers = np.arange(0.5, 50.0, 1.0)
    with pytest.raises(ValueError, match="no cone edge"):
        fit_cone(centers, np.full_like(centers, 3.0), P)


# ---- on a frame -----------------------------------------------------------------

@pytest.mark.parametrize("half", [25.0, 30.0, 35.0])
def test_estimate_on_a_rendered_frame_finds_the_edge(half):
    observed, depth, camera = wall_frame(half=half)
    shader = build_shader(light(spline_block(), profile_bootstrap=dict(PARAMS)), {})
    fit = estimate_cone(observed, depth, camera, shader, P)
    assert fit.theta_out == pytest.approx(half, abs=1.5)
    assert fit.theta_in < fit.theta_out


def test_saturated_pixels_are_left_out():
    observed, depth, camera = wall_frame()
    shader = build_shader(light(spline_block(), profile_bootstrap=dict(PARAMS)), {})
    clipped = observed.clone()
    clipped[:, H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10] = 1.0  # a clipped spot on the axis
    fit = estimate_cone(clipped, depth, camera, shader, P)
    assert fit.theta_out == pytest.approx(30.0, abs=1.5)


def test_bootstrap_restarts_the_spline_in_place():
    observed, depth, camera = wall_frame()
    shader = build_shader(light(spline_block(), profile_bootstrap=dict(PARAMS)), {})
    check_shader_supports_bootstrap(shader)
    param = shader.learnable_params()["angular.knot_values"]
    tensor = param.tensor
    assert bool((tensor.detach() == 1.0).all())  # flat start
    viewpoint = SimpleNamespace(original_image=observed, depth=depth, fx=camera.fx, fy=camera.fy,
                                cx=camera.cx, cy=camera.cy)
    fit = bootstrap_spline(shader, viewpoint, P)
    assert param.tensor is tensor  # the optimizer holds this object
    assert tensor.is_leaf and tensor.requires_grad
    v = dict(zip(KNOTS[1:], tensor.detach().tolist()))
    assert v[8] == pytest.approx(1.0)
    assert v[int(math.ceil(fit.theta_out)) + 1] == pytest.approx(0.0, abs=1e-6)
    # Non-increasing and the prior target follows the new start.
    values = tensor.detach().tolist()
    assert all(b <= a for a, b in zip(values, values[1:]))
    torch.testing.assert_close(param.start, tensor.detach())
    # The shader now renders a cone: nothing outside, full on the axis.
    phi = shader.angular(torch.cos(torch.tensor([0.0, math.radians(fit.theta_out + 2.0)])))
    torch.testing.assert_close(phi, torch.tensor([1.0, 0.0]), atol=1e-5, rtol=0)


def test_config_builds_with_a_flat_start_and_the_block(monkeypatch):
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/rgbd_lambert_selfcal_bootstrap.yaml")
    p = read_profile_bootstrap(cfg["Light"])
    assert p is not None
    assert cfg["Light"]["angular"]["init"] == {"type": "uniform"}
    # Thresholds shared with the SLAM masks stay equal to them.
    assert p.sat_8bit == cfg["LightTracking"]["saturation_mask"]["threshold_8bit"]
    assert p.min_cos_nl == cfg["LightTracking"]["pixel_weight"]["min_cos_nl"]
    # Nothing of the lamp comes from the data's params file.
    assert cfg["Light"]["pose"]["t_CL"]["source"] == "value"
    assert cfg["Light"]["pose"]["R_CL"]["source"] == "value"
    assert cfg["Light"]["intensity"]["source"] == "value"
    assert cfg["Light"]["ambient"]["c"]["source"] == "value"


# ---- the backend hook -----------------------------------------------------------

def test_backend_hook_does_nothing_without_the_block():
    from utils.slam_backend import BackEnd

    # Original MonoGS (no shader) and light models without the block: no call
    # reaches the shader, here an object that would raise on any attribute.
    fake = SimpleNamespace(profile_bootstrap=None)
    BackEnd.bootstrap_light_profile(fake, 0, None)


def test_backend_hook_restarts_the_spline():
    from utils.slam_backend import BackEnd

    observed, depth, camera = wall_frame()
    shader = build_shader(light(spline_block(), profile_bootstrap=dict(PARAMS)), {})
    viewpoint = SimpleNamespace(original_image=observed, depth=depth, fx=camera.fx, fy=camera.fy,
                                cx=camera.cx, cy=camera.cy)
    fake = SimpleNamespace(profile_bootstrap=P, shader=shader)
    BackEnd.bootstrap_light_profile(fake, 0, viewpoint)
    assert float(shader.angular.values().min()) == pytest.approx(0.0, abs=1e-6)


def test_settings_survive_the_pickling_of_the_backend():
    # The backend object is pickled into its own process (spawn).
    p = pickle.loads(pickle.dumps(read_profile_bootstrap({"profile_bootstrap": dict(PARAMS)})))
    assert p.grid_deg == PARAMS["grid_deg"]


def test_a_restarted_shader_state_reaches_a_frontend_copy():
    observed, depth, camera = wall_frame()
    cfg = light(spline_block(), profile_bootstrap=dict(PARAMS))
    backend, frontend = build_shader(cfg, {}), pickle.loads(pickle.dumps(build_shader(cfg, {})))
    viewpoint = SimpleNamespace(original_image=observed, depth=depth, fx=camera.fx, fy=camera.fy,
                                cx=camera.cx, cy=camera.cy)
    bootstrap_spline(backend, viewpoint, P)
    assert not bool((frontend.angular.values() == backend.angular.values()).all())
    frontend.load_learnable_state(backend.learnable_state())
    torch.testing.assert_close(frontend.angular.values(), backend.angular.values())
