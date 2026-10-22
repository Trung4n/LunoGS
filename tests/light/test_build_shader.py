"""light_models.build_shader: strict reading of the Light block (no torch needed)."""
import glob
import os
import pickle

import pytest
import yaml

from light_models import build_shader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config(light=None, spherical_harmonics=False):
    cfg = {"Training": {"spherical_harmonics": spherical_harmonics}}
    if light is not None:
        cfg["Light"] = light
    return cfg


def test_missing_light_block_raises():
    with pytest.raises(KeyError, match="Light"):
        build_shader(_config())


def test_missing_enabled_key_raises():
    with pytest.raises(KeyError, match="Light.enabled"):
        build_shader(_config(light={}))


@pytest.mark.parametrize("value", ["false", 0, None])
def test_enabled_must_be_bool(value):
    with pytest.raises(TypeError):
        build_shader(_config(light={"enabled": value}))


def test_disabled_returns_none():
    assert build_shader(_config(light={"enabled": False})) is None


def test_enabled_rejects_spherical_harmonics():
    with pytest.raises(ValueError, match="spherical_harmonics"):
        build_shader(
            _config(
                light={"enabled": True, "shader": {"type": "identity"}},
                spherical_harmonics=True,
            )
        )


def test_enabled_requires_shader_block():
    with pytest.raises(KeyError, match="Light.shader"):
        build_shader(_config(light={"enabled": True}))


def test_enabled_requires_shader_type():
    with pytest.raises(KeyError, match="Light.shader.type"):
        build_shader(_config(light={"enabled": True, "shader": {}}))


def test_identity_shader_returns_albedo_unchanged():
    shader = build_shader(_config(light={"enabled": True, "shader": {"type": "identity"}}))
    albedo = object()
    out = shader({"albedo": albedo, "depth": None, "opacity": None}, None)
    assert out == {"radiance_linear": albedo}


def test_shader_survives_pickling():
    # Backend and GUI processes are started with "spawn".
    shader = build_shader(_config(light={"enabled": True, "shader": {"type": "identity"}}))
    clone = pickle.loads(pickle.dumps(shader))
    assert clone({"albedo": 1}, None) == {"radiance_linear": 1}


def test_colocated_requires_its_components():
    # colocated reads intensity/pose/components from the Light block.
    with pytest.raises(KeyError, match="Light.intensity"):
        build_shader(
            _config(light={"enabled": True, "shader": {"type": "colocated"}}),
            params_data={},
        )


@pytest.mark.parametrize("name", ["Identity", "lambert", None])
def test_unknown_shader_type_raises(name):
    with pytest.raises(ValueError, match="Light.shader.type"):
        build_shader(_config(light={"enabled": True, "shader": {"type": name}}))


def test_identity_rejects_parameters():
    with pytest.raises(ValueError, match="Unknown keys"):
        build_shader(
            _config(light={"enabled": True, "shader": {"type": "identity", "gain": 2}})
        )



@pytest.mark.parametrize(
    "path",
    sorted(
        glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
        + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))
    ),
)
def test_every_root_config_states_baseline_explicitly(path):
    with open(path, "r") as f:
        cfg = yaml.full_load(f)
    assert cfg["Light"]["enabled"] is False, path
    # Background moved from slam.py into the config (docs/DECISIONS.md D9):
    # the baseline value must stay exactly the old hard-coded one.
    assert cfg["pipeline_params"]["background"] == [0, 0, 0], path
    # Loss options that reproduce the baseline (docs/DECISIONS.md D2, D3).
    assert cfg["LightTracking"] == {
        "loss_color_space": "srgb",
        "exposure_affine": True,
        "pixel_weight": {"enabled": False},
        "saturation_mask": {"enabled": False},
        "min_gradient": {"enabled": False},
    }, path




@pytest.mark.parametrize(
    "path",
    sorted(
        glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
        + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))
    ),
)
def test_every_depth_config_states_alpha(path):
    # The RGB-D loss requires Training.alpha (D47); original MonoGS silently
    # used 0.95 when it was missing, so that is the value a config without
    # one had (TUM / EuRoC state their own 0.9).
    with open(path, "r") as f:
        cfg = yaml.full_load(f)
    if cfg["Dataset"]["sensor_type"] == "monocular":
        return
    assert isinstance(cfg["Training"]["alpha"], float), path


# Stand-in for light_params.json with every key the configs read: the
# estimated values (SLAM) and the Blender config values (verification, D13).
# Arbitrary numbers, only the structure matters here.
_FAKE_PARAMS = {
    "intensity": {"K_estimated_step3": 30.0, "K_theory": 30.0},
    "t_CL_m": {"estimated_opencv": [0.0, -0.05, 0.0], "config_opencv": [0.0, -0.05, 0.0]},
    "R_CL": {"estimated": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             "config": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    "cone": {"half_angle_deg_estimated": 30.0, "blend_estimated": 0.35,
             "half_angle_deg_config": 30.0, "blend_config": 0.35},
    "ambient": {"c_estimated": 0.12, "c_config": 0.12},
}


# ---- configs/light: the experiment and verification configs (D53) ----------

LIGHT_CONFIGS = sorted(glob.glob(os.path.join(REPO, "configs", "light", "*.yaml")))
# Settings files read by a tool next to an experiment config, not runnable.
_SETTINGS_ONLY = {"verify_tracking.yaml"}


def test_light_config_set():
    # The kept set after the cleanup of D53; a new experiment is a new file
    # here and a new line in this set.
    assert {os.path.basename(p) for p in LIGHT_CONFIGS} == {
        "rgbd_nocos.yaml",
        "rgbd_lambert.yaml",
        "verify_shading.yaml",
        "verify_roundtrip.yaml",
        "verify_tracking.yaml",
        "verify_reparam.yaml",
    }


@pytest.mark.parametrize(
    "path", [p for p in LIGHT_CONFIGS if os.path.basename(p) not in _SETTINGS_ONLY]
)
def test_every_light_config_resolves_and_builds(monkeypatch, path):
    # Resolved exactly as slam.py does (inherit_from relative to the repo
    # root); every runnable config is RGB-D (D47, D53) and builds its shader.
    pytest.importorskip("torch")
    from light_models import read_light_tracking
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config(os.path.relpath(path, REPO))
    assert cfg["Dataset"]["sensor_type"] == "depth"
    read_light_tracking(cfg)
    assert build_shader(cfg, _FAKE_PARAMS) is not None


def test_verify_reparam_is_the_rgbd_baseline_with_an_identity_shader(monkeypatch):
    from light_models import read_albedo_init, read_light_tracking
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/verify_reparam.yaml")
    assert type(build_shader(cfg)).__name__ == "IdentityShader"
    assert read_albedo_init(cfg).strategy == "observed"
    lt = read_light_tracking(cfg)
    assert (lt.loss_color_space, lt.exposure_affine) == ("srgb", True)
    # Everything else must be the baseline's.
    base = load_config("configs/rgbd/replica/office0.yaml")
    for key in ("Light", "LightTracking", "Verify", "inherit_from"):
        cfg.pop(key, None)
        base.pop(key, None)
    assert cfg == base


def test_main_model_rgbd_nocos(monkeypatch):
    pytest.importorskip("torch")
    from light_models import read_light_tracking
    from light_models.albedo_init import get_albedo_init
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/rgbd_nocos.yaml")
    assert cfg["Training"]["alpha"] == 0.95
    assert cfg["Light"]["cosine"] == {"type": "none"}
    lt = read_light_tracking(cfg)
    assert lt.exposure_affine is False and lt.pixel_weight.enabled is False
    sm = lt.saturation_mask
    assert (sm.enabled, sm.mode, sm.apply_to) == (True, "pixel", ("tracking", "mapping"))
    # The saturation threshold is the one of the verification (D16, D44).
    shading = load_config("configs/light/verify_shading.yaml")
    assert sm.threshold_8bit == shading["Verify"]["layer1"]["saturation_8bit"]
    assert type(build_shader(cfg, _FAKE_PARAMS)).__name__ == "ColocatedShader"
    get_albedo_init(cfg)


def test_lambert_differs_from_the_main_model_only_by_the_cosine(monkeypatch):
    from light_models import read_light_tracking
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    lam = load_config("configs/light/rgbd_lambert.yaml")
    nocos = load_config("configs/light/rgbd_nocos.yaml")
    assert lam["Light"]["cosine"] == {
        "type": "lambert", "warmup_keyframes": 0,
        "normal_source": {"type": "sensor_depth_fd", "stencil_px": 1},
    }
    pw = read_light_tracking(lam).pixel_weight
    assert (pw.enabled, pw.apply_to, pw.min_cos_nl) == (True, ("tracking", "mapping"), 0.3)
    for cfg in (lam, nocos):
        cfg["Light"].pop("cosine")
        cfg["LightTracking"].pop("pixel_weight")
        cfg.pop("inherit_from")
    assert lam == nocos


def test_placement_depth_albedo_init_refused_for_monocular(monkeypatch):
    pytest.importorskip("torch")
    from light_models.albedo_init import get_albedo_init
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/rgbd_nocos.yaml")
    assert cfg["Light"]["init_albedo"]["strategy"] == "placement_depth"
    cfg["Dataset"]["sensor_type"] = "monocular"
    with pytest.raises(ValueError, match="measured"):
        get_albedo_init(cfg)


def test_sensor_depth_normals_refused_for_monocular(monkeypatch):
    # sensor_depth_fd reads Camera.depth: a sensor in RGB-D, ground truth in
    # monocular (loaded anyway, D20), so the builder refuses it there (D49).
    pytest.importorskip("torch")
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/rgbd_lambert.yaml")
    assert build_shader(cfg, _FAKE_PARAMS).requires_sensor_depth is True
    cfg["Dataset"]["sensor_type"] = "monocular"
    with pytest.raises(ValueError, match="monocular"):
        build_shader(cfg, _FAKE_PARAMS)
    # Sources that do not read the sensor are allowed.
    depth_fd = load_config("configs/light/rgbd_lambert.yaml")
    depth_fd["Light"]["cosine"]["normal_source"] = {"type": "depth_fd", "stencil_px": 1}
    depth_fd["Dataset"]["sensor_type"] = "monocular"
    assert build_shader(depth_fd, _FAKE_PARAMS).requires_sensor_depth is False


def test_sensor_and_map_normals_agree_on_a_smooth_map(monkeypatch):
    # When the rasterized depth IS the sensor depth (smooth, same pixel
    # convention), sensor_depth_fd and depth_fd must shade identically: the
    # only difference between them is where n comes from.
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    h, w = 24, 32
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    z = 2.0 / (1.0 - 0.3 * (u - w / 2) / 40.0 + 0.2 * (v - h / 2) / 40.0)
    cam = SimpleNamespace(fx=40.0, fy=40.0, cx=w / 2, cy=h / 2, depth=z.numpy().copy())
    g = {"albedo": torch.full((3, h, w), 0.5, dtype=torch.float64), "depth": z[None],
         "opacity": torch.ones(1, h, w, dtype=torch.float64), "pixel_offset": 0.0}
    out = {}
    for source in ("sensor_depth_fd", "depth_fd"):
        cfg = load_config("configs/light/rgbd_lambert.yaml")
        cfg["Light"]["cosine"]["normal_source"] = {"type": source, "stencil_px": 1}
        out[source] = build_shader(cfg, _FAKE_PARAMS)(g, cam)
    a, b = out["sensor_depth_fd"], out["depth_fd"]
    assert torch.equal(a["light_normal_valid"], b["light_normal_valid"])
    torch.testing.assert_close(a["radiance_linear"], b["radiance_linear"])
