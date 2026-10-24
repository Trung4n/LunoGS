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


# Calibration sensitivity (D57): name -> (path of the ParamSpec in Light,
# expected perturb block).
_SENSITIVITY = {
    "half_angle_m2deg": (("angular", "half_angle_deg"), {"add": -2.0}),
    "half_angle_m1deg": (("angular", "half_angle_deg"), {"add": -1.0}),
    "half_angle_p1deg": (("angular", "half_angle_deg"), {"add": 1.0}),
    "half_angle_p2deg": (("angular", "half_angle_deg"), {"add": 2.0}),
    "tilt_1deg": (("pose", "R_CL"), {"rotate_deg": 1.0, "axis": [1.0, 0.0, 0.0]}),
    "tilt_2deg": (("pose", "R_CL"), {"rotate_deg": 2.0, "axis": [1.0, 0.0, 0.0]}),
    "tilt_5deg": (("pose", "R_CL"), {"rotate_deg": 5.0, "axis": [1.0, 0.0, 0.0]}),
    "tcl_y1cm": (("pose", "t_CL"), {"add": [0.0, 0.01, 0.0]}),
    "tcl_y2cm": (("pose", "t_CL"), {"add": [0.0, 0.02, 0.0]}),
    "k_x080": (("intensity",), {"scale": 0.8}),
    "k_x090": (("intensity",), {"scale": 0.9}),
    "k_x110": (("intensity",), {"scale": 1.1}),
    "k_x120": (("intensity",), {"scale": 1.2}),
    "c_x080": (("ambient", "c"), {"scale": 0.8}),
    "c_x120": (("ambient", "c"), {"scale": 1.2}),
    "blend_x080": (("angular", "blend"), {"scale": 0.8}),
    "blend_x120": (("angular", "blend"), {"scale": 1.2}),
}


def test_light_config_set():
    # The set after the cleanup of D53 plus the component ablations of D54; a
    # new experiment is a new file here and a new line in this set.
    assert {os.path.basename(p) for p in LIGHT_CONFIGS} == {
        "rgbd_nocos.yaml",
        "rgbd_lambert.yaml",
        "rgbd_nocos_angular_uniform.yaml",
        "rgbd_nocos_ambient_none.yaml",
        "rgbd_nocos_ambient_additive.yaml",
        "rgbd_nocos_falloff_constant.yaml",
        *(f"rgbd_nocos_sens_{n}.yaml" for n in _SENSITIVITY),
        "rgbd_nocos_sweep.yaml",
        "rgbd_nocos_learn_half_angle_p2deg.yaml",
        "rgbd_nocos_learn_half_angle_m2deg.yaml",
        "rgbd_nocos_learn_tilt_5deg.yaml",
        "rgbd_nocos_learn_cone.yaml",
        "rgbd_nocos_learn_all.yaml",
        "rgbd_nocos_learn_all_perturbed.yaml",
        "rgbd_nocos_learn_k1.yaml",
        "rgbd_nocos_learn_half_angle_p4deg.yaml",
        "rgbd_nocos_learn_half_angle_p6deg.yaml",
        "rgbd_nocos_learn_tilt_10deg.yaml",
        "rgbd_lambert_learn_all.yaml",
        "rgbd_nocos_spline.yaml",
        "rgbd_nocos_learn_spline_flat.yaml",
        "rgbd_nocos_learn_spline_wide.yaml",
        "rgbd_nocos_learn_spline_narrow.yaml",
        "rgbd_lambert_learn_spline_flat.yaml",
        "rgbd_nocos_selfcal.yaml",
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


@pytest.mark.parametrize(
    "name, block, expected",
    [
        ("rgbd_nocos_angular_uniform", "angular", {"type": "uniform"}),
        ("rgbd_nocos_ambient_none", "ambient", {"type": "none"}),
        ("rgbd_nocos_ambient_additive", "ambient",
         {"type": "additive_const",
          "c": {"source": "params_file", "key": "ambient.c_estimated", "learnable": False}}),
        ("rgbd_nocos_falloff_constant", "falloff",
         {"type": "constant", "distance": {"source": "value", "value": 2.36, "learnable": False}}),
    ],
)
def test_component_ablations_change_one_block(monkeypatch, name, block, expected):
    # D54: each ablation is rgbd_nocos.yaml with exactly ONE Light component
    # changed, so a change in ATE is attributable to that component.
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    abl = load_config(f"configs/light/{name}.yaml")
    main = load_config("configs/light/rgbd_nocos.yaml")
    assert abl["Light"][block] == expected
    assert main["Light"][block] != expected
    for cfg in (abl, main):
        cfg["Light"].pop(block)
        cfg.pop("inherit_from")
    assert abl == main


@pytest.mark.parametrize("name", sorted(_SENSITIVITY))
def test_sensitivity_configs_perturb_one_parameter(monkeypatch, name):
    # D57: each sensitivity run is rgbd_nocos.yaml plus ONE perturb block on
    # one light parameter; the nominal value still comes from the params file.
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    path, perturb = _SENSITIVITY[name]
    sens = load_config(f"configs/light/rgbd_nocos_sens_{name}.yaml")
    main = load_config("configs/light/rgbd_nocos.yaml")
    spec = sens["Light"]
    for part in path:
        spec = spec[part]
    assert spec.pop("perturb") == perturb
    assert spec["source"] == "params_file"
    sens.pop("inherit_from")
    main.pop("inherit_from")
    assert sens == main


def test_sweep_config_reads_the_configured_light_position(monkeypatch):
    # D58: the b-sweep levels only have the CONFIGURED t_CL in their params
    # file; everything else is rgbd_nocos.yaml.
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    sweep = load_config("configs/light/rgbd_nocos_sweep.yaml")
    main = load_config("configs/light/rgbd_nocos.yaml")
    assert sweep["Light"]["pose"]["t_CL"] == {
        "source": "params_file", "key": "t_CL_m.config_opencv", "learnable": False}
    for cfg in (sweep, main):
        cfg["Light"]["pose"].pop("t_CL")
        cfg.pop("inherit_from")
    assert sweep == main


@pytest.mark.parametrize("name, learned", [
    ("rgbd_nocos_learn_half_angle_p2deg", ["angular.half_angle_deg"]),
    ("rgbd_nocos_learn_half_angle_m2deg", ["angular.half_angle_deg"]),
    ("rgbd_nocos_learn_tilt_5deg", ["pose.R_CL"]),
    ("rgbd_nocos_learn_cone", ["angular.half_angle_deg", "pose.R_CL"]),
    ("rgbd_nocos_learn_all", ["ambient.c", "angular.blend", "angular.half_angle_deg", "pose.R_CL", "pose.t_CL"]),
    ("rgbd_nocos_learn_all_perturbed", ["ambient.c", "angular.blend", "angular.half_angle_deg", "pose.R_CL", "pose.t_CL"]),
    ("rgbd_nocos_learn_k1", ["ambient.c", "angular.blend", "angular.half_angle_deg", "pose.R_CL", "pose.t_CL"]),
    ("rgbd_nocos_learn_half_angle_p4deg", ["angular.half_angle_deg"]),
    ("rgbd_nocos_learn_half_angle_p6deg", ["angular.half_angle_deg"]),
    ("rgbd_nocos_learn_tilt_10deg", ["pose.R_CL"]),
])
def test_self_calibration_configs(monkeypatch, name, learned):
    # D60: the learned parameters, and nothing else becomes learnable.
    pytest.importorskip("torch")
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    sh = build_shader(load_config(f"configs/light/{name}.yaml"), _FAKE_PARAMS)
    assert sorted(sh.learnable_params()) == learned
    assert build_shader(load_config("configs/light/rgbd_nocos.yaml"), _FAKE_PARAMS).learnable_params() == {}


@pytest.mark.parametrize(
    "path",
    sorted(
        glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
        + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))
    ),
)
def test_every_root_config_states_color_refinement(path):
    # slam.py requires Results.color_refinement (D61); True = original MonoGS
    # behaviour of --eval, --skip_color_refinement / --skip_render_eval set it
    # to False.
    with open(path, "r") as f:
        cfg = yaml.full_load(f)
    assert cfg["Results"]["color_refinement"] is True, path


def test_stage_a_starting_values(monkeypatch):
    # D65 stage A: K is never learned (gauge anchor); the perturbed run starts
    # wrong in every learned parameter; learn_k1 uses K = 1 and a guessed c'.
    pytest.importorskip("torch")
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)

    def build(name):
        return build_shader(load_config(f"configs/light/{name}.yaml"), _FAKE_PARAMS)

    good, bad, k1 = build("rgbd_nocos_learn_all"), build("rgbd_nocos_learn_all_perturbed"), build("rgbd_nocos_learn_k1")
    for sh in (good, bad, k1):
        assert "intensity" not in sh.learnable_params()
    g, b = good.learnable_state(), bad.learnable_state()
    assert b["angular.half_angle_deg"] == pytest.approx(g["angular.half_angle_deg"] + 2.0)
    assert b["angular.blend"] == pytest.approx(0.8 * g["angular.blend"])
    assert b["ambient.c"] == pytest.approx(1.5 * g["ambient.c"])
    assert b["pose.t_CL"][1] == pytest.approx(g["pose.t_CL"][1] + 0.01)
    assert float(k1.intensity) == 1.0
    assert k1.learnable_state()["ambient.c"] == pytest.approx(0.01)
    cfg = load_config("configs/light/rgbd_nocos_learn_k1.yaml")
    assert cfg["Light"]["init_albedo"]["min_shading"] < 0.01  # below the start c'
    assert set(cfg["Light"]["intensity"]) == {"source", "value", "learnable"}


def test_ambient_c_cannot_be_negative():
    # Checked at construction and after every update of a learned c (D65).
    pytest.importorskip("torch")
    import light_models.ambient  # noqa: F401  (registers the forms)
    from light_models.registry import build_component

    negative = {"source": "value", "value": -0.1, "learnable": False}
    for form in ("multiplicative_const", "additive_const"):
        with pytest.raises(ValueError, match="ambient.c"):
            build_component("ambient", {"type": form, "c": negative}, "Light.ambient", None)


@pytest.mark.parametrize("name", ["learn_all", "learn_spline_flat"])
def test_lambert_learn_diagnostic_is_nocos_plus_the_lambert_of_rgbd_lambert(monkeypatch, name):
    # D66 / D69 diagnostics: the only differences with the nocos learning
    # config are the n . l term and its pixel weight, copied from
    # rgbd_lambert.yaml.
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    diag = load_config(f"configs/light/rgbd_lambert_{name}.yaml")
    learn = load_config(f"configs/light/rgbd_nocos_{name}.yaml")
    lam = load_config("configs/light/rgbd_lambert.yaml")
    assert diag["Light"]["cosine"] == lam["Light"]["cosine"]
    assert diag["LightTracking"]["pixel_weight"] == lam["LightTracking"]["pixel_weight"]
    for cfg in (diag, learn):
        cfg["Light"].pop("cosine")
        cfg["LightTracking"].pop("pixel_weight")
        cfg.pop("inherit_from")
    assert diag == learn


@pytest.mark.parametrize("name, learned", [
    ("rgbd_nocos_spline", []),
    ("rgbd_nocos_learn_spline_flat", ["angular.knot_values"]),
    ("rgbd_nocos_learn_spline_wide", ["angular.knot_values"]),
    ("rgbd_nocos_learn_spline_narrow", ["angular.knot_values"]),
])
def test_spline_configs_change_only_the_angular_profile(monkeypatch, name, learned):
    # D67: rgbd_nocos.yaml with the angular block replaced by a spline (and a
    # schedule when it learns); 1 deg knots from 0 to 50 deg.
    pytest.importorskip("torch")
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config(f"configs/light/{name}.yaml")
    sh = build_shader(cfg, _FAKE_PARAMS)
    assert sorted(sh.learnable_params()) == learned
    assert cfg["Light"]["angular"]["knots_deg"] == list(range(0, 51))
    main = load_config("configs/light/rgbd_nocos.yaml")
    for c in (cfg, main):
        c["Light"].pop("angular")
        c["Light"].pop("schedule", None)
        c.pop("inherit_from")
    assert cfg == main


def test_selfcal_reads_only_the_light_position_from_the_params_file(monkeypatch):
    # D70: self-calibration mode. The params file holds only t_CL (the mount's
    # design value); K = 1, c' guessed, spline from flat, R_CL from identity.
    pytest.importorskip("torch")
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/rgbd_nocos_selfcal.yaml")
    sh = build_shader(cfg, {"t_CL_m": {"config_opencv": [0.0, -0.05, 0.0]}})
    assert sorted(sh.learnable_params()) == [
        "ambient.c", "angular.knot_values", "pose.R_CL", "pose.t_CL"]
    after = {p: lp.after_keyframes for p, lp in sh.learnable_params().items()}
    assert after == {"ambient.c": 3, "angular.knot_values": 3, "pose.R_CL": 10, "pose.t_CL": 10}
    assert cfg["Light"]["intensity"] == {"source": "value", "value": 1.0, "learnable": False}
    assert cfg["Light"]["angular"]["init"] == {"type": "uniform"}
