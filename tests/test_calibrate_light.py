"""tools/calibrate (docs/DECISIONS.md D106): light pre-calibration from a
flat-target clip. CPU only, small images.

    python -m pytest tests/test_calibrate_light.py
"""
import copy
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(REPO, "tools", "calibrate")
sys.path.insert(0, CAL)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CAL, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fl = _load("fit_light")
ms = _load("make_synthetic")

torch = pytest.importorskip("torch")
from utils.config_utils import load_config  # noqa: E402

# Parameters of the test lamp (values of this test, not of any dataset).
PARAMS = {"K": 30.0, "t": [0.0, -0.05, 0.0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          "half": 30.0, "blend": 0.35, "c": 0.12}
INTR = {"fx": 40.0, "fy": 40.0, "cx": 39.5, "cy": 22.5, "width": 80, "height": 46}


def calib_block(**over):
    c = {"target_albedo": [0.05, 0.05, 0.05], "plane_inlier_m": 0.01, "plane_iters": 3, "sat_8bit": 250,
         "loss_color_space": "linear", "stages": [150, 300, 300]}
    c.update(over)
    return c


def test_read_calibration_requires_every_key_and_rejects_extras():
    assert fl.read_calibration({"Calibration": calib_block()}).stages == [150, 300, 300]
    for key in fl.CALIBRATION_KEYS:
        block = calib_block()
        del block[key]
        with pytest.raises(KeyError):
            fl.read_calibration({"Calibration": block})
    with pytest.raises(ValueError):
        fl.read_calibration({"Calibration": calib_block(extra=1)})
    with pytest.raises(ValueError):
        fl.read_calibration({"Calibration": calib_block(loss_color_space="log")})
    with pytest.raises(ValueError):
        fl.read_calibration({"Calibration": calib_block(stages=[])})


def test_fit_plane_ignores_outliers_and_faces_the_camera():
    rays = fl.pixel_grid(INTR, INTR["height"], INTR["width"])
    n_true = np.array([0.3, -0.2, -1.0])
    n_true /= np.linalg.norm(n_true)
    d_true = float(n_true @ np.array([0, 0, 2.0]))
    z = fl.plane_depth(n_true, d_true, rays)
    z[:10, :10] = 0.5  # an object in front of the target
    pts = rays * z[..., None]
    n, d, inl = fl.fit_plane(pts, z > 0, 0.01, 3)
    np.testing.assert_allclose(n, n_true, atol=1e-6)
    assert d == pytest.approx(d_true, abs=1e-6)
    assert not inl[:10, :10].any() and inl[20:, 20:].all()


def _light(learn):
    """Light block of the test lamp; learn=True: the datasheet-style start of
    configs/calib/fit_spot_datasheet.yaml (perturbed, every parameter learned)."""
    fixed = lambda key: {"source": "params_file", "key": key, "learnable": False}  # noqa: E731
    light = {
        "shader": {"type": "colocated"},
        "intensity": fixed("K"),
        "pose": {"t_CL": fixed("t"), "R_CL": dict(fixed("R"), perturb={"rotate_deg": 3.0, "axis": [1.0, 0.0, 0.0]})},
        "falloff": {"type": "inverse_square"},
        "angular": {"type": "blender_spot", "half_angle_deg": fixed("half"), "blend": fixed("blend")},
        "cosine": {"type": "lambert", "normal_source": {"type": "gbuffer"}, "warmup_keyframes": 0},
        "ambient": {"type": "multiplicative_const", "c": fixed("c")},
        "gbuffer": {"opacity_thr": 0.5},
    }
    if not learn:
        return light
    fit = load_config(os.path.join(REPO, "configs", "calib", "fit_spot_datasheet.yaml"))["Light"]
    light = copy.deepcopy(fit)
    rekey = {"t_CL_m.config_opencv": "t", "cone.half_angle_deg_config": "half", "cone.blend_config": "blend"}
    for block in (light["pose"]["t_CL"], light["angular"]["half_angle_deg"], light["angular"]["blend"]):
        block["key"] = rekey[block["key"]]
    return light


@pytest.mark.parametrize("loss_space", ["linear", "srgb"])
def test_synthetic_clip_recovers_the_lamp(tmp_path, loss_space):
    synthetic = {"intrinsics": INTR, "depth_scale": 6553.5, "n_frames": 16, "distance_m": [0.8, 4.0],
                 "max_tilt_deg": 60, "target_albedo": [0.05, 0.05, 0.05], "jpeg_quality": None,
                 "depth_noise_m": 0.0, "seed": 0}
    truth = ms.make({"Light": _light(False), "Synthetic": synthetic}, PARAMS, str(tmp_path), log=lambda s: None)
    truth = truth["light"]

    from light_models.shaders import make_shader

    light = _light(True)
    calib = fl.read_calibration({"Calibration": calib_block(loss_color_space=loss_space)})
    shader = make_shader(light, lambda: PARAMS)
    intr, frames = fl.load_clip(str(tmp_path))
    prepared = fl.prepare(frames, intr, calib, "cpu")
    fl.fit(shader, prepared, intr, calib, log=lambda s: None)
    got = fl.light_values(shader, light, PARAMS)

    assert got["intensity"] == pytest.approx(truth["intensity"], rel=0.03)
    assert got["angular"]["half_angle_deg"] == pytest.approx(truth["angular"]["half_angle_deg"], abs=0.3)
    np.testing.assert_allclose(got["light_axis_tilt_deg"], truth["light_axis_tilt_deg"], atol=0.3)
    np.testing.assert_allclose(got["t_CL_m"], truth["t_CL_m"], atol=0.005)
    assert got["ambient_of_calibration_room"]["c"] == pytest.approx(PARAMS["c"], rel=0.05)


def test_fit_refuses_a_schedule_with_more_stages_than_configured(tmp_path):
    from light_models.shaders import make_shader

    light = _light(True)
    shader = make_shader(light, lambda: PARAMS)
    calib = fl.read_calibration({"Calibration": calib_block(stages=[10, 10])})  # schedule uses stage 2
    with pytest.raises(ValueError, match="stage 2"):
        fl.fit(shader, [], INTR, calib, log=lambda s: None)


def test_light_values_report_the_tilt_of_the_axis():
    from light_models.shaders import make_shader

    light = _light(False)
    got = fl.light_values(make_shader(light, lambda: PARAMS), light, PARAMS)
    # 3 deg about the camera x axis turns the axis towards -y.
    np.testing.assert_allclose(got["light_axis_tilt_deg"], [0.0, -3.0], atol=1e-9)
    assert math.isclose(np.linalg.det(np.array(got["R_CL"])), 1.0, rel_tol=1e-9)
