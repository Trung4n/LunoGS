"""tools/verify/eval_map.py (docs/DECISIONS.md D79): the pure parts of the map
quality evaluation (the rendering itself needs CUDA).

    python -m pytest tests/test_eval_map.py
"""
import importlib.util
import json
import math
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("eval_map", os.path.join(REPO, "tools", "verify", "eval_map.py"))
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


def test_srgb_round_trip_and_psnr():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(em.linear_to_srgb(em.srgb_to_linear(x)), x, atol=1e-9)
    a = np.zeros((4, 4, 3))
    b = a + 0.1
    assert em.psnr(a, b) == pytest.approx(20.0)
    assert em.psnr(a, a) == math.inf
    mask = np.zeros((4, 4), bool)
    assert math.isnan(em.psnr(a, b, mask))
    mask[0, 0] = True
    b[1:, :] = 1.0  # outside the mask: ignored
    assert em.psnr(a, b, mask) == pytest.approx(20.0)


def test_gain_fit_recovers_a_per_channel_scale():
    rng = np.random.default_rng(0)
    pred = rng.random((20, 30, 3))
    target = pred * np.array([2.0, 0.5, 1.5])
    mask = np.ones((20, 30), bool)
    fit = em.GainFit()
    fit.add(pred[:10], target[:10], mask[:10])
    fit.add(pred[10:], target[10:], mask[10:])
    np.testing.assert_allclose(fit.gain(), [2.0, 0.5, 1.5])
    assert (em.GainFit().gain() == 1.0).all()  # nothing accumulated


def test_cone_mask_follows_the_light_axis():
    # 100 x 100 image, focal 50 px: the image edge is ~45 deg off the axis
    intr = {"fx": 50.0, "fy": 50.0, "cx": 50.0, "cy": 50.0}
    depth = np.full((100, 100), 2.0)
    depth[0, 0] = 0.0
    col = lambda deg: int(50 + 50 * math.tan(math.radians(deg)))  # noqa: E731
    m = em.cone_mask(depth, intr, [0, 0, 0], [0, 0, 1], 30.0)
    assert not m[0, 0]  # no depth
    assert m[49, 49] and m[49, col(25)] and not m[49, col(35)]
    # tilting the axis by 20 deg about y moves the cone sideways
    t = 20.0
    m2 = em.cone_mask(depth, intr, [0, 0, 0], [math.sin(math.radians(t)), 0, math.cos(math.radians(t))], 10.0)
    assert m2[49, col(t)] and not m2[49, 49]
    # an offset light (t_CL) shifts the apex: at 2 m, 0.5 m to the right is ~14 deg
    m3 = em.cone_mask(depth, intr, [0.5, 0, 0], [0, 0, 1], 5.0)
    assert m3[49, col(14)] and not m3[49, 49]


def test_eval_frames_skip_keyframes():
    assert em.eval_frames(21, [0, 10, 12], 5) == [5, 15]


def test_learned_state_from_csv(tmp_path):
    p = tmp_path / "light_learned.csv"
    p.write_text("frame_idx,n_keyframes,ambient.c,angular.knot_values[0],angular.knot_values[1],"
                 "pose.R_CL[0],pose.R_CL[1],ambient.c.n_at_bound\n"
                 "5,1,0.01,1,1,0,0,0\n100,20,0.005,0.9,0.5,0.1,-0.2,3\n")
    assert em.learned_state(str(p)) == {"ambient.c": 0.005, "angular.knot_values": [0.9, 0.5],
                                        "pose.R_CL": [0.1, -0.2]}


def test_table_formats_runs(tmp_path):
    m = {"label": "nocos", **{k: 0.5 for k, _ in em.COLUMNS}}
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    out = em.table([str(tmp_path / "metrics.json")])
    assert out.splitlines()[0].startswith("| run | lit PSNR") and "| nocos | 0.50 |" in out
