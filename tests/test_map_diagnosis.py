"""Map diagnosis tools (docs/DECISIONS.md D81): tools/verify/model_ceiling.py,
tools/analysis/diagnose_map.py, tools/verify/eval_records.py. CPU only.

    python -m pytest tests/test_map_diagnosis.py
"""
import csv
import importlib.util
import json
import math
import os
import zipfile

import numpy as np
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, *parts):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mc = _load("model_ceiling", "tools", "verify", "model_ceiling.py")
dm = _load("diagnose_map", "tools", "analysis", "diagnose_map.py")
er = _load("eval_records", "tools", "verify", "eval_records.py")


# ---- model_ceiling -------------------------------------------------------------


def test_voxel_keys_group_points_of_one_cell():
    pts = np.array([[0.001, 0.001, 0.001], [0.019, 0.010, 0.005], [0.021, 0.0, 0.0], [-0.001, 0.0, 0.0]])
    k = mc.voxel_keys(pts, 0.02)
    assert k[0] == k[1] and len({k[0], k[2], k[3]}) == 3


def test_fit_per_group_is_least_squares_per_key():
    keys = np.array([5, 5, 9])
    pred = np.array([1.0, 2.0, 4.0])
    obs = np.array([2.0, 4.0, 2.0])    # key 5: k = 2 exactly; key 9: k = 0.5
    uniq, k, inv = mc.fit_per_group(keys, obs * pred, pred * pred)
    assert uniq.tolist() == [5, 9] and np.allclose(k, [2.0, 0.5]) and inv.tolist() == [0, 0, 1]


def test_pooled_psnr():
    assert mc.pooled_psnr(0.01 * 4, 4) == pytest.approx(20.0)
    assert math.isnan(mc.pooled_psnr(0.0, 0)) and mc.pooled_psnr(0.0, 3) == float("inf")


# ---- diagnose_map ----------------------------------------------------------------


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def test_pose_errors_ignore_a_rigid_offset_and_find_a_local_error():
    gt = {}
    for i in range(20):
        m = np.eye(4)
        m[:3, :3] = _rot_z(3 * i)
        m[:3, 3] = [0.1 * i, 0.05 * i * i / 20, 0.3 * math.sin(i / 3)]
        gt[i] = m
    world = np.eye(4)
    world[:3, :3] = _rot_z(40)
    world[:3, 3] = [1.0, -2.0, 0.5]
    poses = {i: (world @ gt[i], gt[i]) for i in gt}       # same trajectory, other world frame
    err = dm.pose_errors(poses, rpe_delta=2)
    assert max(e[0] for e in err.values()) < 1e-9 and max(e[1] for e in err.values()) < 1e-4  # deg; arccos round-off
    assert math.isnan(err[19][2]) and err[0][2] < 1e-9
    moved = dict(poses)
    est = poses[10][0].copy()
    est[:3, 3] += world[:3, :3] @ np.array([0.0, 0.0, 0.05])   # 5 cm off at frame 10
    moved[10] = (est, gt[10])
    err = dm.pose_errors(moved, rpe_delta=2)
    assert err[10][0] == pytest.approx(0.05, abs=0.01) and err[10][0] > 5 * err[0][0]


def test_spearman_and_terciles():
    x = [1, 2, 3, 4, 5, 6]
    assert dm.spearman(x, [10, 20, 30, 40, 50, 60]) == pytest.approx(1.0)
    assert dm.spearman(x, [6, 5, 4, 3, 2, float("nan")]) == pytest.approx(-1.0)
    rows = [{"f": v, "lit_psnr_in_cone": 30.0 - v} for v in x]
    t = dm.terciles(rows, "f", "lit_psnr_in_cone")
    assert [b["n"] for b in t] == [2, 2, 2] and t[0]["metric_mean"] > t[2]["metric_mean"]


def test_local_path_maps_wsl_and_windows():
    if os.name == "nt":
        assert dm.local_path("/mnt/d/CodeForYou/x") == "D:/CodeForYou/x"
    else:
        assert dm.local_path("D:/CodeForYou/x") == "/mnt/d/CodeForYou/x"
    assert dm.local_path("record/x") == "record/x"


def test_ceiling_csv_and_figure(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "ceiling_per_frame.csv"
    path.write_text("frame,nocos_in_cone\n5,30.0\n10,nan\n15,31.0\n")
    label, x, y = dm.read_ceiling(f"nocos ceiling={path}:nocos_in_cone")
    assert label == "nocos ceiling" and x == [5, 15] and y == [30.0, 31.0]
    rows = [{"frame": f, "lit_psnr_in_cone": 25.0 + f / 10, "ate_t_m": 0.01 + f / 1e4, "sat_fraction": 0.02}
            for f in (5, 10, 15)]
    runs = [("a", rows, [(1, 1, 1000), (15, 3, 2000)], "/d/lit"), ("b", rows, [(1, 1, 900)], "/d/lit")]
    dm.figure(str(tmp_path / "fig.png"), runs, [(label, x, y)])
    assert (tmp_path / "fig.png").stat().st_size > 10_000


# ---- eval_records ----------------------------------------------------------------


def _fake_run(root, name, dataset_path):
    run = root / name
    (run / "record" / "maps").mkdir(parents=True)
    (run / "record" / "meta.json").write_text("{}")
    cfg = {"inherit_from": "configs/light/rgbd_nocos.yaml", "Dataset": {"dataset_path": dataset_path},
           "Light": {"enabled": True}}
    (run / "config.yml").write_text(yaml.safe_dump(cfg))
    return run


def test_extract_resolved_config_and_scene(tmp_path):
    from utils.record_zip import zip_run

    name_dir = _fake_run(tmp_path / "results" / "rgbd_nocos__room0_lit", "2026-10-31-09-05-00",
                         "/kaggle/input/replica-monogs/room0/lit")
    inbox = tmp_path / "inbox"
    zip_run(str(name_dir), str(inbox))
    zips, dirs = er.find_inputs(str(inbox))
    assert [os.path.basename(z) for z in zips] == ["rgbd_nocos__room0_lit__2026-10-31-09-05-00.zip"] and dirs == []
    run = er.extract(zips[0], str(tmp_path / "record"))
    assert os.path.isfile(os.path.join(run, "record", "meta.json"))
    assert er.extract(zips[0], str(tmp_path / "record")) == run      # already extracted: kept
    path, cfg = er.resolved_config(run)
    with open(path, encoding="utf-8") as f:
        assert "inherit_from" not in yaml.safe_load(f)
    assert er.scene_variant(cfg) == ("room0", "lit")
    assert er.label_of(os.path.basename(run), "lit") == "rgbd_nocos (lit, 10-31 09:05)"


def test_extract_refuses_a_foreign_zip(tmp_path):
    bad = tmp_path / "x.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "no")
    with pytest.raises(SystemExit):
        er.extract(str(bad), str(tmp_path / "record"))


# ---- figure_maps -----------------------------------------------------------------


def test_figure_grid_layout():
    fm = _load("figure_maps", "tools", "verify", "figure_maps.py")
    img = np.full((34, 60, 3), 0.5)
    canvas = fm.grid([[img, img, img], [img, None, img]], [["a", "b", "c"], ["d", "e", "f"]],
                     tile_h=68, label_h=20, gap=4)
    assert canvas.size == (3 * 120 + 2 * 4, 2 * (68 + 20) + 4)
    assert canvas.getpixel((130, 20 + 68 + 4 + 20 + 30)) == (255, 255, 255)   # the empty tile
    assert fm.nearest_frame(1003, [995, 1005, 1010]) == 1005


def test_segment_means_and_live_psnr(tmp_path):
    from PIL import Image

    rows = [{"frame": f, "v": float(f)} for f in (0, 100, 399, 400, 850)] + [{"frame": 900, "v": float("nan")}]
    assert dm.segment_means(rows, "v", 400) == {"0-400": pytest.approx(499 / 3), "400-800": 400.0, "800-1200": 850.0}
    (tmp_path / "images").mkdir()
    obs = np.full((4, 4, 3), 100, np.uint8)
    Image.fromarray(obs).save(tmp_path / "images" / "000005_obs.png")
    Image.fromarray(obs + 10).save(tmp_path / "images" / "000005_render.png")
    assert dm.live_psnr(str(tmp_path), 5) == pytest.approx(-10 * math.log10((10 / 255) ** 2))
    assert math.isnan(dm.live_psnr(str(tmp_path), 6))
