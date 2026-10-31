"""utils/run_recorder.py and tools/viz/render_replay.py (docs/DECISIONS.md D76):
recording a run for offline replay, and turning the recording into a video
(CPU point renderer; the Gaussian renderer needs CUDA).

    python -m pytest tests/test_run_recorder.py
"""
import csv
import glob
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from utils.run_recorder import (  # noqa: E402
    RunRecorder,
    camera_to_world,
    open_recorder,
    read_record_config,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"enabled": True, "frame_every": 5, "map_every_keyframes": 10, "image_width": 32, "zip_dir": None}
H, W = 34, 60


# ---- config ------------------------------------------------------------------


def test_disabled_record_is_none_and_keys_are_validated():
    assert read_record_config({"record": {**CFG, "enabled": False}}) is None
    assert read_record_config({"record": CFG}) == {"frame_every": 5, "map_every_keyframes": 10, "image_width": 32,
                                                   "zip_dir": None}
    assert read_record_config({"record": {**CFG, "zip_dir": "z"}})["zip_dir"] == "z"
    with pytest.raises(KeyError):
        read_record_config({})
    for bad in ({**CFG, "frame_every": 0}, {**CFG, "enabled": "yes"}, {**CFG, "extra": 1},
                {**CFG, "zip_dir": ""}, {**CFG, "zip_dir": 3},
                {k: v for k, v in CFG.items() if k != "image_width"},
                {k: v for k, v in CFG.items() if k != "zip_dir"}):
        with pytest.raises((ValueError, TypeError, KeyError)):
            read_record_config({"record": bad})


@pytest.mark.parametrize("path", sorted(
    glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
    + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))))
def test_every_root_config_states_record_off(path):
    # Off = original MonoGS; slam.py --record switches it on (D76).
    with open(path) as f:
        cfg = yaml.full_load(f)
    assert read_record_config(cfg["Results"]) is None, path


def test_record_needs_saved_results():
    config = {"Results": {"record": CFG, "save_results": False, "save_dir": None}}
    with pytest.raises(ValueError, match="save_results"):
        open_recorder(config, None, None, torch.zeros(3))


# ---- recording ---------------------------------------------------------------


def _pose(t):
    """World-to-camera R, T of a camera at position t looking along +z."""
    return torch.eye(3), -torch.tensor(t, dtype=torch.float32)


def _viewpoint(i):
    R, T = _pose([0.01 * i, 0.0, 0.0])
    Rg, Tg = _pose([0.01 * i, 0.002 * i, 0.0])
    return SimpleNamespace(R=R, T=T, R_gt=Rg, T_gt=Tg, original_image=torch.rand(3, H, W))


def _render_pkg(with_albedo):
    pkg = {"render": torch.rand(3, H, W), "depth": torch.full((1, H, W), 2.0)}
    if with_albedo:
        pkg["albedo"] = 3.0 * torch.rand(3, H, W)  # K = 1 scale: above 1
    return pkg


def _gaussians(n):
    g = torch.Generator().manual_seed(n)
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([2.0, 1.0, 1.0]) + torch.tensor([-0.5, -0.5, 2.0])
    return SimpleNamespace(
        get_xyz=xyz, get_albedo=torch.rand(n, 3, generator=g), get_opacity=torch.ones(n, 1),
        get_scaling=torch.full((n, 3), 0.01), get_rotation=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        unique_kfIDs=torch.zeros(n, dtype=torch.int32))


def _record(tmp_path, with_albedo=True, n_frames=21):
    save_dir = tmp_path / "rgbd_nocos__office0_lit" / "2026-10-28"
    rec = RunRecorder(CFG, str(save_dir), intrinsics={"fx": 30.0, "fy": 30.0, "cx": W / 2, "cy": H / 2,
                                                      "width": W, "height": H},
                      map_color="linear_albedo" if with_albedo else "srgb", background=[0.0, 0.0, 0.0])
    cameras, kfs = {}, []
    for i in range(n_frames):
        vp = _viewpoint(i)
        cameras[i] = vp
        if i == 0:
            kfs.append(0)
            rec.frame(0, vp, None, len(kfs))
            rec.map_snapshot(len(kfs), 0, _gaussians(100))
            continue
        rec.frame(i, vp, _render_pkg(with_albedo), len(kfs))
        if i % 2 == 0:  # a keyframe every second frame
            kfs.append(i)
            rec.map_snapshot(len(kfs), i, _gaussians(100 + i))
    rec.finish(cameras, kfs, _gaussians(200), n_frames - 1)
    return save_dir


def test_refined_map_is_written_after_finish(tmp_path):
    rec = RunRecorder(CFG, str(tmp_path / "run"), intrinsics={"fx": 30.0, "fy": 30.0, "cx": W / 2, "cy": H / 2,
                                                             "width": W, "height": H},
                      map_color="linear_albedo", background=[0.0, 0.0, 0.0])
    rec.finish({}, [], _gaussians(50), 0)
    rec.refined_map(20, 7, _gaussians(60))   # slam.py calls it after the color refinement (D81)
    with np.load(tmp_path / "run" / "record" / "maps" / "final_refined.npz") as z:
        assert len(z["xyz"]) == 60 and int(z["frame_idx"]) == 20 and int(z["n_keyframes"]) == 7


def test_recording_layout(tmp_path):
    run = _record(tmp_path)
    root = run / "record"
    meta = json.loads((root / "meta.json").read_text())
    assert meta["run"] == "rgbd_nocos__office0_lit" and meta["map_color"] == "linear_albedo"
    with open(root / "frames.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["frame_idx"]) for r in rows] == list(range(21))
    # images every 5 frames, never for the untracked initial frame
    assert [int(r["frame_idx"]) for r in rows if r["has_images"] == "1"] == [5, 10, 15, 20]
    for kind in ("obs", "render", "albedo", "depth"):
        assert (root / "images" / f"000010_{kind}.png").is_file()
    assert float(rows[10]["albedo_scale"]) > 1.0  # unbounded albedo scaled for display
    # map snapshots: initial keyframe, every 10 keyframes, final
    maps = sorted(os.path.basename(p) for p in glob.glob(str(root / "maps" / "*.npz")))
    assert maps == ["final.npz", "kf00001.npz", "kf00010.npz"]
    with np.load(root / "maps" / "kf00010.npz") as z:
        assert z["xyz"].shape == (118, 3) and int(z["n_keyframes"]) == 10 and int(z["frame_idx"]) == 18
    with open(root / "frames_final.csv") as f:
        final = list(csv.DictReader(f))
    assert sum(r["is_keyframe"] == "1" for r in final) == 11
    # poses are camera-to-world: position of frame 10 is (0.1, 0, 0)
    assert float(rows[10]["est_03"]) == pytest.approx(0.1)
    assert float(rows[10]["gt_13"]) == pytest.approx(0.02)


def test_camera_to_world_inverts_world_to_camera():
    R = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T = torch.tensor([1.0, 2.0, 3.0])
    c2w = camera_to_world(R, T)
    assert np.allclose(c2w[:, :3] @ (R.numpy() @ np.array([4.0, 5, 6]) + T.numpy()) + c2w[:, 3], [4, 5, 6])


# ---- replay ------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")


def _replay():
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_replay", os.path.join(REPO, "tools", "viz", "render_replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_look_at_and_projection():
    rr = _replay()
    up = np.array([0.0, -1.0, 0.0])  # OpenCV: up is -y
    c2w = rr.look_at(np.zeros(3), np.array([0.0, 0.0, 5.0]), up)
    assert np.allclose(c2w[:3, :3], np.eye(3), atol=1e-9)
    K = rr.intrinsics(200, 100, 90.0)
    uv, front = rr.project(np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -1.0]]), c2w, K)
    assert np.allclose(uv[0], [100, 50]) and front.tolist() == [True, False]
    # a point 1 m to the right at 1 m depth lands on the image edge for 90 deg FOV
    uv, _ = rr.project(np.array([[1.0, 0.0, 1.0]]), c2w, K)
    assert uv[0, 0] == pytest.approx(200.0)
    assert np.allclose(rr.world_up([c2w, c2w]), up)


def test_follow_camera_sits_behind_and_above():
    rr = _replay()
    up = np.array([0.0, -1.0, 0.0])
    cam = np.eye(4)
    cam[:3, 3] = [1.0, 0.0, 0.0]
    f = rr.FollowCamera(up, back=1.0, height=0.5, ahead=1.0, smoothing=1.0)
    v = f.update(cam)
    assert np.allclose(v[:3, 3], [1.0, -0.5, -1.0])
    assert v[2, 2] > 0.8  # still looking forward


def test_replay_video_from_a_recording(tmp_path):
    rr = _replay()
    a = _record(tmp_path / "a", with_albedo=True)
    b = _record(tmp_path / "b", with_albedo=False)
    out = tmp_path / "replay.gif"
    rr.main(["--run", str(a), "--run", str(b), "--label", "light", "--label", "MonoGS",
             "--out", str(out), "--renderer", "points", "--map_width", "240", "--map_height", "180",
             "--panel_width", "100", "--view", "overview"])
    from PIL import Image

    im = Image.open(out)
    assert im.n_frames == 4  # frames 5, 10, 15, 20
    assert im.size == ((240 + 200) // 2, (180 * 2) // 2)
    # the light run shows its albedo, the baseline run has none
    run_a, run_b = rr.Run(str(a)), rr.Run(str(b))
    assert run_a.image(10, "albedo") is not None and run_b.image(10, "albedo") is None
    assert run_a.color_gain > 0 and run_b.color_gain == 1.0
    assert math.isfinite(run_a.color_gain)
    # the legend is written next to the video, not drawn on it
    notes = (tmp_path / "replay.txt").read_text(encoding="utf-8")
    assert "light" in notes and "MonoGS" in notes and "3D map (orbit)" in notes and "overview" in notes


def test_both_views_with_default_names_from_a_record_folder(tmp_path):
    rr = _replay()
    a = _record(tmp_path, with_albedo=True)
    # the record/ folder itself, as downloaded on its own
    rr.main(["--run", str(a / "record"), "--renderer", "points", "--map_width", "160", "--map_height", "90",
             "--panel_width", "60", "--no_h264"])
    videos = a / "videos"
    names = sorted(os.listdir(videos))
    assert names == ["rgbd_nocos__office0_lit_follow.mp4", "rgbd_nocos__office0_lit_follow.txt",
                     "rgbd_nocos__office0_lit_overview.mp4", "rgbd_nocos__office0_lit_overview.txt"]
    cap = cv2.VideoCapture(str(videos / "rgbd_nocos__office0_lit_follow.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    cap.release()


def test_orbit_camera_circles_the_centre():
    rr = _replay()
    up = np.array([0.0, 0.0, 1.0])
    c = np.array([1.0, 2.0, 0.5])
    poses = [rr.orbit_pose(c, up, 2.0, math.radians(30), a) for a in (0.0, math.pi / 2, math.pi)]
    for p in poses:
        assert np.linalg.norm(p[:3, 3] - c) == pytest.approx(2.0)
        assert (p[:3, 3] - c) @ up == pytest.approx(1.0)  # 2 * sin(30 deg)
        assert np.allclose(p[:3, 2], (c - p[:3, 3]) / 2.0)  # looks at the centre
    assert np.allclose(poses[0][:2, 3] + poses[2][:2, 3], 2 * c[:2])  # opposite sides


def test_pack_records_zips_each_run(tmp_path):
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location("pack_records", os.path.join(REPO, "tools", "viz", "pack_records.py"))
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    run = _record(tmp_path / "results")  # results/rgbd_nocos__office0_lit/2026-10-28/record
    (run / "config.yml").write_text("x: 1")
    (run / "timing.csv").write_text("kind,index,ms,iters\n")
    pr.main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "zips")])
    path = tmp_path / "zips" / "rgbd_nocos__office0_lit__2026-10-28.zip"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        z.extractall(tmp_path / "unzipped")
    root = "rgbd_nocos__office0_lit__2026-10-28/"
    assert root + "config.yml" in names and root + "record/meta.json" in names
    assert root + "timing.csv" in names and root + "light_learned.csv" not in names
    assert any(n.startswith(root + "record/maps/") for n in names)
    # unzipped, the folder is directly a --run of the replay tool
    assert _replay().Run(str(tmp_path / "unzipped" / root)).image_frames == [5, 10, 15, 20]
