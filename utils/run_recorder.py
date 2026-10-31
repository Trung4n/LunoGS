# Recording of a SLAM run for offline replay videos (docs/DECISIONS.md D76).
# The hardware at hand cannot run the live GUI, so a run on Kaggle writes
# what the GUI would have shown, and tools/viz/render_replay.py turns it into
# a video afterwards. Logging only: nothing here is read back by SLAM.
#
# Config, Results.record (every key required):
#   enabled: <bool>                 slam.py --record sets it to true
#   frame_every: <n>                images every n frames (pose: every frame)
#   map_every_keyframes: <n>        map snapshot every n keyframes
#   image_width: <px>               width of the stored images
#   zip_dir: <path> | null          folder for <experiment>__<time>.zip of
#                                   the run, written by slam.py at the end
#                                   (utils/record_zip.py); null = no zip
#
# Layout under <save_dir>/record/:
#   meta.json            intrinsics, image size, colour space of the map, ...
#   frames.csv           per tracked frame: the LIVE pose estimate (right
#                        after tracking, before any later BA correction) and
#                        the reference pose, both camera-to-world 3x4
#   frames_final.csv     per frame at the end of the run: final pose estimate
#                        (keyframes BA-refined), reference pose, keyframe flag
#   images/<idx>_obs.png       observed image (sRGB)
#   images/<idx>_render.png    rendered image at the tracked pose (sRGB)
#   images/<idx>_albedo.png    rendered albedo (light model only), divided by
#                              the frame's albedo_scale (in frames.csv), sRGB
#   images/<idx>_depth.png     rendered depth, uint16 millimetres
#   maps/<tag>.npz       Gaussian map snapshot: xyz, colour, opacity, scale,
#                        rotation (activated values) and the creating keyframe
#
# The reference (ground-truth) pose is written for the replay only, to draw
# the reference trajectory next to the estimate; SLAM never reads it here.

import csv
import json
import os

import numpy as np
import torch
from PIL import Image

from utils.color_space import linear2sRGB
from utils.config_utils import require_key

RECORD_DIR = "record"
_INT_KEYS = ("frame_every", "map_every_keyframes", "image_width")
# Percentile of the albedo used to scale the stored albedo image into [0, 1]
# for display (the albedo is unbounded: K = 1 runs reach ~2-5). Display only.
ALBEDO_DISPLAY_PERCENTILE = 99.0


def read_record_config(results_cfg):
    """Results.record as a dict, or None when disabled."""
    rec = require_key(results_cfg, "record", "Results")
    if not isinstance(rec, dict):
        raise TypeError(f"Results.record must be a block, got {rec!r}")
    extra = set(rec) - {"enabled", "zip_dir", *_INT_KEYS}
    if extra:
        raise ValueError(f"Results.record: unexpected keys {sorted(extra)}")
    enabled = require_key(rec, "enabled", "Results.record")
    if not isinstance(enabled, bool):
        raise TypeError(f"Results.record.enabled must be true or false, got {enabled!r}")
    out = {}
    for key in _INT_KEYS:
        v = require_key(rec, key, "Results.record")
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"Results.record.{key} must be an integer >= 1, got {v!r}")
        out[key] = v
    zip_dir = require_key(rec, "zip_dir", "Results.record")
    if zip_dir is not None and (not isinstance(zip_dir, str) or not zip_dir):
        raise ValueError(f"Results.record.zip_dir must be a folder path or null, got {zip_dir!r}")
    out["zip_dir"] = zip_dir
    return out if enabled else None


def camera_to_world(R, T):
    """3x4 camera-to-world matrix from a Camera's world-to-camera R, T."""
    w2c = np.eye(4)
    w2c[:3, :3] = R.detach().cpu().numpy()
    w2c[:3, 3] = T.detach().cpu().numpy()
    return np.linalg.inv(w2c)[:3]


def _pose_cols(prefix):
    return [f"{prefix}_{r}{c}" for r in range(3) for c in range(4)]


def _rgb_uint8(img, width):
    """(3,H,W) tensor in [0, 1] -> (h,w,3) uint8 resized to `width`."""
    a = (img.detach().float().clamp(0.0, 1.0) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(a)
    height = max(1, round(im.height * width / im.width))
    return im.resize((width, height), Image.BILINEAR)


def _depth_uint16(depth, width):
    d = depth.detach().float().reshape(depth.shape[-2:]).cpu().numpy()
    mm = np.clip(d * 1000.0 + 0.5, 0, 65535).astype(np.uint16)
    im = Image.fromarray(mm)
    height = max(1, round(im.height * width / im.width))
    return im.resize((width, height), Image.NEAREST)


class RunRecorder:
    def __init__(self, cfg, save_dir, *, intrinsics, map_color, background):
        self.cfg = cfg
        self.root = os.path.join(save_dir, RECORD_DIR)
        os.makedirs(os.path.join(self.root, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "maps"), exist_ok=True)
        meta = {
            "version": 1,
            "intrinsics": intrinsics,
            "image_width": cfg["image_width"],
            "frame_every": cfg["frame_every"],
            "map_every_keyframes": cfg["map_every_keyframes"],
            # "linear_albedo": light model (colours are the linear albedo);
            # "srgb": original MonoGS (colours are sRGB radiance).
            "map_color": map_color,
            "background": background,
            # results/<experiment>__<scene>/<time>/: the default video label.
            "run": os.path.basename(os.path.dirname(os.path.normpath(save_dir))),
        }
        with open(os.path.join(self.root, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self._file = open(os.path.join(self.root, "frames.csv"), "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["frame_idx", "n_keyframes", "has_images", "albedo_scale"]
                           + _pose_cols("est") + _pose_cols("gt"))
        self._maps = set()

    # ---- per frame ----

    def frame(self, frame_idx, viewpoint, render_pkg, n_keyframes):
        """Pose of a frame right after tracking (render_pkg None: a frame
        that was not tracked, e.g. the initial one; no images then)."""
        est = camera_to_world(viewpoint.R, viewpoint.T)
        gt = camera_to_world(viewpoint.R_gt, viewpoint.T_gt)
        images = render_pkg is not None and frame_idx % self.cfg["frame_every"] == 0
        albedo_scale = ""
        if images:
            albedo_scale = self._save_images(frame_idx, viewpoint, render_pkg)
        self._csv.writerow([frame_idx, n_keyframes, int(images), albedo_scale]
                           + [f"{v:.7g}" for v in est.ravel()] + [f"{v:.7g}" for v in gt.ravel()])
        self._file.flush()

    def _save_images(self, frame_idx, viewpoint, render_pkg):
        w = self.cfg["image_width"]
        base = os.path.join(self.root, "images", f"{frame_idx:06d}")
        _rgb_uint8(viewpoint.original_image, w).save(base + "_obs.png")
        _rgb_uint8(render_pkg["render"], w).save(base + "_render.png")
        _depth_uint16(render_pkg["depth"], w).save(base + "_depth.png")
        if "albedo" not in render_pkg:
            return ""
        albedo = render_pkg["albedo"].detach().float()
        positive = albedo[albedo > 0]
        scale = float(torch.quantile(positive[:1_000_000], ALBEDO_DISPLAY_PERCENTILE / 100.0)) \
            if positive.numel() else 1.0
        scale = scale if scale > 0 else 1.0
        _rgb_uint8(linear2sRGB((albedo / scale).clamp(0.0, 1.0)), w).save(base + "_albedo.png")
        return f"{scale:.6g}"

    # ---- map snapshots ----

    def map_snapshot(self, n_keyframes, frame_idx, gaussians):
        """Called when the backend has mapped a new keyframe: snapshot of the
        map at the initial keyframe and every map_every_keyframes keyframes."""
        if n_keyframes in self._maps:
            return
        if n_keyframes != 1 and n_keyframes % self.cfg["map_every_keyframes"] != 0:
            return
        self._maps.add(n_keyframes)
        self._save_map(f"kf{n_keyframes:05d}", frame_idx, n_keyframes, gaussians)

    def refined_map(self, frame_idx, n_keyframes, gaussians):
        """The map after slam.py's offline color refinement (poses fixed):
        maps/final_refined.npz, next to the online final.npz."""
        self._save_map("final_refined", frame_idx, n_keyframes, gaussians)

    def _save_map(self, tag, frame_idx, n_keyframes, gaussians):
        def arr(t, dtype):
            return t.detach().float().cpu().numpy().astype(dtype)

        np.savez_compressed(
            os.path.join(self.root, "maps", tag + ".npz"),
            frame_idx=np.int64(frame_idx),
            n_keyframes=np.int64(n_keyframes),
            xyz=arr(gaussians.get_xyz, np.float32),
            color=arr(gaussians.get_albedo, np.float16),
            opacity=arr(gaussians.get_opacity[:, 0], np.float16),
            scaling=arr(gaussians.get_scaling, np.float16),
            rotation=arr(gaussians.get_rotation, np.float16),
            kf_id=gaussians.unique_kfIDs.detach().cpu().numpy().astype(np.int32),
        )

    # ---- end of run ----

    def finish(self, cameras, kf_indices, gaussians, frame_idx):
        kfs = set(int(k) for k in kf_indices)
        with open(os.path.join(self.root, "frames_final.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "is_keyframe"] + _pose_cols("est") + _pose_cols("gt"))
            for idx in sorted(cameras):
                cam = cameras[idx]
                est = camera_to_world(cam.R, cam.T)
                gt = camera_to_world(cam.R_gt, cam.T_gt)
                w.writerow([idx, int(idx in kfs)] + [f"{v:.7g}" for v in est.ravel()]
                           + [f"{v:.7g}" for v in gt.ravel()])
        if gaussians is not None:
            self._save_map("final", frame_idx, len(kf_indices), gaussians)
        self._file.close()


def open_recorder(config, dataset, shader, background):
    """RunRecorder for this run, or None when Results.record is disabled."""
    cfg = read_record_config(config["Results"])
    if cfg is None:
        return None
    if not config["Results"]["save_results"]:
        raise ValueError("Results.record.enabled needs Results.save_results (run with --eval)")
    intrinsics = {k: float(getattr(dataset, k)) for k in ("fx", "fy", "cx", "cy")}
    intrinsics.update(width=int(dataset.width), height=int(dataset.height))
    return RunRecorder(
        cfg,
        config["Results"]["save_dir"],
        intrinsics=intrinsics,
        map_color="srgb" if shader is None else "linear_albedo",
        background=[float(v) for v in background.detach().cpu().reshape(-1)],
    )
