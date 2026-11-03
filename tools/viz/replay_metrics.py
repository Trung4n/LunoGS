"""Numbers of a recorded run for the replay app (docs/DECISIONS.md D100).

ATE   position error after a global Umeyama alignment of the estimated
      trajectory onto the reference, as utils/eval_utils.evaluate_evo:
      Sim(3) for monocular runs, SE(3) for RGB-D (Dataset.sensor_type of
      the run's config.yml; without it no alignment is chosen). Two series:
        live      every tracked frame, pose right after tracking (frames.csv);
        keyframe  the keyframes at the end of the run (frames_final.csv),
                  which is what eval_ate reports.
      The alignment of each series is fitted on that whole series.

Image metrics  PSNR / SSIM / LPIPS of the RECORDED rendered image (at the
      tracked pose, at tracking time, `image_width` px, 8-bit sRGB) against
      the recorded observed image, over all pixels, as tools/verify/eval_map.py
      (SSIM of gaussian_splatting, LPIPS AlexNet of torchmetrics). They follow
      the run over time; they are not the final map quality of eval_map.py.
      Computed on the CPU in a background thread (the GPU stays with the
      renderer); SSIM needs torch, LPIPS needs torchmetrics.

Display only: nothing here feeds back into SLAM. The reference trajectory is
read for evaluation, as in eval_ate.
"""
import math
import os
import sys
import threading

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.trajectory_analysis import umeyama  # noqa: E402  (numpy only)


# ---------------------------------------------------------------- ATE


def alignment(config):
    """'sim3' (monocular) or 'se3' (RGB-D), as eval_ate's correct_scale;
    None without config.yml."""
    if config is None:
        return None
    sensor = config["Dataset"]["sensor_type"]
    return "sim3" if sensor == "monocular" else "se3"


def aligned_errors(frames, est, gt, with_scale):
    """(N,) position error (m) of est[f] after aligning all est positions onto
    gt (Umeyama), or None with fewer than 3 frames."""
    if len(frames) < 3:
        return None
    p = np.array([est[f][:3, 3] for f in frames], dtype=np.float64)
    q = np.array([gt[f][:3, 3] for f in frames], dtype=np.float64)
    s, r, t = umeyama(p, q, with_scale=with_scale)
    return np.linalg.norm(s * p @ r.T + t - q, axis=1)


def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def ate_summary(run):
    """{"align": 'se3'|'sim3'|None, "live": [[frame, cm]], "live_rmse": cm,
    "keyframe": [[frame, cm]], "keyframe_rmse": cm} of a render_replay.Run.
    Series that cannot be computed are empty lists / None."""
    align = alignment(run.config)
    out = {"align": align, "live": [], "live_rmse": None, "keyframe": [], "keyframe_rmse": None}
    if align is None:
        return out
    with_scale = align == "sim3"
    frames = sorted(f for f in run.est_live if f in run.gt)
    e = aligned_errors(frames, run.est_live, run.gt, with_scale)
    if e is not None:
        out["live"] = [[f, round(100 * float(v), 3)] for f, v in zip(frames, e)]
        out["live_rmse"] = 100 * _rmse(e)
    kfs = [k for k in run.keyframes if k in run.est_final and k in run.gt]
    e = aligned_errors(kfs, run.est_final, run.gt, with_scale)
    if e is not None:
        out["keyframe"] = [[f, round(100 * float(v), 3)] for f, v in zip(kfs, e)]
        out["keyframe_rmse"] = 100 * _rmse(e)
    return out


# ---------------------------------------------------------------- image metrics


def psnr(a, b):
    """PSNR (dB) of two images in [0, 1] (as eval_map.psnr, all pixels)."""
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def _metric_functions():
    """(ssim(a, b) or None, lpips(a, b) or None, {metric: why it is missing}).
    a, b: (H,W,3) float arrays in [0, 1]."""
    missing = {}
    try:
        import torch

        from gaussian_splatting.utils.loss_utils import ssim as ssim_t
    except ImportError as exc:
        return None, None, {"ssim": f"cần torch ({exc})", "lpips": f"cần torch ({exc})"}

    def tensor(a):
        return torch.tensor(np.ascontiguousarray(a.transpose(2, 0, 1)), dtype=torch.float32)[None]

    def ssim(a, b):
        with torch.no_grad():
            return float(ssim_t(tensor(a), tensor(b)))

    lpips = None
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        net = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

        def lpips(a, b):
            with torch.no_grad():
                return float(net(tensor(a), tensor(b)))
    except Exception as exc:  # noqa: BLE001 - missing package or weights: reported to the UI
        missing["lpips"] = f"cần torchmetrics và trọng số AlexNet ({exc.__class__.__name__}: {exc})"
    return ssim, lpips, missing


class ImageMetrics:
    """Background computation of the image metrics of one run."""

    _compute_lock = threading.Lock()  # one run at a time (CPU bound)

    def __init__(self, run):
        self.run = run
        self.state = "idle"
        self.rows = []      # [frame, psnr, ssim | None, lpips | None]
        self.missing = {}
        self.message = ""
        self.total = len(run.image_frames)
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.state != "idle":
                return
            self.state = "running"
        threading.Thread(target=self._job, daemon=True).start()

    def _load(self, frame, kind):
        from PIL import Image

        path = os.path.join(self.run.dir, "images", f"{frame:06d}_{kind}.png")
        return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0

    def _job(self):
        try:
            with ImageMetrics._compute_lock:
                ssim, lpips, self.missing = _metric_functions()
                for f in self.run.image_frames:
                    obs, ren = self._load(f, "obs"), self._load(f, "render")
                    lp = None
                    if lpips is not None:
                        try:
                            lp = lpips(ren, obs)
                        except RuntimeError as exc:  # e.g. images too small for AlexNet
                            self.missing["lpips"] = f"LPIPS lỗi ({exc})"
                            lpips = None
                    row = [f, psnr(ren, obs), None if ssim is None else ssim(ren, obs), lp]
                    with self._lock:
                        self.rows.append(row)
            self.state = "done"
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            self.message = f"{exc.__class__.__name__}: {exc}"
            self.state = "error"

    def payload(self):
        with self._lock:
            rows = list(self.rows)
        out = {"state": self.state, "done": len(rows), "total": self.total, "missing": self.missing,
               "message": self.message}
        for k, name in enumerate(("psnr", "ssim", "lpips"), start=1):
            out[name] = [[r[0], round(r[k], 5)] for r in rows if r[k] is not None and math.isfinite(r[k])]
        return out
