# Run-time diagnostics of the light model. Logging only: nothing here feeds
# back into tracking or mapping.
#
# Albedo above 1 (docs/DECISIONS.md D1). The albedo is not clamped above 1 on
# purpose, so that a mismatch between the calibrated light intensity and the
# data shows up instead of being hidden: a large or growing fraction of
# Gaussians above 1 means the fixed intensity is too low for the observed
# images (or the Replica base color was baked too bright). The bound 1 is the
# definition of a physical albedo, not a tunable threshold.
#
# Two stages per keyframe, so the source of the excess can be told apart:
#   new - the Gaussians just created for this keyframe (Light.init_albedo);
#   map - every Gaussian after this keyframe's mapping.
# Printed with Log (no square brackets in the message: Log uses rich, which
# would swallow "[new]" / "[map]" as markup) and, when the run saves
# results, appended to
# <save_dir>/albedo_stats.csv.
#
# No torch import: only tensor methods of the input are used.

import math
import os

from utils.logging_utils import Log

CSV_NAME = "albedo_stats.csv"
_COLUMNS = (
    "frame_idx",
    "stage",
    "n_gaussians",
    "frac_any_gt1",
    "frac_r_gt1",
    "frac_g_gt1",
    "frac_b_gt1",
    "max",
    "mean",
)


def albedo_stats(albedo):
    """albedo: (N,3) tensor. Fractions are over Gaussians (not pixels)."""
    n = int(albedo.shape[0])
    if n == 0:
        return {"n_gaussians": 0}
    above = albedo > 1.0
    per_channel = above.float().mean(dim=0)
    return {
        "n_gaussians": n,
        "frac_any_gt1": above.any(dim=1).float().mean().item(),
        "frac_r_gt1": per_channel[0].item(),
        "frac_g_gt1": per_channel[1].item(),
        "frac_b_gt1": per_channel[2].item(),
        "max": albedo.max().item(),
        "mean": albedo.mean().item(),
    }


def log_albedo_stats(save_dir, frame_idx, stage, albedo):
    """Logs albedo_stats for one (keyframe, stage). save_dir None = no file
    (Results.save_results: false); the console line is always printed."""
    stats = albedo_stats(albedo.detach())
    row = {"frame_idx": frame_idx, "stage": stage, **stats}
    if stats["n_gaussians"] == 0:
        Log(f"albedo stage={stage} kf {frame_idx}: no Gaussians", tag="Light")
    else:
        Log(
            f"albedo stage={stage} kf {frame_idx}: "
            f"{stats['frac_any_gt1']:.2%} of {stats['n_gaussians']} Gaussians > 1 "
            f"(max {stats['max']:.3f}, mean {stats['mean']:.3f})",
            tag="Light",
        )
    if save_dir is None:
        return
    path = os.path.join(save_dir, CSV_NAME)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(_COLUMNS) + "\n")
        f.write(",".join(_format(row.get(c)) for c in _COLUMNS) + "\n")


EXPOSURE_CSV_NAME = "exposure_stats.csv"
_EXPOSURE_COLUMNS = ("frame_idx", "stage", "exposure_a", "exposure_b", "gain_exp_a")


def log_exposure(save_dir, frame_idx, stage, viewpoint):
    """Per-keyframe affine exposure (Camera.exposure_a/b), for EVERY run,
    baseline included: reading the values changes no computation.

    Why: with a co-located light the affine term can absorb the brightness
    change caused by the moving lamp (docs/DECISIONS.md D6, step 4.3). The
    stages mirror the albedo log: `new` = value tracked by the frontend when
    the keyframe arrives, `map` = after this keyframe's mapping.
    """
    a = float(viewpoint.exposure_a.detach().item())
    b = float(viewpoint.exposure_b.detach().item())
    gain = math.exp(a)
    Log(
        f"exposure stage={stage} kf {frame_idx}: a={a:+.4f} (gain {gain:.4f}) b={b:+.4f}",
        tag="Light",
    )
    if save_dir is None:
        return
    row = {"frame_idx": frame_idx, "stage": stage, "exposure_a": a,
           "exposure_b": b, "gain_exp_a": gain}
    path = os.path.join(save_dir, EXPOSURE_CSV_NAME)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(_EXPOSURE_COLUMNS) + "\n")
        f.write(",".join(_format(row[c]) for c in _EXPOSURE_COLUMNS) + "\n")


PIXEL_USAGE_CSV_NAME = "pixel_usage.csv"
_PIXEL_USAGE_COLUMNS = (
    "frame_idx",
    "frac_saturated_px",
    "frac_saturated_values",
    "frac_removed_saturation",
    "frac_removed_light_weight",
    "frac_light_valid",
    "frac_normal_valid",
    "frac_cos_below_min",
    "opacity_p10",
    "opacity_median",
    "frac_used_monogs_masks",
    "frac_used_tracking",
)


def log_pixel_usage(save_dir, frame_idx, stats):
    """One row per tracked frame (utils/slam_utils.pixel_usage_stats), next
    to albedo_stats.csv; CSV only, no console line (one per frame would
    drown the log). Empty cells = criterion disabled. Written only when an
    extension mask is enabled (stats is None otherwise) and results are
    saved. frac_saturated_px / _values are reported whatever the mode, so the
    two modes can be compared from one run."""
    if stats is None or save_dir is None:
        return
    path = os.path.join(save_dir, PIXEL_USAGE_CSV_NAME)
    write_header = not os.path.exists(path)
    row = {"frame_idx": frame_idx, **stats}
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(_PIXEL_USAGE_COLUMNS) + "\n")
        f.write(",".join(_format(row.get(c)) for c in _PIXEL_USAGE_COLUMNS) + "\n")


LEARNED_CSV_NAME = "light_learned.csv"


def _flatten(state):
    # {"angular.half_angle_deg": 31.2, "pose.R_CL": [0.1, -0.3]} ->
    # [("angular.half_angle_deg", 31.2), ("pose.R_CL[0]", 0.1), ...]
    out = []
    for path in sorted(state):
        v = state[path]
        if isinstance(v, list):
            for i, x in enumerate(v):
                out.append((f"{path}[{i}]", float(x)))
        else:
            out.append((path, float(v)))
    return out


def log_learned_light(save_dir, frame_idx, n_keyframes, state):
    """Learned light parameters after a keyframe's mapping (D60): console
    line + <save_dir>/light_learned.csv. pose.R_CL is the tilt of the light
    axis in degrees (about camera x, y)."""
    flat = _flatten(state)
    Log(
        f"learned light kf {frame_idx} ({n_keyframes} keyframes): "
        + ", ".join(f"{name}={v:.4f}" for name, v in flat),
        tag="Light",
    )
    if save_dir is None:
        return
    path = os.path.join(save_dir, LEARNED_CSV_NAME)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(["frame_idx", "n_keyframes"] + [n for n, _ in flat]) + "\n")
        f.write(",".join([str(frame_idx), str(n_keyframes)] + [_format(v) for _, v in flat]) + "\n")


def summarize_csv_logs(save_dir):
    """End-of-run summary of the per-frame / per-keyframe CSVs of this module,
    as text lines printed into the log (so the numbers survive when the
    result folder is lost, e.g. a Kaggle session). Missing files are skipped.
    Pure reading; never used by tracking or mapping."""
    import csv

    lines = []

    def rows(name):
        path = os.path.join(save_dir, name)
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def quantiles(values):
        v = sorted(values)
        pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # noqa: E731
        return f"p5 {pick(0.05):.4f}  median {pick(0.5):.4f}  p95 {pick(0.95):.4f}"

    usage = rows(PIXEL_USAGE_CSV_NAME)
    if usage:
        lines.append(f"pixel usage over {len(usage)} tracked frames:")
        for col in _PIXEL_USAGE_COLUMNS[1:]:
            vals = [float(r[col]) for r in usage if r.get(col)]
            if vals:
                lines.append(f"  {col:26s} {quantiles(vals)}")
    expo = [r for r in rows(EXPOSURE_CSV_NAME) if r["stage"] == "map"]
    if expo:
        gains = [float(r["gain_exp_a"]) for r in expo]
        lines.append(f"exposure gain exp(a) over {len(expo)} keyframes: {quantiles(gains)}")
    learned = rows(LEARNED_CSV_NAME)
    if learned:
        cols = [c for c in learned[0] if c not in ("frame_idx", "n_keyframes")]
        lines.append(f"learned light parameters over {len(learned)} keyframes (first -> middle -> last):")
        mid = learned[len(learned) // 2]
        for c in cols:
            lines.append(
                f"  {c:28s} {float(learned[0][c]):.4f} -> {float(mid[c]):.4f} -> {float(learned[-1][c]):.4f}"
            )
    albedo = [r for r in rows(CSV_NAME) if r["stage"] == "map" and r.get("mean")]
    if albedo:
        last = albedo[-1]
        lines.append(
            f"albedo, map after the last keyframe {last['frame_idx']}: mean {float(last['mean']):.3f}, "
            f"max {float(last['max']):.2f}, fraction > 1 {float(last['frac_any_gt1']):.1%}"
        )
    return lines


def _format(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
