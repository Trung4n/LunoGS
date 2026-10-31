"""Why is a recorded map worse than another? (docs/DECISIONS.md D81)

Joins, per evaluation frame of each run:
  - map quality from tools/verify/eval_map.py (<map_eval>/per_frame.csv);
  - the pose error of that frame: final estimate (record/frames_final.csv)
    after one SE(3) alignment of all frames to the reference, translation (m)
    and rotation (deg), plus the relative translation error over
    `--rpe_delta` frames (local consistency);
  - the fraction of saturated pixels of the observed image (any channel >=
    `--sat_8bit`, on the dataset frame the SLAM read);
and the number of Gaussians of every map snapshot (record/maps/kf*.npz).

Writes <map_eval>/diagnosis.json (rank correlations of the map metrics with
each factor, metrics per tercile of the factor) and
<map_eval>/diagnosis_per_frame.csv per run, and one figure of the factors
over time for all runs (--figure).

The reference poses are read for evaluation only (CLAUDE.md section 3).

    python tools/analysis/diagnose_map.py --sat_8bit 250 --rpe_delta 10 --segment 400 \\
        --run "MonoGS, even"=record/office0_even_rgbd,results/map_eval/office0/even,<dataset>/even \\
        --run nocos=record/office0_lit_nocos_rgbd,results/map_eval/office0/nocos,<dataset>/lit \\
        --figure results/map_eval/office0/diagnosis.png
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.trajectory_analysis import umeyama  # noqa: E402

# Categorical slots 1..8 in fixed order (dataviz reference palette, light).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
METRICS = ("lit_psnr", "lit_psnr_in_cone", "lit_psnr_out_cone", "albedo_psnr")
FACTORS = ("ate_t_m", "ate_r_deg", "rpe_t_m", "sat_fraction")


def read_poses(record_dir):
    """frame -> (est c2w 4x4, gt c2w 4x4), and the keyframe set."""
    out, kfs = {}, set()
    with open(os.path.join(record_dir, "frames_final.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mats = []
            for prefix in ("est", "gt"):
                m = np.eye(4)
                m[:3] = np.array([float(row[f"{prefix}_{r}{c}"]) for r in range(3) for c in range(4)]).reshape(3, 4)
                mats.append(m)
            i = int(row["frame_idx"])
            out[i] = tuple(mats)
            if row["is_keyframe"] == "1":
                kfs.add(i)
    return out, kfs


def pose_errors(poses, rpe_delta):
    """{frame: (ate_t, ate_r_deg, rpe_t or nan)} after one SE(3) alignment."""
    ids = sorted(poses)
    est = np.stack([poses[i][0] for i in ids])
    gt = np.stack([poses[i][1] for i in ids])
    s, r, t = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale=False)
    al = est.copy()
    al[:, :3, :3] = r @ est[:, :3, :3]
    al[:, :3, 3] = (s * (r @ est[:, :3, 3].T)).T + t
    ate_t = np.linalg.norm(al[:, :3, 3] - gt[:, :3, 3], axis=1)
    rel = np.einsum("nji,njk->nik", gt[:, :3, :3], al[:, :3, :3])
    ate_r = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out = {}
    pos = {i: n for n, i in enumerate(ids)}
    for n, i in enumerate(ids):
        j = pos.get(i + rpe_delta)
        rpe = float("nan")
        if j is not None:
            d_est = np.linalg.inv(al[n]) @ al[j]
            d_gt = np.linalg.inv(gt[n]) @ gt[j]
            rpe = float(np.linalg.norm((np.linalg.inv(d_gt) @ d_est)[:3, 3]))
        out[i] = (float(ate_t[n]), float(ate_r[n]), rpe)
    return out


def live_psnr(record_dir, frame):
    """PSNR of the image rendered right after tracking (the map AT THAT TIME)
    against the observed one, both as stored by the recorder (reduced width);
    nan when the frame has no stored images."""
    from PIL import Image

    base = os.path.join(record_dir, "images", f"{frame:06d}")
    if not os.path.isfile(base + "_render.png"):
        return float("nan")
    obs = np.asarray(Image.open(base + "_obs.png").convert("RGB"), np.float64) / 255.0
    ren = np.asarray(Image.open(base + "_render.png").convert("RGB"), np.float64) / 255.0
    mse = float(((obs - ren) ** 2).mean())
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def segment_means(rows, key, step):
    """{"<lo>-<hi>": mean of key over frames in [lo, hi)} per `step` frames."""
    last = max(r["frame"] for r in rows)
    out = {}
    for lo in range(0, last + 1, step):
        v = [r[key] for r in rows if lo <= r["frame"] < lo + step and np.isfinite(r[key])]
        if v:
            out[f"{lo}-{lo + step}"] = float(np.mean(v))
    return out


def saturated_fraction(dataset, frame, sat_8bit):
    from PIL import Image

    a = np.asarray(Image.open(os.path.join(dataset, "results", f"frame{frame:06d}.jpg")).convert("RGB"))
    return float((a >= sat_8bit).any(-1).mean())


def gaussian_counts(record_dir):
    """[(frame_idx, n_keyframes, n_gaussians)] of every map snapshot (the
    keyframe ones and the final map)."""
    out = []
    paths = glob.glob(os.path.join(record_dir, "maps", "kf*.npz")) + [os.path.join(record_dir, "maps", "final.npz")]
    for p in sorted(p for p in paths if os.path.isfile(p)):
        with np.load(p) as z:
            out.append((int(z["frame_idx"]), int(z["n_keyframes"]), int(len(z["xyz"]))))
    return sorted(out)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])


def terciles(rows, factor, metric):
    v = np.array([r[factor] for r in rows], float)
    m = np.array([r[metric] for r in rows], float)
    ok = np.isfinite(v) & np.isfinite(m)
    v, m = v[ok], m[ok]
    if len(v) < 3:
        return []
    edges = np.quantile(v, [0, 1 / 3, 2 / 3, 1])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v >= lo) & (v <= hi)
        out.append({"factor_range": [float(lo), float(hi)], "n": int(sel.sum()), "metric_mean": float(m[sel].mean())})
    return out


def diagnose(label, record_dir, map_eval_dir, dataset, sat_8bit, rpe_delta, segment_step):
    rec = record_dir if os.path.isfile(os.path.join(record_dir, "meta.json")) else os.path.join(record_dir, "record")
    poses, kfs = read_poses(rec)
    errs = pose_errors(poses, rpe_delta)
    with open(os.path.join(map_eval_dir, "per_frame.csv"), newline="", encoding="utf-8") as f:
        evals = list(csv.DictReader(f))
    rows = []
    for e in evals:
        i = int(e["frame"])
        ate_t, ate_r, rpe = errs[i]
        row = {"frame": i, **{m: float(e[m]) for m in METRICS},
               "ate_t_m": ate_t, "ate_r_deg": ate_r, "rpe_t_m": rpe,
               "sat_fraction": saturated_fraction(dataset, i, sat_8bit),
               "live_psnr_stored_px": live_psnr(rec, i)}
        rows.append(row)
    counts = gaussian_counts(rec)
    summary = {
        "label": label, "record": record_dir, "map_eval": map_eval_dir, "dataset": dataset,
        "sat_8bit": sat_8bit, "rpe_delta_frames": rpe_delta, "frames": len(rows),
        "ate_t_rmse_all_frames_m": float(np.sqrt(np.mean([errs[i][0] ** 2 for i in errs]))),
        "ate_t_rmse_keyframes_m": float(np.sqrt(np.mean([errs[i][0] ** 2 for i in kfs]))) if kfs else None,
        "sat_fraction_median": float(np.median([r["sat_fraction"] for r in rows])),
        # forgetting: the map right after tracking vs the final map, same
        # frames (live at the recorder's reduced width, hence a little higher)
        "live_psnr_stored_px_mean": float(np.nanmean([r["live_psnr_stored_px"] for r in rows])),
        "final_lit_psnr_mean": float(np.mean([r["lit_psnr"] for r in rows])),
        "segments_frames": segment_step,
        "live_psnr_by_segment": segment_means(rows, "live_psnr_stored_px", segment_step),
        "final_lit_psnr_by_segment": segment_means(rows, "lit_psnr", segment_step),
        "final_in_cone_by_segment": segment_means(rows, "lit_psnr_in_cone", segment_step),
        "gaussians_final": counts[-1][2] if counts else None,
        "gaussians_max": max(c[2] for c in counts) if counts else None,
        "spearman": {m: {fct: spearman([r[fct] for r in rows], [r[m] for r in rows]) for fct in FACTORS}
                     for m in METRICS},
        "terciles": {fct: terciles(rows, fct, "lit_psnr_in_cone") for fct in FACTORS},
        "gaussian_counts": [{"frame": f, "n_keyframes": k, "n": n} for f, k, n in counts],
    }
    with open(os.path.join(map_eval_dir, "diagnosis_per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(map_eval_dir, "diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary, rows, counts


def local_path(p):
    """A path written in WSL (/mnt/d/...) as seen from Windows, and back."""
    if os.name == "nt" and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:{p[6:]}"
    if os.name != "nt" and len(p) > 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}".replace("\\", "/")
    return p


def read_ceiling(spec):
    """label=csv:column -> (label, frames, values) of model_ceiling.py's
    per-frame csv."""
    label, rest = spec.split("=", 1)
    path, column = rest.rsplit(":", 1)
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if np.isfinite(float(r[column]))]
    return label, [int(r["frame"]) for r in rows], [float(r[column]) for r in rows]


def _end_labels(ax, ends, ink, min_gap_pt=9.0):
    """Direct labels right of the line ends, pushed apart vertically so they
    never overlap (needs the axes limits final: call after plotting)."""
    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform
    gap = min_gap_pt * ax.figure.dpi / 72.0
    x_text = ax.get_xlim()[1]
    placed = []
    for yd, label in sorted((to_disp((x, y))[1], label) for label, x, y in ends):
        if placed and yd < placed[-1][0] + gap:
            yd = placed[-1][0] + gap
        placed.append((yd, label))
    xd = to_disp((x_text, 0))[0] + 4
    for yd, label in placed:
        ax.text(*to_data((xd, yd)), label, fontsize=8, color=ink, va="center", ha="left", clip_on=False)


def figure(path, runs, ceilings=()):
    """Four stacked panels sharing the frame axis, one quantity (one y-axis)
    each; runs in fixed categorical order, model ceilings dashed gray; legend
    plus end labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    refs = ("#4d4c47", "#9a9890")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.titlesize": 9.5,
                         "axes.titleweight": "bold", "axes.titlecolor": ink, "axes.titlelocation": "left"})
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 9.6), sharex=True)
    for ax in axes:
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01)
    fig.canvas.draw()

    def panel(ax, title, series, log=False):
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ends = []
        for label, x, y, style in series:
            ax.plot(x, y, label=label, **style)
            ends.append((label, x[-1], y[-1]))
        fig.canvas.draw()
        _end_labels(ax, ends, ink)

    psnr = [(label, [r["frame"] for r in rows], [r["lit_psnr_in_cone"] for r in rows],
             {"color": SERIES[k], "lw": 1.4}) for k, (label, rows, counts, _) in enumerate(runs)]
    psnr += [(label, x, y, {"color": refs[k % 2], "lw": 1.2, "ls": (0, (4, 2))})
             for k, (label, x, y) in enumerate(ceilings)]
    panel(axes[0], "lit PSNR inside the light cone (dB)", psnr)
    panel(axes[1], "Camera position error after SE(3) alignment (cm, log scale)",
          [(label, [r["frame"] for r in rows], [100 * r["ate_t_m"] for r in rows], {"color": SERIES[k], "lw": 1.4})
           for k, (label, rows, counts, _) in enumerate(runs)], log=True)
    # one line per input sequence: runs on the same dataset see the same images
    seen, sat = set(), []
    for k, (label, rows, counts, dataset) in enumerate(runs):
        if dataset in seen:
            continue
        seen.add(dataset)
        sat.append((f"input: {os.path.basename(os.path.normpath(dataset))}", [r["frame"] for r in rows],
                    [100 * r["sat_fraction"] for r in rows], {"color": refs[len(sat) % 2], "lw": 1.3}))
    panel(axes[2], "Saturated pixels in the observed image (%)", sat)
    panel(axes[3], "Gaussians in the map (thousands)",
          [(label, [c[0] for c in counts], [c[2] / 1000 for c in counts], {"color": SERIES[k], "lw": 1.4})
           for k, (label, rows, counts, _) in enumerate(runs)])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False,
               bbox_to_anchor=(0.47, 0.995))
    axes[-1].set_xlabel("frame")
    fig.subplots_adjust(right=0.79, hspace=0.42, top=0.90)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", default=[],
                    help="label=record_dir,map_eval_dir,dataset_dir (repeat; figure order = slot order)")
    ap.add_argument("--metrics", nargs="*", default=[],
                    help="metrics.json of eval_map.py (label, record and dataset read from it), after the --run ones")
    ap.add_argument("--ceiling", action="append", default=[],
                    help="label=<model_ceiling per-frame csv>:<column>, drawn dashed in the PSNR panel")
    ap.add_argument("--sat_8bit", type=int, required=True, help="saturated if any channel >= this")
    ap.add_argument("--rpe_delta", type=int, required=True, help="frames between the poses of the relative error")
    ap.add_argument("--segment", type=int, required=True, help="frames per segment of the time summaries")
    ap.add_argument("--figure", default=None)
    args = ap.parse_args(argv)
    sources = []
    for spec in args.run:
        label, rest = spec.split("=", 1)
        sources.append((label, *rest.split(",")))
    for pattern in args.metrics:
        for p in sorted(glob.glob(pattern)):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            if "record" not in m:
                raise SystemExit(f"{p}: no record path (evaluated before D81); give it with --run")
            sources.append((m["label"], local_path(m["record"]), os.path.dirname(p), local_path(m["dataset"])))
    if not sources:
        raise SystemExit("nothing to diagnose: give --run or --metrics")
    runs = []
    for label, record_dir, map_eval_dir, dataset in sources:
        summary, rows, counts = diagnose(label, record_dir, map_eval_dir, dataset, args.sat_8bit, args.rpe_delta,
                                         args.segment)
        runs.append((label, rows, counts, dataset))
        sp = summary["spearman"]["lit_psnr_in_cone"]
        print(f"{label}: ATE all frames {100 * summary['ate_t_rmse_all_frames_m']:.2f} cm, saturated median "
              f"{100 * summary['sat_fraction_median']:.1f} %, Gaussians {summary['gaussians_final']}; "
              f"rank corr. of in-cone PSNR with " + ", ".join(f"{k} {v:+.2f}" for k, v in sp.items()))
        print(f"  PSNR right after tracking {summary['live_psnr_stored_px_mean']:.2f} -> final map "
              f"{summary['final_lit_psnr_mean']:.2f}; by segment: " + ", ".join(
                  f"{k} {summary['live_psnr_by_segment'].get(k, float('nan')):.1f}->{v:.1f}"
                  for k, v in summary["final_lit_psnr_by_segment"].items()))
    if args.figure:
        figure(args.figure, runs, [read_ceiling(c) for c in args.ceiling])
        print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
