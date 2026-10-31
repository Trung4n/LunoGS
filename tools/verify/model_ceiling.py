"""Multi-view ceiling of a light model (docs/DECISIONS.md D81).

How well can a map whose colour is FIXED per surface point reproduce the
observed images, if the shading is computed by a given light model? The
light moves with the camera, so a model that misses a view-dependent term
(nocos has no n . l) cannot be absorbed by any per-point colour: the point
needs a different colour in every view.

For each model (a config of configs/light/, built with the same
light_models/ shader SLAM uses) and every frame of a ground-truth bundle
(tools/verify/export_gt_npy.py):

    pred(p) = albedo_gt(p) * shading_model(p)          (linear)

Surface points are grouped into world voxels of `--voxel_m` (GT depth and
pose), and each voxel gets ONE scalar k_v, the least-squares fit of
obs = k_v * pred over all its unsaturated observations, i.e. the best
view-independent correction a map could learn. The ground-truth albedo keeps
the texture inside a voxel, so only the shading is averaged. Reported per
model, as PSNR of k_v * pred against the observed lit image (sRGB, pooled
over frames), inside the cone / outside / saturated, also per frame
(<out>_per_frame.csv, voxel variant):
  voxel   per-voxel k_v (voxels seen in >= --min_views frames)
  global  one k for the whole sequence
plus the spread of obs / pred within a voxel (view dependence).

GROUND-TRUTH CONSUMER: tools/verify/ only, never imported by the SLAM path
(CLAUDE.md section 3). CPU only.

    python tools/verify/model_ceiling.py --bundle <export dir>/office0 \\
        --params-file <dataset>/light_params.json \\
        --model nocos=configs/light/rgbd_nocos.yaml \\
        --model lambert_gt=configs/light/verify_ceiling_lambert.yaml \\
        --voxel_m 0.02 --min_views 3 --sat_8bit 250 --out results/map_eval/office0/ceiling.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

REGIONS = ("in_cone", "out_cone", "saturated", "all")


def voxel_keys(points, voxel_m):
    """(N,3) world points -> (N,) int64 voxel keys."""
    ijk = np.floor(points / voxel_m).astype(np.int64)
    off, m = 1 << 20, 1 << 21
    return ((ijk[:, 0] + off) * m + (ijk[:, 1] + off)) * m + (ijk[:, 2] + off)


def fit_per_group(keys, op, pp):
    """Least-squares scalar per key from sums of obs*pred and pred^2.
    Returns (unique keys, k per key, number of contributions per key)."""
    uniq, inv = np.unique(keys, return_inverse=True)
    num = np.bincount(inv, weights=op)
    den = np.bincount(inv, weights=pp)
    k = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    return uniq, k, inv


def pooled_psnr(sq_err_sum, n):
    if n == 0:
        return float("nan")
    mse = sq_err_sum / n
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def frame_model_data(config, params, frame_dir, sat_8bit):
    """One frame: valid-pixel world points, observed (linear, sRGB), the
    model prediction (linear), region masks."""
    import torch

    from light_models import build_shader
    from shading_layer1 import camera_from_config, gbuffer_from_gt, light_geometry, load_frame
    from utils.color_space import sRGB2Linear

    rgb8, alb8, z, n_gt = load_frame(frame_dir)
    c2w = np.load(os.path.join(frame_dir, "c2w.npy"))
    cam = camera_from_config(config)
    gb = gbuffer_from_gt(alb8, z, n_gt)
    pred = build_shader(config, params)(gb, cam)["radiance_linear"].permute(1, 2, 0).numpy()
    cos_th, cos_a, _, pts = light_geometry(config, params, cam, z)
    valid = (z > 0) & (np.abs(n_gt).sum(-1) > 0)
    cam_pts = pts.permute(1, 2, 0).numpy()[valid]
    world = cam_pts @ c2w[:3, :3].T + c2w[:3, 3]
    obs_srgb = rgb8[valid] / 255.0
    obs_lin = sRGB2Linear(torch.from_numpy(obs_srgb)).numpy()
    sat = (rgb8[valid] >= sat_8bit).any(-1)
    cone = cos_th[valid] >= cos_a
    return world, obs_srgb, obs_lin, pred[valid], sat, cone


def run(args):
    from light_models.params import load_params_file
    from utils.color_space import linear2sRGB
    from utils.config_utils import load_config
    import torch

    params = load_params_file(args.params_file)
    models = dict(m.split("=", 1) for m in args.model)
    frames = sorted(d for d in os.listdir(args.bundle) if d.startswith("frame"))
    if args.max_frames:
        frames = frames[: args.max_frames]
    result = {"bundle": args.bundle, "frames": len(frames), "voxel_m": args.voxel_m,
              "min_views": args.min_views, "sat_8bit": args.sat_8bit, "models": {}}
    per_frame = {}
    for name, cfg_path in models.items():
        config = load_config(cfg_path)
        data = lambda fr: frame_model_data(config, params, os.path.join(args.bundle, fr), args.sat_8bit)  # noqa: E731
        # pass 1: per-voxel sums over unsaturated observations, reduced per
        # frame (one row per voxel and frame) to keep the memory small
        keys_all, op_all, pp_all, fid_all = [], [], [], []
        for f_i, fr in enumerate(frames):
            world, _, obs_lin, pred, sat, _ = data(fr)
            use = ~sat
            u, inv = np.unique(voxel_keys(world[use], args.voxel_m), return_inverse=True)
            keys_all.append(u)
            op_all.append(np.bincount(inv, weights=(obs_lin[use] * pred[use]).sum(-1)))
            pp_all.append(np.bincount(inv, weights=(pred[use] ** 2).sum(-1)))
            fid_all.append(np.full(len(u), f_i, np.int32))
        keys_cat = np.concatenate(keys_all)
        op_cat, pp_cat = np.concatenate(op_all), np.concatenate(pp_all)
        uniq, k_vox, inv = fit_per_group(keys_cat, op_cat, pp_cat)
        # rows are unique per (voxel, frame): the count per voxel is its views
        views = np.bincount(inv, minlength=len(uniq))
        k_global = float(op_cat.sum() / pp_cat.sum())

        # pass 2 (shades again): errors per region, plus the within-voxel
        # spread of obs/pred
        acc = {v: {r: [0.0, 0] for r in REGIONS} for v in ("voxel", "global")}
        log_ratio_sum = np.zeros(len(uniq))
        log_ratio_sq = np.zeros(len(uniq))
        log_n = np.zeros(len(uniq))
        for fr in frames:
            world, obs_srgb, obs_lin, pred, sat, cone = data(fr)
            keys = voxel_keys(world, args.voxel_m)
            pos = np.clip(np.searchsorted(uniq, keys), 0, len(uniq) - 1)
            ok = (uniq[pos] == keys) & (views[pos] >= args.min_views)
            row = per_frame.setdefault(int(fr[len("frame"):]), {})
            for variant, k in (("voxel", np.where(ok, k_vox[pos], np.nan)), ("global", np.full(len(keys), k_global))):
                p = linear2sRGB(torch.from_numpy(np.clip(k[:, None] * pred, 0, 1))).numpy()
                sq = ((p - obs_srgb) ** 2).mean(-1)
                # same pixels for both variants: voxels seen in >= min_views frames
                has = np.isfinite(sq) & ok
                for region, m in (("in_cone", cone & ~sat), ("out_cone", ~cone & ~sat),
                                  ("saturated", sat), ("all", np.ones_like(sat))):
                    sel = has & m
                    acc[variant][region][0] += float(sq[sel].sum())
                    acc[variant][region][1] += int(sel.sum())
                    if variant == "voxel":
                        row[f"{name}_{region}"] = pooled_psnr(float(sq[sel].sum()), int(sel.sum()))
            r_ok = ok & ~sat & (pred.sum(-1) > 0) & (obs_lin.sum(-1) > 0)
            lr = np.log(obs_lin[r_ok].sum(-1) / pred[r_ok].sum(-1))
            np.add.at(log_ratio_sum, pos[r_ok], lr)
            np.add.at(log_ratio_sq, pos[r_ok], lr * lr)
            np.add.at(log_n, pos[r_ok], 1)
        multi = log_n >= 2
        sd = np.sqrt(np.maximum(log_ratio_sq[multi] / log_n[multi] - (log_ratio_sum[multi] / log_n[multi]) ** 2, 0))
        res = {
            "config": cfg_path,
            "k_global": k_global,
            "voxels": int(len(uniq)),
            "voxels_min_views": int((views >= args.min_views).sum()),
            "views_median": float(np.median(views)),
            "within_voxel_sd_log_ratio": {"median": float(np.median(sd)), "p90": float(np.percentile(sd, 90))},
        }
        for variant in acc:
            for region in REGIONS:
                s, n = acc[variant][region]
                res[f"psnr_{variant}_{region}"] = pooled_psnr(s, n)
                res[f"px_{variant}_{region}"] = n
        result["models"][name] = res
        print(f"{name}: " + "  ".join(f"{v}/{r} {res[f'psnr_{v}_{r}']:.2f}" for v in acc for r in REGIONS)
              + f"  sd(log obs/pred) median {res['within_voxel_sd_log_ratio']['median']:.3f}", flush=True)
    if args.out:
        import csv

        cols = ["frame"] + sorted({c for r in per_frame.values() for c in r})
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(os.path.splitext(args.out)[0] + "_per_frame.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in sorted(per_frame):
                w.writerow([i] + [f"{per_frame[i].get(c, float('nan')):.4f}" for c in cols[1:]])
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True, help="<export dir>/<room> of export_gt_npy.py")
    ap.add_argument("--params-file", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=config.yaml (repeat)")
    ap.add_argument("--voxel_m", type=float, required=True)
    ap.add_argument("--min_views", type=int, required=True)
    ap.add_argument("--sat_8bit", type=int, required=True, help="saturated if any channel >= this")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
