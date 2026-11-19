"""Accuracy of the normals the shading uses, against a reference
(docs/DECISIONS.md D149).

For every evaluation frame of a recorded run (slam.py --record), at its final
estimated pose:
  map     the run's normal source on the final map: for shortest_axis, the
          blended shortest axis of the flat Gaussians (rendered from the map);
          for the sensor-depth sources, their normals of the run's dataset
          frame (what the SLAM used);
  frame   the bilateral sensor-depth normals of the run's dataset frame (the
          per-frame baseline, D146), when --frame_source is given;
  ref     the reference: 1-pixel edge-aware differences of the CLEAN depth
          (--reference, a dataset folder with noise-free depth*.png, as the
          comparison of D146), or later the ground-truth normal pass.
Reported: median angle and the fraction of pixels off by more than 30 deg,
over the pixels where both have a normal, and the fraction of reference
pixels where the map has none.

GROUND-TRUTH CONSUMER (clean depth as a reference): tools/verify/ only, never
imported by the SLAM path (CLAUDE.md section 3). Needs CUDA and the built
rasterizer (WSL MonoGS env).

    python tools/verify/map_normals.py --record <run>/record \\
        --config configs/light/rgbd_lambert_calib_ambgrid_cell0p5_parallel_fused_gaussnormal.yaml \\
        --dataset <.../office0/lit_depthnoise_heavy> --reference <.../office0/lit> --out normals.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "tools", "verify"), os.path.join(REPO, "tools", "viz")):
    if p not in sys.path:
        sys.path.insert(0, p)


def angle_stats(n_a, ok_a, n_b, ok_b, off_deg):
    """(median angle deg, fraction > off_deg, n pixels) over ok_a & ok_b;
    normals (3,H,W) unit, numpy."""
    both = ok_a & ok_b
    if not both.any():
        return float("nan"), float("nan"), 0
    cos = np.clip((n_a * n_b).sum(0)[both], -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return float(np.median(ang)), float((ang > off_deg).mean()), int(both.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, help="the run's dataset folder (its sensor depth)")
    ap.add_argument("--reference", required=True, help="dataset folder with the noise-free depth")
    ap.add_argument("--frame_source", default=None,
                    help="YAML/JSON block of a sensor-depth normal source for the per-frame baseline, "
                         "e.g. '{type: sensor_depth_bilateral_fd_edge, stencil_px: 4, trusted_pixels_only: "
                         "false, bilateral: {radius_px: 7, sigma_space_px: 3.5, sigma_range_rel: 0.06}}'")
    ap.add_argument("--map", default="final", help="map snapshot in record/maps: final, final_refined, kfNNNNN")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--min_opacity", type=float, required=True, help="rendered opacity for a pixel to count")
    ap.add_argument("--off_deg", type=float, required=True, help="angle counted as wrong")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    import yaml
    from PIL import Image

    from eval_map import MapRenderer, eval_frames, read_final
    from light_models import build_shader
    from light_models.gbuffer import pixel_rays, surface_points
    from light_models.normals import EdgeAwareDepthFiniteDifference
    from light_models.registry import build_component
    from utils.config_utils import load_config

    rec = args.record if os.path.isfile(os.path.join(args.record, "meta.json")) else os.path.join(args.record, "record")
    with open(os.path.join(rec, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    config = load_config(args.config)
    config["Dataset"]["dataset_path"] = args.dataset
    depth_scale = config["Dataset"]["Calibration"]["depth_scale"]
    shader = build_shader(config)
    if shader is None or not hasattr(shader, "cosine") or not hasattr(shader.cosine, "normals"):
        raise SystemExit("the config's light model reads no normal")
    shader.set_keyframe_count(10 ** 9)  # past every warm-up: the full model
    normals = shader.cosine.normals
    frame_src = None
    if args.frame_source:
        frame_src = build_component("normals", yaml.safe_load(args.frame_source), "--frame_source", None)
    ref_fd = EdgeAwareDepthFiniteDifference(1)

    with np.load(os.path.join(rec, "maps", args.map + ".npz")) as z:
        snap = {k: z[k] for k in z.files}
    poses, kfs = read_final(rec)
    frames = eval_frames(len(poses), kfs, args.interval)
    if args.max_frames:
        frames = frames[: args.max_frames]
    renderer = MapRenderer(snap, shader, meta["background"], meta["intrinsics"])

    def depth_of(folder, i):
        return np.asarray(Image.open(os.path.join(folder, "results", f"depth{i:06d}.png")),
                          np.float64) / depth_scale

    def np3(t):
        t = t.detach()
        return (t if t.dtype == torch.bool else t.float()).cpu().numpy()

    rows = []
    for i in frames:
        depth = depth_of(args.dataset, i)
        cam = renderer._camera(poses[i], depth)
        with torch.no_grad():
            pkg = renderer.render(cam, renderer.model, renderer.pipe, renderer.bg, shader=shader,
                                  defer_shading=True)
            g = pkg["gbuffer"]
            points, valid = surface_points(g, cam, shader.opacity_thr)
            n_map, ok_map = normals(points, valid, g, cam)
            ok_map = ok_map & (g["opacity"][0] > args.min_opacity)
            z = torch.tensor(depth_of(args.reference, i), dtype=torch.float32, device="cuda")
            h, w = z.shape
            rays = pixel_rays(cam, h, w, "cuda", torch.float32, 0.0)
            n_ref, ok_ref = ref_fd(rays * z[None], z > 0, {}, cam)
            row = {"frame": i}
            row["map_median_deg"], row["map_off_frac"], row["map_pixels"] = angle_stats(
                np3(n_map), np3(ok_map), np3(n_ref), np3(ok_ref), args.off_deg)
            row["map_missing_frac"] = float((np3(ok_ref) & ~np3(ok_map)).sum() / max(np3(ok_ref).sum(), 1))
            if frame_src is not None:
                n_fr, ok_fr = frame_src(points, valid, g, cam)
                row["frame_median_deg"], row["frame_off_frac"], row["frame_pixels"] = angle_stats(
                    np3(n_fr), np3(ok_fr), np3(n_ref), np3(ok_ref), args.off_deg)
                # The same pixels as the map, for a like-for-like comparison.
                row["frame_on_map_median_deg"], row["frame_on_map_off_frac"], _ = angle_stats(
                    np3(n_fr), np3(ok_fr) & np3(ok_map), np3(n_ref), np3(ok_ref), args.off_deg)
        rows.append(row)
        print(json.dumps(row), flush=True)

    def pooled(key):
        vals = [r[key] for r in rows if key in r and not math.isnan(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    keys = [k for k in rows[0] if k.endswith(("_deg", "_frac"))] if rows else []
    summary = {"frames": len(rows), "median_over_frames": {k: pooled(k) for k in keys},
               "config": args.config, "dataset": args.dataset, "reference": args.reference,
               "map": args.map, "off_deg": args.off_deg, "min_opacity": args.min_opacity}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
