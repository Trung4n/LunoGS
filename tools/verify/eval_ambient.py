"""How well a run recovered the ambient light of a scene with static lamps
(docs/DECISIONS.md D121, D122). Reads ground-truth passes: verification only,
never imported by the SLAM path (CLAUDE.md section 3).

Truth, from the passes of the data repo at the frames of the pass
ambient_only (lamp off, EXR, linear):
    E_true = I_ambient_only / albedo_pass, per channel, mean over the channels
    with albedo_pass >= --min_albedo_channel, on pixels whose albedo sums to
    more than --min_albedo_sum (the rule of the data repo's README).
Prediction, from the final map rendered at the estimated pose:
    E_pred = c + M / A, per channel, same channel rule on the map's albedo A,
    with M the rasterized A_i (E_i - c) of ambient per_gaussian; E_pred = c
    for a constant ambient. M / A is the opacity-weighted mean of E_i - c,
    weighted by albedo too, on pixels where several Gaussians overlap.

Only K*A and E/K are observable, so E is compared in the units of the
calibrated K (a gauge close to 1 here: K fitted 30.42, theory 30.40), and the
product A*E, which needs no gauge, against I_ambient_only. Outside the cone
only that product is observed, so E there may be wrong and the product right.

    OPENCV_IO_ENABLE_OPENEXR is set here, before importing cv2 (the MonoGS env
    has no OpenEXR package). Needs CUDA.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", ".."), os.path.dirname(__file__)]
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def channel_mean(values, reliable):
    """(H,W,3) values -> (H,W) mean over the reliable channels; pixels without
    any reliable channel are NaN."""
    n = reliable.sum(-1)
    total = np.where(reliable, values, 0.0).sum(-1)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def true_irradiance(ambient, albedo, min_channel):
    reliable = albedo >= min_channel
    ratio = np.divide(ambient, albedo, out=np.zeros_like(ambient), where=reliable)
    return channel_mean(ratio, reliable)


def predicted_irradiance(albedo, residual, c, min_channel):
    """(H,W) c + M / A. residual None = a constant ambient: c wherever the map
    has a reliable albedo. c: scalar or 3 values."""
    reliable = albedo >= min_channel
    c = np.broadcast_to(np.asarray(c, np.float64), (3,))
    if residual is None:
        return channel_mean(np.broadcast_to(c, albedo.shape), reliable)
    excess = np.divide(residual, albedo, out=np.zeros_like(albedo), where=reliable)
    return channel_mean(c + excess, reliable)


def error_stats(pred, true, reference):
    """Over the pixels given (1-D arrays): how close pred is to true, next to
    the same numbers for a reference prediction."""
    if pred.size == 0:
        raise ValueError("no pixel to compare")
    corr = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 else float("nan")
    return {
        "pixels": int(pred.size),
        "true_median": float(np.median(true)),
        "pred_median": float(np.median(pred)),
        "abs_err_median": float(np.median(np.abs(pred - true))),
        "abs_err_reference": float(np.median(np.abs(reference - true))),
        "rel_err_median": float(np.median(np.abs(pred - true) / true)),
        "rel_err_reference": float(np.median(np.abs(reference - true) / true)),
        "pearson": corr,
    }


def by_true_level(pred, true, edges):
    """Median prediction per range of the true value: the bias at each level."""
    rows = []
    for lo, hi in zip([0.0] + list(edges), list(edges) + [float("inf")]):
        m = (true >= lo) & (true < hi)
        if m.any():
            rows.append({"true_from": lo, "true_to": hi, "pixels": int(m.sum()),
                         "true_median": float(np.median(true[m])), "pred_median": float(np.median(pred[m]))})
    return rows


def run(args):
    import cv2
    import torch
    from PIL import Image

    import eval_map
    from light_models import build_shader
    from light_models.params import load_params_file
    from utils.config_utils import load_config

    config = load_config(args.config)
    config["Dataset"]["dataset_path"] = args.dataset
    shader = build_shader(config)
    if args.light_csv:
        shader.load_learnable_state(eval_map.learned_state(args.light_csv))
    rec = args.record if os.path.isfile(os.path.join(args.record, "meta.json")) else os.path.join(args.record, "record")
    with open(os.path.join(rec, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    intr = meta["intrinsics"]
    depth_scale = config["Dataset"]["Calibration"]["depth_scale"]
    params = load_params_file(args.light_params)
    t_cl = params["t_CL_m"]["estimated_opencv"]
    axis = np.asarray(params["R_CL"]["estimated"], np.float64) @ np.array([0.0, 0.0, 1.0])
    half_angle = params["cone"]["half_angle_deg_estimated"]

    with np.load(os.path.join(rec, "maps", args.map + ".npz")) as z:
        snap = {k: z[k] for k in z.files}
    poses, _ = eval_map.read_final(rec)
    renderer = eval_map.MapRenderer(snap, shader, meta["background"], intr)
    c = float(shader.ambient.c.mean())

    def load_rgb(path):
        return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0

    collected = {"all": [], "in_cone": [], "out_cone": []}
    product = []
    frames = [i for i in range(0, len(poses), args.every)
              if os.path.isfile(os.path.join(args.ambient_dir, f"ambient{i:06d}.exr"))]
    if not frames:
        raise SystemExit(f"no frame of {args.ambient_dir} matches the run")
    figure = {}
    for i in frames:
        bgr = cv2.imread(os.path.join(args.ambient_dir, f"ambient{i:06d}.exr"), cv2.IMREAD_UNCHANGED)
        ambient = bgr[..., ::-1].astype(np.float64)
        alb_gt = eval_map.srgb_to_linear(load_rgb(os.path.join(args.renders, "passes", "albedo_png", f"albedo{i:06d}.png")))
        depth = np.asarray(Image.open(os.path.join(args.dataset, "results", f"depth{i:06d}.png")), np.float64) / depth_scale
        cam = renderer._camera(poses[i], depth)
        with torch.no_grad():
            pkg = renderer.render(cam, renderer.model, renderer.pipe, renderer.bg, shader=shader)
        hwc = lambda t: t.detach().float().permute(1, 2, 0).cpu().numpy().astype(np.float64)  # noqa: E731
        albedo = hwc(pkg["albedo"])
        opacity = hwc(pkg["opacity"])[..., 0]
        residual = hwc(pkg["light_ambient_residual"]) if "light_ambient_residual" in pkg else None
        e_true = true_irradiance(ambient, alb_gt, args.min_albedo_channel)
        e_pred = predicted_irradiance(albedo, residual, c, args.min_albedo_channel)
        ok = (alb_gt.sum(-1) > args.min_albedo_sum) & (opacity > args.min_opacity) & (depth > 0)
        ok &= np.isfinite(e_true) & np.isfinite(e_pred) & (e_true > 0)
        cone = eval_map.cone_mask(depth, intr, t_cl, axis, half_angle)
        collected["all"].append((e_pred[ok], e_true[ok]))
        collected["in_cone"].append((e_pred[ok & cone], e_true[ok & cone]))
        collected["out_cone"].append((e_pred[ok & ~cone], e_true[ok & ~cone]))
        # The ambient radiance of the map, no gauge: outside the cone, what the
        # image actually constrains.
        model_ambient = albedo * c + (residual if residual is not None else 0.0)
        product.append((model_ambient[ok].ravel(), ambient[ok].ravel()))
        if i in args.figure_frames:
            figure[i] = (np.where(ok, e_true, np.nan), np.where(ok, e_pred, np.nan))

    out = {"record": rec, "map": args.map, "frames": len(frames), "c": c}
    for name, parts in collected.items():
        pred = np.concatenate([p for p, _ in parts])
        true = np.concatenate([t for _, t in parts])
        out[name] = error_stats(pred, true, np.full_like(true, c))
        out[name]["by_true_level"] = by_true_level(pred, true, args.levels)
    pred = np.concatenate([p for p, _ in product])
    true = np.concatenate([t for _, t in product])
    out["ambient_radiance_abs_err_median"] = float(np.median(np.abs(pred - true)))
    out["ambient_radiance_pearson"] = float(np.corrcoef(pred, true)[0, 1])
    extras = {k[len("extra_"):]: v for k, v in snap.items() if k.startswith("extra_")}
    if "ambient_residual" in extras:
        e = c + np.asarray(extras["ambient_residual"], np.float64)
        out["per_gaussian_E"] = {"gaussians": int(e.shape[0]), "percentiles_5_50_95": np.percentile(e, [5, 50, 95]).tolist(),
                                 "fraction_below_0": float((e < 0).mean())}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ambient.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, len(figure), figsize=(4 * len(figure), 5), squeeze=False)
        for col, (i, (t, p)) in enumerate(sorted(figure.items())):
            for row, (img, name) in enumerate(((t, "E true"), (p, "E of the map"))):
                axes[row][col].imshow(img, vmin=0, vmax=args.figure_vmax, cmap="viridis")
                axes[row][col].set_title(f"{name}, frame {i}", fontsize=9)
                axes[row][col].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ambient.png"), dpi=110)
        plt.close(fig)
    for name in collected:
        s = out[name]
        print(f"{name:9s} true {s['true_median']:.3f} map {s['pred_median']:.3f} | rel err {s['rel_err_median']:.3f} "
              f"(constant c {s['rel_err_reference']:.3f}) | r {s['pearson']:.3f} | {s['pixels']} px")
    print(f"ambient radiance: abs err {out['ambient_radiance_abs_err_median']:.4f}, r {out['ambient_radiance_pearson']:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True, help="record folder (or the run folder holding record/)")
    ap.add_argument("--config", required=True, help="the run's resolved config")
    ap.add_argument("--dataset", required=True, help="MonoGS dataset folder of the variant (depth)")
    ap.add_argument("--renders", required=True, help="Replica 03_renders/<scene> (albedo pass)")
    ap.add_argument("--ambient_dir", required=True, help="the pass ambient_only of the variant")
    ap.add_argument("--light_params", required=True, help="light_params.json for the cone mask")
    ap.add_argument("--light_csv", default=None, help="light_learned.csv of the run")
    ap.add_argument("--map", required=True, choices=["final", "final_refined"])
    ap.add_argument("--every", type=int, required=True, help="use every n-th frame")
    ap.add_argument("--min_albedo_channel", type=float, required=True)
    ap.add_argument("--min_albedo_sum", type=float, required=True)
    ap.add_argument("--min_opacity", type=float, required=True)
    ap.add_argument("--levels", type=float, nargs="+", required=True, help="edges of the E ranges of the bias table")
    ap.add_argument("--figure_frames", type=int, nargs="*", required=True)
    ap.add_argument("--figure_vmax", type=float, required=True)
    ap.add_argument("--out", required=True)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
