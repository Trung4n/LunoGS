"""Map quality of a recorded run (docs/DECISIONS.md D79).

Renders the FINAL map of a recording (slam.py --record, record/maps/final.npz)
at the final estimated pose of every evaluation frame (every `--interval`
frames, keyframes skipped, as MonoGS's eval_rendering) and compares:

  lit     rendered image (through the run's light model, or the plain
          rasterized colours for original MonoGS) vs the observed image the
          SLAM read (dataset results/frameNNNNNN.jpg); PSNR / SSIM / LPIPS
          over all pixels, and PSNR inside / outside the light cone;
  albedo  the map's light-free colour (rendered albedo for a light model; the
          rasterized colour for MonoGS, i.e. radiance with the light baked
          in) vs the ground-truth albedo pass (base colour);
  even    the same map colour vs the image rendered under even light only
          ("relighting": what the map looks like without the lamp).

The map colour is defined up to a scale (K, the gain of the ambient light,
the n . l absorbed by nocos), so for albedo and even one gain per colour
channel is fitted by least squares in linear space over all evaluation
frames (reported) before the comparison. Pixels count when the rendered
opacity is > `--min_opacity` (coverage reported).

The cone mask (theta <= estimated half-angle, from the dataset depth and the
estimated light pose of light_params.json) is the same for every run.

GROUND-TRUTH CONSUMER (albedo / even passes): tools/verify/ only, never
imported by the SLAM path (CLAUDE.md section 3). Needs CUDA and the built
rasterizer (run it in the WSL MonoGS env).

    python tools/verify/eval_map.py run --record record/office0_lit_nocos_rgbd \\
        --config configs/light/rgbd_nocos.yaml \\
        --dataset /mnt/d/CodeForYou/Replica/04_datasets/monogs/office0/lit \\
        --renders /mnt/d/CodeForYou/Replica/03_renders/office0 --out results/map_eval/nocos
    python tools/verify/eval_map.py table results/map_eval/*/metrics.json
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "tools", "viz")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------- pure helpers


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def psnr(a, b, mask=None):
    """PSNR (dB) of images in [0, 1], (H,W,3); mask (H,W) bool or None."""
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        d = d[mask]
    mse = float(d.mean())
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


class GainFit:
    """Per-channel least-squares gain g minimising sum (target - g pred)^2,
    accumulated over frames (linear values)."""

    def __init__(self):
        self.tp = np.zeros(3)
        self.pp = np.zeros(3)

    def add(self, pred, target, mask):
        p, t = pred[mask], target[mask]
        self.tp += (t * p).sum(axis=0)
        self.pp += (p * p).sum(axis=0)

    def gain(self):
        return np.where(self.pp > 0, self.tp / np.maximum(self.pp, 1e-12), 1.0)


def cone_mask(depth, intr, t_cl, axis, half_angle_deg):
    """(H,W) bool: pixels whose surface point (from the metric z-depth) lies
    within the light cone. Pixel centres at +0.5 (OpenCV); depth 0 = no point."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u + 0.5 - intr["cx"]) / intr["fx"] * depth
    y = (v + 0.5 - intr["cy"]) / intr["fy"] * depth
    p = np.stack([x, y, depth], axis=-1) - np.asarray(t_cl, np.float64)
    cos = (p @ np.asarray(axis, np.float64)) / np.maximum(np.linalg.norm(p, axis=-1), 1e-9)
    return (depth > 0) & (cos >= math.cos(math.radians(half_angle_deg)))


def eval_frames(n_frames, keyframes, interval):
    """MonoGS eval_rendering's frames: every `interval`, keyframes skipped."""
    kfs = set(keyframes)
    return [i for i in range(0, n_frames - 1, interval) if i not in kfs]


def read_final(record_dir):
    """{frame: (c2w 4x4 final estimate)}, keyframe list."""
    poses, kfs = {}, []
    with open(os.path.join(record_dir, "frames_final.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = np.eye(4)
            m[:3] = np.array([float(row[f"est_{r}{c}"]) for r in range(3) for c in range(4)]).reshape(3, 4)
            i = int(row["frame_idx"])
            poses[i] = m
            if row["is_keyframe"] == "1":
                kfs.append(i)
    return poses, sorted(kfs)


def learned_state(csv_path):
    """Last row of light_learned.csv as a learnable state {path: value|list}."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: empty")
    state = {}
    for k, v in rows[-1].items():
        if k in ("frame_idx", "n_keyframes") or k.endswith(".n_at_bound"):
            continue
        if "[" in k:
            name, i = k[:-1].split("[")
            state.setdefault(name, {})[int(i)] = float(v)
        else:
            state[k] = float(v)
    return {k: ([v[i] for i in sorted(v)] if isinstance(v, dict) else v) for k, v in state.items()}


# ---------------------------------------------------------------- rendering


class MapRenderer:
    """Final map snapshot -> rendered image / albedo / opacity at a pose,
    through the run's shader (or none for original MonoGS)."""

    def __init__(self, snap, shader, background, intr):
        import torch
        from types import SimpleNamespace

        from gaussian_splatting.gaussian_renderer import render
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from gaussian_splatting.utils.sh_utils import RGB2SH
        from utils.camera_utils import Camera

        self.torch, self.render, self.Camera = torch, render, Camera
        self.shader, self.intr = shader, intr
        self.pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        self.bg = torch.tensor(background, dtype=torch.float32, device="cuda")
        f = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda")  # noqa: E731
        m = GaussianModel(0)
        m._xyz = f(snap["xyz"])
        m._scaling = torch.log(f(snap["scaling"]).clamp_min(1e-8))
        m._rotation = f(snap["rotation"])
        op = f(snap["opacity"]).clamp(1e-6, 1 - 1e-6)[:, None]
        m._opacity = torch.log(op / (1 - op))
        # The recorded colour is get_albedo = max(SH2RGB(dc), 0): back to dc.
        m._features_dc = RGB2SH(f(snap["color"]))[:, None, :]
        m._features_rest = torch.zeros(m._xyz.shape[0], 0, 3, device="cuda")
        self.model = m

    def __call__(self, c2w, sensor_depth):
        torch, it = self.torch, self.intr
        w, h = int(it["width"]), int(it["height"])
        fovx = 2 * math.atan(w / (2 * it["fx"]))
        fovy = 2 * math.atan(h / (2 * it["fy"]))
        cam = self.Camera.init_from_gui(0, torch.eye(4, device="cuda"), fovx, fovy,
                                        it["fx"], it["fy"], it["cx"], it["cy"], h, w)
        w2c = np.linalg.inv(c2w)
        cam.update_RT(torch.tensor(w2c[:3, :3], dtype=torch.float32, device="cuda"),
                      torch.tensor(w2c[:3, 3], dtype=torch.float32, device="cuda"))
        cam.depth = sensor_depth  # read only by sensor-depth normal sources (lambert)
        with torch.no_grad():
            pkg = self.render(cam, self.model, self.pipe, self.bg, shader=self.shader)
        hwc = lambda t: t.detach().float().permute(1, 2, 0).cpu().numpy()  # noqa: E731
        image = np.clip(hwc(pkg["render"]), 0, 1)          # sRGB, as SLAM compares it
        opacity = hwc(pkg["opacity"])[..., 0]
        if "albedo" in pkg:
            colour = hwc(pkg["albedo"])                    # linear albedo (opacity-weighted)
        else:
            colour = srgb_to_linear(np.clip(hwc(pkg["render"]), 0, 1))  # baked radiance
        return image, colour, opacity


# ---------------------------------------------------------------- run


def run(args):
    import torch
    from PIL import Image
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from gaussian_splatting.utils.loss_utils import ssim as ssim_fn
    from light_models import build_shader
    from light_models.params import load_params_file
    from utils.config_utils import load_config

    rec = args.record if os.path.isfile(os.path.join(args.record, "meta.json")) else os.path.join(args.record, "record")
    with open(os.path.join(rec, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    intr = meta["intrinsics"]
    config = load_config(args.config)
    config["Dataset"]["dataset_path"] = args.dataset
    depth_scale = config["Dataset"]["Calibration"]["depth_scale"]
    shader = build_shader(config)
    if shader is not None and args.light_csv:
        shader.load_learnable_state(learned_state(args.light_csv))
    if (shader is None) != (meta["map_color"] == "srgb"):
        raise SystemExit(f"--config does not match the recording (map colour {meta['map_color']})")
    params = load_params_file(args.light_params or os.path.join(args.dataset, "light_params.json"))
    t_cl = params["t_CL_m"]["estimated_opencv"]
    axis = np.asarray(params["R_CL"]["estimated"], np.float64) @ np.array([0.0, 0.0, 1.0])
    half_angle = params["cone"]["half_angle_deg_estimated"]

    with np.load(os.path.join(rec, "maps", "final.npz")) as z:
        snap = {k: z[k] for k in z.files}
    poses, kfs = read_final(rec)
    frames = eval_frames(len(poses), kfs, args.interval)
    if args.max_frames:
        frames = frames[: args.max_frames]
    renderer = MapRenderer(snap, shader, meta["background"], intr)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    tt = lambda a: torch.tensor(np.ascontiguousarray(a.transpose(2, 0, 1)), dtype=torch.float32,  # noqa: E731
                                device="cuda")[None]

    def load_rgb(path):
        return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0

    def frame_data(i):
        obs = load_rgb(os.path.join(args.dataset, "results", f"frame{i:06d}.jpg"))
        depth = np.asarray(Image.open(os.path.join(args.dataset, "results", f"depth{i:06d}.png")),
                           np.float64) / depth_scale
        image, colour, opacity = renderer(poses[i], depth)
        alb_gt = srgb_to_linear(load_rgb(os.path.join(args.renders, "passes", "albedo_png", f"albedo{i:06d}.png")))
        even_gt = srgb_to_linear(load_rgb(os.path.join(args.renders, "even", "rgb_png", f"frame{i:06d}.png")))
        return obs, depth, image, colour, opacity > args.min_opacity, alb_gt, even_gt

    # pass 1: render, lit metrics, gain fits
    fit_alb, fit_even = GainFit(), GainFit()
    rows = []
    for n, i in enumerate(frames):
        obs, depth, image, colour, covered, alb_gt, even_gt = frame_data(i)
        cone = cone_mask(depth, intr, t_cl, axis, half_angle)
        fit_alb.add(colour, alb_gt, covered)
        fit_even.add(colour, even_gt, covered)
        with torch.no_grad():
            s = float(ssim_fn(tt(image), tt(obs)))
            lp = float(lpips(tt(image), tt(obs)))
        rows.append({"frame": i, "lit_psnr": psnr(image, obs), "lit_ssim": s, "lit_lpips": lp,
                     "lit_psnr_in_cone": psnr(image, obs, cone & covered),
                     "lit_psnr_out_cone": psnr(image, obs, ~cone & (depth > 0) & covered),
                     "coverage": float(covered.mean()), "cone_fraction": float(cone.mean())})
        if n % 50 == 0:
            print(f"  pass 1: frame {i} ({n + 1}/{len(frames)})", flush=True)

    # pass 2 (renders again: keeping every frame would take GB): albedo / even
    # after the fitted gains, compared in sRGB like the images
    g_alb, g_even = fit_alb.gain(), fit_even.gain()
    figure_frames = {min(frames, key=lambda f: abs(f - x)) for x in args.figure_frames}
    for n, row in enumerate(rows):
        i = row["frame"]
        obs, depth, image, colour, covered, alb_gt, even_gt = frame_data(i)
        for name, gain, gt in (("albedo", g_alb, alb_gt), ("even", g_even, even_gt)):
            pred = linear_to_srgb(np.clip(colour * gain, 0, 1))
            ref = linear_to_srgb(gt)
            pred_m = np.where(covered[..., None], pred, 0.0)
            ref_m = np.where(covered[..., None], ref, 0.0)
            with torch.no_grad():
                row[f"{name}_ssim"] = float(ssim_fn(tt(pred_m), tt(ref_m)))
                row[f"{name}_lpips"] = float(lpips(tt(pred_m), tt(ref_m)))
            row[f"{name}_psnr"] = psnr(pred, ref, covered)
        if i in figure_frames:
            save_figure(args.out, i, obs, image, linear_to_srgb(np.clip(colour * g_alb, 0, 1)),
                        linear_to_srgb(alb_gt), linear_to_srgb(np.clip(colour * g_even, 0, 1)),
                        linear_to_srgb(even_gt), os.path.basename(os.path.normpath(args.dataset)))
        if n % 50 == 0:
            print(f"  pass 2: frame {i} ({n + 1}/{len(rows)})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = lambda k: float(np.nanmean([r[k] for r in rows]))  # noqa: E731
    metrics = {
        "label": args.label or meta["run"], "run": meta["run"], "map_color": meta["map_color"],
        "frames": len(rows), "interval": args.interval, "gaussians": int(len(snap["xyz"])),
        "gain_albedo": g_alb.tolist(), "gain_even": g_even.tolist(),
        **{k: mean(k) for k in rows[0] if k != "frame"},
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def save_figure(out, frame, obs, image, alb, alb_gt, even, even_gt, obs_name):
    from PIL import Image, ImageDraw

    tiles = [(f"observed ({obs_name})", obs), ("rendered", image), ("albedo, map", alb), ("albedo, GT", alb_gt),
             ("even, map", even), ("even, GT", even_gt)]
    h, w = obs.shape[:2]
    canvas = Image.new("RGB", (3 * w, 2 * (h + 18)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for k, (name, img) in enumerate(tiles):
        x, y = (k % 3) * w, (k // 3) * (h + 18)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)), (x, y + 18))
        draw.text((x + 4, y + 3), name, fill=(220, 220, 220))
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    canvas.save(os.path.join(out, "figures", f"frame{frame:06d}.png"))


# ---------------------------------------------------------------- table


COLUMNS = [("lit_psnr", "lit PSNR"), ("lit_psnr_in_cone", "in cone"), ("lit_psnr_out_cone", "out of cone"),
           ("lit_ssim", "lit SSIM"), ("lit_lpips", "lit LPIPS"), ("albedo_psnr", "albedo PSNR"),
           ("albedo_ssim", "albedo SSIM"), ("even_psnr", "even PSNR"), ("even_ssim", "even SSIM"),
           ("coverage", "coverage")]


def table(paths):
    """Markdown table of several metrics.json."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    head = "| run | " + " | ".join(name for _, name in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 1)]
    for m in rows:
        lines.append(f"| {m['label']} | " + " | ".join(
            f"{m[k]:.3f}" if "ssim" in k or "lpips" in k or k == "coverage" else f"{m[k]:.2f}" for k, _ in COLUMNS) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate one recording")
    r.add_argument("--record", required=True, help="record folder (or the run folder holding record/)")
    r.add_argument("--config", required=True, help="the run's config (light model, depth scale)")
    r.add_argument("--dataset", required=True, help="MonoGS dataset folder of the lit sequence")
    r.add_argument("--renders", required=True, help="Replica 03_renders/<scene> (albedo and even passes)")
    r.add_argument("--light_csv", default=None, help="light_learned.csv of a run that learned light parameters")
    r.add_argument("--light_params", default=None,
                   help="light_params.json for the cone mask (default: <dataset>/light_params.json; give the lit "
                   "sequence's for a run on the even sequence)")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default=None)
    r.add_argument("--interval", type=int, default=5, help="as MonoGS eval_rendering")
    r.add_argument("--min_opacity", type=float, default=0.5, help="rendered opacity for a pixel to count")
    r.add_argument("--max_frames", type=int, default=None)
    r.add_argument("--figure_frames", type=int, nargs="*", default=[500, 1000, 1500])
    t = sub.add_parser("table", help="markdown table of metrics.json files")
    t.add_argument("metrics", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        print(table([p for g in args.metrics for p in sorted(glob.glob(g))]))


if __name__ == "__main__":
    main()
