"""Side-by-side figure of recorded maps for the report (docs/DECISIONS.md D81).

For every chosen frame, two rows:
  lit     the observed image, then each run's map rendered at that run's
          final estimated pose of the frame (through its light model);
  albedo  the ground-truth albedo pass, then each run's map colour after the
          per-channel gain fitted by eval_map.py (gain_albedo of metrics.json),
          so every run is shown on the albedo's scale.
The runs are evaluated recordings: eval_map.py's metrics.json gives the
record, dataset, config (and learned light) of each run. The observed image
is the first run's dataset frame.

GROUND-TRUTH CONSUMER (albedo pass): tools/verify/ only (CLAUDE.md section 3).
Needs CUDA (WSL MonoGS env).

    python tools/verify/figure_maps.py --renders /mnt/d/CodeForYou/Replica/03_renders/office0 \\
        --run "MonoGS (lit)=results/map_eval/office0/lit/metrics.json" \\
        --run "Ours (nocos)=results/map_eval/office0/nocos/metrics.json" \\
        --frames 300 1000 1600 --out results/map_eval/office0/figure_maps.png
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE, os.path.join(REPO, "tools", "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)


def nearest_frame(frame, available):
    return min(available, key=lambda f: abs(f - frame))


def grid(tiles, header, tile_h, label_h=22, gap=4, bg=(255, 255, 255)):
    """tiles: rows of (H,W,3) float images in [0,1] (None = empty); header:
    rows of labels. Returns a PIL image with labels above every tile."""
    from PIL import Image, ImageDraw

    n_cols = max(len(r) for r in tiles)
    h0, w0 = next(t for r in tiles for t in r if t is not None).shape[:2]
    tile_w = round(w0 * tile_h / h0)
    W = n_cols * tile_w + (n_cols - 1) * gap
    H = len(tiles) * (tile_h + label_h) + (len(tiles) - 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    for r, (row, labels) in enumerate(zip(tiles, header)):
        y = r * (tile_h + label_h + gap)
        for c, (img, label) in enumerate(zip(row, labels)):
            x = c * (tile_w + gap)
            draw.text((x + 2, y + 5), label, fill=(31, 31, 30))
            if img is not None:
                tile = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
                canvas.paste(tile.resize((tile_w, tile_h), Image.BILINEAR), (x, y + label_h))
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="label=<eval_map metrics.json> (repeat, column order)")
    ap.add_argument("--renders", required=True, help="Replica 03_renders/<scene> (albedo pass)")
    ap.add_argument("--frames", type=int, nargs="+", required=True,
                    help="frames to show (the nearest evaluated non-keyframe of the first run is used)")
    ap.add_argument("--tile_height", type=int, required=True, help="pixels per tile")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch  # noqa: F401  (CUDA context before the renderers)
    from PIL import Image

    import eval_map as em
    from diagnose_map import local_path
    from light_models import build_shader
    from utils.config_utils import load_config

    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if "record" not in m:
            raise SystemExit(f"{path}: evaluated before D81 (no record path); evaluate it again with eval_map.py")
        rec, dataset = local_path(m["record"]), local_path(m["dataset"])
        with open(os.path.join(rec, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        config = load_config(m["config"])
        config["Dataset"]["dataset_path"] = dataset
        shader = build_shader(config)
        if shader is not None and m.get("light_csv"):
            shader.load_learnable_state(em.learned_state(local_path(m["light_csv"])))
        with np.load(os.path.join(rec, "maps", "final.npz")) as z:
            snap = {k: z[k] for k in z.files}
        poses, kfs = em.read_final(rec)
        runs.append({"label": label, "dataset": dataset, "poses": poses, "kfs": set(kfs),
                     "renderer": em.MapRenderer(snap, shader, meta["background"], meta["intrinsics"]),
                     "gain": np.asarray(m["gain_albedo"]), "depth_scale": config["Dataset"]["Calibration"]["depth_scale"]})

    first = runs[0]
    usable = sorted(f for f in first["poses"] if f not in first["kfs"])
    load = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float64) / 255.0  # noqa: E731
    tiles, header = [], []
    for want in args.frames:
        i = nearest_frame(want, usable)
        obs = load(os.path.join(first["dataset"], "results", f"frame{i:06d}.jpg"))
        alb_gt = load(os.path.join(args.renders, "passes", "albedo_png", f"albedo{i:06d}.png"))
        lit_row, alb_row = [obs], [alb_gt]
        for r in runs:
            depth = np.asarray(Image.open(os.path.join(r["dataset"], "results", f"depth{i:06d}.png")),
                               np.float64) / r["depth_scale"]
            image, colour, _ = r["renderer"](r["poses"][i], depth)
            lit_row.append(image)
            alb_row.append(em.linear_to_srgb(np.clip(colour * r["gain"], 0, 1)))
        tiles += [lit_row, alb_row]
        header += [[f"frame {i}: observed"] + [r["label"] for r in runs],
                   ["albedo, ground truth"] + [f"{r['label']}: map colour" for r in runs]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    grid(tiles, header, args.tile_height).save(args.out)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
