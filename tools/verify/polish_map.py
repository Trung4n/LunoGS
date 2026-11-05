"""Offline albedo polishing of a recorded map: how much of the albedo error is
under-optimisation, how much pose drift (docs/DECISIONS.md D105).

Loads record/maps/final.npz (geometry fixed), optimises ONLY the albedo
(f_dc) with the loss of BackEnd.color_refinement (get_loss_refinement: L1 +
D-SSIM in LightTracking.loss_color_space, mapping weight, Adam with
opt_params.feature_lr, lambda_dssim from the config) on a set of training
frames, at a chosen pose source:

  --poses est     the run's final estimates (frames_final.csv);
  --poses refit   frames tracked again against the final map first
                  (MapRenderer.refit_pose, tracking loss, D88): removes the
                  pose error RELATIVE to the map, e.g. of a drifted revisit.
                  --refit_set used: training AND evaluation frames (the eval
                  frames are then tracked on their own image, test-time pose
                  optimisation: not MonoGS's protocol);
                  --refit_set train: training frames only, eval frames keep
                  the run's estimates (MonoGS's protocol; what a final
                  keyframe-pose + albedo refinement inside SLAM could give).
No reference-pose mode: the map was built along the drifted trajectory, so
rendering it at reference poses would be misaligned by the drift itself.

  --frames keyframes   the run's keyframes (as the SLAM refinement);
  --frames all         every `--train_stride`-th of the frames that are NOT
                       evaluation frames of eval_map (those stay held out).

Writes <out>/record/{meta.json, frames_final.csv, maps/final.npz} with the
polished map and the poses used, so that

    python tools/verify/eval_map.py run --record <out> --config <config> ...

evaluates it exactly like a recording. Reads only the SLAM inputs (no ground
truth); tools/verify/ only. Needs CUDA (WSL MonoGS env).

    python tools/verify/polish_map.py --run <run dir> --dataset <lit dataset>
        --poses est --frames all --train_stride 2 --iters 26000 --seed 0 --out <dir>
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_map as em  # noqa: E402


def write_frames_final(path, poses, kfs):
    kfs = set(kfs)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "is_keyframe"] + [f"est_{r}{c}" for r in range(3) for c in range(4)])
        for i in sorted(poses):
            w.writerow([i, int(i in kfs)] + [f"{v:.9g}" for v in poses[i][:3].ravel()])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder holding record/ and config_resolved.yml")
    ap.add_argument("--dataset", required=True, help="MonoGS dataset folder of the lit sequence")
    ap.add_argument("--poses", required=True, choices=["est", "refit"])
    ap.add_argument("--frames", required=True, choices=["keyframes", "all"])
    ap.add_argument("--train_stride", type=int, default=1, help="with --frames all: every n-th non-eval frame")
    ap.add_argument("--iters", type=int, required=True, help="optimisation steps (one frame per step)")
    ap.add_argument("--interval", type=int, default=5, help="eval frames held out (as eval_map --interval)")
    ap.add_argument("--refit_set", choices=["used", "train"], default=None, help="required with --poses refit")
    ap.add_argument("--refit_steps", type=int, default=None, help="default: Training.tracking_itr_num")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if (args.poses == "refit") != (args.refit_set is not None):
        ap.error("--refit_set is required with --poses refit and only allowed with it")

    import torch

    from light_models import build_shader
    from utils.config_utils import load_config
    from utils.slam_utils import get_loss_refinement

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rec = os.path.join(args.run, "record")
    with open(os.path.join(rec, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    config = load_config(os.path.join(args.run, "config_resolved.yml"))
    config["Dataset"]["dataset_path"] = args.dataset
    config["Training"]["monocular"] = config["Dataset"]["sensor_type"] == "monocular"
    depth_scale = config["Dataset"]["Calibration"]["depth_scale"]
    lr = config["opt_params"]["feature_lr"]
    lambda_dssim = config["opt_params"]["lambda_dssim"]
    shader = build_shader(config)
    if shader is None:
        raise SystemExit("polish_map needs a light-model run (albedo map)")

    est, kfs = em.read_final(rec)
    n = len(est)
    eval_set = set(em.eval_frames(n, kfs, args.interval))
    if args.frames == "keyframes":
        train = list(kfs)
    else:
        train = [i for i in range(n) if i not in eval_set][:: args.train_stride]
    used = sorted(set(train) | eval_set)
    poses = dict(est)  # every frame: eval_map derives its eval frames from the count

    with np.load(os.path.join(rec, "maps", "final.npz")) as z:
        snap = {k: z[k] for k in z.files}
    r = em.MapRenderer(snap, shader, meta["background"], meta["intrinsics"])

    images, depths = {}, {}

    def load(i):
        if i not in images:
            images[i] = np.asarray(Image.open(os.path.join(args.dataset, "results", f"frame{i:06d}.jpg")).convert("RGB"))
            depths[i] = np.asarray(Image.open(os.path.join(args.dataset, "results", f"depth{i:06d}.png")))
        return images[i], depths[i].astype(np.float64) / depth_scale

    t0 = time.time()
    if args.poses == "refit":
        steps = args.refit_steps or config["Training"]["tracking_itr_num"]
        refit = used if args.refit_set == "used" else sorted(train)
        for k, i in enumerate(refit):
            img, d = load(i)
            poses[i], _ = r.refit_pose(poses[i], d, img.astype(np.float64) / 255.0, config, steps)
            if k % 200 == 0:
                print(f"refit {k}/{len(refit)} frames, {time.time() - t0:.0f} s", flush=True)

    m = r.model
    m._features_dc = torch.nn.Parameter(m._features_dc.detach().clone().requires_grad_(True))
    for name in ("_xyz", "_scaling", "_rotation", "_opacity"):
        getattr(m, name).requires_grad_(False)
    opt = torch.optim.Adam([{"params": [m._features_dc], "lr": lr}], eps=1e-15)

    def camera(i):
        img, d = load(i)
        cam = r._camera(poses[i], d)
        cam.original_image = torch.tensor(np.ascontiguousarray(img.transpose(2, 0, 1)), dtype=torch.float32,
                                          device="cuda") / 255.0
        return cam

    print(f"{len(train)} training frames ({args.frames}), {len(eval_set)} held-out eval frames, poses {args.poses}, "
          f"{args.iters} steps, albedo lr {lr}", flush=True)
    t1 = time.time()
    for it in range(1, args.iters + 1):
        cam = camera(random.choice(train))
        pkg = r.render(cam, m, r.pipe, r.bg, shader=shader)
        loss = get_loss_refinement(config, pkg, cam, lambda_dssim)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it % 2000 == 0:
            print(f"step {it}/{args.iters}, loss {loss.item():.5f}, {it / (time.time() - t1):.1f} it/s", flush=True)

    out_rec = os.path.join(args.out, "record")
    os.makedirs(os.path.join(out_rec, "maps"), exist_ok=True)
    shutil.copy(os.path.join(rec, "meta.json"), out_rec)
    with torch.no_grad():
        snap["color"] = m.get_albedo.detach().cpu().numpy().astype(snap["color"].dtype)
    np.savez_compressed(os.path.join(out_rec, "maps", "final.npz"), **snap)
    write_frames_final(os.path.join(out_rec, "frames_final.csv"), poses, kfs)
    with open(os.path.join(args.out, "polish.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "train_frames": len(train), "albedo_lr": lr, "lambda_dssim": lambda_dssim,
                   "seconds": time.time() - t0}, f, indent=2)
    print(f"done in {time.time() - t0:.0f} s -> {out_rec}", flush=True)


if __name__ == "__main__":
    main()
