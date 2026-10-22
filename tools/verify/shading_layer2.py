"""Layer 2 of the 4.2 verification (docs/DECISIONS.md D15, D16): as layer 1,
but the shader gets RASTERIZED maps, as in SLAM, instead of ground-truth
passes.

    python tools/verify/shading_layer2.py --config configs/light/verify_shading.yaml \
        --params-file <scene>/light_params.json --bundle <export dir>/<room> --frames 0 500

Gaussians are created at the ground-truth pose from the ground-truth depth
and albedo, through the SAME code SLAM uses (extend_from_pcd_seq with an
albedo_map, one Gaussian per pixel), their opacity set to
Verify.layer2.gaussian_opacity; then render() with the configured shader
rasterizes albedo, depth and opacity and shades them. Variants per frame:
  nocos    - cosine: none
  depth_fd - cosine: lambert, normals from the rasterized depth / opacity
compared with the rendered lit PNG over the layer-1 pixel selection AND
rasterized opacity >= Verify.layer2.opacity_thr. A layer-1 match with a
layer-2 mismatch points at the G-buffer handling (opacity weighting, depth
division, back-projection, depth_fd on rasterized depth). cosine: none is
expected to be off on surfaces not facing the lamp (n.l < 1): that gap is
the model, not the code. Needs CUDA.
"""
import argparse
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from munch import munchify  # noqa: E402

import shading_layer1 as l1  # noqa: E402
from albedo_roundtrip import make_camera  # noqa: E402
from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402
from light_models import build_shader  # noqa: E402
from light_models.params import load_params_file  # noqa: E402
from utils.config_utils import load_config, require_key  # noqa: E402


def build_map(config, frame_dir, uid, device):
    rgb8, alb8, z, n_gt = l1.load_frame(frame_dir)
    c2w = np.load(os.path.join(frame_dir, "c2w.npy"))
    gb = l1.gbuffer_from_gt(alb8, z, n_gt)
    color = torch.from_numpy(alb8 / 255.0).permute(2, 0, 1).float().to(device)
    cam = make_camera(config, c2w, color, uid, device)
    v2 = require_key(require_key(config, "Verify", ""), "layer2", "Verify")
    g_op = require_key(v2, "gaussian_opacity", "Verify.layer2")

    cfg = copy.deepcopy(config)
    cfg["Dataset"]["pcd_downsample_init"] = 1  # one Gaussian per valid pixel
    gaussians = GaussianModel(0, config=cfg)
    gaussians.init_lr(6.0)  # as slam.py; learning rates are irrelevant here
    gaussians.training_setup(munchify(cfg["opt_params"]))
    gaussians.extend_from_pcd_seq(
        cam, kf_id=uid, init=True, depthmap=z.astype(np.float32),
        albedo_map=gb["albedo"].float().to(device),
    )
    with torch.no_grad():
        gaussians._opacity.fill_(float(torch.logit(torch.tensor(g_op))))
        # Enlarged so the map covers every pixel (see tracking_probe.py, D41).
        gaussians._scaling.add_(
            math.log(require_key(v2, "gaussian_scale_mult", "Verify.layer2"))
        )
    return rgb8, z, n_gt, gb, cam, gaussians


def check_frame(config, params, frame_dir, uid, device):
    rgb8, z, n_gt, gb, cam, gaussians = build_map(config, frame_dir, uid, device)
    thr = require_key(config["Verify"]["layer2"], "opacity_thr", "Verify.layer2")
    pipe = munchify(config["pipeline_params"])
    bg = torch.tensor(config["pipeline_params"]["background"], dtype=torch.float32, device=device)

    # Layer-1 selection (ground-truth geometry) and cone regions.
    lcam = l1.camera_from_config(config)
    cos_th, cos_a, core_half, pts = l1.light_geometry(config, params, lcam, z)
    gt_cos = build_shader(config, params)(gb, lcam)["light_cos_nl"][0].numpy()
    base = l1.selection(config, rgb8, gb["albedo"], z, n_gt, gt_cos, pts)
    margin = config["Verify"]["layer1"]["core_margin_deg"]
    regions = {
        "core": cos_th >= np.cos(np.radians(core_half - margin)),
        "rim": (cos_th > cos_a) & (cos_th < np.cos(np.radians(core_half))),
        "outside": cos_th <= cos_a,
    }
    variants = {}
    stencils = require_key(config["Verify"]["layer2"], "depth_fd_stencils", "Verify.layer2")
    cosines = [("nocos", {"type": "none"})] + [
        (f"depth_fd_k{k}", {"type": "lambert", "warmup_keyframes": 0,
                            "normal_source": {"type": "depth_fd", "stencil_px": k}})
        for k in stencils
    ]
    for name, cosine in cosines:
        cfg = copy.deepcopy(config)
        cfg["Light"]["cosine"] = cosine
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg, shader=build_shader(cfg, params))
        pred = pkg["radiance_linear"].permute(1, 2, 0).double().cpu().numpy()
        op = pkg["opacity"][0].cpu().numpy()
        mask = base & (op >= thr)
        variants[name] = {r: l1.summarize(pred, rgb8, mask & m) for r, m in regions.items()}
        if "light_normal_valid" in pkg:
            # What the loss sees with pixel_weight on: pixels whose normal is
            # invalid get cos = 0 (ambient only) and are excluded there (D41).
            nv = pkg["light_normal_valid"][0].cpu().numpy()
            variants[name + "_normal_valid_only"] = {
                r: l1.summarize(pred, rgb8, mask & m & nv) for r, m in regions.items()
            }
            n_cmp = int(mask.sum())
            variants[name + "_normal_valid_only"]["normal_valid_frac"] = (
                float((mask & nv).sum() / n_cmp) if n_cmp else None
            )
        variants[name]["opacity_ge_thr_frac"] = float((op >= thr).mean())
        variants[name]["opacity_median_selected"] = float(np.median(op[mask])) if mask.any() else None
        # Same prediction as if the map covered the pixel fully (albedo is
        # opacity-weighted, D5). The Gaussians built here from ground truth
        # sit on the data's pixel grid, i.e. half a pixel off the rasterizer's
        # (D29), so they never cover a rasterized pixel completely; a gap
        # that disappears here is that construction, not the shader.
        full = pred / np.where(op > 0, op, 1.0)[..., None]
        variants[name + "_full_coverage"] = {
            r: l1.summarize(full, rgb8, mask & m) for r, m in regions.items()
        }
    return variants


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--params-file", required=True)
    ap.add_argument("--bundle", required=True, help="<export dir>/<room>")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("shading_layer2.py needs CUDA (rasterizer)")
    config = load_config(args.config)
    params = load_params_file(args.params_file)
    results = {}
    for i in args.frames:
        res = check_frame(config, params, os.path.join(args.bundle, "frame%06d" % i), i, "cuda")
        results[i] = res
        print(f"frame {i}")
        for variant, regions in res.items():
            if "opacity_ge_thr_frac" in regions:
                print(f"  {variant}: rasterized opacity >= thr on "
                      f"{regions['opacity_ge_thr_frac']:.1%} of the image, median opacity "
                      f"of the compared pixels {regions['opacity_median_selected']:.3f}")
            elif regions.get("normal_valid_frac") is not None:
                print(f"  {variant}: valid normal on {regions['normal_valid_frac']:.1%} "
                      "of the compared pixels")
            else:
                print(f"  {variant}:")
            for region in ("core", "rim", "outside"):
                s = regions[region]
                if s["n_px"] == 0:
                    print(f"    {region:7s} no pixels")
                    continue
                print(
                    f"    {region:7s} n={s['n_px']:6d}  ratio obs/pred median "
                    f"{s['ratio_median']:.4f} [p5 {s['ratio_p5']:.4f}, p95 {s['ratio_p95']:.4f}]"
                    f"  |err| 8-bit median {s['abs_err8_median']:.1f} p95 {s['abs_err8_p95']:.1f}"
                )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
