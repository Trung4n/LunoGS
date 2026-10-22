"""Layer 1 of the 4.2 verification (docs/DECISIONS.md D15, D16, D18):
ground-truth depth, normals and albedo -> the SAME light_models/ shader SLAM
uses -> predicted image, compared with the rendered lit image (PNG).

    python tools/verify/shading_layer1.py --config configs/light/verify_shading.yaml \
        --params-file <scene>/light_params.json --bundle <export dir>/<room> --frames 0 500

CPU only. The bundle comes from tools/verify/export_gt_npy.py (run in the
dataset repo's environment). Two variants per frame:
  gt_normals - Light.cosine.normal_source as configured (gbuffer = the
               ground-truth normals of the bundle);
  depth_fd   - same config with normal_source: depth_fd on the ground-truth
               depth, i.e. what SLAM computes (from rasterized depth).
Reported per region of the cone (core, rim, outside) over the pixels kept by
Verify.layer1 (copied from check_physics.py):
  ratio = sum_c I_obs / sum_c I_pred   (1 = exact; linear)
  |sRGB8(I_pred) - I_obs8|             (gray levels)
Shadowed pixels are not excluded (no mask exported), hence median and p95.
"""
import argparse
import copy
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from light_models import build_shader  # noqa: E402
from light_models.gbuffer import pixel_rays  # noqa: E402
from light_models.normals import DepthFiniteDifference  # noqa: E402
from light_models.params import load_params_file, resolve_param  # noqa: E402
from utils.color_space import linear2sRGB, sRGB2Linear  # noqa: E402
from utils.config_utils import load_config, require_key  # noqa: E402

DTYPE = torch.float64


def gray8(rgb8):
    # Pillow "L" integer formula, as Replica/scripts/physics_check/common.py
    # gray8 (the dark threshold is defined on it).
    a = rgb8.astype(np.int64)
    return (a[..., 0] * 19595 + a[..., 1] * 38470 + a[..., 2] * 7471 + 0x8000) >> 16


def camera_from_config(config):
    c = config["Dataset"]["Calibration"]
    return type("Cam", (), {"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"]})()


def load_frame(frame_dir):
    rgb8 = np.asarray(Image.open(os.path.join(frame_dir, "rgb_lit.png")).convert("RGB"))
    alb8 = np.asarray(Image.open(os.path.join(frame_dir, "albedo.png")).convert("RGB"))
    z = np.load(os.path.join(frame_dir, "depth_m.npy")).astype(np.float64)
    n = np.load(os.path.join(frame_dir, "normal_cam.npy")).astype(np.float64)
    return rgb8, alb8, z, n


def gbuffer_from_gt(alb8, z, n):
    albedo = sRGB2Linear(torch.from_numpy(alb8 / 255.0).permute(2, 0, 1).to(DTYPE))
    valid = z > 0
    n_valid = valid & (np.abs(n).sum(-1) > 0)
    return {
        "albedo": albedo,
        "depth": torch.from_numpy(z)[None],
        "opacity": torch.from_numpy(valid.astype(np.float64))[None],
        # Ground-truth passes follow the data's pixel convention (D29).
        "pixel_offset": 0.0,
        "normal": torch.from_numpy(n).permute(2, 0, 1),
        "normal_valid": torch.from_numpy(n_valid),
    }


def light_geometry(config, params, cam, z):
    """cos(theta) and the core/rim/outside regions, from the same config
    values as the shader (used only to split the report by region)."""
    light = config["Light"]
    t = torch.tensor(resolve_param(light["pose"]["t_CL"], "t", params).value, dtype=DTYPE)
    r = torch.tensor(resolve_param(light["pose"]["R_CL"], "R", params).value, dtype=DTYPE)
    axis = r[:, 2] / r[:, 2].norm()
    ang = light["angular"]
    half = resolve_param(ang["half_angle_deg"], "a", params).value
    blend = resolve_param(ang["blend"], "b", params).value
    zt = torch.from_numpy(z)
    pts = pixel_rays(cam, z.shape[0], z.shape[1], "cpu", DTYPE, 0.0) * zt[None]
    v = pts - t.view(3, 1, 1)
    cos_th = ((v * axis.view(3, 1, 1)).sum(0) / v.norm(dim=0)).numpy()
    cos_a = math.cos(math.radians(half))
    core_half = math.degrees(math.acos(min(1.0, cos_a + (1.0 - cos_a) * blend)))
    return cos_th, cos_a, core_half, pts


def selection(config, rgb8, albedo_lin, z, n_gt, cos_nl, pts):
    v1 = require_key(require_key(config, "Verify", ""), "layer1", "Verify")
    sat = require_key(v1, "saturation_8bit", "Verify.layer1")
    dark = require_key(v1, "dark_gray_8bit", "Verify.layer1")
    cos_min = require_key(v1, "cos_min", "Verify.layer1")
    alb_min = require_key(v1, "albedo_min_sum_linear", "Verify.layer1")
    planar = require_key(v1, "planar_deg", "Verify.layer1")
    valid = z > 0
    # check_physics.py's planar test uses 1-pixel differences (common.depth_normals).
    n_fd, n_fd_ok = DepthFiniteDifference(stencil_px=1)(pts, torch.from_numpy(valid), None, None)
    cos_ang = (n_fd.permute(1, 2, 0).numpy() * n_gt).sum(-1).clip(-1, 1)
    is_planar = n_fd_ok.numpy() & (np.degrees(np.arccos(cos_ang)) < planar)
    return (
        valid
        & is_planar
        & (rgb8.max(-1) < sat)
        & (gray8(rgb8) > dark)
        & (cos_nl >= cos_min)
        & (albedo_lin.sum(0).numpy() >= alb_min)
    )


def summarize(pred_lin, rgb8, mask):
    obs_lin = sRGB2Linear(torch.from_numpy(rgb8 / 255.0)).numpy()
    ratio = obs_lin.sum(-1)[mask] / np.maximum(pred_lin.sum(-1)[mask], 1e-12)
    pred8 = np.rint(linear2sRGB(torch.from_numpy(pred_lin).clamp(0, 1)).numpy() * 255.0)
    err8 = np.abs(pred8 - rgb8.astype(np.float64))[mask].ravel()
    if not mask.any():
        return {"n_px": 0}
    return {
        "n_px": int(mask.sum()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p5": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "abs_err8_median": float(np.median(err8)),
        "abs_err8_p95": float(np.percentile(err8, 95)),
    }


def check_frame(config, params, frame_dir):
    rgb8, alb8, z, n_gt = load_frame(frame_dir)
    cam = camera_from_config(config)
    gb = gbuffer_from_gt(alb8, z, n_gt)
    cos_th, cos_a, core_half, pts = light_geometry(config, params, cam, z)
    core_margin = require_key(config["Verify"]["layer1"], "core_margin_deg", "Verify.layer1")
    regions = {
        "core": cos_th >= math.cos(math.radians(core_half - core_margin)),
        "rim": (cos_th > cos_a) & (cos_th < math.cos(math.radians(core_half))),
        "outside": cos_th <= cos_a,
    }
    variants = {}
    k = require_key(config["Verify"]["layer1"], "depth_fd_stencil_px", "Verify.layer1")
    for name, normal_source in (
        ("gt_normals", None),
        ("depth_fd", {"type": "depth_fd", "stencil_px": k}),
    ):
        cfg = copy.deepcopy(config)
        if normal_source is not None:
            cfg["Light"]["cosine"]["normal_source"] = normal_source
        out = build_shader(cfg, params)(gb, cam)
        pred = out["radiance_linear"].permute(1, 2, 0).numpy()
        # Selection always uses the ground-truth n.l (as check_physics.py).
        gt_out = build_shader(config, params)(gb, cam)
        mask = selection(config, rgb8, gb["albedo"], z, n_gt,
                         gt_out["light_cos_nl"][0].numpy(), pts)
        variants[name] = {r: summarize(pred, rgb8, mask & m) for r, m in regions.items()}
    return variants


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--params-file", required=True)
    ap.add_argument("--bundle", required=True, help="<export dir>/<room>")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    config = load_config(args.config)
    params = load_params_file(args.params_file)
    results = {}
    for i in args.frames:
        res = check_frame(config, params, os.path.join(args.bundle, "frame%06d" % i))
        results[i] = res
        print(f"frame {i}")
        for variant, regions in res.items():
            for region, s in regions.items():
                if s["n_px"] == 0:
                    print(f"  {variant:10s} {region:7s}   no pixels")
                    continue
                print(
                    f"  {variant:10s} {region:7s} n={s['n_px']:6d}  ratio obs/pred median "
                    f"{s['ratio_median']:.4f} [p5 {s['ratio_p5']:.4f}, p95 {s['ratio_p95']:.4f}]"
                    f"  |err| 8-bit median {s['abs_err8_median']:.1f} p95 {s['abs_err8_p95']:.1f}"
                )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
