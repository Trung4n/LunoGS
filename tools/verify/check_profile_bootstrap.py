"""How well does the first-frame estimate of the emission profile work
(light_models/profile_bootstrap.py, docs/DECISIONS.md D112)? Treats many frames
of a sequence as if each were the first one and compares the fitted cone with
the profile the lamp was rendered with.

The estimate reads only the image and the sensor depth; the *_config keys of
light_params.json are read here only to score it (tools/verify/, never imported
by the SLAM path, CLAUDE.md section 3). CPU is enough.

    python tools/verify/check_profile_bootstrap.py --config configs/light/rgbd_lambert_selfcal_bootstrap.yaml
        --dataset <dataset of the lit sequence> --step 50 --out <result.json>
"""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Scored on this grid of angles; a display grid, not a parameter of the model.
THETA_GRID_DEG = np.arange(0.0, 50.5, 0.5)


def blender_profile(theta_deg, half_angle_deg, blend):
    cos_a = math.cos(math.radians(half_angle_deg))
    x = np.clip((np.cos(np.radians(theta_deg)) - cos_a) / ((1.0 - cos_a) * blend), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smoothstep_profile(theta_deg, theta_in, theta_out):
    x = np.clip((theta_out - theta_deg) / (theta_out - theta_in), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def profile_errors(estimate, truth, theta=THETA_GRID_DEG):
    """(max |error|, mean |error|) of a profile sampled on the grid."""
    e = np.abs(estimate - truth)
    return float(e.max()), float(e.mean())


def percentiles(values):
    return [float(np.percentile(values, q)) for q in (0, 5, 50, 95, 100)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--step", type=int, required=True, help="every n-th frame is tried as a first frame")
    ap.add_argument("--guess_deg", type=float, nargs=2, required=True,
                    help="theta_in theta_out of the hand guess, scored as a reference")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch

    from light_models.params import load_params_file
    from light_models import read_profile_bootstrap
    from light_models.profile_bootstrap import check_shader_supports_bootstrap, estimate_cone
    from light_models.shaders import make_shader
    from utils.config_utils import load_config

    config = load_config(args.config)
    params = load_params_file(os.path.join(args.dataset, config["Light"]["params_file"]))
    light = config["Light"]
    shader = make_shader(light, lambda: params)
    check_shader_supports_bootstrap(shader)
    p = read_profile_bootstrap(light)
    if p is None:
        raise SystemExit(f"{args.config} has no Light.profile_bootstrap")
    cal = config["Dataset"]["Calibration"]
    half, blend = params["cone"]["half_angle_deg_config"], params["cone"]["blend_config"]
    truth = blender_profile(THETA_GRID_DEG, half, blend)
    flat = profile_errors(np.ones_like(truth), truth)
    guess = profile_errors(smoothstep_profile(THETA_GRID_DEG, *args.guess_deg), truth)

    results = "results"
    frames = sorted(f for f in os.listdir(os.path.join(args.dataset, results)) if f.startswith("frame"))
    rows, failures = [], []
    for name in frames[:: args.step]:
        i = int(name[len("frame"):].split(".")[0])
        rgb = np.asarray(Image.open(os.path.join(args.dataset, results, name)).convert("RGB"))
        depth = np.asarray(Image.open(os.path.join(args.dataset, results, f"depth{i:06d}.png")),
                           np.float64) / cal["depth_scale"]
        # A new object per frame: the sensor-normal source caches per camera.
        camera = SimpleNamespace(fx=cal["fx"], fy=cal["fy"], cx=cal["cx"], cy=cal["cy"], depth=depth)
        observed = torch.tensor(rgb.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        try:
            fit = estimate_cone(observed, depth, camera, shader, p)
        except ValueError as e:
            failures.append({"frame": i, "error": str(e)})
            continue
        worst, mean = profile_errors(smoothstep_profile(THETA_GRID_DEG, fit.theta_in, fit.theta_out), truth)
        rows.append({"frame": i, "theta_in": fit.theta_in, "theta_out": fit.theta_out, "fit_error": fit.error,
                     "contrast": fit.level_core / fit.level_out, "profile_err_max": worst, "profile_err_mean": mean})

    if not rows:
        raise SystemExit(f"no frame gave a fit: {failures[:3]}")
    summary = {
        "frames_tried": len(rows) + len(failures), "failures": len(failures),
        "truth": {"half_angle_deg": half, "blend": blend},
        "percentiles_0_5_50_95_100": {k: percentiles([r[k] for r in rows])
                                      for k in ("theta_in", "theta_out", "contrast", "profile_err_max",
                                                "profile_err_mean")},
        "frame0": rows[0] if rows[0]["frame"] == 0 else None,
        "reference_err_max_mean": {"flat": flat, f"guess_{args.guess_deg[0]:g}_{args.guess_deg[1]:g}": guess},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "failures": failures}, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
