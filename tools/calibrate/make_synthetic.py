"""Synthetic calibration clip for fit_light.py (docs/DECISIONS.md D106): a
flat target of uniform albedo rendered with the SLAM light model itself, at
random distances and tilts, then degraded like the real data: clipped to
[0, 1], sRGB-encoded, 8-bit, JPEG; depth quantised to the 16-bit PNG of the
datasets, optionally with Gaussian noise.

It tests whether the fit is well posed (which parameters it recovers and
which ones trade off) BEFORE a clip is rendered in Blender. It cannot test
how well the light model matches the renderer: the clip is made by the same
model that is fitted.

The config holds the TRUE light (a Light block, every value fixed) and a
Synthetic block:
    intrinsics {fx, fy, cx, cy, width, height}, depth_scale,
    n_frames, distance_m [min, max] (target distance along the optical axis),
    max_tilt_deg (target normal vs the optical axis), target_albedo [r, g, b],
    jpeg_quality (null = PNG), depth_noise_m (std of white noise, 0 = none), seed

    python tools/calibrate/make_synthetic.py --config configs/calib/synthetic_office0_lamp.yaml
        --params <light_params.json> --out <clip dir>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fit_light as fl  # noqa: E402
from utils.config_utils import load_config, require_key  # noqa: E402

SYNTHETIC_KEYS = ("intrinsics", "depth_scale", "n_frames", "distance_m", "max_tilt_deg", "target_albedo",
                  "jpeg_quality", "depth_noise_m", "seed")


def read_synthetic(config):
    block = require_key(config, "Synthetic", "")
    extra = set(block) - set(SYNTHETIC_KEYS)
    if extra:
        raise ValueError(f"Synthetic: unexpected keys {sorted(extra)}")
    s = {k: require_key(block, k, "Synthetic") for k in SYNTHETIC_KEYS}
    for k in ("fx", "fy", "cx", "cy", "width", "height"):
        require_key(s["intrinsics"], k, "Synthetic.intrinsics")
    lo, hi = s["distance_m"]
    if not 0 < lo <= hi:
        raise ValueError(f"Synthetic.distance_m must be [min, max] with 0 < min <= max, got {s['distance_m']!r}")
    if not 0 <= s["max_tilt_deg"] < 90:
        raise ValueError(f"Synthetic.max_tilt_deg must be in [0, 90), got {s['max_tilt_deg']!r}")
    if s["jpeg_quality"] is not None and not 1 <= s["jpeg_quality"] <= 100:
        raise ValueError(f"Synthetic.jpeg_quality must be null or in [1, 100], got {s['jpeg_quality']!r}")
    if s["depth_noise_m"] < 0:
        raise ValueError(f"Synthetic.depth_noise_m must be >= 0, got {s['depth_noise_m']!r}")
    return s


def random_plane(rng, distance_m, max_tilt_deg):
    """Plane through (0, 0, D), normal tilted from -z by up to max_tilt_deg in
    a random direction: (n, d) with n . x = d, facing the camera."""
    dist = rng.uniform(*distance_m)
    tilt = math.radians(max_tilt_deg) * math.sqrt(rng.uniform())  # uniform over the disc of tilts
    phi = rng.uniform(0, 2 * math.pi)
    n = np.array([math.sin(tilt) * math.cos(phi), math.sin(tilt) * math.sin(phi), -math.cos(tilt)])
    return n, float(n @ np.array([0.0, 0.0, dist]))


def make(config, params, out, log=print):
    import torch

    from light_models.shaders import make_shader
    from utils.color_space import linear2sRGB

    s = read_synthetic(config)
    light_cfg = require_key(config, "Light", "")
    shader = make_shader(light_cfg, lambda: params)
    if shader.learnable_params():
        raise ValueError("the true light of a synthetic clip must be fixed (no learnable parameter)")
    intr = s["intrinsics"]
    h, w = intr["height"], intr["width"]
    rays = fl.pixel_grid(intr, h, w)
    rng = np.random.default_rng(s["seed"])
    cam = fl.camera_of(intr)
    albedo = torch.tensor(s["target_albedo"], dtype=torch.float32).view(3, 1, 1).expand(3, h, w)
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    ids, planes = [], []
    for i in range(s["n_frames"]):
        n, d = random_plane(rng, s["distance_m"], s["max_tilt_deg"])
        z = fl.plane_depth(n, d, rays)
        valid = z > 0
        gbuffer = {
            "albedo": albedo,
            "depth": torch.tensor(z, dtype=torch.float32)[None],
            "opacity": torch.ones(1, h, w),
            "pixel_offset": 0.0,
            "normal": torch.tensor(np.broadcast_to(n, (h, w, 3)).transpose(2, 0, 1).copy(), dtype=torch.float32),
            "normal_valid": torch.tensor(valid),
        }
        with torch.no_grad():
            rad = shader(gbuffer, cam)["radiance_linear"].clamp(0.0, 1.0)  # the sensor clips
            srgb = linear2sRGB(rad).clamp(0.0, 1.0)
        rgb = np.round(srgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        fid = f"{i:06d}"
        if s["jpeg_quality"] is None:
            Image.fromarray(rgb).save(os.path.join(out, "rgb", fid + ".png"))
        else:
            Image.fromarray(rgb).save(os.path.join(out, "rgb", fid + ".jpg"), quality=s["jpeg_quality"])
        zn = z + (rng.normal(0.0, s["depth_noise_m"], z.shape) if s["depth_noise_m"] > 0 else 0.0)
        code = np.where(valid, np.round(np.clip(zn, 0, None) * s["depth_scale"]), 0)
        Image.fromarray(np.clip(code, 0, 65535).astype(np.uint16)).save(os.path.join(out, "depth", fid + ".png"))
        ids.append(fid)
        planes.append({"normal": n.tolist(), "d": d})
    with open(os.path.join(out, "clip.json"), "w", encoding="utf-8") as f:
        json.dump({"intrinsics": intr, "depth_scale": s["depth_scale"], "frames": ids}, f, indent=2)
    truth = {"light": fl.light_values(shader, light_cfg, params), "planes": planes, "synthetic": s}
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
    log(f"{len(ids)} frames -> {out}")
    return truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from light_models.params import load_params_file

    make(load_config(args.config), load_params_file(args.params), args.out)


if __name__ == "__main__":
    main()
