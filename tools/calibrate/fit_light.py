"""Pre-calibration of the co-located light from a calibration clip
(docs/DECISIONS.md D106), after the camera-light calibration of DarkGS
(neuralight): a flat target of uniform albedo seen from many distances and
angles by the moving camera + lamp rig; the light parameters are fitted to the
photometric error over the target.

Adapted to RGB-D (instead of AprilTags + PnP): the target plane is fitted to
the SENSOR depth of every frame (least squares, iteratively keeping the points
within Calibration.plane_inlier_m of the plane), and the pixels on it are
shaded at the plane's depth with the plane's normal. Only what a real rig gives
is read: the 8-bit sRGB image and the sensor depth of each frame.

The model is the SLAM light model itself: the Light block of the config is
built with light_models (ColocatedShader) and its learnable ParamSpecs are
fitted with LightLearner. Light.schedule is read as STAGES here:
`after_keyframes: s` means "unfrozen from stage s on" (stage s runs
Calibration.stages[s] full-batch steps), so a profile can be fitted first and
the pose of the light afterwards, as in neuralight.

Gauge (D65, D73): only K x target albedo is observable; the target albedo is
FIXED to Calibration.target_albedo (the nominal reflectance of the target) and
K is fitted in those units. The ambient light of the calibration room is fitted
too but belongs to that room: it is exported separately and not meant to be
reused in a scene.

Clip folder layout (make_synthetic.py writes it; a real capture must too):
    clip.json   {"intrinsics": {fx, fy, cx, cy, width, height},
                 "depth_scale": <depth PNG units per metre>, "frames": [<id>, ...]}
    rgb/<id>.jpg | rgb/<id>.png   8-bit sRGB image
    depth/<id>.png                16-bit depth, value / depth_scale = metres, 0 = none
Intrinsics follow the data's convention: integer pixel centres (offset 0).

    python tools/calibrate/fit_light.py --config configs/calib/fit_spot_datasheet.yaml
        --params <json with the starting values> --clip <clip dir> --out <result.json> [--device cuda]
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.config_utils import load_config, require_key  # noqa: E402

LOSS_COLOR_SPACES = ("linear", "srgb")
CALIBRATION_KEYS = ("target_albedo", "plane_inlier_m", "plane_iters", "sat_8bit", "loss_color_space", "stages")


def read_calibration(config):
    """The Calibration block, every key required, nothing else allowed."""
    block = require_key(config, "Calibration", "")
    if not isinstance(block, dict):
        raise TypeError(f"Calibration must be a block, got {block!r}")
    extra = set(block) - set(CALIBRATION_KEYS)
    if extra:
        raise ValueError(f"Calibration: unexpected keys {sorted(extra)}")
    c = {k: require_key(block, k, "Calibration") for k in CALIBRATION_KEYS}
    a = c["target_albedo"]
    if (not isinstance(a, list) or len(a) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in a)):
        raise ValueError(f"Calibration.target_albedo must be 3 numbers > 0, got {a!r}")
    v = c["plane_inlier_m"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"Calibration.plane_inlier_m must be a number > 0, got {v!r}")
    v = c["plane_iters"]
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"Calibration.plane_iters must be an integer >= 1, got {v!r}")
    v = c["sat_8bit"]
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 255:
        raise ValueError(f"Calibration.sat_8bit must be an integer in [1, 255], got {v!r}")
    if c["loss_color_space"] not in LOSS_COLOR_SPACES:
        raise ValueError(f"Calibration.loss_color_space must be one of {LOSS_COLOR_SPACES}, got {c['loss_color_space']!r}")
    s = c["stages"]
    if (not isinstance(s, list) or not s
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in s)):
        raise ValueError(f"Calibration.stages must be a non-empty list of integers >= 1, got {s!r}")
    return SimpleNamespace(**c)


# ---------------------------------------------------------------- geometry

def pixel_grid(intr, height, width):
    """(H,W,3) K^-1 [u, v, 1] with integer pixel centres (data convention)."""
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    return np.stack([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], np.ones_like(u)], -1)


def fit_plane(points, valid, inlier_m, iters):
    """Least-squares plane n . x = d of the valid points (H,W,3), re-fitted
    `iters` times on the points within inlier_m of the previous plane. The
    normal faces the camera (n . x < 0 for the points, d < 0). Returns
    (n, d, inliers (H,W) bool)."""
    mask = valid.copy()
    for _ in range(iters):
        p = points[mask]
        if len(p) < 3:
            raise ValueError("plane fit: fewer than 3 points")
        centre = p.mean(0)
        n = np.linalg.svd(p - centre, full_matrices=False)[2][-1]
        if n @ centre > 0:
            n = -n
        d = float(n @ centre)
        mask = valid & (np.abs(points @ n - d) < inlier_m)
    return n, d, mask


def plane_depth(n, d, rays):
    """(H,W) z-depth of the plane along each ray (0 where the ray misses it)."""
    nr = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(nr < 0, d / nr, 0.0)
    return np.where(z > 0, z, 0.0)


# ---------------------------------------------------------------- data

def load_clip(clip_dir):
    with open(os.path.join(clip_dir, "clip.json"), encoding="utf-8") as f:
        meta = json.load(f)
    intr = require_key(meta, "intrinsics", "clip.json")
    for k in ("fx", "fy", "cx", "cy", "width", "height"):
        require_key(intr, k, "clip.json intrinsics")
    scale = require_key(meta, "depth_scale", "clip.json")
    frames = []
    for fid in require_key(meta, "frames", "clip.json"):
        rgb_path = next((os.path.join(clip_dir, "rgb", f"{fid}{ext}") for ext in (".jpg", ".png")
                         if os.path.isfile(os.path.join(clip_dir, "rgb", f"{fid}{ext}"))), None)
        if rgb_path is None:
            raise FileNotFoundError(f"{clip_dir}/rgb/{fid}.jpg|png")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.asarray(Image.open(os.path.join(clip_dir, "depth", f"{fid}.png")), np.float64) / scale
        if rgb.shape[:2] != (intr["height"], intr["width"]) or depth.shape != rgb.shape[:2]:
            raise ValueError(f"frame {fid}: image size does not match clip.json")
        frames.append(SimpleNamespace(id=fid, rgb=rgb, depth=depth))
    return intr, frames


def prepare(frames, intr, calib, device):
    """Per frame: the fitted plane, the G-buffer of the target pixels at the
    plane depth / normal, the observation and the pixel mask."""
    import torch

    from utils.color_space import sRGB2Linear

    h, w = intr["height"], intr["width"]
    rays = pixel_grid(intr, h, w)
    albedo = torch.tensor(calib.target_albedo, dtype=torch.float32, device=device).view(3, 1, 1).expand(3, h, w)
    out = []
    for fr in frames:
        valid = fr.depth > 0
        n, d, inl = fit_plane(rays * fr.depth[..., None], valid, calib.plane_inlier_m, calib.plane_iters)
        z = plane_depth(n, d, rays)
        sat = (fr.rgb >= calib.sat_8bit).any(-1)
        mask = inl & (z > 0) & ~sat
        t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
        srgb = t(fr.rgb.transpose(2, 0, 1)) / 255.0
        gbuffer = {
            "albedo": albedo,
            "depth": t(z)[None],
            "opacity": t(np.ones((h, w)))[None],
            "pixel_offset": 0.0,  # maps in the data's own pixel convention (gbuffer.py, D29)
            "normal": t(np.broadcast_to(n, (h, w, 3)).transpose(2, 0, 1)),
            "normal_valid": t(mask) > 0.5,
        }
        out.append(SimpleNamespace(
            id=fr.id, gbuffer=gbuffer, mask=t(mask)[None] > 0.5,
            target=srgb if calib.loss_color_space == "srgb" else sRGB2Linear(srgb),
            plane_normal=n.tolist(), plane_d=d, n_pixels=int(mask.sum()),
            inlier_fraction=float(inl[valid].mean()) if valid.any() else 0.0))
    return out


def camera_of(intr):
    return SimpleNamespace(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])


# ---------------------------------------------------------------- fit

def frame_loss(shader, fr, cam, calib):
    """Mean L1 over the frame's target pixels in the loss colour space."""
    from utils.color_space import linear2sRGB

    pred = shader(fr.gbuffer, cam)["radiance_linear"]
    if calib.loss_color_space == "srgb":
        pred = linear2sRGB(pred)
    m = fr.mask.expand_as(pred)
    return (pred - fr.target).abs()[m].mean()


def fit(shader, prepared, intr, calib, log=print):
    """Full-batch fit of the shader's learnable parameters, stage by stage."""
    import torch

    from light_models.learning import LightLearner

    learnable = shader.learnable_params()
    if not learnable:
        raise ValueError("nothing to calibrate: no learnable light parameter in the Light block")
    max_after = max(p.after_keyframes for p in learnable.values())
    if max_after >= len(calib.stages):
        raise ValueError(f"Light.schedule unfreezes a parameter at stage {max_after}, "
                         f"but Calibration.stages has only {len(calib.stages)} stages")
    learner = LightLearner(learnable)
    shader.set_learning(True)
    cam = camera_of(intr)
    history = []
    for stage, steps in enumerate(calib.stages):
        n = stage + 1  # "keyframe count" of LightLearner: params with after_keyframes < n are active
        shader.set_keyframe_count(n)
        log(f"stage {stage}: {steps} steps, active {learner.active(n)}")
        for step in range(steps):
            loss = sum(frame_loss(shader, fr, cam, calib) for fr in prepared) / len(prepared)
            prior = learner.prior_loss(n)
            total = loss + prior if prior is not None else loss
            total.backward()
            learner.step(n)
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                history.append({"stage": stage, "step": step, "loss": float(loss)})
                log(f"  step {step}: L1 {float(loss):.6f}")
    shader.set_learning(False)
    with torch.no_grad():
        per_frame = [float(frame_loss(shader, fr, cam, calib)) for fr in prepared]
    return history, per_frame


# ---------------------------------------------------------------- export

def _value(t):
    return t.detach().cpu().tolist()


def light_values(shader, light_cfg, params_data):
    """The light as fitted (or as configured), in plain numbers: what a SLAM
    config can read back as fixed values (step 3 of D106)."""
    import torch

    from light_models.params import resolve_param

    r_init = torch.tensor(resolve_param(light_cfg["pose"]["R_CL"], "Light.pose.R_CL", params_data).value,
                          dtype=torch.float64)
    if shader.tilt_deg is not None:
        tilt = shader.tilt_deg.detach().to(torch.float64) * (math.pi / 180.0)
        z = torch.zeros((), dtype=torch.float64)
        skew = torch.stack([torch.stack([z, z, tilt[1]]), torch.stack([z, z, -tilt[0]]),
                            torch.stack([-tilt[1], tilt[0], z])])
        r = torch.linalg.matrix_exp(skew) @ r_init
    else:
        r = r_init
    axis = r[:, 2] / r[:, 2].norm()
    out = {
        "intensity": _value(shader.intensity.view(-1)) if shader.intensity.numel() > 1 else float(shader.intensity),
        "t_CL_m": _value(shader.t_CL),
        "R_CL": r.tolist(),
        "light_axis_tilt_deg": [math.degrees(math.atan2(float(axis[0]), float(axis[2]))),
                                math.degrees(math.atan2(float(axis[1]), float(axis[2])))],
    }
    for kind, comp in (("angular", shader.angular), ("falloff", shader.falloff)):
        block = {"type": light_cfg[kind]["type"]}
        for name in getattr(comp, "_param_specs", {}):
            block[name] = _value(getattr(comp, name))
        if hasattr(comp, "knot_values"):  # spline (D67)
            block["knots_deg"] = list(comp.knots_deg)
            block["knot_values"] = _value(comp.values())
        out[kind] = block
    amb = {"type": light_cfg["ambient"]["type"]}
    for name in getattr(shader.ambient, "_param_specs", {}):
        amb[name] = _value(getattr(shader.ambient, name))
    out["ambient_of_calibration_room"] = amb
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="calibration config: Light block (what is fitted) + Calibration")
    ap.add_argument("--params", required=True, help="JSON with the starting values the Light block reads")
    ap.add_argument("--clip", required=True, help="calibration clip folder (clip.json, rgb/, depth/)")
    ap.add_argument("--out", required=True, help="result JSON")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_frames", type=int, default=None, help="use only the first N frames of clip.json")
    args = ap.parse_args(argv)

    from light_models.params import load_params_file
    from light_models.shaders import make_shader

    config = load_config(args.config)
    calib = read_calibration(config)
    light_cfg = require_key(config, "Light", "")
    params = load_params_file(args.params)
    shader = make_shader(light_cfg, lambda: params)
    start = light_values(shader, light_cfg, params)
    intr, frames = load_clip(args.clip)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    t0 = time.time()
    prepared = prepare(frames, intr, calib, args.device)
    print(f"{len(prepared)} frames, target pixels per frame: median {int(np.median([p.n_pixels for p in prepared]))}, "
          f"plane inlier fraction: min {min(p.inlier_fraction for p in prepared):.3f}", flush=True)
    history, per_frame = fit(shader, prepared, intr, calib, log=lambda s: print(s, flush=True))
    result = {
        "light": light_values(shader, light_cfg, params),
        "start": start,
        "fit": {"frames": len(prepared), "final_l1": float(np.mean(per_frame)),
                "per_frame_l1": dict(zip([p.id for p in prepared], per_frame)),
                "history": history, "seconds": time.time() - t0,
                "loss_color_space": calib.loss_color_space},
        "provenance": {"config": os.path.abspath(args.config), "params": os.path.abspath(args.params),
                       "clip": os.path.abspath(args.clip), "frames": [p.id for p in prepared]},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"done in {time.time() - t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
