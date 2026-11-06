# Coarse emission profile from the FIRST frame, before any map exists
# (docs/DECISIONS.md D112), for the self-calibration mode: it replaces a flat
# or hand-guessed start of the spline, whose first keyframes cost most of the
# ATE of the mode (D109, D111). Only what a sensor gives is read: the observed
# image and the sensor depth of that frame (RGB-D). Nothing about the lamp is
# assumed here: distance, direction and normals come from the shader's own
# components and current values.
#
# Per pixel, I = A (K Phi(theta) F(d) n.l + c), so with
#     r = I / (F(d) n.l)
# r is about A K Phi(theta) where the lamp dominates. A varies per pixel, but
# the median of r over a band of theta averages much of that out, giving a
# curve r(theta): flat in the core, falling at the edge of the cone, low
# outside. The core LEVEL of a single frame is noisy (it depends on which
# surfaces the band sees); the POSITION of the edge is not. So a smoothstep
# (theta_in, theta_out) is fitted to the curve, with both levels taken as
# medians of the curve, and only the SHAPE is kept: Phi(0) = 1.
#
# Absent Light.profile_bootstrap (read by light_models.read_profile_bootstrap)
# = nothing happens, the behaviour before D112. A frame where no edge can be
# found is an error, not a silent fallback.

from types import SimpleNamespace

import numpy as np
import torch

from light_models.gbuffer import pixel_rays
from utils.color_space import sRGB2Linear

def check_shader_supports_bootstrap(shader):
    """The shader must have the parts the estimate reads and writes."""
    if not hasattr(getattr(shader, "angular", None), "restart_from"):
        raise ValueError("Light.profile_bootstrap needs Light.angular.type: spline")
    if "angular.knot_values" not in shader.learnable_params():
        raise ValueError("Light.profile_bootstrap needs a learnable spline: Light.angular.learn.learnable true")
    if getattr(shader.cosine, "normals", None) is None or not shader.cosine.requires_sensor_depth:
        raise ValueError("Light.profile_bootstrap needs a lambert cosine with a sensor-depth normal source")


def radial_profile(observed_srgb, depth, camera, shader, p):
    """(centers_deg, medians, counts) of r = I / (F(d) n.l) per band of theta,
    bands of p.bin_deg degrees with at least p.min_pixels_per_bin pixels.
    observed_srgb: (3,H,W) in [0, 1] as stored; depth: (H,W) metres, 0 = none."""
    device, dtype = observed_srgb.device, observed_srgb.dtype
    z = torch.as_tensor(depth, device=device, dtype=dtype)
    h, w = z.shape
    points = pixel_rays(camera, h, w, device, dtype, 0.0) * z[None]  # sensor depth: offset 0 (D29)
    valid = z > 0
    with torch.no_grad():
        normal, normal_ok = shader.cosine.normals(points, valid, {}, camera)
        t = shader.t_CL.detach().to(device, dtype).view(3, 1, 1)
        axis = shader._light_axis(device, dtype).detach().view(3, 1, 1)
        v = points - t
        dist = v.norm(dim=0).clamp_min(torch.finfo(dtype).tiny)  # no pixel at the lamp: only guards depth 0
        cos_nl = -(normal * v).sum(0) / dist
        cos_theta = (v * axis).sum(0) / dist
        falloff = shader.falloff(dist)
        codes = torch.round(observed_srgb * 255.0)  # 8-bit codes as stored (CLAUDE.md section 6)
        saturated = (codes >= p.sat_8bit).any(0)
        keep = valid & normal_ok & (cos_nl >= p.min_cos_nl) & ~saturated
        linear = sRGB2Linear(observed_srgb).mean(0)
        r = (linear[keep] / (falloff[keep] * cos_nl[keep])).cpu().numpy()
        theta = torch.rad2deg(torch.acos(cos_theta[keep].clamp(-1.0, 1.0))).cpu().numpy()
    if r.size == 0:
        raise ValueError("profile bootstrap: no usable pixel in the frame")
    index = np.floor(theta / p.bin_deg).astype(np.int64)
    centers, medians, counts = [], [], []
    for i in np.unique(index):
        band = r[index == i]
        if band.size >= p.min_pixels_per_bin:
            centers.append((i + 0.5) * p.bin_deg)
            medians.append(np.median(band))
            counts.append(band.size)
    return np.array(centers), np.array(medians), np.array(counts)


def _smoothstep_of_angle(theta, theta_in, theta_out):
    x = np.clip((theta_out - theta) / (theta_out - theta_in), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def fit_cone(centers, medians, p):
    """Grid search of (theta_in, theta_out) minimising the L1 distance, in log,
    between the curve and the smoothstep between its two levels: the median
    inside theta_in and the median beyond theta_out. Raises if no pair has
    bands on both sides with a contrast of at least p.min_contrast. Returns a
    namespace theta_in, theta_out, level_core, level_out, error."""
    grid = np.arange(0.0, float(centers.max()) + p.grid_deg, p.grid_deg)
    best = None
    for theta_in in grid:
        core = medians[centers <= theta_in]
        if core.size == 0:
            continue
        level_core = np.median(core)
        for theta_out in grid[grid >= theta_in + p.min_ramp_deg]:
            outside = medians[centers >= theta_out]
            if outside.size == 0:
                continue
            level_out = np.median(outside)
            if level_core < p.min_contrast * level_out:
                continue
            model = level_out + (level_core - level_out) * _smoothstep_of_angle(centers, theta_in, theta_out)
            # In log r: the albedo seen by a band is a factor, a term here,
            # small next to the drop at the edge of the cone.
            error = np.abs(np.log(medians) - np.log(model)).mean()
            if best is None or error < best.error:
                best = SimpleNamespace(theta_in=float(theta_in), theta_out=float(theta_out),
                                       level_core=float(level_core), level_out=float(level_out),
                                       error=float(error))
    if best is None:
        raise ValueError(
            f"profile bootstrap: no cone edge found (need a contrast of {p.min_contrast} between the core "
            "and the outside of the cone in this frame)"
        )
    return best


def estimate_cone(observed_srgb, depth, camera, shader, p):
    centers, medians, _ = radial_profile(observed_srgb, depth, camera, shader, p)
    return fit_cone(centers, medians, p)


def bootstrap_spline(shader, viewpoint, p):
    """Restarts the shader's learnable spline from the cone estimated on this
    frame. Returns the fit (for the log)."""
    fit = estimate_cone(viewpoint.original_image, viewpoint.depth, viewpoint, shader, p)

    def profile(cos_theta):
        theta = torch.rad2deg(torch.acos(cos_theta.clamp(-1.0, 1.0)))
        x = ((fit.theta_out - theta) / (fit.theta_out - fit.theta_in)).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    shader.angular.restart_from(profile)
    param = shader.learnable_params()["angular.knot_values"]
    param.start = param.tensor.detach().clone()  # the value the prior refers to
    return fit
