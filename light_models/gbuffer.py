# Surface points from the rasterized G-buffer (docs/DECISIONS.md D5, D21).
#
# The rasterizer (black background) returns opacity-weighted sums:
# depth = sum_i w_i z_i, opacity = sum_i w_i. The z-depth of a pixel is
# therefore depth / opacity, and only where opacity is large enough to trust
# it (Light.gbuffer.opacity_thr). Back-projection with the camera
# intrinsics:  x_C = z * K^-1 [u, v, 1]^T  (OpenCV camera frame; z is the
# depth along the optical axis, NOT the distance to the camera or the light).
#
# Pixels below the threshold are not dropped: they are shaded at a fallback
# depth, the median z of the trusted pixels of the same image (detached).
# That keeps shading finite there (dividing a tiny depth by a tiny opacity
# would put the point at the lamp and blow up 1/d^2) while the albedo and
# opacity of those pixels still receive gradient, which mapping needs to grow
# the map into new regions. Their depth receives no gradient.

import torch


_RAYS = {}


def pixel_rays(camera, height, width, device, dtype, pixel_offset):
    """(3,H,W) K^-1 [u + o, v + o, 1] for pixel (row v, column u), K in the
    OpenCV convention of the data (integer pixel centers). o = pixel_offset
    is where the map's pixel i sits on that grid: 0 for maps in the data's
    own convention (ground-truth passes, the placement depth), 0.5 for
    rasterized maps (RASTERIZER_PIXEL_OFFSET, docs/DECISIONS.md D29)."""
    # Memoized: every render asks for the same rays (docs/DECISIONS.md D128).
    # Few keys: one per intrinsics, size and offset. Not modified by callers.
    key = (height, width, str(device), dtype, pixel_offset, camera.cx, camera.cy, camera.fx, camera.fy)
    rays = _RAYS.get(key)
    if rays is None:
        v, u = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        x = (u + pixel_offset - camera.cx) / camera.fx
        y = (v + pixel_offset - camera.cy) / camera.fy
        rays = torch.stack([x, y, torch.ones_like(x)], dim=0)
        _RAYS[key] = rays
    return rays


def _lower_median(values, mask):
    """torch.median(values[mask]) (the lower of the two middle values) without
    the boolean indexing, which synchronizes with the GPU. 0-d tensor; NaN when
    the mask is empty."""
    flat = torch.where(mask, values, torch.full_like(values, float("inf"))).flatten()
    count = mask.sum()
    ordered = torch.sort(flat).values
    # index_select, not ordered[tensor]: a 0-d tensor index synchronizes.
    middle = torch.div(count - 1, 2, rounding_mode="floor").clamp_min(0)
    pick = ordered.index_select(0, middle.view(1))[0]
    return torch.where(count > 0, pick, torch.full_like(pick, float("nan")))


def surface_points(gbuffer, camera, opacity_thr):
    """Returns (points (3,H,W) in the camera frame, valid (H,W) bool)."""
    depth = gbuffer["depth"][0]
    opacity = gbuffer["opacity"][0]
    valid = (opacity >= opacity_thr) & (depth > 0)
    # Division only where opacity >= opacity_thr > 0 (no epsilon).
    z = depth / torch.where(valid, opacity, torch.ones_like(opacity))

    with torch.no_grad():
        # Fallback: median z of the trusted pixels; if none is trusted (e.g. a
        # nearly empty map), over every pixel with some coverage. Exact zeros
        # are "no Gaussian". Chosen on the GPU without synchronizing (D128).
        covered = (opacity > 0) & (depth > 0)
        z_covered = depth / torch.where(covered, opacity, torch.ones_like(opacity))
        pool = torch.where(valid.any(), valid, covered)
        fallback = _lower_median(z_covered, pool)
        # No pixel covered at all: the albedo is 0 everywhere, so the radiance
        # is 0 for any finite positive depth; 1 only keeps the shading finite.
        fallback = torch.where(pool.any(), fallback, torch.ones_like(fallback))
    z = torch.where(valid, z, fallback)
    h, w = depth.shape
    # Required: which pixel convention the maps follow (D29).
    offset = gbuffer["pixel_offset"]
    rays = pixel_rays(camera, h, w, depth.device, depth.dtype, offset)
    return rays * z[None], valid
