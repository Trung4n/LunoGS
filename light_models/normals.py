# NormalSource: per-pixel surface normals in the camera frame, (3,H,W), plus
# a validity mask (H,W). Normals face the camera (n . x_C < 0).
#
# Call: source(points, valid, gbuffer, camera). requires_sensor_depth: True
# for sources that read the observed frame's depth (camera.depth); the shader
# builder refuses them for monocular (light_models.build_shader).

import torch

from light_models.gbuffer import pixel_rays
from light_models.registry import register


@register("normals", "depth_fd", options=("stencil_px",))
class DepthFiniteDifference:
    """Normals from central differences of the back-projected points, i.e.
    of the rasterized depth AFTER division by opacity (gbuffer.py, D5/D15):
    n = (P[u+k]-P[u-k]) x (P[v+k]-P[v-k]), k = stencil_px, flipped to face
    the camera. k = 1 is the construction of Replica/scripts/physics_check/
    common.py (depth_normals). A wider stencil averages out the bumps of a
    surface made of discrete Gaussians, which make 1-pixel differences
    unusable (layer 2 and tracking probe, docs/DECISIONS.md D40), at the cost
    of more pixels invalid near depth edges. Invalid if the pixel or one of
    its 4 neighbours at distance k is invalid, within k of the image border,
    or if the cross product is exactly zero."""

    requires_sensor_depth = False

    def __init__(self, stencil_px):
        if isinstance(stencil_px, bool) or not isinstance(stencil_px, int) or stencil_px < 1:
            raise ValueError(
                f"Light.cosine.normal_source.stencil_px must be an integer >= 1, got {stencil_px!r}"
            )
        self.k = stencil_px

    def __call__(self, points, valid, gbuffer, camera):
        k = self.k
        du = torch.zeros_like(points)
        dv = torch.zeros_like(points)
        du[:, :, k:-k] = points[:, :, 2 * k:] - points[:, :, :-2 * k]
        dv[:, k:-k, :] = points[:, 2 * k:, :] - points[:, :-2 * k, :]
        ok = valid.clone()
        ok[:, k:-k] &= valid[:, 2 * k:] & valid[:, :-2 * k]
        ok[k:-k, :] &= valid[2 * k:, :] & valid[:-2 * k, :]
        ok[:k, :] = False
        ok[-k:, :] = False
        ok[:, :k] = False
        ok[:, -k:] = False

        n = torch.cross(du, dv, dim=0)
        norm = n.norm(dim=0)
        ok &= norm > 0
        # Division only where the norm is known to be > 0 (no epsilon).
        n = n / torch.where(ok, norm, torch.ones_like(norm))
        facing_away = (n * points).sum(0) > 0
        n = torch.where(facing_away[None], -n, n)
        n = torch.where(ok[None], n, torch.zeros_like(n))
        return n, ok


def _shift(x, k, dim, forward):
    """x shifted by k along dim (forward: element i holds x[i+k]), and the
    (H,W) mask of the elements that have such a neighbour in the image."""
    n = x.shape[dim]
    out = torch.zeros_like(x)
    has = torch.zeros(x.shape[-2:], dtype=torch.bool, device=x.device)
    hdim = dim - (x.dim() - 2)  # the same axis in the (H,W) mask
    if forward:
        out.narrow(dim, 0, n - k).copy_(x.narrow(dim, k, n - k))
        has.narrow(hdim, 0, n - k).fill_(True)
    else:
        out.narrow(dim, k, n - k).copy_(x.narrow(dim, 0, n - k))
        has.narrow(hdim, k, n - k).fill_(True)
    return out, has


@register("normals", "depth_fd_edge", options=("stencil_px",))
class EdgeAwareDepthFiniteDifference(DepthFiniteDifference):
    """depth_fd with ONE-SIDED differences chosen per pixel and axis: of
    P[u+k]-P[u] and P[u]-P[u-k] the one with the smaller depth step |dz|,
    i.e. the side that does not cross a depth edge (docs/DECISIONS.md D94).

    Why: the central difference at a pixel next to an occlusion edge takes
    a point of the OTHER surface; on office0, 98% of the squared shading
    error of sensor_depth_fd normals is within 1 pixel of a depth step
    (D93). On a plane both sides give the same direction, so this matches
    depth_fd there; the baseline is k instead of 2k (noisier on noisy
    depth). A side is usable if its neighbour is valid and inside the image;
    the pixel is invalid only if an axis has no usable side, or the cross
    product is exactly zero. Ties (equal |dz|) take the forward side."""

    def __call__(self, points, valid, gbuffer, camera):
        k = self.k
        diffs = []
        ok = valid.clone()
        for dim in (2, 1):  # u (columns), then v (rows)
            nxt, has_n = _shift(points, k, dim, forward=True)
            prv, has_p = _shift(points, k, dim, forward=False)
            v_n, _ = _shift(valid, k, dim - 1, forward=True)
            v_p, _ = _shift(valid, k, dim - 1, forward=False)
            fwd, bwd = nxt - points, points - prv
            use_f = has_n & v_n
            use_b = has_p & v_p
            take_f = use_f & (~use_b | (fwd[2].abs() <= bwd[2].abs()))
            diffs.append(torch.where(take_f[None], fwd, bwd))
            ok &= use_f | use_b
        n = torch.cross(diffs[0], diffs[1], dim=0)
        norm = n.norm(dim=0)
        ok &= norm > 0
        # Division only where the norm is known to be > 0 (no epsilon).
        n = n / torch.where(ok, norm, torch.ones_like(norm))
        facing_away = (n * points).sum(0) > 0
        n = torch.where(facing_away[None], -n, n)
        n = torch.where(ok[None], n, torch.zeros_like(n))
        return n, ok


@register("normals", "gbuffer")
class GBufferNormals:
    """Normals supplied in the G-buffer itself (gbuffer["normal"],
    gbuffer["normal_valid"]). SLAM does not provide them; used by the
    verification scripts to feed ground-truth normals through the SAME
    shading code (docs/DECISIONS.md D15, D18)."""

    requires_sensor_depth = False

    def __call__(self, points, valid, gbuffer, camera):
        if "normal" not in gbuffer or "normal_valid" not in gbuffer:
            raise KeyError(
                "normal_source 'gbuffer' needs gbuffer['normal'] and "
                "gbuffer['normal_valid'] (only the verification scripts supply them)"
            )
        return gbuffer["normal"], gbuffer["normal_valid"] & valid


@register("normals", "sensor_depth_fd", options=("stencil_px", "trusted_pixels_only"))
class SensorDepthFiniteDifference:
    """depth_fd applied to the OBSERVED frame's sensor depth (camera.depth,
    RGB-D only), not to the rasterized depth (docs/DECISIONS.md D49).

    Why: the rasterizer gives every Gaussian ONE depth, that of its center,
    over its whole footprint, so the rasterized depth of a SLAM map is a
    staircase whose steps are as wide as the Gaussians; finite differences
    of it give wrong normals at every step edge (21% of the pixels with
    n . l < 0.3 in RGB-D SLAM vs 1-3% on ground truth, D48). The sensor depth
    has no steps.

    Consequences: n is a constant of the observation (no gradient), so the
    pose still reaches the shading through the rasterized depth (distance
    and direction to the lamp) and the albedo, as before. In tracking, n is
    the normal of the frame being tracked, in its own camera frame: exactly
    the n that the shading at the true pose needs. The sensor depth is in the
    data's pixel convention (offset 0) while the rasterized maps sit half a
    pixel off (D29); the normal at pixel i is used with the light direction
    of rasterized pixel i, half a pixel apart, which is negligible for a
    direction. Pixels without a sensor reading (depth 0) have no normal.

    trusted_pixels_only (docs/DECISIONS.md D99): true = the normals are also
    restricted to the pixels the G-buffer trusts (`valid`: rasterized
    opacity >= Light.gbuffer.opacity_thr), the behaviour before D99. Then a
    low-opacity hole of the map gets no normal, Lambert gives it n . l = 0
    and it renders ambient-only (black spots), and LightTracking.pixel_weight
    drops it from both losses. false = the normal depends on the sensor
    depth only; the distance and direction to the lamp of such a pixel still
    come from the G-buffer's fallback depth (D21)."""

    requires_sensor_depth = True
    fd_class = DepthFiniteDifference

    def __init__(self, stencil_px, trusted_pixels_only):
        if not isinstance(trusted_pixels_only, bool):
            raise ValueError(
                "Light.cosine.normal_source.trusted_pixels_only must be true or false, "
                f"got {trusted_pixels_only!r}"
            )
        self.trusted_pixels_only = trusted_pixels_only
        self.fd = self.fd_class(stencil_px)
        # Runtime cache, not configuration: normals of the last camera seen
        # (tracking renders the same frame for every iteration). One entry
        # only: mapping cycles through many keyframes.
        self._cache_key = None
        self._cache = None

    def __getstate__(self):
        # Shaders are pickled to the backend / GUI processes: no CUDA tensors.
        state = self.__dict__.copy()
        state["_cache_key"] = None
        state["_cache"] = None
        return state

    def __call__(self, points, valid, gbuffer, camera):
        depth = getattr(camera, "depth", None)
        if depth is None:
            raise ValueError(
                "normal_source 'sensor_depth_fd' needs the observed frame's sensor "
                "depth (camera.depth): RGB-D only"
            )
        if getattr(camera, "frame_cache", None) is not None:
            # Tracking: once per frame, in the frame's static buffers (D130).
            from utils.perf import frame_constant

            n, ok = frame_constant(camera, ("sensor_normals", type(self).__name__, self.fd_class.__name__),
                                   lambda: self._from_sensor(depth, camera, points, gbuffer))
            return n, (ok & valid) if self.trusted_pixels_only else ok
        # The key holds the camera and depth objects themselves (not their
        # id(), which Python reuses after garbage collection).
        cached = self._cache_key
        hit = (
            cached is not None
            and cached[0] is camera
            and cached[1] is depth
            and cached[2:] == (points.device, points.dtype)
        )
        if not hit:
            self._cache = self._from_sensor(depth, camera, points, gbuffer)
            self._cache_key = (camera, depth, points.device, points.dtype)
        n, ok = self._cache
        return n, (ok & valid) if self.trusted_pixels_only else ok

    def _from_sensor(self, depth, camera, points, gbuffer):
        z = depth if torch.is_tensor(depth) else torch.from_numpy(depth)
        z = self._prefilter(z.to(points.device, points.dtype).reshape(points.shape[1:]))
        h, w = z.shape
        rays = pixel_rays(camera, h, w, points.device, points.dtype, 0.0)
        # 0 = no sensor reading (the dataset's convention), not a threshold.
        return tuple(self.fd(rays * z[None], z > 0, gbuffer, camera))

    def _prefilter(self, z):
        """The sensor depth the differences are taken on (none here)."""
        return z


@register("normals", "sensor_depth_fd_edge", options=("stencil_px", "trusted_pixels_only"))
class SensorDepthEdgeAwareFiniteDifference(SensorDepthFiniteDifference):
    """sensor_depth_fd with the edge-aware one-sided differences of
    depth_fd_edge (docs/DECISIONS.md D94). RGB-D only."""

    fd_class = EdgeAwareDepthFiniteDifference


def bilateral_depth(z, radius_px, sigma_space_px, sigma_range_rel):
    """Edge-preserving smoothing of a depth image (H,W; 0 = no reading),
    for normals of NOISY sensor depth (docs/DECISIONS.md D146). Weighted mean
    of the readings in the (2r+1)^2 window, weight = exp(-d^2 / 2 sigma_s^2)
    (pixel distance d) * exp(-(ln z' - ln z)^2 / 2 sigma_r^2): the range term
    is on log depth, so sigma_range_rel is a RELATIVE depth difference (the
    sensor noise grows with depth). Pixels without a reading neither
    contribute nor receive a value. Where the window is one-sided (within r
    of the image border, next to holes and depth edges) the mean of a slanted
    surface is biased towards its inner side, a few degrees on the normals of
    a steep plane; inside, a plane is unchanged. One pass of (2r+1)^2 shifted
    adds: run once per frame (frame_constant / the normal cache)."""
    valid = z > 0
    logz = torch.where(valid, torch.log(torch.where(valid, z, torch.ones_like(z))), torch.zeros_like(z))
    h, w = z.shape
    r = radius_px
    pad_log = torch.nn.functional.pad(logz[None, None], (r, r, r, r))[0, 0]
    pad_val = torch.nn.functional.pad(valid[None, None].to(z.dtype), (r, r, r, r))[0, 0]
    num = torch.zeros_like(z)
    den = torch.zeros_like(z)
    inv_s = 1.0 / (2.0 * sigma_space_px ** 2)
    inv_r = 1.0 / (2.0 * sigma_range_rel ** 2)
    dx2 = torch.arange(-r, r + 1, device=z.device, dtype=z.dtype) ** 2
    for dy in range(-r, r + 1):
        # One row of the window at once: (H, W, 2r+1) views of the padded
        # band (unfold copies nothing), so 2r+1 passes instead of (2r+1)^2.
        nb_log = pad_log[r + dy:r + dy + h].unfold(1, 2 * r + 1, 1)
        nb_val = pad_val[r + dy:r + dy + h].unfold(1, 2 * r + 1, 1)
        wgt = nb_val * torch.exp(-(dx2 + dy * dy) * inv_s - (nb_log - logz[..., None]) ** 2 * inv_r)
        num += (wgt * nb_log).sum(-1)
        den += wgt.sum(-1)
    # The centre itself always weighs 1 where valid, so den > 0 there.
    out = torch.exp(num / torch.where(valid, den, torch.ones_like(den)))
    return torch.where(valid, out, torch.zeros_like(z))


@register("normals", "sensor_depth_bilateral_fd_edge",
          options=("stencil_px", "trusted_pixels_only", "bilateral"))
class SensorDepthBilateralEdgeAwareFiniteDifference(SensorDepthEdgeAwareFiniteDifference):
    """sensor_depth_fd_edge on the sensor depth smoothed by bilateral_depth
    (docs/DECISIONS.md D146). RGB-D only. On noisy depth, 1-pixel differences
    turn the noise into normals off by 35 deg (Kinect-v1 level) to 64 deg
    (heavy level) (D145); smoothing first keeps the depth edges, so the
    edge-aware differences still see them. bilateral: {radius_px,
    sigma_space_px, sigma_range_rel}, all required."""

    def __init__(self, stencil_px, trusted_pixels_only, bilateral):
        super().__init__(stencil_px, trusted_pixels_only)
        where = "Light.cosine.normal_source.bilateral"
        if not isinstance(bilateral, dict):
            raise TypeError(f"{where} must be a block, got {bilateral!r}")
        keys = ("radius_px", "sigma_space_px", "sigma_range_rel")
        extra = set(bilateral) - set(keys)
        if extra:
            raise ValueError(f"{where}: unexpected keys {sorted(extra)}")
        missing = [k for k in keys if k not in bilateral]
        if missing:
            raise KeyError(f"{where}: missing keys {missing}")
        r = bilateral["radius_px"]
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError(f"{where}.radius_px must be an integer >= 1, got {r!r}")
        for k in ("sigma_space_px", "sigma_range_rel"):
            v = bilateral[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                raise ValueError(f"{where}.{k} must be a positive number, got {v!r}")
        self.bilateral = (r, float(bilateral["sigma_space_px"]), float(bilateral["sigma_range_rel"]))

    def _prefilter(self, z):
        return bilateral_depth(z, *self.bilateral)


def quaternion_axes(q):
    """(N,3,3) rotation matrices of (N,4) quaternions (w, x, y, z), on q's
    device; column j is the Gaussian's j-th local axis in the world frame (the
    covariance is R S S^T R^T, gaussian_model.build_covariance_from_scaling_rotation).
    The same matrix as general_utils.build_rotation, which only builds on CUDA."""
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def quaternion_from_z(n):
    """(N,4) unit quaternions (w, x, y, z) of the shortest rotation taking the
    local z axis to the unit vectors n (N,3). n and -n give the same flat
    Gaussian, so n is flipped to n_z >= 0 first: then 1 + n_z >= 1 and the
    rotation is never the degenerate half turn."""
    n = torch.where(n[:, 2:3] < 0, -n, n)
    q = torch.stack([1 + n[:, 2], -n[:, 1], n[:, 0], torch.zeros_like(n[:, 0])], 1)
    return q / q.norm(dim=1, keepdim=True)


@register("normals", "shortest_axis", options=("min_agreement", "init_flatness", "init", "supervision", "tracking_refresh_every",
                  "raster_gradient"))
class ShortestAxis:
    """Normals from the map itself: each Gaussian's shortest axis, flipped to
    face the camera, rasterized like a colour (docs/DECISIONS.md D149). The
    blended vector of a pixel is normalized; the pixel has a normal where the
    G-buffer trusts it (opacity >= Light.gbuffer.opacity_thr) and the blended
    vector keeps at least min_agreement of the opacity, |sum w_i n_i| >=
    min_agreement * sum w_i (the Gaussians covering it roughly agree).

    Meaningful only for FLAT Gaussians: Training.scale_regularizer must be
    `disc` (the backend refuses `isotropic`, whose spheres have no shortest
    axis). New Gaussians start flat, oriented by the normals of the `init`
    block (another normal source, e.g. sensor_depth_bilateral_fd_edge for
    RGB-D) on the placement depth, with their normal axis init_flatness times
    their in-plane size (GaussianModel.create_pcd_from_image_and_depth). The
    placement de-shading (albedo_init.shading_at_depth, gbuffer["placement"])
    uses the same init normals, so it stays the inverse of how the new
    Gaussians are shaded.

    Unlike the sensor-depth sources, the normal of a pixel is not a constant
    of the observation: it is rendered from the map. Mapping rasterizes it at
    every render, with gradient to the Gaussians' rotation, scale, position
    and opacity. Tracking rasterizes it once per frame, at the frame's
    initial pose, and keeps it fixed (utils/perf.frame_constant): normals
    that move with the pose pulled tracking off with the rasterized-depth
    normals (D48, D146 rastfd5), and once per frame keeps the CUDA graphs and
    the compiled shading unchanged. The colour refinement (geometry frozen)
    does the same per keyframe.
    tracking_refresh_every (required; null = never): re-render the normals,
    still without gradient, at the current pose estimate after every this
    many tracking iterations. Why: tracking starts from the previous frame's
    pose (Training.motion_model: none), and the inter-frame motion of office0
    is 2.4 px median, 4.8 px p90 at the image; normals rendered there are
    shifted against the observation, which the sensor-depth sources never
    are (they come from the observed frame). Kaggle, once per frame: RPE
    higher than `_bilateral_fill` at every depth-noise level (D149).
    raster_gradient (required): `all` = the mapping losses reach the
    Gaussians' rotation, position, opacity and scale (and the keyframe pose)
    through the rasterized normals; `rotation` = only the rotation, through
    the colours, the rest of the normal raster detached. Why: with `all` the
    shading and the supervision can move and fade Gaussians to fit the
    normals, and the depth rendered from the map on heavy depth noise stays
    at one noisy frame's error (17.8 mm vs 10.1 for the spheres of
    `_bilateral_fill`, D149).
    supervision (required; null = none): {weight, source}. In mapping, the
    penalty weight * mean (1 - n . n_s) over the pixels where both have a
    normal, n the map's normal (with gradient) and n_s that of `source`, a
    normal source of the keyframe's observation (e.g. the bilateral sensor
    normals). Added to the mapping loss with the ambient's L1
    (light_map_penalty), never in tracking or the refinement. Why: the
    photometric and depth losses alone turn the rotations away from the
    surface (local test, heavy depth noise, 60 frames: 15 deg after the
    init with the rotations frozen, 22-25 deg when they learn); every
    keyframe that sees a Gaussian pulls it towards its own noisy normal, so
    the map averages the noise of independent frames. RGB-D only; monocular
    will need another source (null meanwhile)."""

    KEY = "gaussian_normal"

    def __init__(self, min_agreement, init_flatness, init, supervision, tracking_refresh_every, raster_gradient):
        where = "Light.cosine.normal_source"
        if raster_gradient not in ("all", "rotation"):
            raise ValueError(f"{where}.raster_gradient must be 'all' or 'rotation', got {raster_gradient!r}")
        self.raster_gradient = raster_gradient
        r = tracking_refresh_every
        if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 1):
            raise ValueError(f"{where}.tracking_refresh_every must be null or an integer >= 1, got {r!r}")
        self.tracking_refresh_every = r
        for name, v in (("min_agreement", min_agreement), ("init_flatness", init_flatness)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
                raise ValueError(f"{where}.{name} must be in (0, 1], got {v!r}")
        self.min_agreement = float(min_agreement)
        self.init_flatness = float(init_flatness)
        if not isinstance(init, dict):
            raise TypeError(f"{where}.init must be a block {{type: <normal source>, ...}}, got {init!r}")
        from light_models.registry import build_component

        if init.get("type") == "shortest_axis":
            raise ValueError(f"{where}.init cannot be shortest_axis itself")
        self.init = build_component("normals", init, f"{where}.init", None)
        self.requires_sensor_depth = self.init.requires_sensor_depth
        self.supervision = None
        if supervision is not None:
            sw = f"{where}.supervision"
            if not isinstance(supervision, dict) or set(supervision) != {"weight", "source"}:
                raise ValueError(f"{sw} must be null or a block {{weight, source}}, got {supervision!r}")
            w = supervision["weight"]
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not w > 0:
                raise ValueError(f"{sw}.weight must be > 0 (null for no supervision), got {w!r}")
            src = supervision["source"]
            if not isinstance(src, dict):
                raise TypeError(f"{sw}.source must be a normal source block, got {src!r}")
            if src.get("type") == "shortest_axis":
                raise ValueError(f"{sw}.source cannot be shortest_axis itself")
            self.supervision = (float(w), build_component("normals", src, f"{sw}.source", None))
            self.requires_sensor_depth = self.requires_sensor_depth or self.supervision[1].requires_sensor_depth

    # ---- what render() rasterizes ----

    def frame_rasters(self, gaussians, camera):
        """{gbuffer key: zero-argument function -> (N,3) colours}: the
        shortest axis of every Gaussian, facing the camera, in the camera
        frame. A function so that tracking, which reads the frame's copy,
        computes nothing on its other iterations."""
        def colours():
            xyz = gaussians.get_xyz
            R = camera.R.to(xyz.device, xyz.dtype)  # world -> camera
            T = camera.T.to(xyz.device, xyz.dtype)
            axes = quaternion_axes(gaussians.get_rotation)
            k = gaussians.get_scaling.argmin(dim=1)
            n_w = axes.gather(2, k.view(-1, 1, 1).expand(-1, 3, 1))[:, :, 0]
            n_c = n_w @ R.T
            x_c = xyz @ R.T + T
            # Facing the camera: n . x_C < 0 (x_C = the ray to the centre).
            flip = (n_c * x_c).sum(1, keepdim=True) > 0
            return torch.where(flip, -n_c, n_c)

        return {self.KEY: colours}

    # ---- normal source ----

    def __call__(self, points, valid, gbuffer, camera):
        if gbuffer.get("placement", False):
            return self.init(points, valid, gbuffer, camera)
        from utils.perf import FRAME_RASTER, frame_constant, map_frozen

        if map_frozen(camera):
            def missing():
                raise KeyError("shortest_axis: the frame's normal raster was not rendered")

            raw = frame_constant(camera, (FRAME_RASTER, self.KEY), missing)
        elif self.KEY in gbuffer:
            raw = gbuffer[self.KEY]
        else:
            raise KeyError(
                f"normal_source 'shortest_axis' needs gbuffer[{self.KEY!r}] (rendered by "
                "gaussian_renderer.render) or gbuffer['placement']"
            )
        length = raw.norm(dim=0)
        opacity = gbuffer["opacity"][0]
        ok = valid & (length >= self.min_agreement * opacity) & (length > 0)
        # Division only where length > 0 (no epsilon).
        n = raw / torch.where(ok, length, torch.ones_like(length))[None]
        facing_away = (n * points).sum(0) > 0
        n = torch.where(facing_away[None], -n, n)
        n = torch.where(ok[None], n, torch.zeros_like(n))
        return n, ok

    def penalty(self, n, ok, points, valid, gbuffer, camera):
        """Mapping penalty pulling the map's normals towards the supervision
        source's (see the class doc); None when there is none to add (no
        supervision, frozen map, placement shading)."""
        from utils.perf import map_frozen

        if self.supervision is None or map_frozen(camera) or gbuffer.get("placement", False):
            return None
        weight, source = self.supervision
        n_s, ok_s = source(points, valid, gbuffer, camera)
        both = (ok & ok_s).to(n.dtype)
        mis = 1.0 - (n * n_s).sum(0)
        return weight * (mis * both).sum() / both.sum().clamp_min(1.0)

    def initial_orientation(self, camera, depth):
        """(normals (3,H,W) in the camera frame, valid (H,W)) for the new
        Gaussians of a keyframe, from the init source on the placement depth
        (H,W tensor, 0 = no Gaussian), in the data's pixel convention."""
        h, w = depth.shape
        rays = pixel_rays(camera, h, w, depth.device, depth.dtype, 0.0)
        valid = depth > 0
        gbuffer = {"depth": depth[None], "opacity": valid.to(depth.dtype)[None],
                   "pixel_offset": 0.0, "placement": True}
        return self.init(rays * depth[None], valid, gbuffer, camera)
