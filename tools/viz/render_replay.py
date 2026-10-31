"""Replay video of recorded SLAM runs (docs/DECISIONS.md D76).

Reads the record/ folder written by slam.py --record (utils/run_recorder.py)
and draws, frame by frame, what the live GUI would have shown:

  left   the Gaussian map as it grows, from a virtual camera ('follow':
         behind the estimated camera; 'overview': fixed above the
         trajectory), with the estimated and reference trajectories, the
         current frustums and the latest keyframes;
  right  observed, rendered, albedo (light model) and depth images, the
         learned emission profile, and the current map from a third-person
         camera circling the scene.

The legend is not drawn on the video: it is written next to it, in
<video>.txt. Several runs (e.g. MonoGS on `lit` vs the light model) are
stacked vertically, synchronised on the frame index.

    # both views, default names (<record's folder>/videos/<run>_<view>.mp4)
    python tools/viz/render_replay.py --run record/office0_even_rgbd
    # comparison
    python tools/viz/render_replay.py --run <MonoGS lit run> --label "MonoGS (lit)"         --run <nocos run> --label "light model"

The map views use the MonoGS rasterizer when CUDA is available (--renderer
auto), otherwise a CPU preview (one opaque disc per Gaussian). The videos
are re-encoded to H.264 when ffmpeg exists. Display only: nothing here feeds
back into SLAM.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Colours (BGR, OpenCV) of the overlays.
COL_EST = (40, 40, 230)
COL_GT = (60, 200, 60)
COL_KF = (230, 140, 40)
COL_TEXT = (235, 235, 235)
COL_PANEL_BG = (30, 30, 30)


# ---------------------------------------------------------------- loading


def _read_poses(path, prefix):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = [f"{prefix}_{r}{c}" for r in range(3) for c in range(4)]
    poses = {}
    for row in rows:
        m = np.eye(4)
        m[:3] = np.array([float(row[c]) for c in cols]).reshape(3, 4)
        poses[int(row["frame_idx"])] = m
    return rows, poses


class Run:
    """One recorded run: metadata, poses, images and map snapshots."""

    def __init__(self, run_dir, label=None):
        # Either the run folder (results/<experiment>/<time>) or its record/
        # folder itself (e.g. downloaded on its own).
        self.dir = run_dir if os.path.isfile(os.path.join(run_dir, "meta.json")) else os.path.join(run_dir, "record")
        if not os.path.isfile(os.path.join(self.dir, "meta.json")):
            raise FileNotFoundError(f"{run_dir}: no record/meta.json (run slam.py with --record)")
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.label = label or self.meta["run"]
        rows, self.est_live = _read_poses(os.path.join(self.dir, "frames.csv"), "est")
        _, self.gt = _read_poses(os.path.join(self.dir, "frames.csv"), "gt")
        self.n_keyframes = {int(r["frame_idx"]): int(r["n_keyframes"]) for r in rows}
        self.image_frames = sorted(int(r["frame_idx"]) for r in rows if r["has_images"] == "1")
        final = os.path.join(self.dir, "frames_final.csv")
        self.keyframes = []
        if os.path.isfile(final):
            frows, _ = _read_poses(final, "est")
            self.keyframes = [int(r["frame_idx"]) for r in frows if r["is_keyframe"] == "1"]
        self.maps = []  # (frame_idx, path), the final map last
        for path in sorted(glob.glob(os.path.join(self.dir, "maps", "kf*.npz"))):
            with np.load(path) as z:
                self.maps.append((int(z["frame_idx"]), path))
        final_map = os.path.join(self.dir, "maps", "final.npz")
        if os.path.isfile(final_map):
            with np.load(final_map) as z:
                self.maps.append((int(z["frame_idx"]), final_map))
        self.maps.sort(key=lambda m: m[0])
        self.light = _read_light(os.path.join(os.path.dirname(os.path.normpath(self.dir)), "light_learned.csv"))             or _read_light(os.path.join(self.dir, "light_learned.csv"))
        self._map_cache = (None, None)
        self.color_gain = self._color_gain()

    def map_at(self, frame_idx):
        """Latest map snapshot taken at or before frame_idx (dict) or None."""
        path = None
        for idx, p in self.maps:
            if idx <= frame_idx:
                path = p
        if path is None:
            return None
        if self._map_cache[0] != path:
            with np.load(path) as z:
                self._map_cache = (path, {k: z[k] for k in z.files})
        return self._map_cache[1]

    def _color_gain(self):
        # Linear albedo is unbounded (K = 1 runs reach ~2-5): scale the map
        # colours once per run so that the 99th percentile maps to white.
        if self.meta["map_color"] != "linear_albedo" or not self.maps:
            return 1.0
        with np.load(self.maps[-1][1]) as z:
            c = z["color"].astype(np.float32)
        p = float(np.percentile(c[c > 0], 99)) if (c > 0).any() else 1.0
        return 1.0 / p if p > 0 else 1.0

    def display_colors(self, snap):
        c = snap["color"].astype(np.float32)
        if self.meta["map_color"] == "linear_albedo":
            c = linear_to_srgb(np.clip(c * self.color_gain, 0.0, 1.0))
        return np.clip(c, 0.0, 1.0)

    def image(self, frame_idx, kind):
        path = os.path.join(self.dir, "images", f"{frame_idx:06d}_{kind}.png")
        if not os.path.isfile(path):
            return None
        import cv2

        flag = cv2.IMREAD_UNCHANGED if kind == "depth" else cv2.IMREAD_COLOR
        return cv2.imread(path, flag)


def _read_light(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


# ---------------------------------------------------------------- geometry


def look_at(eye, target, up):
    """Camera-to-world 4x4, OpenCV convention (x right, y down, z forward)."""
    z = target - eye
    z = z / np.linalg.norm(z)
    # OpenCV: y points down (-up) and x = y cross z.
    x = np.cross(-up, z)
    if np.linalg.norm(x) < 1e-8:  # looking along `up`
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def world_up(poses):
    """Average 'up' of the cameras (-y axis of each camera, OpenCV)."""
    ups = np.array([-p[:3, 1] for p in poses])
    u = ups.mean(axis=0)
    return u / np.linalg.norm(u)


def overview_pose(positions, up, elevation, distance_scale):
    """Fixed camera above the trajectory looking at its centre."""
    centre = positions.mean(axis=0)
    flat = positions - centre
    flat = flat - np.outer(flat @ up, up)
    # Look along the trajectory's second principal direction, so the longer
    # extent spans the image width.
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    side = vt[1] if np.linalg.norm(vt[1]) > 0 else vt[0]
    extent = max(float(np.linalg.norm(flat, axis=1).max()), 0.5)
    eye = centre + distance_scale * extent * (math.cos(elevation) * side + math.sin(elevation) * up)
    return look_at(eye, centre, up)


class FollowCamera:
    """Camera behind and above the estimated camera, smoothed over time."""

    def __init__(self, up, back, height, ahead, smoothing):
        self.up, self.back, self.height, self.ahead, self.smoothing = up, back, height, ahead, smoothing
        self.eye = self.target = None

    def update(self, c2w):
        p, z = c2w[:3, 3], c2w[:3, 2]
        eye = p - self.back * z + self.height * self.up
        target = p + self.ahead * z
        if self.eye is None:
            self.eye, self.target = eye, target
        else:
            a = self.smoothing
            self.eye = (1 - a) * self.eye + a * eye
            self.target = (1 - a) * self.target + a * target
        return look_at(self.eye, self.target, self.up)


def intrinsics(width, height, hfov_deg):
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


def project(points, c2w, K):
    """(N,3) world points -> (N,2) pixels and (N,) in-front mask."""
    w2c = np.linalg.inv(c2w)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 1e-3
    uv = (pc[:, :2] / np.where(front, z, 1.0)[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, front


def frustum_lines(c2w, intr, depth):
    """World-space segments of a camera frustum drawn `depth` metres deep."""
    w, h = intr["width"], intr["height"]
    corners = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        d = np.array([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], 1.0]) * depth
        corners.append(c2w[:3, :3] @ d + c2w[:3, 3])
    o = c2w[:3, 3]
    segs = [(o, c) for c in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return segs


# ---------------------------------------------------------------- drawing


def _visible_runs(uv, ok):
    """Maximal runs of consecutive drawable points, as int32 (n,1,2) arrays."""
    runs, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            runs.append(uv[start:i])
            start = None
    if start is not None:
        runs.append(uv[start:])
    return [np.round(r).astype(np.int32).reshape(-1, 1, 2) for r in runs if len(r) >= 2]


def draw_polyline(img, points, c2w, K, color, thickness):
    """Polyline through world points; pieces behind the camera (or absurdly
    far off-screen) are skipped. One projection, one cv2.polylines call."""
    import cv2

    if len(points) < 2:
        return
    uv, front = project(np.asarray(points, dtype=np.float64), c2w, K)
    ok = front & np.all(np.abs(uv) < 1e5, axis=1)
    runs = _visible_runs(uv, ok)
    if runs:
        cv2.polylines(img, runs, False, color, thickness, cv2.LINE_AA)


def draw_segments(img, segs, c2w, K, color, thickness):
    """Independent segments [(a, b), ...] projected together."""
    import cv2

    if not segs:
        return
    pts = np.asarray([p for seg in segs for p in seg], dtype=np.float64)
    uv, front = project(pts, c2w, K)
    ok = (front & np.all(np.abs(uv) < 1e5, axis=1)).reshape(-1, 2).all(axis=1)
    lines = np.round(uv).astype(np.int32).reshape(-1, 2, 2)[ok]
    if len(lines):
        cv2.polylines(img, list(lines.reshape(-1, 2, 1, 2)), False, color, thickness, cv2.LINE_AA)


def render_points(snap, colors, c2w, K, size, background, max_radius=8, footprint=3.0):
    """CPU fallback (no CUDA): every Gaussian drawn as an opaque disc whose
    pixel radius follows its largest scale (footprint x sigma), nearest disc
    wins per pixel (z-buffer). No alpha blending: a preview, not a render."""
    w, h = size
    img = np.empty((h, w, 3), np.float32)
    img[:] = background
    keep = snap["opacity"].astype(np.float32) > 0.5
    xyz = snap["xyz"][keep].astype(np.float64)
    uv, front = project(xyz, c2w, K)
    w2c = np.linalg.inv(c2w)
    z = (xyz @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    sigma = snap["scaling"][keep].astype(np.float32).max(axis=1)
    ok = front & (uv[:, 0] > -max_radius) & (uv[:, 0] < w + max_radius)         & (uv[:, 1] > -max_radius) & (uv[:, 1] < h + max_radius)
    u, v, z, sigma, col = uv[ok, 0], uv[ok, 1], z[ok], sigma[ok], colors[keep][ok]
    r = np.clip(np.ceil(K[0, 0] * footprint * sigma / z), 1, max_radius).astype(int)
    pix, depth, cidx = [], [], []
    for rr in np.unique(r):
        sel = np.nonzero(r == rr)[0]
        dy, dx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disc = dx ** 2 + dy ** 2 <= rr * rr + rr
        dx, dy = dx[disc], dy[disc]
        px = (np.round(u[sel])[:, None] + dx[None]).astype(int)
        py = (np.round(v[sel])[:, None] + dy[None]).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pix.append((py * w + px)[inside])
        depth.append(np.broadcast_to(z[sel][:, None], px.shape)[inside])
        cidx.append(np.broadcast_to(sel[:, None], px.shape)[inside])
    if pix:
        pix, depth, cidx = np.concatenate(pix), np.concatenate(depth), np.concatenate(cidx)
        order = np.lexsort((depth, pix))
        first = np.unique(pix[order], return_index=True)[1]
        win = order[first]
        img.reshape(-1, 3)[pix[win]] = col[cidx[win]]
    return (np.clip(img, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)  # RGB -> BGR


class GaussianRenderer:
    """Map view through the MonoGS rasterizer (display colours as SH degree 0)."""

    def __init__(self, background):
        import torch
        from types import SimpleNamespace

        from gaussian_splatting.gaussian_renderer import render
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from gaussian_splatting.utils.sh_utils import RGB2SH
        from utils.camera_utils import Camera

        self.torch, self.render, self.GaussianModel, self.Camera = torch, render, GaussianModel, Camera
        self.RGB2SH = RGB2SH
        self.pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        self.bg = torch.tensor(background, dtype=torch.float32, device="cuda")
        # The snapshot itself is kept (not its id): an id can be reused once
        # the old dict is garbage collected.
        self._snap, self._model, self._colors = None, None, None

    def _model_for(self, snap, colors):
        torch = self.torch
        dev = "cuda"
        f = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)  # noqa: E731
        if self._snap is not snap:
            m = self.GaussianModel(0)
            m._xyz = f(snap["xyz"])
            m._scaling = torch.log(f(snap["scaling"]).clamp_min(1e-8))
            m._rotation = f(snap["rotation"])
            op = f(snap["opacity"]).clamp(1e-6, 1 - 1e-6)[:, None]
            m._opacity = torch.log(op / (1 - op))
            m._features_rest = torch.zeros(m._xyz.shape[0], 0, 3, device=dev)
            self._model, self._snap, self._colors = m, snap, None
        if self._colors is not colors:
            # Colours as the degree-0 SH coefficient: the rasterizer splats
            # C0 * dc + 0.5. (render()'s override_color is never used: it
            # tests colors_precomp, just set to None, instead of
            # override_color.) Re-uploaded when the colour mode changes.
            self._model._features_dc = self.RGB2SH(f(colors))[:, None, :]
            self._colors = colors
        return self._model

    def __call__(self, snap, colors, c2w, K, size):
        torch = self.torch
        w, h = size
        model = self._model_for(snap, colors)
        w2c = np.linalg.inv(c2w)
        fovx = 2 * math.atan(w / (2 * K[0, 0]))
        fovy = 2 * math.atan(h / (2 * K[1, 1]))
        cam = self.Camera.init_from_gui(0, torch.eye(4, device="cuda"), fovx, fovy,
                                        K[0, 0], K[1, 1], K[0, 2], K[1, 2], h, w)
        cam.update_RT(torch.tensor(w2c[:3, :3], dtype=torch.float32, device="cuda"),
                      torch.tensor(w2c[:3, 3], dtype=torch.float32, device="cuda"))
        with torch.no_grad():
            out = self.render(cam, model, self.pipe, self.bg, shader=None)["render"]
        img = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return img[:, :, ::-1].copy()


def colorize_depth(depth_mm, depth_max_m):
    import cv2

    d = np.clip(depth_mm.astype(np.float32) / 1000.0 / depth_max_m, 0, 1)
    img = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth_mm == 0] = 0
    return img


def fit(img, size):
    import cv2

    w, h = size
    out = np.empty((h, w, 3), np.uint8)
    out[:] = COL_PANEL_BG
    if img is None:
        return out
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def put_text(img, lines, org=(8, 20), scale=0.5):
    import cv2

    x, y = org
    for line in lines:
        # Dark outline from thin text shifted by 1 px (a thicker stroke changes
        # the glyph widths in OpenCV 5 and misaligns the outline).
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            cv2.putText(img, line, (x + ox, y + oy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, COL_TEXT, 1, cv2.LINE_AA)
        y += int(26 * scale / 0.5)


def light_panel(run, n_kf, size):
    """Learned emission profile Phi(theta) (spline knots) at this keyframe
    count, or the learned scalar values as text."""
    import cv2

    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = COL_PANEL_BG
    if run.light is None:
        put_text(img, ["light parameters: fixed / none"])
        return img
    row = run.light[0]
    for r in run.light:
        if int(r["n_keyframes"]) <= n_kf:
            row = r
    knots = sorted((int(k.split("[")[1][:-1]), float(v)) for k, v in row.items()
                   if k.startswith("angular.knot_values["))
    lines = [f"learned light (kf {row['n_keyframes']})"]
    for k, v in row.items():
        if k not in ("frame_idx", "n_keyframes") and not k.startswith("angular.knot_values["):
            lines.append(f"{k} {float(v):.4g}")
    if knots:
        # knot i is at (i + 1) degrees (Phi(0) = 1 fixed), see angular.Spline.
        xs = [0.0] + [i + 1.0 for i, _ in knots]
        ys = [1.0] + [v for _, v in knots]
        x0, y0, x1, y1 = 40, 50, w - 10, h - 25
        cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
        pts = [(int(x0 + (x1 - x0) * x / xs[-1]), int(y1 - (y1 - y0) * y)) for x, y in zip(xs, ys)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (80, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Phi", (5, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_TEXT, 1)
        cv2.putText(img, f"0..{int(xs[-1])} deg", (x0, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_TEXT, 1)
        put_text(img, lines[:2], org=(8, 18), scale=0.42)
    else:
        put_text(img, lines, scale=0.45)
    return img


# ---------------------------------------------------------------- composing


def orbit_pose(centre, up, radius, elevation, angle):
    """Camera on a circle around `centre`, `elevation` above the horizon."""
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    horiz = math.cos(angle) * e1 + math.sin(angle) * e2
    eye = centre + radius * (math.cos(elevation) * horiz + math.sin(elevation) * up)
    return look_at(eye, centre, up)


def view_defaults(args, view):
    """(ceiling margin or None, trail length or None) of a view."""
    ceiling = args.ceiling if args.ceiling is not None else (0.3 if view == "overview" else None)
    trail = args.trail if args.trail is not None else (300 if view == "follow" else None)
    return ceiling, trail


class RowRenderer:
    """One run's row of the video: map view plus the side panels."""

    def __init__(self, run, args, view, make_renderer):
        self.run, self.args, self.view = run, args, view
        # Separate renderers: each caches the map snapshot it last uploaded.
        self.map_renderer, self.orbit_renderer = make_renderer(run), make_renderer(run)
        self.map_size = (args.map_width, args.map_height)
        self.K = intrinsics(args.map_width, args.map_height, args.fov)
        self.panel_size = (args.panel_width, args.map_height // 3)
        self.K_orbit = intrinsics(*self.panel_size, args.orbit_fov)
        gts = [run.gt[i] for i in sorted(run.gt)]
        self.up = world_up(gts)
        positions = np.array([p[:3, 3] for p in gts])
        self.overview = overview_pose(positions, self.up, math.radians(args.elevation), args.distance)
        self.follow = FollowCamera(self.up, args.back, args.height, args.ahead, args.smoothing)
        self.intr = run.meta["intrinsics"]
        self.centre = positions.mean(axis=0)
        flat = positions - self.centre
        flat = flat - np.outer(flat @ self.up, self.up)
        self.extent = max(float(np.linalg.norm(flat, axis=1).max()), 0.5)
        # Gaussians above the highest camera + margin are hidden in views from
        # above (the ceiling would block them); None = keep everything.
        top = float((positions @ self.up).max())
        ceiling, self.trail = view_defaults(args, view)
        self.ceiling = None if ceiling is None else top + ceiling
        self.orbit_ceiling = top + args.orbit_ceiling
        self._cuts = {}

    def _cut(self, snap, height):
        if height is None or snap is None:
            return snap
        key = (id(snap), height)
        if key not in self._cuts:
            keep = snap["xyz"].astype(np.float64) @ self.up <= height
            cut = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(keep) else v)
                   for k, v in snap.items()}
            self._cuts = {k: v for k, v in self._cuts.items() if k[0] == id(snap)}
            self._cuts[key] = cut
        return self._cuts[key]

    def _draw_cameras(self, img, frame_idx, c2w, K, trail, kf_window, thickness):
        run = self.run
        upto = [i for i in sorted(run.est_live) if i <= frame_idx]
        if trail is not None:
            upto = [i for i in upto if i > frame_idx - trail]
        draw_polyline(img, [run.gt[i][:3, 3] for i in upto], c2w, K, COL_GT, thickness)
        draw_polyline(img, [run.est_live[i][:3, 3] for i in upto], c2w, K, COL_EST, thickness)
        # Frustums of the latest keyframes only (like the window shown by
        # the live GUI); all keyframes clutter the view.
        if kf_window:
            recent = [kf for kf in run.keyframes if kf <= frame_idx and kf in run.est_live]
            for kf in recent[-kf_window:]:
                draw_segments(img, frustum_lines(run.est_live[kf], self.intr, self.args.kf_frustum),
                              c2w, K, COL_KF, 1)
        draw_segments(img, frustum_lines(run.gt[frame_idx], self.intr, self.args.frustum), c2w, K, COL_GT,
                      thickness)
        draw_segments(img, frustum_lines(run.est_live[frame_idx], self.intr, self.args.frustum), c2w, K, COL_EST,
                      thickness)

    def map_view(self, frame_idx):
        run = self.run
        est = run.est_live[frame_idx]
        c2w = self.overview if self.view == "overview" else self.follow.update(est)
        snap = self._cut(run.map_at(frame_idx), self.ceiling)
        if snap is None:
            img = np.zeros((self.map_size[1], self.map_size[0], 3), np.uint8)
        else:
            img = self.map_renderer(snap, run.display_colors(snap), c2w, self.K, self.map_size)
        self._draw_cameras(img, frame_idx, c2w, self.K, self.trail, self.args.kf_window, 2)
        n_g = 0 if snap is None else len(snap["xyz"])
        err = np.linalg.norm(est[:3, 3] - run.gt[frame_idx][:3, 3]) * 100
        put_text(img, [run.label, f"frame {frame_idx}  keyframes {run.n_keyframes.get(frame_idx, 0)}",
                       f"Gaussians {n_g}  position error {err:.1f} cm"], scale=0.55)
        return img

    def orbit_view(self, frame_idx, n):
        """The current map from a third-person camera circling the scene."""
        a = self.args
        w, h = self.panel_size
        angle = 2 * math.pi * n / a.orbit_period
        c2w = orbit_pose(self.centre, self.up, a.orbit_distance * self.extent, math.radians(a.orbit_elevation), angle)
        snap = self._cut(self.run.map_at(frame_idx), self.orbit_ceiling)
        if snap is None:
            img = np.empty((h, w, 3), np.uint8)
            img[:] = COL_PANEL_BG
        else:
            img = self.orbit_renderer(snap, self.run.display_colors(snap), c2w, self.K_orbit, (w, h))
        self._draw_cameras(img, frame_idx, c2w, self.K_orbit, None, 0, 1)
        put_text(img, ["3D map (orbit)"], scale=0.45)
        return img

    def panels(self, frame_idx, n):
        run, a = self.run, self.args
        pw = self.panel_size
        depth = run.image(frame_idx, "depth")
        tiles = [
            ("observed", run.image(frame_idx, "obs")),
            ("rendered", run.image(frame_idx, "render")),
            ("albedo" if run.meta["map_color"] == "linear_albedo" else "(no albedo: baked colours)",
             run.image(frame_idx, "albedo")),
            ("rendered depth", None if depth is None else colorize_depth(depth, a.depth_max)),
        ]
        out = []
        for name, im in tiles:
            t = fit(im, pw)
            put_text(t, [name], scale=0.45)
            out.append(t)
        out.append(light_panel(run, run.n_keyframes.get(frame_idx, 0), pw))
        out.append(self.orbit_view(frame_idx, n))
        col1 = np.vstack(out[0::2])
        col2 = np.vstack(out[1::2])
        return np.hstack([col1, col2])

    def __call__(self, frame_idx, n):
        import cv2

        m = self.map_view(frame_idx)
        p = self.panels(frame_idx, n)
        if p.shape[0] != m.shape[0]:
            p = cv2.resize(p, (p.shape[1], m.shape[0]))
        return np.hstack([m, p])


# ---------------------------------------------------------------- output


def cuda_renderer_available():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import diff_gaussian_rasterization  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means "no"
        return False
    return True


def default_output(runs, view):
    """<folder holding the first run's record/>/videos/<run>[__vs__<run>]_<view>.mp4"""
    folder = os.path.join(os.path.dirname(os.path.normpath(runs[0].dir)), "videos")
    name = "__vs__".join(r.meta["run"] for r in runs)
    return os.path.join(folder, f"{name}_{view}.mp4")


def _wsl_to_windows(path):
    """/mnt/d/a/b -> D:\\a\\b (for a Windows program called from WSL)."""
    import re

    m = re.match(r"^/mnt/([a-z])/(.*)$", os.path.abspath(path))
    if not m:
        raise ValueError(f"{path} is not on a Windows drive (/mnt/<drive>/...)")
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")


def find_ffmpeg():
    """(command, path converter) for ffmpeg, or None. Inside WSL without a
    Linux ffmpeg, the Windows ffmpeg.exe on the PATH (WSL interop) is used,
    with the file paths converted to Windows paths."""
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg, (lambda p: p)
    exe = shutil.which("ffmpeg.exe") if os.name != "nt" else None
    if exe is not None:
        return exe, _wsl_to_windows
    return None


def to_h264(path):
    """Re-encode OpenCV's mp4v file to H.264 in place when ffmpeg exists.
    mp4v (MPEG-4 part 2) plays in VLC only: messengers (Messenger, Zalo),
    phones, browsers and PowerPoint need H.264. faststart puts the index
    first so the video starts before it is fully downloaded."""
    import subprocess

    found = find_ffmpeg()
    if found is None:
        print("ffmpeg not found: the video stays mp4v (plays in VLC only)")
        return False
    ffmpeg, conv = found
    tmp = path + ".h264.mp4"
    try:
        src, dst = conv(path), conv(tmp)
    except ValueError as exc:  # e.g. Windows ffmpeg from WSL, file outside /mnt/<drive>
        print(f"{exc}: the video stays mp4v (plays in VLC only)")
        return False
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264",
                    "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
                    dst], check=True)
    os.replace(tmp, path)
    return True


def write_notes(path, runs, args, view, renderer, n_frames):
    """Legend of the video in <video>.txt next to it, instead of on screen."""
    from datetime import datetime

    ceiling, trail = view_defaults(args, view)
    lines = [
        f"Video replay SLAM: {os.path.basename(path)}",
        f"Tạo bởi tools/viz/render_replay.py lúc {datetime.now():%Y-%m-%d %H:%M}; "
        f"{n_frames} khung hình, {args.fps:g} khung hình/giây.",
        "",
        "Các lần chạy (từ trên xuống):",
    ]
    for i, r in enumerate(runs, 1):
        m = r.meta
        color = (f"albedo tuyến tính (mô hình đèn), nhân x{r.color_gain:.2f} để hiển thị"
                 if m["map_color"] == "linear_albedo" else "radiance sRGB (MonoGS gốc: màu đã nung ánh sáng)")
        lines += [f"  {i}. {r.label}",
                  f"     thư mục   : {m['run']}",
                  f"     màu bản đồ: {color}",
                  f"     ghi ảnh mỗi {m['frame_every']} frame, chụp bản đồ mỗi {m['map_every_keyframes']} keyframe"]
    view_txt = {"follow": "camera ảo đi sau và phía trên camera ước lượng (có làm mượt)",
                "overview": "camera cố định phía trên quỹ đạo, nhìn toàn phòng"}[view]
    lines += [
        "",
        f"Khung lớn bên trái: bản đồ Gaussian tại thời điểm hiện tại, góc nhìn '{view}': {view_txt}.",
        "  Đỏ        : quỹ đạo và frustum camera ƯỚC LƯỢNG (pose ngay sau tracking).",
        "  Xanh lá   : quỹ đạo và frustum camera THAM CHIẾU (ground truth, chỉ để so sánh).",
        f"  Xanh dương: {args.kf_window} keyframe gần nhất.",
        "  Chữ       : tên lần chạy; số frame, số keyframe; số Gaussian và sai số vị trí hiện tại (cm),",
        "              tức khoảng cách giữa camera ước lượng và camera tham chiếu.",
        "  Quỹ đạo   : " + (f"{trail} frame gần nhất." if trail is not None else "toàn bộ từ đầu chuỗi."),
    ]
    if ceiling is not None:
        lines.append(f"  Gaussian cao hơn camera cao nhất + {ceiling:g} m bị ẩn (trần nhà che góc nhìn từ trên).")
    lines += [
        "",
        "Các ô bên phải:",
        "  observed       : ảnh đầu vào (rgb) của frame.",
        "  rendered       : ảnh render từ bản đồ tại pose ước lượng (qua mô hình đèn nếu có).",
        "  albedo         : albedo render (chỉ mô hình đèn), chia cho phân vị 99 của từng frame.",
        f"  rendered depth : độ sâu render, thang màu 0-{args.depth_max:g} m (đen = không có).",
        "  learned light  : giản đồ phát sáng Phi(theta) đang học (spline, 0-50 độ) và các tham số đèn",
        "                   học được; 'fixed / none' nếu tham số đèn cố định hoặc không có mô hình đèn.",
        f"  3D map (orbit) : bản đồ hiện tại nhìn từ góc thứ ba quay vòng ({args.orbit_period} khung hình một vòng);",
        f"                   màu đỏ / xanh lá như trên; ẩn Gaussian cao hơn camera cao nhất + {args.orbit_ceiling:g} m.",
        "",
        "Renderer: " + ("rasterizer Gaussian của MonoGS (CUDA)." if renderer == "gaussian" else
                        "xem thử trên CPU: mỗi Gaussian là một đĩa đặc có z-buffer, không trộn độ trong suốt "
                        "(hình lốm đốm hơn render thật)."),
    ]
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render_video(runs, args, view, renderer, out):
    import cv2

    frames = sorted(set.intersection(*[set(r.image_frames) for r in runs]))[:: args.every]
    if args.max_frames is not None:
        frames = [f for f in frames if f <= args.max_frames]
    if args.min_frame is not None:
        frames = [f for f in frames if f >= args.min_frame]
    if not frames:
        raise SystemExit("no common recorded image frame")

    def make_renderer(run):
        if renderer == "gaussian":
            return GaussianRenderer(run.meta["background"])
        bg = np.array(run.meta["background"], np.float32)
        return lambda snap, colors, c2w, K, size: render_points(snap, colors, c2w, K, size, bg)  # noqa: E731

    rows = [RowRenderer(r, args, view, make_renderer) for r in runs]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    gif = out.lower().endswith(".gif")
    writer, gif_frames = None, []
    for n, f in enumerate(frames):
        frame = np.vstack([row(f, n) for row in rows])
        if gif:
            from PIL import Image

            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(small[:, :, ::-1]))
        else:
            if writer is None:
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if n % 50 == 0:
            print(f"  [{view}] frame {f} ({n + 1}/{len(frames)})", flush=True)
    if gif:
        gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:], duration=int(1000 / args.fps), loop=0)
    else:
        writer.release()
        if not args.no_h264:
            to_h264(out)
    write_notes(out, runs, args, view, renderer, len(frames))
    print(f"wrote {out} ({len(frames)} frames) and its notes {os.path.splitext(out)[0]}.txt")


def build_parser():
    """Command-line options (also the defaults of tools/viz/replay_app.py's
    video export)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="run folder (results/<experiment>/<time>) or its record/ folder; repeatable")
    ap.add_argument("--label", action="append", default=[], help="one per --run, in order")
    ap.add_argument("--out", default=None,
                    help=".mp4 or .gif; with several views the view name is appended "
                    "(default: <folder of the record>/videos/<run>_<view>.mp4)")
    ap.add_argument("--view", nargs="+", choices=("follow", "overview"), default=["follow", "overview"])
    ap.add_argument("--renderer", choices=("auto", "gaussian", "points"), default="auto",
                    help="auto: the MonoGS rasterizer when CUDA is available, else points")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--no_h264", action="store_true", help="keep OpenCV's mp4v encoding")
    ap.add_argument("--map_width", type=int, default=960)
    ap.add_argument("--map_height", type=int, default=540)
    ap.add_argument("--panel_width", type=int, default=320)
    ap.add_argument("--fov", type=float, default=70.0, help="horizontal FOV of the map view (deg)")
    ap.add_argument("--back", type=float, default=1.2, help="follow: metres behind the camera")
    ap.add_argument("--height", type=float, default=0.6, help="follow: metres above the camera")
    ap.add_argument("--ahead", type=float, default=1.0, help="follow: look-at point ahead (m)")
    ap.add_argument("--smoothing", type=float, default=0.15, help="follow: 0..1, 1 = no smoothing")
    ap.add_argument("--elevation", type=float, default=55.0, help="overview: degrees above the horizon")
    ap.add_argument("--distance", type=float, default=1.6, help="overview: distance in trajectory extents")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="hide Gaussians higher than the highest camera + this many metres "
                    "(default: 0.3 for overview, off for follow)")
    ap.add_argument("--orbit_period", type=int, default=240, help="orbit panel: frames per turn")
    ap.add_argument("--orbit_elevation", type=float, default=40.0, help="orbit panel: degrees above the horizon")
    ap.add_argument("--orbit_distance", type=float, default=1.8, help="orbit panel: distance in trajectory extents")
    ap.add_argument("--orbit_fov", type=float, default=60.0, help="orbit panel: horizontal FOV (deg)")
    ap.add_argument("--orbit_ceiling", type=float, default=0.3, help="orbit panel: like --ceiling")
    ap.add_argument("--frustum", type=float, default=0.15, help="current frustum depth (m)")
    ap.add_argument("--kf_frustum", type=float, default=0.08, help="keyframe frustum depth (m)")
    ap.add_argument("--kf_window", type=int, default=10, help="draw the frustums of the last n keyframes")
    ap.add_argument("--trail", type=int, default=None,
                    help="draw the trajectories over the last n frames only "
                    "(default: 300 for follow, the whole run for overview)")
    ap.add_argument("--depth_max", type=float, default=5.0, help="depth colour scale (m)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th recorded image frame")
    ap.add_argument("--max_frames", type=int, default=None, help="stop after this frame index")
    ap.add_argument("--min_frame", type=int, default=None, help="start at this frame index")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.label and len(args.label) != len(args.run):
        ap.error("give one --label per --run, or none")
    runs = [Run(d, args.label[i] if args.label else None) for i, d in enumerate(args.run)]
    renderer = args.renderer
    if renderer == "auto":
        renderer = "gaussian" if cuda_renderer_available() else "points"
        print(f"renderer: {renderer}" + ("" if renderer == "gaussian" else " (no CUDA rasterizer: CPU preview)"))
    for view in args.view:
        if args.out is None:
            out = default_output(runs, view)
        elif len(args.view) > 1:
            stem, ext = os.path.splitext(args.out)
            out = f"{stem}_{view}{ext}"
        else:
            out = args.out
        render_video(runs, args, view, renderer, out)


if __name__ == "__main__":
    main()
