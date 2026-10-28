"""Replay video of recorded SLAM runs (docs/DECISIONS.md D76).

Reads <run>/record/ written by slam.py --record (utils/run_recorder.py) and
draws, frame by frame, what the live GUI would have shown:

  left   the Gaussian map as it grows, seen from a virtual camera, with the
         estimated trajectory (red), the reference trajectory (green), the
         current camera frustums and the keyframes (blue);
  right  observed image, rendered image, rendered albedo (light model),
         rendered depth, the learned emission profile (when the angular
         profile is learned) and a text panel (frame, keyframes, Gaussians,
         live position error).

Several runs (e.g. original MonoGS on `lit` vs the light model) are stacked
vertically and synchronised on the frame index.

    python tools/viz/render_replay.py --run results/rgbd_nocos__office0_lit/<time> \
        --run results/office0__office0_lit/<time> --label "light model" \
        --label "MonoGS" --out office0.mp4

The map view uses the MonoGS rasterizer (needs CUDA and the built
submodules); --renderer points draws the Gaussian centres with numpy instead
(no GPU, for a quick look). Display only: nothing here feeds back into SLAM.
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
        self.dir = os.path.join(run_dir, "record")
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"{run_dir}: no record/ folder (run slam.py with --record)")
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
        self.light = _read_light(os.path.join(run_dir, "light_learned.csv"))
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


def draw_polyline(img, points, c2w, K, color, thickness):
    import cv2

    if len(points) < 2:
        return
    uv, front = project(np.asarray(points), c2w, K)
    for i in range(len(uv) - 1):
        if front[i] and front[i + 1]:
            a, b = uv[i], uv[i + 1]
            if np.all(np.abs(np.concatenate([a, b])) < 1e5):
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thickness, cv2.LINE_AA)


def draw_segments(img, segs, c2w, K, color, thickness):
    for a, b in segs:
        draw_polyline(img, [a, b], c2w, K, color, thickness)


def render_points(snap, colors, c2w, K, size, background):
    """CPU fallback: Gaussian centres splatted as 2x2 points, far to near."""
    w, h = size
    img = np.empty((h, w, 3), np.float32)
    img[:] = background
    keep = snap["opacity"].astype(np.float32) > 0.5
    uv, front = project(snap["xyz"][keep].astype(np.float64), c2w, K)
    c = colors[keep]
    w2c = np.linalg.inv(c2w)
    z = (snap["xyz"][keep] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    order = np.argsort(-z[ok])
    u = uv[ok][order].astype(int)
    col = c[ok][order]
    for dy in (0, 1):
        for dx in (0, 1):
            img[u[:, 1] + dy, u[:, 0] + dx] = col
    return (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB -> BGR


class GaussianRenderer:
    """Map view through the MonoGS rasterizer, colours precomputed."""

    def __init__(self, background):
        import torch
        from types import SimpleNamespace

        from gaussian_splatting.gaussian_renderer import render
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from utils.camera_utils import Camera

        self.torch, self.render, self.GaussianModel, self.Camera = torch, render, GaussianModel, Camera
        self.pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        self.bg = torch.tensor(background, dtype=torch.float32, device="cuda")
        self._snap_id, self._model = None, None

    def _model_for(self, snap, colors):
        torch = self.torch
        if self._snap_id is not id(snap):
            m = self.GaussianModel(0)
            dev = "cuda"
            f = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)  # noqa: E731
            m._xyz = f(snap["xyz"])
            m._scaling = torch.log(f(snap["scaling"]).clamp_min(1e-8))
            m._rotation = f(snap["rotation"])
            op = f(snap["opacity"]).clamp(1e-6, 1 - 1e-6)[:, None]
            m._opacity = torch.log(op / (1 - op))
            n = m._xyz.shape[0]
            m._features_dc = torch.zeros(n, 1, 3, device=dev)
            m._features_rest = torch.zeros(n, 0, 3, device=dev)
            self._model, self._colors, self._snap_id = m, f(colors), id(snap)
        return self._model, self._colors

    def __call__(self, snap, colors, c2w, K, size):
        torch = self.torch
        w, h = size
        model, col = self._model_for(snap, colors)
        w2c = np.linalg.inv(c2w)
        fovx = 2 * math.atan(w / (2 * K[0, 0]))
        fovy = 2 * math.atan(h / (2 * K[1, 1]))
        cam = self.Camera.init_from_gui(0, torch.eye(4, device="cuda"), fovx, fovy,
                                        K[0, 0], K[1, 1], K[0, 2], K[1, 2], h, w)
        cam.update_RT(torch.tensor(w2c[:3, :3], dtype=torch.float32, device="cuda"),
                      torch.tensor(w2c[:3, 3], dtype=torch.float32, device="cuda"))
        with torch.no_grad():
            out = self.render(cam, model, self.pipe, self.bg, override_color=col, shader=None)["render"]
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
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
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


class RowRenderer:
    """One run's row of the video: map view plus the side panels."""

    def __init__(self, run, args, map_renderer):
        self.run, self.args, self.map_renderer = run, args, map_renderer
        self.map_size = (args.map_width, args.map_height)
        self.K = intrinsics(args.map_width, args.map_height, args.fov)
        gts = [run.gt[i] for i in sorted(run.gt)]
        self.up = world_up(gts)
        positions = np.array([p[:3, 3] for p in gts])
        self.overview = overview_pose(positions, self.up, math.radians(args.elevation), args.distance)
        self.follow = FollowCamera(self.up, args.back, args.height, args.ahead, args.smoothing)
        self.intr = run.meta["intrinsics"]
        # Height above which Gaussians are hidden (the ceiling blocks a view
        # from above); None = keep everything.
        self.ceiling = None
        if args.ceiling is not None:
            self.ceiling = float((positions @ self.up).max()) + args.ceiling
        self._cut_cache = (None, None)

    def _cut(self, snap):
        if self.ceiling is None or snap is None:
            return snap
        if self._cut_cache[0] is not snap:
            keep = snap["xyz"].astype(np.float64) @ self.up <= self.ceiling
            cut = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(keep) else v)
                   for k, v in snap.items()}
            self._cut_cache = (snap, cut)
        return self._cut_cache[1]

    def map_view(self, frame_idx):
        run = self.run
        est = run.est_live[frame_idx]
        c2w = self.overview if self.args.view == "overview" else self.follow.update(est)
        snap = self._cut(run.map_at(frame_idx))
        if snap is None:
            img = np.zeros((self.map_size[1], self.map_size[0], 3), np.uint8)
        else:
            img = self.map_renderer(snap, run.display_colors(snap), c2w, self.K, self.map_size)
        upto = [i for i in sorted(run.est_live) if i <= frame_idx]
        draw_polyline(img, [run.gt[i][:3, 3] for i in upto], c2w, self.K, COL_GT, 2)
        draw_polyline(img, [run.est_live[i][:3, 3] for i in upto], c2w, self.K, COL_EST, 2)
        # Frustums of the latest keyframes only (like the window shown by
        # the live GUI); all keyframes clutter the view.
        recent = [kf for kf in run.keyframes if kf <= frame_idx and kf in run.est_live]
        for kf in recent[-self.args.kf_window:]:
            draw_segments(img, frustum_lines(run.est_live[kf], self.intr, self.args.kf_frustum),
                          c2w, self.K, COL_KF, 1)
        draw_segments(img, frustum_lines(run.gt[frame_idx], self.intr, self.args.frustum), c2w, self.K, COL_GT, 2)
        draw_segments(img, frustum_lines(est, self.intr, self.args.frustum), c2w, self.K, COL_EST, 2)
        n_g = 0 if snap is None else len(snap["xyz"])
        err = np.linalg.norm(est[:3, 3] - run.gt[frame_idx][:3, 3]) * 100
        put_text(img, [run.label, f"frame {frame_idx}  keyframes {run.n_keyframes.get(frame_idx, 0)}",
                       f"Gaussians {n_g}  position error {err:.1f} cm",
                       "red: estimate  green: reference  blue: last keyframes"], scale=0.55)
        return img

    def panels(self, frame_idx):
        run, a = self.run, self.args
        pw = (a.panel_width, a.map_height // 3)
        tiles = [
            ("observed", run.image(frame_idx, "obs")),
            ("rendered", run.image(frame_idx, "render")),
            ("albedo" if run.meta["map_color"] == "linear_albedo" else "(no albedo: baked colours)",
             run.image(frame_idx, "albedo")),
            ("rendered depth", None),
        ]
        depth = run.image(frame_idx, "depth")
        tiles[3] = ("rendered depth", None if depth is None else colorize_depth(depth, a.depth_max))
        out = []
        for name, im in tiles:
            t = fit(im, pw)
            put_text(t, [name], scale=0.45)
            out.append(t)
        out.append(light_panel(run, run.n_keyframes.get(frame_idx, 0), pw))
        info = np.empty((pw[1], pw[0], 3), np.uint8)
        info[:] = COL_PANEL_BG
        put_text(info, [f"map colours: {run.meta['map_color']}",
                        f"map gain x{run.color_gain:.2f}" if run.meta["map_color"] == "linear_albedo" else "",
                        f"view: {a.view}"], scale=0.45)
        out.append(info)
        col1 = np.vstack(out[0::2])
        col2 = np.vstack(out[1::2])
        return np.hstack([col1, col2])

    def __call__(self, frame_idx):
        import cv2

        m = self.map_view(frame_idx)
        p = self.panels(frame_idx)
        if p.shape[0] != m.shape[0]:
            p = cv2.resize(p, (p.shape[1], m.shape[0]))
        return np.hstack([m, p])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="results/<experiment>/<time> (repeatable)")
    ap.add_argument("--label", action="append", default=[], help="one per --run, in order")
    ap.add_argument("--out", required=True, help=".mp4 (or .gif)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--view", choices=("follow", "overview"), default="follow")
    ap.add_argument("--renderer", choices=("gaussian", "points"), default="gaussian")
    ap.add_argument("--map_width", type=int, default=960)
    ap.add_argument("--map_height", type=int, default=540)
    ap.add_argument("--panel_width", type=int, default=320)
    ap.add_argument("--fov", type=float, default=70.0, help="horizontal FOV of the map view (deg)")
    ap.add_argument("--back", type=float, default=1.2, help="follow: metres behind the camera")
    ap.add_argument("--height", type=float, default=0.6, help="follow: metres above the camera")
    ap.add_argument("--ahead", type=float, default=1.0, help="follow: look-at point ahead (m)")
    ap.add_argument("--smoothing", type=float, default=0.15, help="follow: 0..1, 1 = no smoothing")
    ap.add_argument("--elevation", type=float, default=55.0, help="overview: degrees above the horizon")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="hide Gaussians higher than the highest camera + this many metres "
                    "(default: 0.3 for --view overview, off for follow)")
    ap.add_argument("--distance", type=float, default=1.6, help="overview: distance in trajectory extents")
    ap.add_argument("--frustum", type=float, default=0.15, help="current frustum depth (m)")
    ap.add_argument("--kf_frustum", type=float, default=0.08, help="keyframe frustum depth (m)")
    ap.add_argument("--kf_window", type=int, default=10, help="draw the frustums of the last n keyframes")
    ap.add_argument("--depth_max", type=float, default=5.0, help="depth colour scale (m)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th recorded image frame")
    ap.add_argument("--max_frames", type=int, default=None, help="stop after this frame index")
    args = ap.parse_args(argv)
    if args.ceiling is None and args.view == "overview":
        args.ceiling = 0.3
    import cv2

    if args.label and len(args.label) != len(args.run):
        ap.error("give one --label per --run, or none")
    runs = [Run(d, args.label[i] if args.label else None) for i, d in enumerate(args.run)]
    frames = sorted(set.intersection(*[set(r.image_frames) for r in runs]))[:: args.every]
    if args.max_frames is not None:
        frames = [f for f in frames if f <= args.max_frames]
    if not frames:
        raise SystemExit("no common recorded image frame")
    def map_renderer(run):
        # One per run: the renderer caches the uploaded map snapshot.
        if args.renderer == "gaussian":
            return GaussianRenderer(run.meta["background"])
        bg = np.array(run.meta["background"], np.float32)
        return lambda snap, colors, c2w, K, size: render_points(snap, colors, c2w, K, size, bg)  # noqa: E731

    rows = [RowRenderer(r, args, map_renderer(r)) for r in runs]

    gif = args.out.lower().endswith(".gif")
    writer, gif_frames = None, []
    for n, f in enumerate(frames):
        frame = np.vstack([row(f) for row in rows])
        if gif:
            from PIL import Image

            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(small[:, :, ::-1]))
        else:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if n % 50 == 0:
            print(f"frame {f} ({n + 1}/{len(frames)})", flush=True)
    if gif:
        gif_frames[0].save(args.out, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / args.fps), loop=0)
    else:
        writer.release()
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
