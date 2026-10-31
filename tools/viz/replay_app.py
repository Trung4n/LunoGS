"""Replay app: browse recorded SLAM runs in the browser (docs/DECISIONS.md D77).

    py tools/viz/replay_app.py record/office0_even_rgbd [more records ...]

Opens a local web page (black UI) to play a run recorded with slam.py
--record back in time: the Gaussian map as it grows from a follow / overview
/ orbiting / free (mouse) camera, the observed, rendered, albedo and depth
images, position error and learned light parameters over time, side by side
comparison of two runs, screenshots and video export.

On Windows the launcher starts the server inside WSL, where the MonoGS env
has CUDA and the Gaussian rasterizer, and opens the browser on Windows. If
that is not possible it runs here with the CPU preview renderer (one opaque
disc per Gaussian). Options: --cpu forces the CPU renderer, --wsl_distro /
--wsl_python select the WSL side, --port the local port.

Display only: nothing here feeds back into SLAM. The reference trajectory
in the recordings is drawn for comparison, as in tools/viz/render_replay.py.
"""
import argparse
import glob
import json
import math
import os
import re
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STATIC = os.path.join(HERE, "replay_app")
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# numpy, OpenCV and render_replay are imported by load_heavy() only when the
# server starts: the Windows launcher needs nothing beyond the standard
# library (it runs the server inside WSL).
np = rr = None


def load_heavy():
    global np, rr
    if rr is None:
        import numpy
        import render_replay

        np, rr = numpy, render_replay

VIEW_MODES = ("follow", "overview", "orbit", "free")
COLOR_MODES = ("color", "keyframe", "opacity", "height")
IMAGE_KINDS = ("obs", "render", "albedo", "depth")
# Largest map image the server renders (pixels), whatever the browser asks.
MAX_RENDER = (1920, 1080)


# ---------------------------------------------------------------- paths


def windows_to_wsl(path):
    """D:\\a\\b -> /mnt/d/a/b (absolute Windows paths only)."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not m:
        return path.replace("\\", "/")
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


def windows_path(path):
    """/mnt/d/a/b -> D:\a\b, for display on Windows (other paths unchanged)."""
    m = re.match(r"^/mnt/([a-z])/(.*)$", path)
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\") if m else path


def local_path(path):
    """A path typed in the UI or given on the command line, usable here (a
    Windows path is converted when the server runs inside WSL/Linux)."""
    path = path.strip().strip('"')
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", path):
        return windows_to_wsl(path)
    return path


# ---------------------------------------------------------------- one run


def _defaults():
    """Display parameters, same defaults as tools/viz/render_replay.py."""
    load_heavy()
    return rr.build_parser().parse_args(["--run", "_"])


class AppRun:
    """A loaded recording plus what the app needs on top of render_replay.Run."""

    def __init__(self, path, label, renderer_kind):
        self.path = os.path.abspath(path)
        self.run = rr.Run(self.path, label)
        r = self.run
        self.renderer_kind = renderer_kind
        self.d = _defaults()
        self.all_frames = sorted(r.est_live)
        gts = [r.gt[i] for i in self.all_frames]
        self.up = rr.world_up(gts)
        pos = np.array([p[:3, 3] for p in gts])
        self.positions = pos
        self.centre = pos.mean(axis=0)
        flat = pos - self.centre
        flat = flat - np.outer(flat @ self.up, self.up)
        self.extent = max(float(np.linalg.norm(flat, axis=1).max()), 0.5)
        self.top = float((pos @ self.up).max())
        self.err_cm = {i: float(np.linalg.norm(r.est_live[i][:3, 3] - r.gt[i][:3, 3]) * 100) for i in self.all_frames}
        follow = rr.FollowCamera(self.up, self.d.back, self.d.height, self.d.ahead, self.d.smoothing)
        self.follow_poses = {i: follow.update(r.est_live[i]) for i in self.all_frames}
        self.overview = rr.overview_pose(pos, self.up, math.radians(self.d.elevation), self.d.distance)
        self.intr = r.meta["intrinsics"]
        self.max_frame = max(self.all_frames)
        self._renderers = {}
        self._cuts = {}
        self._colors = {}
        self._gaussian_counts = None
        self._snap = (None, None)
        self._payloads = {}

    # ---- summary for the UI ----

    def gaussian_counts(self):
        if self._gaussian_counts is None:
            out = []
            for idx, path in self.run.maps:
                with np.load(path) as z:
                    out.append([idx, int(z["xyz"].shape[0])])
            self._gaussian_counts = out
        return self._gaussian_counts

    def light_series(self):
        rows = self.run.light or []
        out = []
        for row in rows:
            knots = sorted((int(k.split("[")[1][:-1]), float(v)) for k, v in row.items()
                           if k.startswith("angular.knot_values["))
            scalars = {k: float(v) for k, v in row.items()
                       if k not in ("frame_idx", "n_keyframes") and not k.startswith("angular.knot_values[")}
            out.append({"frame": int(row["frame_idx"]), "n_keyframes": int(row["n_keyframes"]),
                        # knot i is at (i + 1) degrees, Phi(0) = 1 fixed (angular.Spline)
                        "phi": [1.0] + [v for _, v in knots] if knots else None, "scalars": scalars})
        return out

    def summary(self):
        r = self.run
        eye = self.overview[:3, 3]
        return {
            "label": r.label,
            "run": r.meta["run"],
            "path": self.path,
            "map_color": r.meta["map_color"],
            "color_gain": r.color_gain,
            "frame_every": r.meta["frame_every"],
            "frames": r.image_frames,
            "keyframes": r.keyframes,
            "n_keyframes": {str(f): r.n_keyframes.get(f, 0) for f in r.image_frames},
            "error": [[f, round(self.err_cm[f], 3)] for f in self.all_frames],
            "gaussians": self.gaussian_counts(),
            "light": self.light_series(),
            "up": self.up.tolist(),
            "centre": self.centre.tolist(),
            "extent": self.extent,
            "overview_eye": eye.tolist(),
            "has_albedo": r.meta["map_color"] == "linear_albedo",
            "renderer": self.renderer_kind,
            "top": self.top,
            "intrinsics": self.intr,
            "orbit_distance": self.d.orbit_distance,
            "orbit_elevation": self.d.orbit_elevation,
        }

    # ---- data for the in-browser (WebGL) views ----

    def snapshot(self, k):
        """Map snapshot number k of run.maps (in frame order), cached."""
        path = self.run.maps[k][1]
        if self._snap[0] != path:
            with np.load(path) as z:
                self._snap = (path, {key: z[key] for key in z.files})
        return self._snap[1]

    def points_payload(self, k, mode):
        """Binary point cloud of snapshot k for the WebGL views (Gaussians with
        opacity > 0.05):
          4 x uint32 header (n, k, frame of the snapshot, 0),
          n x 3 float32 position, n float32 footprint radius (sigma),
          n x 3 uint8 colour of the colour mode, n uint8 opacity."""
        key = (k, mode)
        if key not in self._payloads:
            snap = self.snapshot(k)
            op = snap["opacity"].astype(np.float32)
            keep = op > 0.05
            xyz = np.ascontiguousarray(snap["xyz"][keep], dtype=np.float32)
            # Footprint: geometric mean of the two largest scales (a flat
            # Gaussian on a wall is wide in two axes, thin in the third).
            sc = np.sort(snap["scaling"][keep].astype(np.float32), axis=1)
            rad = np.ascontiguousarray(np.sqrt(sc[:, 1] * sc[:, 2]))
            rgb = np.ascontiguousarray((self.colors(snap, mode)[keep] * 255 + 0.5).astype(np.uint8))
            alpha = np.ascontiguousarray((op[keep] * 255 + 0.5).astype(np.uint8))
            head = struct.pack("<4I", len(xyz), k, self.run.maps[k][0], 0)
            if len(self._payloads) >= 8:
                self._payloads.pop(next(iter(self._payloads)))
            self._payloads[key] = head + xyz.tobytes() + rad.tobytes() + rgb.tobytes() + alpha.tobytes()
        return self._payloads[key]

    def traj_payload(self):
        """Binary trajectories: per frame 25 float32 = frame index, the live
        estimate and the reference (camera-to-world 3x4, row-major each)."""
        r = self.run
        rows = [[f, *r.est_live[f][:3].ravel(), *r.gt[f][:3].ravel()] for f in self.all_frames]
        return np.asarray(rows, dtype=np.float32).tobytes()

    # ---- rendering ----

    def _renderer(self, slot):
        if slot not in self._renderers:
            if self.renderer_kind == "gaussian":
                self._renderers[slot] = rr.GaussianRenderer(self.run.meta["background"])
            else:
                bg = np.array(self.run.meta["background"], np.float32)
                self._renderers[slot] = (
                    lambda snap, colors, c2w, K, size: rr.render_points(snap, colors, c2w, K, size, bg))
        return self._renderers[slot]

    def _cut(self, snap, height):
        if snap is None or height is None:
            return snap
        key = (id(snap), height)
        if key not in self._cuts:
            keep = snap["xyz"].astype(np.float64) @ self.up <= height
            cut = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(keep) else v)
                   for k, v in snap.items()}
            # Keep the source alive with its cuts so ids are not reused.
            self._cuts = {k: v for k, v in self._cuts.items() if k[0] == id(snap)}
            self._cuts[key] = (snap, cut)
        return self._cuts[key][1]

    def colors(self, snap, mode):
        """(N,3) display colours in [0, 1], cached per (snapshot, mode) so the
        Gaussian renderer re-uploads them only when they change."""
        key = (id(snap), mode)
        if key in self._colors:
            return self._colors[key][1]
        import cv2

        if mode == "color":
            c = self.run.display_colors(snap)
        elif mode == "keyframe":
            t = np.clip(snap["kf_id"].astype(np.float32) / max(self.max_frame, 1), 0, 1)
            c = cv2.applyColorMap((t * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1] / 255.0
        elif mode == "opacity":
            c = np.repeat(snap["opacity"].astype(np.float32)[:, None], 3, axis=1)
        elif mode == "height":
            hgt = snap["xyz"].astype(np.float64) @ self.up
            lo, hi = np.percentile(hgt, [2, 98]) if len(hgt) else (0.0, 1.0)
            t = np.clip((hgt - lo) / max(hi - lo, 1e-6), 0, 1)
            c = cv2.applyColorMap((t * 255).astype(np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0, ::-1] / 255.0
        else:
            raise ValueError(f"colour mode {mode!r}")
        c = np.ascontiguousarray(c, dtype=np.float32)
        self._colors = {k: v for k, v in self._colors.items() if k[0] == id(snap)}
        self._colors[key] = (snap, c)
        return c

    def camera(self, frame, q):
        """Camera-to-world pose of the map view for the query `q`."""
        mode = q["mode"]
        if mode == "follow":
            return self.follow_poses[frame]
        if mode == "overview":
            return self.overview
        if mode == "orbit":
            return rr.orbit_pose(self.centre, self.up, self.d.orbit_distance * self.extent,
                                 math.radians(self.d.orbit_elevation), q["angle"])
        return rr.look_at(np.array(q["eye"]), np.array(q["target"]), self.up)

    def render(self, frame, q, slot="main"):
        """BGR image of the map at `frame` for the query `q` (mode, size,
        fov, colour, layers, ceiling, trail, eye/target/angle)."""
        w, h = q["w"], q["h"]
        K = rr.intrinsics(w, h, q["fov"])
        c2w = self.camera(frame, q)
        margin = q["ceiling"]
        snap = self._cut(self.run.map_at(frame), None if margin is None else self.top + margin)
        if snap is None or len(snap["xyz"]) == 0:
            img = np.zeros((h, w, 3), np.uint8)
        else:
            img = self._renderer(slot)(snap, self.colors(snap, q["color"]), c2w, K, (w, h))
        r = self.run
        layers = q["layers"]
        upto = [i for i in self.all_frames if i <= frame]
        if q["trail"]:
            upto = [i for i in upto if i > frame - q["trail"]]
        thick = 2 if w >= 640 else 1
        if "gt" in layers:
            rr.draw_polyline(img, [r.gt[i][:3, 3] for i in upto], c2w, K, rr.COL_GT, thick)
        if "est" in layers:
            rr.draw_polyline(img, [r.est_live[i][:3, 3] for i in upto], c2w, K, rr.COL_EST, thick)
        if "kf" in layers:
            recent = [kf for kf in r.keyframes if kf <= frame and kf in r.est_live][-self.d.kf_window:]
            for kf in recent:
                rr.draw_segments(img, rr.frustum_lines(r.est_live[kf], self.intr, self.d.kf_frustum), c2w, K,
                                 rr.COL_KF, 1)
        if "frustum" in layers:
            if "gt" in layers:
                rr.draw_segments(img, rr.frustum_lines(r.gt[frame], self.intr, self.d.frustum), c2w, K, rr.COL_GT,
                                 thick)
            rr.draw_segments(img, rr.frustum_lines(r.est_live[frame], self.intr, self.d.frustum), c2w, K,
                             rr.COL_EST, thick)
        return img, (0 if snap is None else int(len(snap["xyz"])))

    def image(self, frame, kind, depth_max):
        im = self.run.image(frame, kind)
        if im is None:
            return None
        if kind == "depth":
            im = rr.colorize_depth(im, depth_max)
        return im


# ---------------------------------------------------------------- state


class AppState:
    def __init__(self, renderer_kind):
        load_heavy()
        self.renderer_kind = renderer_kind
        self.runs = []
        self.lock = threading.Lock()  # one renderer call at a time (CUDA)
        self.export = {"state": "idle"}
        self.exported = set()

    def remove(self, index):
        """Unloads run `index` (the indices of the later runs shift down)."""
        if not 0 <= index < len(self.runs):
            raise IndexError(f"no run {index}")
        self.runs.pop(index)

    def load(self, path, label=None):
        path = os.path.abspath(local_path(path))
        for i, run in enumerate(self.runs):
            if run.path == path:
                return i
        run = AppRun(path, label, self.renderer_kind)
        self.runs.append(run)
        return len(self.runs) - 1

    def discover(self, extra_roots=()):
        roots = [os.path.join(REPO, "record"), os.path.join(REPO, "results")]
        roots += [os.path.dirname(r.path) for r in self.runs] + [local_path(x) for x in extra_roots]
        found = {}
        for root in roots:
            if not os.path.isdir(root):
                continue
            for pattern in ("meta.json", "*/meta.json", "*/*/meta.json", "*/*/*/meta.json", "*/*/*/*/meta.json"):
                for meta in glob.glob(os.path.join(root, pattern)):
                    rec = os.path.dirname(os.path.abspath(meta))
                    try:
                        with open(meta, encoding="utf-8") as f:
                            name = json.load(f).get("run", os.path.basename(rec))
                    except (OSError, ValueError):
                        continue
                    found[rec] = name
        return [{"path": p, "run": n} for p, n in sorted(found.items())]

    def start_export(self, run_ids, view, start, end):
        if self.export.get("state") == "running":
            raise RuntimeError("an export is already running")
        if view not in ("follow", "overview"):
            raise ValueError("video export supports the follow and overview views")
        runs = [self.runs[i] for i in run_ids]
        args = _defaults()
        args.min_frame, args.max_frames = start, end
        out = rr.default_output([r.run for r in runs], view)
        stem, ext = os.path.splitext(out)
        out = f"{stem}_{start}-{end}{ext}"
        self.export = {"state": "running", "path": out, "started": time.time()}

        def job():
            try:
                with self.lock:
                    fresh = [rr.Run(r.path, r.run.label) for r in runs]
                    rr.render_video(fresh, args, view, self.renderer_kind, out)
                self.exported.add(os.path.abspath(out))
                self.export = {"state": "done", "path": out, "windows_path": windows_path(out),
                               "h264": out_is_h264(out),
                               "seconds": round(time.time() - self.export["started"])}
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                traceback.print_exc()
                self.export = {"state": "error", "message": repr(exc)}

        threading.Thread(target=job, daemon=True).start()
        return out


# ---------------------------------------------------------------- HTTP


def out_is_h264(path):
    """True when the exported video was re-encoded to H.264 (ffmpeg found)."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    return b"avc1" in head or b"avcC" in head


def _encode(img, ext=".jpg", quality=90):
    import cv2

    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("image encoding failed")
    return buf.tobytes()


def parse_render_query(q):
    """Validated render options from the URL query (dict of lists)."""
    get = lambda k, d=None: q.get(k, [d])[0]  # noqa: E731
    mode = get("mode", "follow")
    if mode not in VIEW_MODES:
        raise ValueError(f"mode must be one of {VIEW_MODES}")
    color = get("color", "color")
    if color not in COLOR_MODES:
        raise ValueError(f"color must be one of {COLOR_MODES}")
    w = int(min(max(int(get("w", 960)), 64), MAX_RENDER[0]))
    h = int(min(max(int(get("h", 540)), 36), MAX_RENDER[1]))
    ceiling = get("ceiling", "")
    out = {
        "mode": mode, "color": color, "w": w, "h": h,
        "fov": float(get("fov", 70)),
        "layers": set(filter(None, get("layers", "est,gt,frustum,kf").split(","))),
        "ceiling": None if ceiling in ("", "none") else float(ceiling),
        "trail": int(get("trail", 0)) or None,
        "angle": float(get("angle", 0)),
    }
    if mode == "free":
        for key in ("eye", "target"):
            vals = [float(v) for v in get(key, "").split(",") if v != ""]
            if len(vals) != 3:
                raise ValueError(f"mode=free needs {key}=x,y,z")
            out[key] = vals
    return out


# The browser cancels requests it no longer needs (an image replaced during
# playback or scrubbing): writing the answer then fails. Not an error.
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class Handler(BaseHTTPRequestHandler):
    state = None  # AppState, set by serve()
    server_version = "ReplayApp/1"

    def log_message(self, fmt, *args):  # quiet: only errors are printed
        pass

    def _send(self, code, body, ctype, headers=()):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except CLIENT_GONE:
            self.close_connection = True

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _static(self, name):
        path = os.path.join(STATIC, name)
        if not os.path.isfile(path):
            return self._error(404, "not found")
        ctype = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}.get(
            os.path.splitext(name)[1], "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype + "; charset=utf-8")

    def do_GET(self):  # noqa: N802 - http.server API
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        st = self.state
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path in ("/app.js", "/app.css"):
                return self._static(url.path[1:])
            if url.path == "/api/ping":
                return self._json({"ok": True, "renderer": st.renderer_kind})
            if url.path == "/api/runs":
                return self._json({"renderer": st.renderer_kind, "runs": [r.summary() for r in st.runs]})
            if url.path == "/api/discover":
                return self._json({"records": st.discover(q.get("root", []))})
            if url.path == "/api/export_status":
                return self._json(st.export)
            if url.path == "/api/points":
                run = st.runs[int(q["run"][0])]
                color = q.get("color", ["color"])[0]
                if color not in COLOR_MODES:
                    return self._error(400, f"color must be one of {COLOR_MODES}")
                with st.lock:  # shares the colour cache with the renderer
                    body = run.points_payload(int(q["snap"][0]), color)
                return self._send(200, body, "application/octet-stream")
            if url.path == "/api/traj":
                return self._send(200, st.runs[int(q["run"][0])].traj_payload(), "application/octet-stream")
            if url.path in ("/api/render", "/api/image"):
                run = st.runs[int(q["run"][0])]
                frame = int(q["frame"][0])
                if url.path == "/api/image":
                    kind = q.get("kind", ["obs"])[0]
                    if kind not in IMAGE_KINDS:
                        return self._error(400, f"kind must be one of {IMAGE_KINDS}")
                    if kind != "depth":
                        # Recorded PNG sent as it is (no decode / re-encode).
                        path = os.path.join(run.run.dir, "images", f"{frame:06d}_{kind}.png")
                        if not os.path.isfile(path):
                            return self._error(404, "no image")
                        with open(path, "rb") as f:
                            return self._send(200, f.read(), "image/png")
                    im = run.image(frame, kind, float(q.get("depth_max", [5.0])[0]))
                    if im is None:
                        return self._error(404, "no image")
                    return self._send(200, _encode(im), "image/jpeg")
                opts = parse_render_query(q)
                if st.export.get("state") == "running":
                    return self._error(503, "exporting a video")
                with st.lock:
                    img, n = run.render(frame, opts, slot=q.get("slot", ["main"])[0])
                body = _encode(img, quality=92 if opts["w"] > 1000 else 88)
                return self._send(200, body, "image/jpeg", [("X-Gaussians", str(n))])
            if url.path == "/api/file":
                path = os.path.abspath(q["path"][0])
                if path not in st.exported or not os.path.isfile(path):
                    return self._error(404, "unknown file")
                with open(path, "rb") as f:
                    body = f.read()
                return self._send(200, body, "video/mp4",
                                  [("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')])
            return self._error(404, "not found")
        except CLIENT_GONE:
            self.close_connection = True
            return None
        except (KeyError, ValueError, IndexError) as exc:
            return self._error(400, repr(exc))
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            traceback.print_exc()
            return self._error(500, repr(exc))

    def do_POST(self):  # noqa: N802 - http.server API
        url = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            st = self.state
            if url.path == "/api/load":
                idx = st.load(body["path"], body.get("label") or None)
                return self._json({"index": idx, "run": st.runs[idx].summary()})
            if url.path == "/api/remove":
                st.remove(int(body["index"]))
                return self._json({"runs": len(st.runs)})
            if url.path == "/api/export":
                out = st.start_export([int(i) for i in body["runs"]], body["view"], int(body["start"]),
                                      int(body["end"]))
                return self._json({"path": out})
            if url.path == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return None
            return self._error(404, "not found")
        except CLIENT_GONE:
            self.close_connection = True
            return None
        except (KeyError, ValueError, IndexError, FileNotFoundError, RuntimeError) as exc:
            return self._error(400, repr(exc))
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            traceback.print_exc()
            return self._error(500, repr(exc))


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # socketserver prints every handler exception; a client that went
        # away is expected (see CLIENT_GONE).
        if isinstance(sys.exc_info()[1], CLIENT_GONE):
            return
        super().handle_error(request, client_address)


def free_port(host, port):
    for p in range(port, port + 50):
        with socket.socket() as s:
            if os.name != "nt":
                # Like the server (allow_reuse_address): a port left in
                # TIME_WAIT by the previous run is usable.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise SystemExit(f"no free port from {port}")


# The browser and the launcher use 127.0.0.1, not "localhost": on Windows
# "localhost" tries IPv6 first and every request to the WSL server then
# waits ~0.2 s before falling back to IPv4.


def serve(records, labels, host, port, renderer_kind, open_browser):
    state = AppState(renderer_kind)
    for i, rec in enumerate(records):
        state.load(rec, labels[i] if i < len(labels) else None)
    Handler.state = state
    httpd = AppServer((host, port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Replay app ({renderer_kind} renderer) at {url}  - Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ---------------------------------------------------------------- launcher


def wsl_base(distro):
    return ["wsl"] + (["-d", distro] if distro else [])


def wsl_has_gpu(distro, python):
    """True when the WSL python has CUDA and the Gaussian rasterizer."""
    probe = f"{python} -c 'import torch, diff_gaussian_rasterization; print(torch.cuda.is_available())'"
    try:
        out = subprocess.run(wsl_base(distro) + ["--", "bash", "-lc", probe], capture_output=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and b"True" in out.stdout


def wsl_command(distro, python, records, labels, port):
    """wsl ... command running this app inside WSL (records as WSL paths)."""
    args = ["tools/viz/replay_app.py", "--inside", "--no_browser", "--port", str(port)]
    for i, rec in enumerate(records):
        args.append(windows_to_wsl(os.path.abspath(rec)))
    for lab in labels:
        args += ["--label", lab]
    cmd = f"{python} " + " ".join(shlex.quote(a) for a in args)
    return wsl_base(distro) + ["--cd", REPO, "--", "bash", "-lc", cmd]


def wait_ready(port, proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_in_wsl(args):
    port = free_port("127.0.0.1", args.port)
    cmd = wsl_command(args.wsl_distro, args.wsl_python, args.records, args.label, port)
    print("Starting the GPU renderer in WSL ...", flush=True)
    proc = subprocess.Popen(cmd)
    try:
        if not wait_ready(port, proc, timeout=300):
            print("The WSL server did not start (see the messages above).")
            proc.terminate()
            return False
        url = f"http://127.0.0.1:{port}/"
        print(f"Open {url}  - Ctrl+C here to stop", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/shutdown", data=b"{}", method="POST")
            urllib.request.urlopen(req, timeout=3)
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="*", help="record folders (or run folders holding record/)")
    ap.add_argument("--label", action="append", default=[], help="display name, one per record, in order")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cpu", action="store_true", help="CPU preview renderer here, no WSL / CUDA")
    ap.add_argument("--no_browser", action="store_true", help="do not open the browser")
    ap.add_argument("--wsl_distro", default=None, help="WSL distribution (default: the default one)")
    ap.add_argument("--wsl_python", default="~/micromamba/envs/MonoGS/bin/python",
                    help="python of the MonoGS env inside WSL")
    ap.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)  # set by the launcher
    args = ap.parse_args(argv)

    if os.name == "nt" and not args.cpu and not args.inside:
        print("Checking the GPU renderer in WSL ...", flush=True)
        if wsl_has_gpu(args.wsl_distro, args.wsl_python):
            if run_in_wsl(args):
                return
        print("WSL GPU renderer not available: CPU preview renderer.", flush=True)
    load_heavy()
    kind = "points"
    if not args.cpu and rr.cuda_renderer_available():
        kind = "gaussian"
    # Inside WSL the launcher already chose the port and waits on it.
    port = args.port if args.inside else free_port("127.0.0.1", args.port)
    serve(args.records, args.label, "127.0.0.1", port, kind, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
