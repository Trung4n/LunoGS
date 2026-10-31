"""tools/viz/replay_app.py (docs/DECISIONS.md D77): the replay web app's
server on a fake recording (CPU renderer), its paths and its WSL launcher.

    python -m pytest tests/test_replay_app.py
"""
import importlib.util
import json
import os
import shlex
import threading
import time
import urllib.request

import pytest

pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ra = _load("replay_app", os.path.join(REPO, "tools", "viz", "replay_app.py"))
rec = _load("test_run_recorder_helpers", os.path.join(REPO, "tests", "test_run_recorder.py"))


# ---- paths and launcher ---------------------------------------------------------


def test_path_conversions():
    assert ra.windows_to_wsl(r"D:\CodeForYou\MonoGS\record\x") == "/mnt/d/CodeForYou/MonoGS/record/x"
    assert ra.windows_to_wsl("C:/a b/c") == "/mnt/c/a b/c"
    assert ra.windows_path("/mnt/d/a/b.mp4") == "D:\\a\\b.mp4"
    assert ra.windows_path("/home/pc/x") == "/home/pc/x"
    if os.name != "nt":
        assert ra.local_path(r'"D:\a\b"') == "/mnt/d/a/b"
    else:
        assert ra.local_path(r'"D:\a\b"') == r"D:\a\b"


def test_wsl_command_runs_the_app_inside_with_wsl_paths():
    cmd = ra.wsl_command("Ubuntu-22.04", "~/env/bin/python", [r"D:\rec\a b"], ["my label"], 8766)
    assert cmd[:3] == ["wsl", "-d", "Ubuntu-22.04"] and cmd[3:5] == ["--cd", ra.REPO]
    assert cmd[5:8] == ["--", "bash", "-lc"]
    parts = shlex.split(cmd[8])
    assert parts[0] == os.path.expanduser("~/env/bin/python") or parts[0].endswith("/env/bin/python")
    assert parts[1:6] == ["tools/viz/replay_app.py", "--inside", "--no_browser", "--port", "8766"]
    assert parts[6].endswith("/rec/a b") and parts[-2:] == ["--label", "my label"]
    assert ra.wsl_command(None, "py", [], [], 1)[:2] == ["wsl", "--cd"]


def test_render_query_is_validated():
    q = ra.parse_render_query({"mode": ["orbit"], "w": ["99999"], "h": ["1"], "ceiling": ["none"], "angle": ["1.5"]})
    assert (q["w"], q["h"]) == ra.MAX_RENDER[:1] + (36,) and q["ceiling"] is None and q["angle"] == 1.5
    assert q["layers"] == {"est", "gt", "frustum", "kf"}
    with pytest.raises(ValueError):
        ra.parse_render_query({"mode": ["sideways"]})
    with pytest.raises(ValueError):
        ra.parse_render_query({"color": ["rainbow"]})
    with pytest.raises(ValueError):  # free mode needs the camera
        ra.parse_render_query({"mode": ["free"]})


# ---- server ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    light = rec._record(tmp / "a", with_albedo=True)
    base = rec._record(tmp / "b", with_albedo=False)
    state = ra.AppState("points")
    state.load(str(light), "light")
    ra.Handler.state = state
    httpd = ra.ThreadingHTTPServer(("127.0.0.1", 0), ra.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield url, state, light, base
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_page_and_run_summary(server):
    url, _, _, _ = server
    status, headers, body = _get(url + "/")
    assert status == 200 and b"SLAM Replay" in body
    assert _get(url + "/app.js")[0] == 200 and _get(url + "/app.css")[0] == 200
    runs = json.loads(_get(url + "/api/runs")[2])
    assert runs["renderer"] == "points"
    r = runs["runs"][0]
    assert r["label"] == "light" and r["frames"] == [5, 10, 15, 20] and r["has_albedo"]
    assert len(r["error"]) == 21 and r["error"][10][1] == pytest.approx(2.0, abs=0.01)  # 2 cm of y offset
    assert [g[1] for g in r["gaussians"]] == [100, 118, 200]  # kf 1, kf 10, final


@pytest.mark.parametrize("mode, extra", [
    ("follow", ""), ("overview", "&ceiling=0.3"), ("orbit", "&angle=0.7"),
    ("free", "&eye=0,-1,-1&target=0,0,2.5"),
])
@pytest.mark.parametrize("color", ["color", "keyframe", "opacity", "height"])
def test_render_every_mode_and_colour(server, mode, extra, color):
    url, _, _, _ = server
    status, headers, body = _get(f"{url}/api/render?run=0&frame=10&w=320&h=180&mode={mode}&color={color}{extra}")
    assert status == 200 and headers["Content-Type"] == "image/jpeg"
    img = cv2.imdecode(__import__("numpy").frombuffer(body, "uint8"), cv2.IMREAD_COLOR)
    assert img.shape == (180, 320, 3)
    # the latest snapshot at frame 10 is the initial keyframe's (kf 10 is at frame 18);
    # the ceiling cut hides the ones above the highest camera + 0.3 m
    n = int(headers["X-Gaussians"])
    assert (0 < n < 100) if "ceiling" in extra else n == 100


def test_images_are_served(server):
    url, _, _, _ = server
    status, headers, body = _get(f"{url}/api/image?run=0&frame=10&kind=obs")
    assert status == 200 and headers["Content-Type"] == "image/png" and body[:4] == b"\x89PNG"
    assert _get(f"{url}/api/image?run=0&frame=10&kind=depth")[1]["Content-Type"] == "image/jpeg"
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{url}/api/image?run=0&frame=11&kind=obs")  # no image at frame 11
    assert e.value.code == 404


def test_bad_requests_get_400(server):
    url, _, _, _ = server
    for q in ("run=0&frame=10&mode=sideways", "run=7&frame=10", "frame=10"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(f"{url}/api/render?{q}")
        assert e.value.code == 400


def test_load_a_second_run_and_export_a_video(server):
    url, state, _, base = server
    out = _post(url + "/api/load", {"path": str(base / "record"), "label": "MonoGS"})
    assert out["index"] == 1 and out["run"]["has_albedo"] is False
    assert _post(url + "/api/load", {"path": str(base / "record")})["index"] == 1  # not loaded twice
    disc = json.loads(_get(url + "/api/discover")[2])["records"]
    assert any(d["path"].endswith(os.path.join("b", "rgbd_nocos__office0_lit", "2026-10-28", "record")) for d in disc)
    path = _post(url + "/api/export", {"runs": [0, 1], "view": "overview", "start": 5, "end": 15})["path"]
    for _ in range(200):
        st = json.loads(_get(url + "/api/export_status")[2])
        if st["state"] != "running":
            break
        time.sleep(0.1)
    assert st["state"] == "done", st
    assert os.path.isfile(path) and path.endswith("_overview_5-15.mp4")
    assert os.path.isfile(os.path.splitext(path)[0] + ".txt")
    status, headers, body = _get(f"{url}/api/file?path={urllib.parse.quote(path)}")
    assert status == 200 and len(body) == os.path.getsize(path)
    with pytest.raises(urllib.error.HTTPError):
        _get(f"{url}/api/file?path={urllib.parse.quote(__file__)}")  # only exported videos
    with pytest.raises(urllib.error.HTTPError):
        _post(url + "/api/export", {"runs": [0], "view": "free", "start": 5, "end": 15})


def test_points_and_trajectory_payloads_for_webgl(server):
    import numpy as np

    url, state, _, _ = server
    _, _, body = _get(f"{url}/api/points?run=0&snap=1&color=keyframe")
    n, k, frame, _ = np.frombuffer(body[:16], "<u4")
    assert (k, frame) == (1, 18) and n == 118  # all fake Gaussians are opaque
    assert len(body) == 16 + 20 * n
    xyz = np.frombuffer(body, "<f4", 3 * n, 16).reshape(n, 3)
    rad = np.frombuffer(body, "<f4", n, 16 + 12 * n)
    op = np.frombuffer(body, "u1", n, 16 + 19 * n)
    snap = state.runs[0].snapshot(1)
    np.testing.assert_allclose(xyz, snap["xyz"], rtol=1e-6)
    np.testing.assert_allclose(rad, 0.01, rtol=1e-2)  # fake scales are all 0.01
    assert (op == 255).all()
    with pytest.raises(urllib.error.HTTPError):
        _get(f"{url}/api/points?run=0&snap=1&color=rainbow")
    _, _, traj = _get(f"{url}/api/traj?run=0")
    t = np.frombuffer(traj, "<f4").reshape(-1, 25)
    assert t.shape[0] == 21 and t[10, 0] == 10
    assert t[10, 4] == pytest.approx(0.1) and t[10, 13 + 7] == pytest.approx(0.02)  # est x, gt y


def test_remove_a_run(server):
    url, state, light, base = server
    n0 = len(state.runs)
    idx = _post(url + "/api/load", {"path": str(base / "record"), "label": "to remove"})["index"]
    assert _post(url + "/api/remove", {"index": idx})["runs"] == n0 - (idx < n0)
    assert all(r.path != os.path.abspath(str(base / "record")) for r in state.runs)
    with pytest.raises(urllib.error.HTTPError):
        _post(url + "/api/remove", {"index": 99})


def test_cancelled_request_is_not_an_error(capsys):
    # The browser cancels image requests during playback: the write fails
    # with BrokenPipeError, which must not print a traceback or retry a 500.
    class Gone:
        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

    h = ra.Handler.__new__(ra.Handler)
    h.wfile, h.request_version, h.requestline, h.command = Gone(), "HTTP/1.1", "GET /api/image HTTP/1.1", "GET"
    h._headers_buffer = []
    h.close_connection = False
    h._send(200, b"x" * 10, "image/png")
    assert h.close_connection is True
    assert capsys.readouterr().err == ""
