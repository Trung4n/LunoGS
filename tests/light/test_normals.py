"""light_models/normals.py: depth_fd with a configurable stencil (D40)."""
import pytest

torch = pytest.importorskip("torch")

from light_models.normals import DepthFiniteDifference  # noqa: E402


def _plane(h=30, w=40, noise=0.0, seed=0):
    # Points of the plane z = 2 + 0.3 x - 0.2 y seen by a pinhole camera.
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    xn, yn = (u - w / 2) / 50.0, (v - h / 2) / 50.0
    z = 2.0 / (1.0 - 0.3 * xn + 0.2 * yn)
    if noise:
        z = z + noise * torch.randn(z.shape, generator=torch.Generator().manual_seed(seed), dtype=z.dtype)
    pts = torch.stack([xn * z, yn * z, z])
    n_true = torch.tensor([0.3, -0.2, -1.0], dtype=torch.float64)
    return pts, n_true / n_true.norm()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_plane_normal_exact_for_any_stencil(k):
    pts, n_true = _plane()
    n, ok = DepthFiniteDifference(stencil_px=k)(pts, torch.ones(pts.shape[1:], dtype=torch.bool), None, None)
    torch.testing.assert_close(n[:, ok], n_true[:, None].expand(3, int(ok.sum())))
    # A border of k pixels has no central difference.
    assert not ok[:k].any() and not ok[:, -k:].any() and ok[k:-k, k:-k].all()


def test_wider_stencil_averages_out_bumps():
    pts, n_true = _plane(noise=0.002)
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)

    def mean_angle(k):
        n, ok = DepthFiniteDifference(stencil_px=k)(pts, valid, None, None)
        c = (n[:, ok] * n_true[:, None]).sum(0).clamp(-1, 1)
        return torch.rad2deg(torch.acos(c)).mean()

    assert mean_angle(4) < 0.5 * mean_angle(1)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "2"])
def test_stencil_validated(k):
    with pytest.raises(ValueError, match="stencil_px"):
        DepthFiniteDifference(stencil_px=k)


def _sensor_camera(h=30, w=40, noise=0.0):
    # Camera whose sensor depth is the plane of _plane (same intrinsics:
    # x = (u - w/2) / 50 at pixel offset 0, the data's convention).
    from types import SimpleNamespace

    pts, n_true = _plane(h, w, noise)
    cam = SimpleNamespace(fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, depth=pts[2].numpy().copy())
    return cam, pts, n_true


def test_sensor_depth_normals_come_from_the_sensor_not_the_map():
    from light_models.normals import SensorDepthFiniteDifference

    cam, pts, n_true = _sensor_camera()
    src = SensorDepthFiniteDifference(stencil_px=1, trusted_pixels_only=True)
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    # The map's points are garbage (a staircase): the result must not care.
    stairs = pts.clone()
    stairs[2] = torch.round(stairs[2] * 10) / 10
    n, ok = src(stairs, valid, None, cam)
    torch.testing.assert_close(n[:, ok], n_true[:, None].expand(3, int(ok.sum())))
    assert ok[1:-1, 1:-1].all()


def test_sensor_depth_normals_respect_validity_and_missing_readings():
    from light_models.normals import SensorDepthFiniteDifference

    cam, pts, _ = _sensor_camera()
    cam.depth[10, 10] = 0.0  # no sensor reading
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    valid[20, 20] = False  # not trusted by the G-buffer
    _, ok = SensorDepthFiniteDifference(stencil_px=1, trusted_pixels_only=True)(pts, valid, None, cam)
    assert not ok[10, 10] and not ok[10, 11] and not ok[11, 10]
    assert not ok[20, 20] and ok[20, 21]


def test_sensor_depth_normals_need_a_depth_and_cache_per_camera():
    import pickle

    from light_models.normals import SensorDepthFiniteDifference

    src = SensorDepthFiniteDifference(stencil_px=1, trusted_pixels_only=True)
    cam, pts, _ = _sensor_camera()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    n1, _ = src(pts, valid, None, cam)
    # Another camera (another frame) must not reuse the cached normals.
    cam2, _, _ = _sensor_camera()
    cam2.depth = cam2.depth[:, ::-1].copy()
    n2, _ = src(pts, valid, None, cam2)
    assert not torch.equal(n1, n2)
    # Pickled to other processes without its cache.
    assert pickle.loads(pickle.dumps(src))._cache is None
    cam.depth = None
    with pytest.raises(ValueError, match="sensor depth"):
        src(pts, valid, None, cam)


# depth_fd_edge / sensor_depth_fd_edge: one-sided differences that do not
# cross depth edges (D94).

def _step(h=30, w=40, u_edge=20, near=1.5, far=3.0):
    # Two fronto-parallel planes: z = near left of column u_edge, far right.
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    z = torch.where(u < u_edge, torch.full_like(u, near), torch.full_like(u, far))
    xn, yn = (u - w / 2) / 50.0, (v - h / 2) / 50.0
    return torch.stack([xn * z, yn * z, z])


@pytest.mark.parametrize("k", [1, 2])
def test_edge_aware_matches_central_on_a_plane(k):
    from light_models.normals import EdgeAwareDepthFiniteDifference

    pts, n_true = _plane()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    n, ok = EdgeAwareDepthFiniteDifference(stencil_px=k)(pts, valid, None, None)
    # One-sided differences reach the image border too.
    assert ok.all()
    torch.testing.assert_close(n, n_true[:, None, None].expand_as(n))


def test_edge_aware_does_not_cross_a_depth_edge():
    from light_models.normals import EdgeAwareDepthFiniteDifference

    pts = _step()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    facing = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    n_c, ok_c = DepthFiniteDifference(stencil_px=1)(pts, valid, None, None)
    n_e, ok_e = EdgeAwareDepthFiniteDifference(stencil_px=1)(pts, valid, None, None)
    rows = slice(1, -1)
    for col in (19, 20):  # the two columns on either side of the step
        # central: tilted by the other surface
        assert (n_c[:, rows, col] * facing[:, None]).sum(0).max() < 0.9
        torch.testing.assert_close(n_e[:, rows, col], facing[:, None].expand(3, 28))
    assert ok_c[rows, 19].all() and ok_e[rows, 19].all()


def test_edge_aware_uses_the_valid_side():
    from light_models.normals import EdgeAwareDepthFiniteDifference

    pts, n_true = _plane()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    valid[:, 10] = False  # e.g. no sensor reading
    n_c, ok_c = DepthFiniteDifference(stencil_px=1)(pts, valid, None, None)
    n_e, ok_e = EdgeAwareDepthFiniteDifference(stencil_px=1)(pts, valid, None, None)
    assert not ok_c[5:-5, 9].any() and not ok_c[5:-5, 11].any()
    assert ok_e[:, 9].all() and ok_e[:, 11].all() and not ok_e[:, 10].any()
    torch.testing.assert_close(n_e[:, ok_e], n_true[:, None].expand(3, int(ok_e.sum())))


def test_sensor_edge_source_uses_edge_aware_differences():
    from light_models.normals import EdgeAwareDepthFiniteDifference, SensorDepthEdgeAwareFiniteDifference

    cam, pts, _ = _sensor_camera()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    src = SensorDepthEdgeAwareFiniteDifference(stencil_px=1, trusted_pixels_only=True)
    assert src.requires_sensor_depth
    n, ok = src(pts, valid, None, cam)
    n_ref, ok_ref = EdgeAwareDepthFiniteDifference(stencil_px=1)(pts, valid, None, None)
    assert torch.equal(ok, ok_ref)
    torch.testing.assert_close(n, n_ref)


# trusted_pixels_only (D99): false = the sensor normal does not depend on the
# G-buffer's trust (map opacity), so holes of the map keep their n . l.

@pytest.mark.parametrize("cls_name", ["SensorDepthFiniteDifference", "SensorDepthEdgeAwareFiniteDifference"])
def test_untrusted_pixels_keep_their_sensor_normal(cls_name):
    from light_models import normals

    cam, pts, n_true = _sensor_camera()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    valid[18:22, 18:22] = False  # a low-opacity hole of the map
    n_t, ok_t = getattr(normals, cls_name)(stencil_px=1, trusted_pixels_only=True)(pts, valid, None, cam)
    n_u, ok_u = getattr(normals, cls_name)(stencil_px=1, trusted_pixels_only=False)(pts, valid, None, cam)
    assert not ok_t[18:22, 18:22].any()
    assert ok_u[18:22, 18:22].all()
    torch.testing.assert_close(n_u[:, 18:22, 18:22], n_true[:, None, None].expand(3, 4, 4))
    # elsewhere both agree
    assert torch.equal(ok_t, ok_u & valid)
    torch.testing.assert_close(n_t[:, ok_t], n_u[:, ok_t])


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_trusted_pixels_only_validated(value):
    from light_models.normals import SensorDepthFiniteDifference

    with pytest.raises(ValueError, match="trusted_pixels_only"):
        SensorDepthFiniteDifference(stencil_px=1, trusted_pixels_only=value)
