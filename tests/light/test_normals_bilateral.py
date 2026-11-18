"""Normals from bilateral-smoothed sensor depth (sensor_depth_bilateral_fd_edge,
docs/DECISIONS.md D146)."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from light_models.normals import (SensorDepthBilateralEdgeAwareFiniteDifference,  # noqa: E402
                                  SensorDepthEdgeAwareFiniteDifference, bilateral_depth)
from light_models.registry import build_component  # noqa: E402

BIL = {"radius_px": 3, "sigma_space_px": 1.5, "sigma_range_rel": 0.06}
CAM = SimpleNamespace(fx=50.0, fy=50.0, cx=19.5, cy=14.5)
H, W = 30, 40


def _block(**over):
    b = {"type": "sensor_depth_bilateral_fd_edge", "stencil_px": 2, "trusted_pixels_only": False,
         "bilateral": dict(BIL)}
    b.update(over)
    return b


def _plane_depth(tilt=0.3, z0=2.0):
    """z of the plane z = z0 + tilt * x seen by CAM (a slanted wall)."""
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64),
                          indexing="ij")
    rx = (u - CAM.cx) / CAM.fx
    return z0 / (1.0 - tilt * rx)


def _normals(src, z):
    pts = torch.zeros(3, H, W, dtype=z.dtype)
    return src._from_sensor(z, CAM, pts, {})


# ---------------------------------------------------------------- config


def test_built_from_the_registry():
    src = build_component("normals", _block(), "Light.cosine.normal_source", {})
    assert isinstance(src, SensorDepthBilateralEdgeAwareFiniteDifference)
    assert src.requires_sensor_depth  # RGB-D only, refused for monocular
    assert src.bilateral == (3, 1.5, 0.06)


@pytest.mark.parametrize("key", sorted(BIL))
def test_every_filter_key_is_required(key):
    bil = {k: v for k, v in BIL.items() if k != key}
    with pytest.raises(KeyError, match=key):
        build_component("normals", _block(bilateral=bil), "x", {})


@pytest.mark.parametrize("bad", [{"radius_px": 0}, {"radius_px": 2.5}, {"sigma_space_px": 0},
                                 {"sigma_range_rel": -0.1}, {"extra": 1}])
def test_bad_filter_values_raise(bad):
    with pytest.raises(ValueError):
        build_component("normals", _block(bilateral={**BIL, **bad}), "x", {})


def test_the_filter_block_is_required():
    b = _block()
    del b["bilateral"]
    with pytest.raises(KeyError):
        build_component("normals", b, "x", {})


# ---------------------------------------------------------------- the filter


def test_constant_depth_is_unchanged_and_holes_stay_holes():
    z = torch.full((H, W), 2.0, dtype=torch.float64)
    z[10:13, 5:9] = 0.0
    out = bilateral_depth(z, 3, 1.5, 0.06)
    assert out[10:13, 5:9].eq(0).all()
    torch.testing.assert_close(out[z > 0], z[z > 0])


def test_holes_do_not_pull_the_neighbours():
    # A hole (0) next to a pixel must not lower its value (log 0 never enters).
    z = torch.full((H, W), 3.0, dtype=torch.float64)
    z[:, 20:] = 0.0
    out = bilateral_depth(z, 3, 1.5, 0.06)
    torch.testing.assert_close(out[:, :20], z[:, :20])


def test_depth_edges_are_kept():
    # A 2 m / 3 m step (50 %, many sigma_range_rel): no mixing across it.
    z = torch.full((H, W), 2.0, dtype=torch.float64)
    z[:, 20:] = 3.0
    out = bilateral_depth(z, 3, 1.5, 0.06)
    assert (out[:, 19] - 2.0).abs().max() < 1e-6
    assert (out[:, 20] - 3.0).abs().max() < 1e-6


def test_noise_is_reduced():
    g = torch.Generator().manual_seed(0)
    z = torch.full((H, W), 2.0, dtype=torch.float64)
    noisy = z + 0.006 * torch.randn(H, W, generator=g, dtype=torch.float64)
    out = bilateral_depth(noisy, 3, 1.5, 0.06)
    inner = (slice(3, -3), slice(3, -3))
    assert (out - z)[inner].std() < 0.35 * (noisy - z)[inner].std()


# ---------------------------------------------------------------- normals


def _angle_deg(n, ref, ok):
    return torch.rad2deg(torch.acos((n * ref).sum(0).clamp(-1, 1)))[ok]


def test_clean_plane_normals_match_the_unfiltered_ones():
    # Inside: the same normals. Within radius + stencil of the image border
    # the window is one-sided, which biases a slanted plane (a few degrees
    # here, on a steep 0.3 slope): kept, documented in bilateral_depth.
    z = _plane_depth()
    ref, ok_ref = _normals(SensorDepthEdgeAwareFiniteDifference(1, False), z)
    n, ok = _normals(build_component("normals", _block(), "x", {}), z)
    border = BIL["radius_px"] + 2
    inner = torch.zeros(H, W, dtype=torch.bool)
    inner[border:-border, border:-border] = True
    assert _angle_deg(n, ref, ok & ok_ref & inner).max() < 0.01
    assert _angle_deg(n, ref, ok & ok_ref).max() < 10.0


def test_noisy_plane_normals_are_much_better_than_unfiltered():
    z = _plane_depth()
    g = torch.Generator().manual_seed(1)
    noisy = z + 0.006 * torch.randn(H, W, generator=g, dtype=torch.float64)
    ref, ok_ref = _normals(SensorDepthEdgeAwareFiniteDifference(1, False), z)
    plain, ok_p = _normals(SensorDepthEdgeAwareFiniteDifference(1, False), noisy)
    filt, ok_f = _normals(build_component("normals", _block(), "x", {}), noisy)
    inner = torch.zeros(H, W, dtype=torch.bool)
    inner[5:-5, 5:-5] = True
    e_plain = _angle_deg(plain, ref, ok_p & ok_ref & inner).median()
    e_filt = _angle_deg(filt, ref, ok_f & ok_ref & inner).median()
    assert e_filt < 0.25 * e_plain
    assert e_filt < 10.0


def test_frame_cache_keys_differ_from_the_unfiltered_source():
    # Both sources share the cache mechanism; their entries must not collide.
    from utils.perf import frame_constant  # noqa: F401  (the key includes the class)

    cam = SimpleNamespace(**vars(CAM), depth=_plane_depth().float(), frame_cache={})
    pts = torch.zeros(3, H, W)
    valid = torch.ones(H, W, dtype=torch.bool)
    a = SensorDepthEdgeAwareFiniteDifference(2, False)(pts, valid, {}, cam)
    b = build_component("normals", _block(), "x", {})(pts, valid, {}, cam)
    assert len(cam.frame_cache) == 2
    assert not math.isclose(float((a[0] - b[0]).abs().sum()), -1.0)
