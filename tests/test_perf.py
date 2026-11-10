"""Speed changes of docs/DECISIONS.md D128 must not change any value: each new
path is checked against the computation it replaced."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gaussian_splatting.utils.graphics_utils import getWorld2View2  # noqa: E402
from light_models.device import on_device  # noqa: E402
from light_models.gbuffer import _lower_median, surface_points  # noqa: E402
from utils import pose_utils  # noqa: E402
from utils.camera_utils import Camera  # noqa: E402
from utils.perf import FramePrefetcher, frame_constant  # noqa: E402


def _rotation(axis, deg, dtype=torch.float32):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    th = math.radians(deg)
    k = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    return (torch.eye(3, dtype=torch.float64) + math.sin(th) * k + (1 - math.cos(th)) * k @ k).to(dtype)


def _camera(r, t):
    proj = torch.eye(4)
    proj[2, 3] = 1.0
    cam = Camera(0, None, None, torch.eye(4), proj, 100.0, 100.0, 32.0, 24.0, 1.0, 0.8, 48, 64, device="cpu")
    cam.update_RT(r, t)
    return cam


def _old_world_to_view(r, t):
    rt = torch.zeros((4, 4))
    rt[:3, :3] = r
    rt[:3, 3] = t
    rt[3, 3] = 1.0
    return torch.linalg.inv(torch.linalg.inv(rt))


# ---------------------------------------------------------------- camera matrices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_camera_matrices_match_the_inverse_based_formulas(dtype):
    r = _rotation([1, 2, 3], 40, dtype)
    t = torch.tensor([0.3, -1.2, 2.5], dtype=dtype)
    cam = _camera(r, t)
    wv_old = _old_world_to_view(r, t).transpose(0, 1)
    fp_old = wv_old.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0)).squeeze(0)
    cc_old = wv_old.inverse()[3, :3]
    # float32 whatever the pose's dtype: the first frame gets the dataset's
    # float64 ground-truth pose, and the rasterizer only takes float32.
    for got in (cam.world_view_transform, cam.full_proj_transform, cam.camera_center):
        assert got.dtype == torch.float32
    torch.testing.assert_close(cam.world_view_transform, wv_old, atol=1e-6, rtol=0)
    torch.testing.assert_close(cam.full_proj_transform, fp_old, atol=1e-6, rtol=0)
    torch.testing.assert_close(cam.camera_center, cc_old, atol=1e-5, rtol=0)


def test_camera_matrix_cache_follows_the_pose():
    cam = _camera(_rotation([0, 0, 1], 10), torch.tensor([1.0, 0.0, 0.0]))
    first = cam.world_view_transform
    assert cam.world_view_transform is first  # cached
    cam.update_RT(_rotation([0, 1, 0], 25), torch.tensor([0.0, 2.0, 0.0]))
    torch.testing.assert_close(cam.camera_center, _old_world_to_view(cam.R, cam.T).transpose(0, 1).inverse()[3, :3],
                               atol=1e-5, rtol=0)
    with torch.no_grad():
        cam.T.add_(1.0)  # in place: version counter
    torch.testing.assert_close(cam.world_view_transform[3, :3], cam.T)


def test_world_to_view_with_translate_and_scale_keeps_the_original_path():
    r, t = _rotation([1, 0, 0], 30), torch.tensor([0.5, 0.2, -0.1])
    out = getWorld2View2(r, t, translate=torch.tensor([1.0, 2.0, 3.0]), scale=2.0)
    rt = torch.eye(4)
    rt[:3, :3], rt[:3, 3] = r, t
    c2w = torch.linalg.inv(rt)
    c2w[:3, 3] = (c2w[:3, 3] + torch.tensor([1.0, 2.0, 3.0])) * 2.0
    torch.testing.assert_close(out, torch.linalg.inv(c2w))


# ---------------------------------------------------------------- SE(3)


def _old_so3(theta):
    w = pose_utils.skew_sym_mat(theta)
    w2 = w @ w
    angle = torch.norm(theta)
    i = torch.eye(3, dtype=theta.dtype)
    if angle < 1e-5:
        return i + w + 0.5 * w2
    return i + (torch.sin(angle) / angle) * w + ((1 - torch.cos(angle)) / (angle**2)) * w2


def _old_v(theta):
    w = pose_utils.skew_sym_mat(theta)
    w2 = w @ w
    angle = torch.norm(theta)
    i = torch.eye(3, dtype=theta.dtype)
    if angle < 1e-5:
        return i + 0.5 * w + (1.0 / 6.0) * w2
    return i + w * ((1.0 - torch.cos(angle)) / (angle**2)) + w2 * ((angle - torch.sin(angle)) / (angle**3))


@pytest.mark.parametrize("theta", [[0.0, 0.0, 0.0], [1e-7, -2e-7, 3e-7], [0.01, -0.02, 0.03], [0.4, 0.1, -0.7]])
def test_so3_and_v_match_the_branching_formulas(theta):
    th = torch.tensor(theta)
    torch.testing.assert_close(pose_utils.SO3_exp(th), _old_so3(th), atol=0, rtol=0)
    torch.testing.assert_close(pose_utils.V(th), _old_v(th), atol=0, rtol=0)


def test_skew_matrix():
    x = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([-0.5, 0.25, 2.0])
    torch.testing.assert_close(pose_utils.skew_sym_mat(x) @ v, torch.linalg.cross(x, v))


@pytest.mark.parametrize("start_dtype", [torch.float32, torch.float64])
def test_host_pose_update_matches_the_gpu_one(start_dtype):
    r, t = _rotation([1, -1, 2], 20, start_dtype), torch.tensor([0.1, 0.2, 0.3], dtype=start_dtype)
    a, b = _camera(r, t), _camera(r.clone(), t.clone())
    for cam in (a, b):
        cam.cam_trans_delta.data = torch.tensor([1e-3, -2e-3, 5e-4])
        cam.cam_rot_delta.data = torch.tensor([2e-3, 1e-3, -3e-3])
    conv_a = pose_utils.update_pose(a)
    conv_b = pose_utils.update_pose_host(b)
    assert bool(conv_a) == conv_b is False
    assert b.R.dtype == b.T.dtype == torch.float32
    torch.testing.assert_close(b.R, a.R, atol=1e-6, rtol=0)
    torch.testing.assert_close(b.T, a.T, atol=1e-6, rtol=0)
    assert b.cam_rot_delta.abs().sum() == 0 and b.cam_trans_delta.abs().sum() == 0
    assert pose_utils.update_pose_host(b) is True  # zero step: converged


# ---------------------------------------------------------------- motion model


def _frontend(model, cameras):
    from utils.slam_frontend import FrontEnd

    fe = object.__new__(FrontEnd)
    fe.cameras, fe.use_every_n_frames, fe.motion_model = cameras, 1, model
    return fe


def test_constant_velocity_extrapolates_a_constant_motion():
    step_r, step_t = _rotation([0, 1, 0.2], 3), torch.tensor([0.02, 0.0, -0.01])
    r0, t0 = _rotation([1, 0, 0], 10, torch.float64), torch.tensor([0.5, 0.1, 0.0], dtype=torch.float64)
    poses = [(r0.float(), t0.float())]
    for _ in range(3):
        r, t = poses[-1]
        poses.append((step_r @ r, step_r @ t + step_t))
    cams = {0: _camera(r0, t0), 1: _camera(*poses[1]), 2: _camera(*poses[2])}  # frame 0 in float64
    r_pred, t_pred = _frontend("constant_velocity", cams).initial_pose(3)
    torch.testing.assert_close(r_pred, poses[3][0], atol=1e-5, rtol=0)
    torch.testing.assert_close(t_pred, poses[3][1], atol=1e-5, rtol=0)
    r_none, t_none = _frontend("none", cams).initial_pose(3)
    assert r_none is cams[2].R and t_none is cams[2].T
    r_first, _ = _frontend("constant_velocity", {0: cams[0]}).initial_pose(1)
    assert r_first is cams[0].R  # one previous frame only


# ---------------------------------------------------------------- device copies, frame constants


def test_on_device_reuses_the_copy_until_the_tensor_changes():
    t = torch.tensor([1.0, 2.0])
    a = on_device(t, torch.device("cpu"), torch.float64)
    assert on_device(t, torch.device("cpu"), torch.float64) is a
    t.copy_(torch.tensor([3.0, 4.0]))
    b = on_device(t, torch.device("cpu"), torch.float64)
    assert b is not a and b.tolist() == [3.0, 4.0]
    assert on_device(t, torch.device("cpu"), torch.float32) is t


def test_on_device_keeps_the_gradient_of_a_learned_parameter():
    leaf = torch.tensor([2.0], requires_grad=True)
    (on_device(leaf, torch.device("cpu"), torch.float64) * 3).sum().backward()
    assert leaf.grad.item() == 3.0


def test_frame_constant_builds_once_per_frame_only_when_enabled():
    calls = []
    cam = SimpleNamespace(frame_cache=None)
    for _ in range(2):
        frame_constant(cam, "k", lambda: calls.append(1) or len(calls))
    assert len(calls) == 2
    cam.frame_cache = {}
    assert frame_constant(cam, "k", lambda: calls.append(1) or len(calls)) == 3
    assert frame_constant(cam, "k", lambda: calls.append(1) or len(calls)) == 3


def test_prefetcher_returns_the_frame_asked():
    class Data:
        def __len__(self):
            return 5

        def __getitem__(self, i):
            return ("frame", i)

    pf = FramePrefetcher(Data())
    assert [pf.get(i) for i in (0, 1, 1, 3, 2, 4)] == [("frame", i) for i in (0, 1, 1, 3, 2, 4)]
    pf.close()


# ---------------------------------------------------------------- G-buffer


@pytest.mark.parametrize("n", [1, 2, 7, 10])
def test_lower_median_matches_torch_median(n):
    values = torch.rand(4, 5)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[torch.randperm(20)[:n]] = True
    mask = mask.view(4, 5)
    assert _lower_median(values, mask).item() == values[mask].median().item()
    assert torch.isnan(_lower_median(values, torch.zeros_like(mask)))


def test_surface_points_with_no_coverage_stay_finite():
    cam = SimpleNamespace(fx=10.0, fy=10.0, cx=2.0, cy=2.0)
    g = {"depth": torch.zeros(1, 4, 4), "opacity": torch.zeros(1, 4, 4), "pixel_offset": 0.5}
    points, valid = surface_points(g, cam, 0.5)
    assert torch.isfinite(points).all() and not valid.any()
    g = {"depth": torch.full((1, 4, 4), 0.2), "opacity": torch.full((1, 4, 4), 0.1), "pixel_offset": 0.5}
    points, valid = surface_points(g, cam, 0.5)  # covered, none trusted: z = depth / opacity
    torch.testing.assert_close(points[2], torch.full((4, 4), 2.0))
