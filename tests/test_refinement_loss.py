"""utils/slam_utils.get_loss_refinement (docs/DECISIONS.md D87): the offline
color refinement uses the mapping weight; without any extension mask it is
the baseline L1 + D-SSIM. Needs CUDA (the masks read the observed image on
the GPU)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from gaussian_splatting.utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.slam_utils import get_loss_refinement  # noqa: E402

LAMBDA_DSSIM = 0.2


def _config(saturation_enabled):
    return {
        "Light": {"enabled": True},
        "LightTracking": {
            "loss_color_space": "srgb",
            "exposure_affine": False,
            "pixel_weight": {"enabled": False},
            "saturation_mask": (
                {"enabled": True, "mode": "pixel", "threshold_8bit": 250, "apply_to": ["tracking", "mapping"]}
                if saturation_enabled else {"enabled": False}
            ),
        },
    }


def _frame():
    g = torch.Generator().manual_seed(0)
    observed = torch.randint(0, 240, (3, 24, 24), generator=g).float() / 255.0
    observed[:, :8, :8] = 1.0  # a saturated patch (codes 255)
    render = torch.rand(3, 24, 24, generator=g)
    return SimpleNamespace(original_image=observed.cuda()), {"render": render.cuda()}


def test_without_masks_it_is_the_baseline_loss():
    vp, pkg = _frame()
    expected = (1 - LAMBDA_DSSIM) * l1_loss(pkg["render"], vp.original_image) + LAMBDA_DSSIM * (
        1 - ssim(pkg["render"], vp.original_image))
    got = get_loss_refinement(_config(False), pkg, vp, LAMBDA_DSSIM)
    torch.testing.assert_close(got, expected)


def test_saturated_observations_give_no_gradient():
    vp, pkg = _frame()
    render = pkg["render"].clone().requires_grad_(True)
    get_loss_refinement(_config(True), {"render": render}, vp, LAMBDA_DSSIM).backward()
    assert render.grad[:, :8, :8].abs().max() == 0          # masked: no pull on the albedo
    assert render.grad[:, 12:, 12:].abs().max() > 0         # elsewhere: refined as before
