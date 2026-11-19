"""Lambert's factor at pixels without a normal (Light.cosine.missing_normal,
docs/DECISIONS.md D147)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import light_models.normals  # noqa: E402,F401  (registers the normal sources)
from light_models.cosine import Lambert  # noqa: E402
from light_models.registry import build_component  # noqa: E402

H, W = 4, 6


def _lambert(**over):
    block = {"type": "lambert", "normal_source": {"type": "gbuffer"}, "warmup_keyframes": 0}
    block.update(over)
    return build_component("cosine", block, "Light.cosine", {})


def _scene():
    """Normals facing the light with n . l = 0.8 on the left half, 0.4 on the
    right half; the last column has no normal."""
    to_light = torch.zeros(3, H, W)
    to_light[2] = 1.0
    n = torch.zeros(3, H, W)
    n[2, :, : W // 2] = 0.8
    n[0, :, : W // 2] = 0.6
    n[2, :, W // 2:] = 0.4
    n[0, :, W // 2:] = (1 - 0.16) ** 0.5
    n_valid = torch.ones(H, W, dtype=torch.bool)
    n_valid[:, -1] = False
    gbuffer = {"normal": n, "normal_valid": n_valid}
    valid = torch.ones(H, W, dtype=torch.bool)
    return torch.zeros(3, H, W), valid, to_light, gbuffer


def test_absent_key_is_the_behaviour_before():
    lam = _lambert()
    assert isinstance(lam, Lambert) and lam.missing_normal == "zero"
    pts, valid, l, g = _scene()
    factor, aux = lam(pts, valid, l, g, SimpleNamespace())
    assert factor[:, -1].eq(0).all()
    torch.testing.assert_close(_lambert(missing_normal="zero")(pts, valid, l, g, None)[0], factor)


def test_bad_value_raises():
    with pytest.raises(ValueError, match="missing_normal"):
        _lambert(missing_normal="one")


def test_frame_mean_fills_the_pixels_without_a_normal():
    pts, valid, l, g = _scene()
    factor, aux = _lambert(missing_normal="frame_mean")(pts, valid, l, g, None)
    known = g["normal_valid"]
    expected = factor[known].mean()  # 3 columns at 0.8, 2 at 0.4
    torch.testing.assert_close(expected, torch.tensor((3 * 0.8 + 2 * 0.4) / 5))
    torch.testing.assert_close(factor[:, -1], expected.expand(H))
    # The pixels still count as having no normal for the loss.
    assert not aux["normal_valid"][:, -1].any()
    assert aux["cos_nl"][:, -1].eq(0).all()


def test_frame_mean_uses_only_trusted_pixels():
    pts, valid, l, g = _scene()
    valid[:, : W // 2] = False  # the 0.8 half is not trusted
    factor, _ = _lambert(missing_normal="frame_mean")(pts, valid, l, g, None)
    torch.testing.assert_close(factor[:, -1], torch.full((H,), 0.4))


def test_frame_without_any_normal_gives_zero():
    pts, valid, l, g = _scene()
    g["normal_valid"][:] = False
    factor, _ = _lambert(missing_normal="frame_mean")(pts, valid, l, g, None)
    assert factor.eq(0).all()


def test_frame_mean_is_differentiable_and_needs_no_host_read():
    pts, valid, l, g = _scene()
    l = l.clone().requires_grad_()
    factor, _ = _lambert(missing_normal="frame_mean")(pts, valid, l, g, None)
    factor.sum().backward()
    assert l.grad is not None and torch.isfinite(l.grad).all()
