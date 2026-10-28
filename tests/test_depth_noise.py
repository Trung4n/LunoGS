"""utils/depth_noise.py (docs/DECISIONS.md D72): synthetic degradation of the
sensor depth for the robustness experiment, and the configs that use it.

    python -m pytest tests/test_depth_noise.py
"""
import os

import numpy as np
import pytest

from utils.depth_noise import apply_depth_noise, edge_mask, read_depth_noise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = {"seed": 3, "max_range_m": 4.0, "axial_sigma": {"a": 0.001, "b": 0.002, "z0": 0.4},
         "edge": {"window_px": 2, "rel_jump": 0.1}}


def _params(**over):
    block = {**BLOCK, **over}
    return read_depth_noise({"depth_noise": block})


def test_absent_block_means_no_noise():
    assert read_depth_noise({"dataset_path": "x"}) is None


@pytest.mark.parametrize("over, error", [
    ({"seed": -1}, ValueError),
    ({"seed": 1.5}, ValueError),
    ({"max_range_m": 0}, ValueError),
    ({"axial_sigma": {"a": 0.001, "b": 0.002}}, KeyError),  # z0 missing
    ({"axial_sigma": {"a": -0.1, "b": 0.002, "z0": 0.4}}, ValueError),
    ({"edge": {"window_px": 0, "rel_jump": 0.1}}, ValueError),
    ({"edge": {"window_px": 2, "rel_jump": 0.1, "extra": 1}}, ValueError),
    ({"typo": 1}, ValueError),
])
def test_block_validated(over, error):
    with pytest.raises(error):
        _params(**over)


def test_missing_key_is_an_error():
    block = dict(BLOCK)
    del block["edge"]
    with pytest.raises(KeyError):
        read_depth_noise({"depth_noise": block})


def test_noise_level_follows_the_axial_model_and_is_deterministic():
    p = _params(edge={"window_px": 1, "rel_jump": 0.5})
    for z in (1.0, 3.0):
        depth = np.full((200, 200), z)
        out = apply_depth_noise(depth, p, frame_idx=7)
        expected = 0.001 + 0.002 * (z - 0.4) ** 2
        assert np.std(out - depth) == pytest.approx(expected, rel=0.05)
        assert abs(np.mean(out - depth)) < 3 * expected / 200
    depth = np.full((50, 50), 2.0)
    np.testing.assert_array_equal(apply_depth_noise(depth, p, 7), apply_depth_noise(depth, p, 7))
    assert not np.array_equal(apply_depth_noise(depth, p, 7), apply_depth_noise(depth, p, 8))
    assert not np.array_equal(apply_depth_noise(depth, p, 7),
                              apply_depth_noise(depth, _params(seed=4, edge={"window_px": 1, "rel_jump": 0.5}), 7))


def test_range_edges_and_missing_readings_are_dropped():
    depth = np.full((40, 40), 2.0)
    depth[:, 20:] = 3.0  # a depth jump of 50 %
    depth[0, 0] = 0.0  # no reading stays no reading
    depth[30:, 30:] = 5.0  # beyond the range
    out = apply_depth_noise(depth, _params(), frame_idx=0)
    assert out[0, 0] == 0
    assert (out[30:, 30:] == 0).all()
    assert (out[10, 18:22] == 0).all()  # within 2 px of the jump
    assert (out[10, 5:15] > 0).all() and (out[10, 24:28] > 0).all()
    assert out.dtype == np.float64 and depth[10, 19] == 2.0  # input untouched


def test_edge_mask_ignores_smooth_slopes():
    ramp = np.tile(np.linspace(1.0, 1.2, 60), (30, 1))  # 0.2 m over 60 px
    assert not edge_mask(ramp, 2, 0.1).any()


# ---- configs ----------------------------------------------------------------

def _load(path):
    from utils.config_utils import load_config

    return load_config(path)


def test_noise_configs_share_one_block_and_change_nothing_else(monkeypatch):
    monkeypatch.chdir(REPO)
    pairs = [("configs/rgbd/replica/depth_noise.yaml", "configs/rgbd/replica/office0.yaml"),
             ("configs/rgbd/replica/depth_only_noise.yaml", "configs/rgbd/replica/depth_only.yaml"),
             ("configs/light/rgbd_nocos_depth_noise.yaml", "configs/light/rgbd_nocos.yaml")]
    blocks = []
    for noisy, clean in pairs:
        a, b = _load(noisy), _load(clean)
        blocks.append(a["Dataset"].pop("depth_noise"))
        assert read_depth_noise({"depth_noise": blocks[-1]}) is not None
        assert "depth_noise" not in b["Dataset"]
        a.pop("inherit_from"), b.pop("inherit_from", None)
        assert a == b
    assert blocks[0] == blocks[1] == blocks[2]


def test_depth_only_is_the_baseline_with_alpha_zero(monkeypatch):
    monkeypatch.chdir(REPO)
    a, b = _load("configs/rgbd/replica/depth_only.yaml"), _load("configs/rgbd/replica/office0.yaml")
    assert a["Training"]["alpha"] == 0.0 and b["Training"]["alpha"] > 0
    for c in (a, b):
        c["Training"].pop("alpha")
        c.pop("inherit_from")
    assert a == b
    assert "Light" not in a or not a["Light"].get("enabled", False)


# ---- heavy level: correlated noise and block dropout (D75) ---------------------

HEAVY = {**BLOCK, "correlated": {"cell_px": 8, "a": 0.01, "b": 0.0, "z0": 0.4},
         "dropout": {"cell_px": 4, "frac": 0.2}}


def test_optional_parts_absent_keep_the_light_level_identical():
    # The white noise is drawn first: adding the optional parts must not
    # change it, and their absence parses to None.
    p = _params()
    assert p["correlated"] is None and p["dropout"] is None
    depth = np.full((60, 80), 2.0)
    light = apply_depth_noise(depth, p, 5)
    only_cor = read_depth_noise({"depth_noise": {**BLOCK, "correlated": {"cell_px": 8, "a": 0.0, "b": 0.0, "z0": 0.4}}})
    np.testing.assert_array_equal(apply_depth_noise(depth, only_cor, 5), light)


@pytest.mark.parametrize("over, error", [
    ({"correlated": {"cell_px": 0, "a": 0.01, "b": 0.0, "z0": 0.4}}, ValueError),
    ({"correlated": {"cell_px": 8, "a": 0.01, "b": 0.0}}, KeyError),
    ({"dropout": {"cell_px": 4, "frac": 1.0}}, ValueError),
    ({"dropout": {"cell_px": 4}}, KeyError),
    ({"dropout": {"cell_px": 4, "frac": 0.1, "x": 1}}, ValueError),
])
def test_optional_parts_validated(over, error):
    with pytest.raises(error):
        _params(**over)


def test_correlated_field_has_unit_variance_and_is_smooth():
    from utils.depth_noise import correlated_field

    f = correlated_field(np.random.default_rng(0), (400, 400), 16)
    assert np.std(f) == pytest.approx(1.0, rel=0.1)
    # neighbouring pixels nearly equal, pixels far apart independent
    assert np.corrcoef(f[:, :-1].ravel(), f[:, 1:].ravel())[0, 1] > 0.9
    assert abs(np.corrcoef(f[:, :-40].ravel(), f[:, 40:].ravel())[0, 1]) < 0.2


def test_block_dropout_removes_whole_blocks_at_the_given_rate():
    p = read_depth_noise({"depth_noise": {**HEAVY, "edge": {"window_px": 1, "rel_jump": 0.5},
                                          "correlated": {"cell_px": 8, "a": 0.0, "b": 0.0, "z0": 0.4}}})
    out = apply_depth_noise(np.full((400, 400), 2.0), p, 3)
    lost = out == 0
    assert lost.mean() == pytest.approx(0.2, abs=0.03)
    blocks = lost.reshape(100, 4, 100, 4)
    assert (blocks.all(axis=(1, 3)) == blocks.any(axis=(1, 3))).all()  # all-or-nothing per block


def test_heavy_configs_share_one_block(monkeypatch):
    monkeypatch.chdir(REPO)
    a = _load("configs/rgbd/replica/depth_only_noise_heavy.yaml")
    b = _load("configs/light/rgbd_nocos_depth_noise_heavy.yaml")
    assert a["Dataset"]["depth_noise"] == b["Dataset"]["depth_noise"]
    p = read_depth_noise(a["Dataset"])
    assert p["correlated"] is not None and p["dropout"] is not None
    for noisy, clean in [(a, _load("configs/rgbd/replica/depth_only.yaml")),
                         (b, _load("configs/light/rgbd_nocos.yaml"))]:
        noisy["Dataset"].pop("depth_noise")
        noisy.pop("inherit_from"), clean.pop("inherit_from", None)
        assert noisy == clean
