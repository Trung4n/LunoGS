"""Light.albedo_lr (docs/DECISIONS.md D97): absolute = opt_params.feature_lr,
relative_to_initial_albedo = relative_lr * the mean albedo of the initial map.

    python -m pytest tests/light/test_albedo_lr.py
"""
import os
from types import SimpleNamespace

import pytest

from light_models import read_albedo_lr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cfg(block):
    return {"Light": {"albedo_lr": block}}


def test_absolute():
    spec = read_albedo_lr(_cfg({"mode": "absolute"}))
    assert spec.mode == "absolute" and spec.relative_lr is None


def test_relative():
    spec = read_albedo_lr(_cfg({"mode": "relative_to_initial_albedo", "relative_lr": 0.002}))
    assert spec.mode == "relative_to_initial_albedo" and spec.relative_lr == 0.002


def test_missing_block_is_refused():
    with pytest.raises(KeyError, match="albedo_lr"):
        read_albedo_lr({"Light": {}})


@pytest.mark.parametrize("block, match", [
    ({}, "mode"),
    ({"mode": "relative"}, "mode"),
    ({"mode": "relative_to_initial_albedo"}, "relative_lr"),
    ({"mode": "relative_to_initial_albedo", "relative_lr": 0}, "relative_lr"),
    ({"mode": "relative_to_initial_albedo", "relative_lr": True}, "relative_lr"),
    ({"mode": "relative_to_initial_albedo", "relative_lr": "0.002"}, "relative_lr"),
    ({"mode": "absolute", "relative_lr": 0.002}, "unexpected"),
    ("absolute", "block"),
])
def test_bad_blocks_are_refused(block, match):
    with pytest.raises((KeyError, ValueError, TypeError), match=match):
        read_albedo_lr(_cfg(block))


def test_relative_config_resolves(monkeypatch):
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/rgbd_lambert_edgefd_albedolr_rel.yaml")
    spec = read_albedo_lr(cfg)
    assert spec.mode == "relative_to_initial_albedo" and spec.relative_lr == pytest.approx(0.0019)
    # the main configs stay absolute
    for name in ("rgbd_nocos", "rgbd_lambert", "rgbd_lambert_featlr_small_edgefd"):
        assert read_albedo_lr(load_config(f"configs/light/{name}.yaml")).mode == "absolute"


def _backend(spec, albedo):
    torch = pytest.importorskip("torch")
    groups = [{"name": n, "lr": 0.0025} for n in ("xyz", "f_dc", "f_rest", "opacity")]
    gaussians = SimpleNamespace(get_albedo=torch.tensor(albedo, dtype=torch.float32),
                                optimizer=SimpleNamespace(param_groups=groups))
    return SimpleNamespace(albedo_lr=spec, gaussians=gaussians), {g["name"]: g for g in groups}


def test_set_albedo_lr_relative():
    slam_backend = pytest.importorskip("utils.slam_backend")
    spec = SimpleNamespace(mode="relative_to_initial_albedo", relative_lr=0.002)
    fake, groups = _backend(spec, [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    slam_backend.BackEnd.set_albedo_lr(fake, 0)
    assert groups["f_dc"]["lr"] == pytest.approx(0.002 * 0.3)
    assert groups["f_rest"]["lr"] == pytest.approx(0.002 * 0.3 / 20.0)
    assert groups["xyz"]["lr"] == 0.0025 and groups["opacity"]["lr"] == 0.0025


@pytest.mark.parametrize("spec", [None, SimpleNamespace(mode="absolute", relative_lr=None)])
def test_set_albedo_lr_absolute_leaves_feature_lr(spec):
    slam_backend = pytest.importorskip("utils.slam_backend")
    fake, groups = _backend(spec, [[0.1, 0.2, 0.3]])
    slam_backend.BackEnd.set_albedo_lr(fake, 0)
    assert all(g["lr"] == 0.0025 for g in groups.values())
