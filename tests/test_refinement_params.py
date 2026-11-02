"""Results.color_refinement_params (docs/DECISIONS.md D89): which Gaussian
parameters the offline color refinement optimizes. All of them in original
MonoGS; only the albedo with a light model.

    python -m pytest tests/test_refinement_params.py
"""
import glob
import os

import pytest
import yaml

pytest.importorskip("torch")

from utils.config_utils import load_config  # noqa: E402
from utils.slam_utils import REFINABLE_GROUPS, read_refinement_params  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("value", [None, [], "f_dc", ["f_dc", "colour"], ["xyz", 3]])
def test_bad_values_are_refused(value):
    with pytest.raises(ValueError, match="color_refinement_params"):
        read_refinement_params({"color_refinement_params": value})


def test_missing_key_is_refused():
    with pytest.raises(KeyError, match="color_refinement_params"):
        read_refinement_params({})


@pytest.mark.parametrize("path", sorted(
    glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
    + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))))
def test_root_configs_refine_everything_as_monogs(path):
    with open(path) as f:
        cfg = yaml.full_load(f)
    assert read_refinement_params(cfg["Results"]) == list(REFINABLE_GROUPS), path


@pytest.mark.parametrize("path", sorted(
    p for p in glob.glob(os.path.join(REPO, "configs", "light", "rgbd_*.yaml"))))
def test_light_configs_refine_only_the_albedo(monkeypatch, path):
    # The shading depends on the geometry: refining it with the photometric
    # loss alone wrecked the map (D89).
    monkeypatch.chdir(REPO)
    cfg = load_config(os.path.relpath(path, REPO))
    assert read_refinement_params(cfg["Results"]) == ["f_dc"], path
