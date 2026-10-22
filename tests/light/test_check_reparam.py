"""tools/verify/check_reparam.py on synthetic run directories."""
import importlib.util
import json
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "check_reparam", os.path.join(REPO, "tools", "verify", "check_reparam.py")
)
check_reparam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_reparam)

DATASET = "datasets/replica/office0/"


def _baseline_cfg(pre_extension=False):
    cfg = {"Dataset": {"dataset_path": DATASET, "sensor_type": "monocular"}}
    if not pre_extension:
        cfg["Light"] = {"enabled": False}
        cfg["LightTracking"] = {"loss_color_space": "srgb", "exposure_affine": True}
    return cfg


def _candidate_cfg(tol=0.002, affine=True, space="srgb"):
    return {
        "Dataset": {"dataset_path": DATASET, "sensor_type": "monocular"},
        "Light": {"enabled": True, "shader": {"type": "identity"}},
        "LightTracking": {"loss_color_space": space, "exposure_affine": affine},
        "Verify": {"reparam": {"ate_abs_tolerance_m": tol}},
    }


def _run(tmp_path, name, cfg, ate):
    d = tmp_path / name
    (d / "plot").mkdir(parents=True)
    (d / "config.yml").write_text(yaml.dump(cfg))
    (d / "plot" / "stats_final.json").write_text(json.dumps({"rmse": ate, "mean": ate}))
    return str(d)


def test_pass_within_tolerance(tmp_path):
    b = [_run(tmp_path, f"b{i}", _baseline_cfg(), a) for i, a in enumerate([0.010, 0.012])]
    c = [_run(tmp_path, "c0", _candidate_cfg(tol=0.002), 0.0125)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 0


def test_fail_outside_tolerance(tmp_path):
    b = [_run(tmp_path, "b0", _baseline_cfg(), 0.010)]
    c = [_run(tmp_path, "c0", _candidate_cfg(tol=0.002), 0.020)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 1


def test_pre_extension_baseline_is_accepted(tmp_path):
    b = [_run(tmp_path, "b0", _baseline_cfg(pre_extension=True), 0.010)]
    c = [_run(tmp_path, "c0", _candidate_cfg(), 0.011)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 0


@pytest.mark.parametrize("tol", [None, 0, -1, "0.01", True])
def test_tolerance_must_be_filled_and_positive(tmp_path, tol):
    b = [_run(tmp_path, "b0", _baseline_cfg(), 0.010)]
    c = [_run(tmp_path, "c0", _candidate_cfg(tol=tol), 0.010)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 2


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate_cfg(affine=False),  # differs from baseline
        _candidate_cfg(space="linear"),  # not the reparam setting
        {**_candidate_cfg(), "Dataset": {"dataset_path": "other/x/", "sensor_type": "monocular"}},
        {**_candidate_cfg(), "Dataset": {"dataset_path": DATASET, "sensor_type": "rgbd"}},
        {**_candidate_cfg(), "Light": {"enabled": False}},
    ],
)
def test_not_comparable(tmp_path, candidate):
    b = [_run(tmp_path, "b0", _baseline_cfg(), 0.010)]
    c = [_run(tmp_path, "c0", candidate, 0.010)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 2


def test_candidate_used_as_baseline_is_rejected(tmp_path):
    b = [_run(tmp_path, "b0", _candidate_cfg(), 0.010)]
    c = [_run(tmp_path, "c0", _candidate_cfg(), 0.010)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 2


def test_missing_stats_is_rejected(tmp_path):
    b = _run(tmp_path, "b0", _baseline_cfg(), 0.010)
    os.remove(os.path.join(b, "plot", "stats_final.json"))
    c = _run(tmp_path, "c0", _candidate_cfg(), 0.010)
    assert check_reparam.main(["--baseline", b, "--candidate", c]) == 2


def test_rgbd_runs_are_compared_among_themselves(tmp_path):
    # The main setting is RGB-D (D47, D53): runs of one sensor type, any type,
    # are comparable (all aligned the same way by evaluate_evo).
    rgbd = {"dataset_path": DATASET, "sensor_type": "depth"}
    b = [_run(tmp_path, "b0", {**_baseline_cfg(), "Dataset": rgbd}, 0.010)]
    c = [_run(tmp_path, "c0", {**_candidate_cfg(tol=0.002), "Dataset": rgbd}, 0.011)]
    assert check_reparam.main(["--baseline", *b, "--candidate", *c]) == 0
