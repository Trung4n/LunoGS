"""light_models/params.py: ParamSpec parsing (no torch needed except tensor())."""
import json

import pytest

from light_models.params import (
    load_params_file,
    lookup_dotted,
    params_file_path,
    resolve_param,
)

DATA = {
    "intensity": {"K_estimated_step3": 30.37, "K_theory": 30.39},
    "t_CL_m": {"estimated_opencv": [0.0, -0.0505, 0.0002]},
    "flag": True,
}


def test_dotted_lookup():
    assert lookup_dotted(DATA, "intensity.K_estimated_step3", "x") == 30.37
    assert lookup_dotted(DATA, "t_CL_m.estimated_opencv", "x") == [0.0, -0.0505, 0.0002]


def test_dotted_lookup_names_the_missing_segment():
    with pytest.raises(KeyError, match="'intensity.K_step9'"):
        lookup_dotted(DATA, "intensity.K_step9", "Light.intensity")


def test_params_file_source():
    spec = resolve_param(
        {"source": "params_file", "key": "intensity.K_estimated_step3", "learnable": False},
        "Light.intensity",
        DATA,
    )
    assert (spec.value, spec.learnable, spec.lr) == (30.37, False, None)


def test_value_source():
    spec = resolve_param({"source": "value", "value": [1, 2, 3], "learnable": False}, "x", None)
    assert spec.value == [1, 2, 3]


@pytest.mark.parametrize(
    "block, err",
    [
        ({"key": "a", "learnable": False}, KeyError),  # no source
        ({"source": "file", "key": "a", "learnable": False}, ValueError),
        ({"source": "params_file", "learnable": False}, KeyError),  # no key
        ({"source": "value", "learnable": False}, KeyError),  # no value
        ({"source": "value", "value": 1}, KeyError),  # no learnable
        ({"source": "value", "value": 1, "learnable": "no"}, TypeError),
        ({"source": "value", "value": 1, "learnable": True}, KeyError),  # no lr
        ({"source": "value", "value": 1, "learnable": True, "lr": 0.1}, KeyError),
        ({"source": "value", "value": 1, "learnable": False, "lr": 0.1}, ValueError),
        ({"source": "value", "value": 1, "key": "a", "learnable": False}, ValueError),
        ({"source": "value", "value": "1", "learnable": False}, TypeError),
        ({"source": "value", "value": True, "learnable": False}, TypeError),
        ({"source": "params_file", "key": "flag", "learnable": False}, TypeError),
        ({"source": "value", "value": 1, "learnable": True, "lr": -1, "prior_weight": 0}, ValueError),
    ],
)
def test_invalid_blocks(block, err):
    with pytest.raises(err):
        resolve_param(block, "x", DATA)


def test_params_file_source_without_file():
    with pytest.raises(ValueError, match="no params file"):
        resolve_param({"source": "params_file", "key": "a", "learnable": False}, "x", None)


def test_learnable_block():
    spec = resolve_param(
        {"source": "value", "value": 2.0, "learnable": True, "lr": 0.01, "prior_weight": 0},
        "x",
        None,
    )
    assert (spec.learnable, spec.lr, spec.prior_weight) == (True, 0.01, 0)
    torch = pytest.importorskip("torch")
    t = spec.tensor()
    assert t.requires_grad and t.dtype == torch.float32


def test_params_file_relative_to_dataset(tmp_path):
    (tmp_path / "light_params.json").write_text(json.dumps(DATA))
    cfg = {"Dataset": {"dataset_path": str(tmp_path)}, "Light": {"params_file": "light_params.json"}}
    assert load_params_file(params_file_path(cfg)) == DATA


def test_params_file_missing(tmp_path):
    cfg = {"Dataset": {"dataset_path": str(tmp_path)}, "Light": {"params_file": "nope.json"}}
    with pytest.raises(FileNotFoundError):
        load_params_file(params_file_path(cfg))


# ---- perturb: deliberate calibration errors (D57) --------------------------

_DATA = {"K": 30.0, "t": [0.0, -0.05, 0.0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


def _spec(key, perturb, **extra):
    from light_models.params import resolve_param

    block = {"source": "params_file", "key": key, "learnable": False, "perturb": perturb, **extra}
    return resolve_param(block, f"Light.{key}", _DATA)


def test_perturb_scale_and_add_keep_the_nominal():
    k = _spec("K", {"scale": 0.8})
    assert k.value == pytest.approx(24.0) and k.nominal == 30.0 and k.perturb == {"scale": 0.8}
    t = _spec("t", {"add": [0.0, 0.01, 0.0]})
    assert t.value == pytest.approx([0.0, -0.04, 0.0]) and t.nominal == [0.0, -0.05, 0.0]


def test_unperturbed_spec_has_no_nominal():
    from light_models.params import resolve_param

    s = resolve_param({"source": "value", "value": 2.0, "learnable": False}, "x", None)
    assert s.value == 2.0 and s.nominal is None and s.perturb is None


def test_perturb_rotate_tilts_the_light_axis_by_the_angle():
    import math

    r = _spec("R", {"rotate_deg": 5.0, "axis": [1.0, 0.0, 0.0]}).value
    # Orthonormal, and the light axis a_L = R e_z moved by exactly 5 deg.
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-12)
    assert math.degrees(math.acos(r[2][2])) == pytest.approx(5.0)
    assert r[0][2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "key, perturb, error",
    [
        ("K", {"scale": 0.0}, ValueError),
        ("K", {"scale": True}, ValueError),
        ("K", {"scale": 0.8, "add": 1.0}, ValueError),  # two operations
        ("K", {}, ValueError),  # none
        ("t", {"add": [0.0, 0.01]}, ValueError),  # wrong shape
        ("K", {"add": [1.0]}, ValueError),  # wrong shape
        ("K", {"rotate_deg": 2.0, "axis": [1.0, 0.0, 0.0]}, ValueError),  # not 3x3
        ("R", {"rotate_deg": 2.0, "axis": [0.0, 0.0, 0.0]}, ValueError),  # zero axis
        ("R", {"rotate_deg": 2.0}, KeyError),  # axis missing
        ("R", {"rotate_deg": 2.0, "axis": [1, 0, 0], "scale": 2.0}, ValueError),
        ("K", {"scale": 0.8, "typo": 1}, ValueError),
        ("K", 0.8, TypeError),
    ],
)
def test_perturb_validated(key, perturb, error):
    with pytest.raises(error):
        _spec(key, perturb)


def test_perturb_combines_with_learnable():
    # A wrong starting value for a learned parameter (self-calibration later).
    s = _spec("K", {"scale": 1.2}, learnable=True, lr=0.01, prior_weight=0.0)
    assert s.learnable and s.value == pytest.approx(36.0)
