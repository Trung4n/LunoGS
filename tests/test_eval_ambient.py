"""tools/verify/eval_ambient.py: the arithmetic, without CUDA or data."""
import importlib.util
import os

import numpy as np
import pytest

_path = os.path.join(os.path.dirname(__file__), "..", "tools", "verify", "eval_ambient.py")
_spec = importlib.util.spec_from_file_location("eval_ambient", _path)
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)


def test_true_irradiance_skips_dark_channels():
    albedo = np.array([[[0.5, 0.25, 0.001]]])
    ambient = np.array([[[0.1, 0.05, 0.9]]])  # the dark channel would give 900
    assert ea.true_irradiance(ambient, albedo, 0.02)[0, 0] == pytest.approx(0.2)


def test_pixel_without_a_reliable_channel_is_nan():
    assert np.isnan(ea.true_irradiance(np.ones((1, 1, 3)), np.zeros((1, 1, 3)), 0.02)[0, 0])


def test_predicted_irradiance_inverts_the_rasterized_residual():
    albedo = np.array([[[0.5, 0.2, 0.4]]])
    e_i = 0.3
    c = 0.1
    residual = albedo * (e_i - c)
    assert ea.predicted_irradiance(albedo, residual, c, 0.02)[0, 0] == pytest.approx(e_i)


def test_constant_ambient_predicts_c_where_the_map_has_albedo():
    albedo = np.array([[[0.5, 0.2, 0.4], [0.0, 0.0, 0.0]]])
    e = ea.predicted_irradiance(albedo, None, 0.12, 0.02)
    assert e[0, 0] == pytest.approx(0.12) and np.isnan(e[0, 1])


def test_error_stats_compare_with_the_reference():
    true = np.array([0.1, 0.2, 0.4])
    s = ea.error_stats(true.copy(), true, np.full(3, 0.2))
    assert s["abs_err_median"] == 0 and s["rel_err_median"] == 0 and s["pearson"] == pytest.approx(1)
    assert s["abs_err_reference"] == pytest.approx(0.1)


def test_by_true_level_bins_the_true_value():
    true = np.array([0.05, 0.15, 0.25, 0.5])
    rows = ea.by_true_level(true * 2, true, [0.1, 0.3])
    assert [r["pixels"] for r in rows] == [1, 2, 1]
    assert rows[1]["pred_median"] == pytest.approx(0.4)
