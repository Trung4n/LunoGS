"""light_models/diagnostics.py: albedo > 1 statistics and CSV log (needs torch)."""
import csv

import pytest

torch = pytest.importorskip("torch")

from light_models.diagnostics import CSV_NAME, albedo_stats, log_albedo_stats  # noqa: E402


def test_stats_count_gaussians_not_channels():
    albedo = torch.tensor(
        [[0.2, 0.3, 0.4], [1.5, 0.1, 0.1], [0.9, 1.2, 1.1], [1.0, 1.0, 1.0]]
    )
    s = albedo_stats(albedo)
    assert s["n_gaussians"] == 4
    assert s["frac_any_gt1"] == pytest.approx(0.5)  # exactly 1.0 is not above
    assert (s["frac_r_gt1"], s["frac_g_gt1"], s["frac_b_gt1"]) == pytest.approx(
        (0.25, 0.25, 0.25)
    )
    assert s["max"] == pytest.approx(1.5)


def test_empty_selection():
    assert albedo_stats(torch.zeros(0, 3)) == {"n_gaussians": 0}


def test_csv_appends_with_single_header(tmp_path):
    log_albedo_stats(str(tmp_path), 0, "new", torch.tensor([[1.2, 0.5, 0.5]]))
    log_albedo_stats(str(tmp_path), 0, "map", torch.tensor([[0.5, 0.5, 0.5]]))
    log_albedo_stats(str(tmp_path), 4, "new", torch.zeros(0, 3))
    with open(tmp_path / CSV_NAME) as f:
        rows = list(csv.DictReader(f))
    assert [(r["frame_idx"], r["stage"]) for r in rows] == [("0", "new"), ("0", "map"), ("4", "new")]
    assert float(rows[0]["frac_any_gt1"]) == 1.0
    assert float(rows[1]["frac_any_gt1"]) == 0.0
    assert rows[2]["n_gaussians"] == "0" and rows[2]["frac_any_gt1"] == ""


def test_no_file_without_save_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_albedo_stats(None, 0, "map", torch.tensor([[1.2, 0.5, 0.5]]))
    assert not (tmp_path / CSV_NAME).exists()


def test_console_line_only_without_csv(tmp_path, monkeypatch):
    # One CSV row per keyframe and stage, no console line (D82); the console
    # line only when nothing is saved, so the numbers are never lost.
    from types import SimpleNamespace

    import light_models.diagnostics as diag

    lines = []
    monkeypatch.setattr(diag, "Log", lambda *a, **k: lines.append(a))
    vp = SimpleNamespace(exposure_a=torch.tensor(0.1), exposure_b=torch.tensor(0.0))
    diag.log_albedo_stats(str(tmp_path), 0, "map", torch.tensor([[1.2, 0.5, 0.5]]))
    diag.log_exposure(str(tmp_path), 0, "map", vp)
    assert lines == [] and (tmp_path / diag.EXPOSURE_CSV_NAME).exists()
    diag.log_albedo_stats(None, 0, "map", torch.tensor([[1.2, 0.5, 0.5]]))
    diag.log_exposure(None, 0, "map", vp)
    assert len(lines) == 2


def test_timing_csv_and_summary(tmp_path):
    # D79: tracking per frame, keyframe mapping, background mapping between keyframes
    from light_models.diagnostics import log_timing, summarize_csv_logs

    d = str(tmp_path)
    log_timing(None, "tracking", 1, 10.0, 5)  # no save dir: nothing written
    for i, ms in enumerate([10.0, 20.0, 30.0], start=1):
        log_timing(d, "tracking", i, ms, 40)
    log_timing(d, "background", 4, 600.0, 12)
    log_timing(d, "keyframe", 4, 250.0, 10)
    log_timing(d, "background", 8, 0.0, 0)
    lines = "\n".join(summarize_csv_logs(d))
    assert "tracking per frame over 3: ms p5 10.0000  median 20.0000  p95 30.0000, mean iterations 40.0" in lines
    assert "keyframe mapping over 1:" in lines
    assert "background mapping: 12 iterations, 50.0 ms per iteration, 1 s in all" in lines
