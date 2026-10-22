"""Re-parameterization test (docs/DECISIONS.md D6): compares the ATE of runs
with the identity shader (candidate) against original MonoGS runs (baseline).

    python tools/verify/check_reparam.py --baseline <run dir> [...] --candidate <run dir> [...]

A run dir is what slam.py writes: <save_dir>/config.yml (fully resolved
config) and <save_dir>/plot/stats_final.json (evo APE statistics). Several
runs per side are allowed because MonoGS sets no seed; their means are
compared. The tolerance is read from the candidates' saved config
(Verify.reparam.ate_abs_tolerance_m), so it is recorded with the runs.

Reads results only, never ground-truth passes. Exit code 0 = PASS, 1 = FAIL,
2 = the runs are not comparable or the config is incomplete.
"""
import argparse
import json
import os
import sys

import yaml


class NotComparable(Exception):
    pass


def load_run(run_dir):
    cfg_path = os.path.join(run_dir, "config.yml")
    stats_path = os.path.join(run_dir, "plot", "stats_final.json")
    for p in (cfg_path, stats_path):
        if not os.path.isfile(p):
            raise NotComparable(f"{p} not found (not a finished slam.py run dir?)")
    with open(cfg_path, "r") as f:
        cfg = yaml.full_load(f)
    with open(stats_path, "r") as f:
        stats = json.load(f)
    if "rmse" not in stats:
        raise NotComparable(f"{stats_path} has no 'rmse'")
    return {"dir": run_dir, "config": cfg, "ate": float(stats["rmse"])}


def _is_baseline(cfg):
    # Runs saved before the light extension existed have no Light block; they
    # are original MonoGS by construction. Otherwise Light.enabled must be false.
    return "Light" not in cfg or cfg["Light"].get("enabled") is False


def _exposure_affine(cfg):
    # Same reasoning: pre-extension runs always applied the affine term.
    if "LightTracking" not in cfg:
        return True
    return cfg["LightTracking"]["exposure_affine"]


def _dataset(cfg):
    return os.path.normpath(cfg["Dataset"]["dataset_path"])


def check(baselines, candidates):
    """Returns a dict with the verdict; raises NotComparable on bad input."""
    if not baselines or not candidates:
        raise NotComparable("need at least one baseline and one candidate run")

    for run in baselines:
        if not _is_baseline(run["config"]):
            raise NotComparable(f"{run['dir']}: Light.enabled is not false")
    for run in candidates:
        cfg = run["config"]
        light = cfg.get("Light", {})
        if light.get("enabled") is not True or light.get("shader", {}).get("type") != "identity":
            raise NotComparable(f"{run['dir']}: not an identity-shader run")
        if cfg["LightTracking"]["loss_color_space"] != "srgb":
            raise NotComparable(
                f"{run['dir']}: loss_color_space must be srgb for this test"
            )

    runs = baselines + candidates
    # One sensor type: evaluate_evo aligns monocular runs with Sim(3)
    # (correct_scale=monocular) and the others with SE(3); ATEs under two
    # alignments are not comparable.
    sensors = {r["config"]["Dataset"]["sensor_type"] for r in runs}
    if len(sensors) != 1:
        raise NotComparable(f"runs use different sensor types: {sorted(sensors)}")
    datasets = {_dataset(r["config"]) for r in runs}
    if len(datasets) != 1:
        raise NotComparable(f"runs use different datasets: {sorted(datasets)}")
    affines = {_exposure_affine(r["config"]) for r in runs}
    if len(affines) != 1:
        raise NotComparable("runs differ in LightTracking.exposure_affine")

    tolerances = set()
    for run in candidates:
        try:
            tolerances.add(run["config"]["Verify"]["reparam"]["ate_abs_tolerance_m"])
        except (KeyError, TypeError):
            raise NotComparable(
                f"{run['dir']}: missing Verify.reparam.ate_abs_tolerance_m"
            ) from None
    if len(tolerances) != 1:
        raise NotComparable(f"candidates disagree on the tolerance: {tolerances}")
    tol = tolerances.pop()
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        raise NotComparable(
            f"Verify.reparam.ate_abs_tolerance_m must be a positive number, got {tol!r}"
        )

    mean_b = sum(r["ate"] for r in baselines) / len(baselines)
    mean_c = sum(r["ate"] for r in candidates) / len(candidates)
    diff = abs(mean_c - mean_b)
    return {
        "baseline_ate": [r["ate"] for r in baselines],
        "candidate_ate": [r["ate"] for r in candidates],
        "baseline_mean": mean_b,
        "candidate_mean": mean_c,
        "abs_diff": diff,
        "tolerance": float(tol),
        "passed": diff <= tol,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        result = check(
            [load_run(d) for d in args.baseline],
            [load_run(d) for d in args.candidate],
        )
    except NotComparable as e:
        print(f"NOT COMPARABLE: {e}")
        return 2
    for side in ("baseline", "candidate"):
        values = ", ".join(f"{v:.6f}" for v in result[f"{side}_ate"])
        print(f"{side:9s} ATE RMSE [m]: {values}  (mean {result[f'{side}_mean']:.6f})")
    print(
        f"|diff| = {result['abs_diff']:.6f} m, tolerance = {result['tolerance']:.6f} m"
        f" -> {'PASS' if result['passed'] else 'FAIL (suspect the color path)'}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
