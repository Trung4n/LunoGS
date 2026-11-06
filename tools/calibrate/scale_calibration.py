"""Calibration of the same lamp at another power (docs/DECISIONS.md D115): a
result JSON of fit_light.py with the intensity multiplied by a known ratio, as a
dimmer with a known setting. Pose and emission profile are properties of the
lamp's mount and optics, so they are kept; the ambient of the calibration room
is kept too, it is not read by SLAM.

The ratio is a statement about the rig, not a measurement here: a real clip
would measure it, as fit_light.py does (D106).

    python tools/calibrate/scale_calibration.py --calibration <result.json>
        --intensity_scale 0.1 --out <dataset>/calib/light_calibrated.json
"""
import argparse
import json
import os


def scaled(calibration, scale, source):
    """A copy of a fit_light.py result with intensity x scale, and its origin recorded."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"intensity scale must be a number > 0, got {scale!r}")
    if "light" not in calibration or "intensity" not in calibration["light"]:
        raise KeyError("not a fit_light.py result: missing light.intensity")
    out = json.loads(json.dumps(calibration))
    k = out["light"]["intensity"]
    out["light"]["intensity"] = [x * scale for x in k] if isinstance(k, list) else k * scale
    out["provenance"] = {"derived_from": source, "intensity_scale": scale, "fit_provenance": calibration.get("provenance")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True, help="result JSON of fit_light.py")
    ap.add_argument("--intensity_scale", type=float, required=True, help="power of this lamp setting / calibrated power")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.calibration, encoding="utf-8") as f:
        calibration = json.load(f)
    result = scaled(calibration, args.intensity_scale, os.path.abspath(args.calibration))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"K {calibration['light']['intensity']} x {args.intensity_scale} -> {result['light']['intensity']}: {args.out}")


if __name__ == "__main__":
    main()
