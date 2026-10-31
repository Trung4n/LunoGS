"""Zip of one recorded run for download (docs/DECISIONS.md D76, D80).

<out>/<experiment>__<time>.zip holds the run's record/ folder plus the small
per-run files below when present. Unzipped, the folder <experiment>__<time>/
is directly a --run of tools/viz/render_replay.py and a --record of
tools/verify/eval_map.py.

Used by slam.py at the end of a --record run (Results.record.zip_dir) and by
tools/viz/pack_records.py. Standard library only.

PNG and npz files are already compressed, so they are stored as they are
(fast); the text files are deflated.
"""
import os
import zipfile

RECORD_DIR = "record"
EXTRA_FILES = (
    "config.yml",
    "light_learned.csv",
    "timing.csv",
    "albedo_stats.csv",
    "exposure_stats.csv",
    "pixel_usage.csv",
)
STORED_EXT = (".png", ".npz", ".mp4", ".gif")


def zip_name(run_dir):
    time = os.path.basename(os.path.normpath(run_dir))
    experiment = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    return f"{experiment}__{time}"


def zip_run(run_dir, out_dir):
    """Writes <out_dir>/<experiment>__<time>.zip and returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    name = zip_name(run_dir)
    path = os.path.join(out_dir, name + ".zip")
    with zipfile.ZipFile(path, "w") as z:
        for root, _, files in os.walk(os.path.join(run_dir, RECORD_DIR)):
            for f in sorted(files):
                full = os.path.join(root, f)
                arc = os.path.join(name, os.path.relpath(full, run_dir))
                kind = zipfile.ZIP_STORED if f.lower().endswith(STORED_EXT) else zipfile.ZIP_DEFLATED
                z.write(full, arc, compress_type=kind)
        for f in EXTRA_FILES:
            full = os.path.join(run_dir, f)
            if os.path.isfile(full):
                z.write(full, os.path.join(name, f), compress_type=zipfile.ZIP_DEFLATED)
    return path
