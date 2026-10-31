"""Zip every recorded run for download (docs/DECISIONS.md D76).

Finds results/<experiment>/<time>/record/ (slam.py --record) and writes one
<out>/<experiment>__<time>.zip per run, holding that run's record/ folder
plus its config.yml and small per-run CSVs (utils/record_zip.py). Unzipped,
the folder <experiment>__<time>/ is directly a --run of
tools/viz/render_replay.py. slam.py --record already writes this zip at the
end of the run (Results.record.zip_dir); this script re-packs runs by hand.

    python tools/viz/pack_records.py --results results --out /kaggle/working/records_zip
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.record_zip import zip_name, zip_run  # noqa: E402  (standard library only)


def find_runs(results):
    """Run folders (results/<experiment>/<time>) that hold a recording."""
    metas = glob.glob(os.path.join(results, "*", "*", "record", "meta.json"))
    return sorted(os.path.dirname(os.path.dirname(m)) for m in metas)


def pack(run_dir, out_dir):
    return zip_run(run_dir, out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="folder holding <experiment>/<time>/ runs")
    ap.add_argument("--out", required=True, help="folder for the .zip files")
    ap.add_argument("--skip_existing", action="store_true", help="do not rewrite a zip that already exists")
    args = ap.parse_args(argv)
    runs = find_runs(args.results)
    if not runs:
        raise SystemExit(f"no recorded run under {args.results} (run slam.py with --record)")
    os.makedirs(args.out, exist_ok=True)
    for run in runs:
        target = os.path.join(args.out, zip_name(run) + ".zip")
        if args.skip_existing and os.path.isfile(target):
            print(f"skip {target}")
            continue
        path = pack(run, args.out)
        print(f"{path}  ({os.path.getsize(path) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
