"""Map evaluation of every downloaded recording, in one command (D81).

For each <experiment>__<scene>_<variant>__<time>.zip (slam.py --record,
Results.record.zip_dir) or already extracted run folder in --inbox:
  1. extracts it to <records>/<name>/ (skipped when already there);
  2. takes the run's own resolved config.yml (inherit_from dropped, so the
     evaluation uses exactly the values of the run), its light_learned.csv
     when the run learned light parameters, and finds the local dataset from
     the run's Dataset.dataset_path (<datasets>/<scene>/<variant>);
  3. runs tools/verify/eval_map.py into <out>/<scene>/<name>/ (skipped when
     its metrics.json exists, unless --force);
then, per scene, prints the main and the breakdown tables and runs
tools/analysis/diagnose_map.py over every evaluated run of the scene
(<out>/<scene>/diagnosis.png).

The cone mask always comes from <datasets>/<scene>/lit/light_params.json,
so it is the same for every run of a scene. Needs CUDA (WSL MonoGS env).

    python tools/verify/eval_records.py --inbox downloads/records --records record \\
        --datasets /mnt/d/CodeForYou/Replica/04_datasets/monogs \\
        --renders /mnt/d/CodeForYou/Replica/03_renders --out results/map_eval \\
        --sat_8bit 250 --rpe_delta 10 --segment 400
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE, os.path.join(REPO, "tools", "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESOLVED_NAME = "config_resolved.yml"


def find_inputs(inbox):
    """Zips and extracted run folders (holding record/meta.json) in inbox."""
    zips = sorted(glob.glob(os.path.join(inbox, "*.zip")))
    dirs = sorted(os.path.dirname(os.path.dirname(m)) for m in glob.glob(os.path.join(inbox, "*", "record", "meta.json")))
    return zips, dirs


def extract(zip_path, records):
    """<records>/<name>/ from <name>.zip (the zip holds <name>/...)."""
    name = os.path.splitext(os.path.basename(zip_path))[0]
    dest = os.path.join(records, name)
    if os.path.isfile(os.path.join(dest, "record", "meta.json")):
        return dest
    with zipfile.ZipFile(zip_path) as z:
        bad = [n for n in z.namelist() if not n.startswith(name + "/") or ".." in n.split("/")]
        if bad:
            raise SystemExit(f"{zip_path}: unexpected entries {bad[:3]} (not a pack of {name})")
        z.extractall(records)
    return dest


def resolved_config(run_dir):
    """The run's config.yml without inherit_from, written next to it: the
    values the run used, not the current files of configs/."""
    src = os.path.join(run_dir, "config.yml")
    if not os.path.isfile(src):
        raise SystemExit(f"{run_dir}: no config.yml (recorded before D80); evaluate it with eval_map.py --config")
    with open(src, encoding="utf-8") as f:
        cfg = yaml.full_load(f)
    cfg.pop("inherit_from", None)
    dst = os.path.join(run_dir, RESOLVED_NAME)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dst, cfg


def scene_variant(cfg):
    """(<scene>, <variant>) from the run's Dataset.dataset_path .../<scene>/<variant>."""
    parts = os.path.normpath(cfg["Dataset"]["dataset_path"].replace("\\", "/")).split(os.sep)
    parts = [p for p in parts if p]
    if len(parts) < 2:
        raise SystemExit(f"cannot read <scene>/<variant> from Dataset.dataset_path {cfg['Dataset']['dataset_path']!r}")
    return parts[-2], parts[-1]


def label_of(name, variant):
    """rgbd_nocos__office0_lit__2026-10-31-09-05-00 -> 'rgbd_nocos (lit, 10-31 09:05)':
    the time tells repeated runs of one config apart."""
    parts = name.split("__")
    stamp = parts[-1].split("-") if len(parts) == 3 else []
    if len(stamp) == 6:
        return f"{parts[0]} ({variant}, {stamp[1]}-{stamp[2]} {stamp[3]}:{stamp[4]})"
    return f"{parts[0]} ({variant})"


def evaluate(run_dir, args):
    """eval_map.py on the run's online map, and on its map after the color
    refinement when the run recorded one (<name>__refined)."""
    import eval_map

    name = os.path.basename(os.path.normpath(run_dir))
    cfg_path, cfg = resolved_config(run_dir)
    scene, variant = scene_variant(cfg)
    maps = [("final", name, label_of(name, variant))]
    if os.path.isfile(os.path.join(run_dir, "record", "maps", "final_refined.npz")):
        maps.append(("final_refined", name + "__refined", label_of(name, variant) + ", refined"))
    for which, out_name, label in maps:
        out = os.path.join(args.out, scene, out_name)
        if os.path.isfile(os.path.join(out, "metrics.json")) and not args.force:
            print(f"skip {out_name}: {out}/metrics.json exists (--force to redo)")
            continue
        argv = ["run", "--record", run_dir, "--config", cfg_path,
                "--dataset", os.path.join(args.datasets, scene, variant),
                "--renders", os.path.join(args.renders, scene),
                "--light_params", os.path.join(args.datasets, scene, "lit", "light_params.json"),
                "--out", out, "--label", label, "--map", which, "--sat_8bit", str(args.sat_8bit)]
        learned = os.path.join(run_dir, "light_learned.csv")
        if os.path.isfile(learned):
            argv += ["--light_csv", learned]
        print(f"evaluating {out_name} ({scene}/{variant})", flush=True)
        eval_map.main(argv)
    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="folder with the downloaded .zip files (or extracted runs)")
    ap.add_argument("--records", required=True, help="where the zips are extracted")
    ap.add_argument("--datasets", required=True, help="MonoGS dataset root: <datasets>/<scene>/<variant>")
    ap.add_argument("--renders", required=True, help="Replica 03_renders root: <renders>/<scene>")
    ap.add_argument("--out", required=True, help="evaluation root: <out>/<scene>/<run>/")
    ap.add_argument("--sat_8bit", type=int, required=True, help="saturated if any channel >= this")
    ap.add_argument("--rpe_delta", type=int, required=True, help="frames of the relative pose error (diagnosis)")
    ap.add_argument("--segment", type=int, required=True, help="frames per segment of the time summaries (diagnosis)")
    ap.add_argument("--force", action="store_true", help="evaluate again runs that already have metrics.json")
    args = ap.parse_args(argv)

    zips, dirs = find_inputs(args.inbox)
    if not zips and not dirs:
        raise SystemExit(f"no .zip or run folder in {args.inbox}")
    os.makedirs(args.records, exist_ok=True)
    runs = [extract(z, args.records) for z in zips] + dirs
    scenes = sorted({evaluate(r, args) for r in runs})

    import diagnose_map
    import eval_map

    for scene in scenes:
        metrics = sorted(glob.glob(os.path.join(args.out, scene, "*", "metrics.json")))
        print(f"\n## {scene}\n")
        print(eval_map.table(metrics))
        print()
        print(eval_map.table(metrics, eval_map.BREAKDOWN))
        with_record = []
        for m in metrics:
            with open(m, encoding="utf-8") as f:
                if "record" in json.load(f):
                    with_record.append(m)
        if with_record:
            diagnose_map.main(["--metrics", *with_record, "--sat_8bit", str(args.sat_8bit),
                               "--rpe_delta", str(args.rpe_delta), "--segment", str(args.segment),
                               "--figure", os.path.join(args.out, scene, "diagnosis.png")])


if __name__ == "__main__":
    main()
