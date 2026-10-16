"""Whole data pipeline for the PCB-defect project (two datasets: bare / solder); up-to-date phases are skipped.

  1 download  chosen Roboflow versions (configs/downloads.csv)    -> data/raw      (needs ROBOFLOW_API_KEY)
  2 extract   one copy per real photo (augment copies merged,     -> data/clean
              tiles / different photos with the same name kept)
  3 build     per family (configs/families.yaml): class map, exclusions, exact-duplicate removal,
              board/scene groups (name rule or SIFT), balanced group split, tiling  -> final_<family>
  4 verify    sanity checks (stops on errors)
  5 review    contact sheets for eyeballing labels                 -> review/<family>
  6 package   notebook + one zip per family                        -> upload_to_drive

The source audit that decided what is used lives in tools/survey_datasets.py + tools/audit_sources.py
(results in data/survey, data/audit).

  python run_pipeline.py                     # everything out of date
  python run_pipeline.py --force build       # rebuild datasets (e.g. after editing configs/families.yaml)
  python run_pipeline.py --adopt             # mark what is on disk as up to date (no work)
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".pipeline"
PHASES = ["download", "extract", "build", "verify", "review", "package"]
FAMILIES = ["bare", "solder"]


def fp(*items):
    h = hashlib.sha1()
    for it in items:
        p = ROOT / it
        if p.is_file():
            h.update(p.read_bytes())
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    h.update(f"{f.relative_to(p)}{f.stat().st_size}".encode())
        else:
            h.update(str(it).encode())
    return h.hexdigest()


def run(*cmd):
    print("   $", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, "-I", *map(str, cmd)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", choices=PHASES)
    ap.add_argument("--adopt", action="store_true", help="mark existing outputs as up to date, run nothing")
    a = ap.parse_args()
    STATE.mkdir(exist_ok=True)
    dirty = set()

    def phase(name, sig, outputs, action, after=()):
        stamp = STATE / f"{name}.json"
        old = json.loads(stamp.read_text())["sig"] if stamp.exists() else None
        forced = a.force and PHASES.index(name.split("_")[0]) >= PHASES.index(a.force)
        if a.adopt:
            if all((ROOT / o).exists() for o in outputs):
                stamp.write_text(json.dumps({"sig": sig}))
                print(f"[{name}] adopted")
            return
        if all((ROOT / o).exists() for o in outputs) and old == sig and not (dirty & set(after)) and not forced:
            print(f"[{name}] SKIP (up to date)")
            return
        print(f"[{name}] RUN", flush=True)
        action()
        stamp.write_text(json.dumps({"sig": sig}))
        dirty.add(name)

    if os.environ.get("ROBOFLOW_API_KEY") and not a.adopt:
        run("tools/download_versions.py")
    else:
        print("[download] ROBOFLOW_API_KEY not set - using what is in data/raw")
    cfg = yaml.safe_load(open(ROOT / "configs/families.yaml", encoding="utf8"))
    used = sorted({s for f in FAMILIES for s in cfg[f]["sources"]})
    if "aaa_hr" in used:   # 2048px aaa cropped to the real photo (its mirrored padding removed), see tools/aaa_hr.py
        phase("extract_aaa_hr_convert", fp("tools/aaa_hr.py", "data/raw/aaa_v1_2048", "data/raw/aaa"),
              ["data/raw/aaa_hr/data.yaml"], lambda: run("tools/aaa_hr.py"))
    for s in used:
        phase(f"extract_{s}", fp("tools/extract_originals.py", "tools/yololib.py", f"data/raw/{s}"), [f"data/clean/{s}/manifest.csv"],
              lambda s=s: (shutil.rmtree(ROOT / "data/clean" / s, ignore_errors=True),
                           run("tools/extract_originals.py", f"data/raw/{s}", f"data/clean/{s}")))
    for f in FAMILIES:
        srcs = list(cfg[f]["sources"])
        phase(f"build_{f}", fp("tools/build_family.py", "configs/families.yaml", *[f"data/clean/{s}/manifest.csv" for s in srcs]),
              [f"final_{f}/data.yaml"], lambda f=f: run("tools/build_family.py", f), after=[f"extract_{s}" for s in srcs])
        phase(f"verify_{f}", fp("tools/verify_dataset.py", f"final_{f}/split_manifest.csv"), [f"final_{f}/qa_report.json"],
              lambda f=f: run("tools/verify_dataset.py", "--root", f"final_{f}"), after=[f"build_{f}"])
        phase(f"review_{f}", fp("tools/make_review.py", f"final_{f}/split_manifest.csv"), [f"review/{f}"],
              lambda f=f: run("tools/make_review.py", f), after=[f"build_{f}"])

    def package():
        out = ROOT / "upload_to_drive"
        out.mkdir(exist_ok=True)
        run("tools/build_notebook.py")                                # PCB_Defects_Colab.ipynb (both models) + code/
        for f in FAMILIES:
            with zipfile.ZipFile(out / f"{f}_data.zip", "w", zipfile.ZIP_STORED) as z:
                for p in sorted((ROOT / f"final_{f}").rglob("*")):
                    if p.is_file():
                        z.write(p, Path(f"final_{f}") / p.relative_to(ROOT / f"final_{f}"))
            print(f"   {f}_data.zip:", round((out / f"{f}_data.zip").stat().st_size / 1e6), "MB")
    phase("package", fp(*[f"final_{f}/split_manifest.csv" for f in FAMILIES], "train/train_solder.py", "infer/predict.py",
                        "tools/verify_dataset.py", "tools/build_notebook.py"),
          ["upload_to_drive/PCB_Defects_Colab.ipynb", *[f"upload_to_drive/{f}_data.zip" for f in FAMILIES]], package,
          after=[f"build_{f}" for f in FAMILIES])
    print("\nPIPELINE DONE ->", ROOT / "upload_to_drive")


if __name__ == "__main__":
    main()
