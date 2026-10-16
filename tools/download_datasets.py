"""Download the solder / PCB-hole datasets from Roboflow Universe (YOLO26 format).

The API key is read from ROBOFLOW_API_KEY only. A dataset whose folder already has
data.yaml is skipped, so re-running after a failure only fetches what is missing.
Datasets download in parallel.

  python tools/download_datasets.py            (run from solder_project/)
"""
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# (workspace, project, version, local folder)
DATASETS = [
    ("rf-projects", "pcb-holes", 1, "rf_pcb_holes"),
    ("test-pygia", "my-first-project-sibea", 1, "pygia_sibea"),
    ("eeproject-ht4h0", "pcb-iuodo", 1, "eeproject_pcb"),
    ("anto-bakvu", "pcb-iwo7t", 1, "anto_pcb"),
    ("nuu-pmlt7", "rex_school-e6sez", 1, "nuu_rex_school"),
    ("kellys-workspace-e9swg", "pbc-holes", 1, "kelly_pbc_holes"),
    ("theodor-nanu-wxxvd", "pcb-holes-8zvga", 1, "theodor_pcb_holes"),
    ("thanakorn-bphfu", "pcb_missing_home-2", 1, "thanakorn_missing_home"),
    ("-nqvmy", "pcb_missing-qfip9", 1, "nqvmy_pcb_missing"),
]


def fetch(item):
    ws, proj, ver, folder = item
    out = RAW / folder
    if (out / "data.yaml").exists():
        return f"SKIP {folder} (already downloaded)"
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    err = ""
    for fmt in ("yolo26", "yolov11", "yolov8"):
        shutil.rmtree(out, ignore_errors=True)
        try:
            rf.workspace(ws).project(proj).version(ver).download(fmt, location=str(out))
            if (out / "data.yaml").exists():
                return f"OK   {folder} ({fmt})"
        except Exception as e:
            err = str(e)[:300]
    return f"FAIL {folder}: {err}"


if __name__ == "__main__":
    if not os.environ.get("ROBOFLOW_API_KEY"):
        sys.exit("set ROBOFLOW_API_KEY first")
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(fetch, DATASETS):
            print(msg, flush=True)
