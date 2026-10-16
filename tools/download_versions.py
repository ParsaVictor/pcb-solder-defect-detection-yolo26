"""Download chosen Roboflow dataset versions robustly (resumable HTTP, retries, parallel).

configs/downloads.csv columns: workspace,project,version,folder,why
A folder that already has data.yaml is skipped. The API key is read from ROBOFLOW_API_KEY.

  python tools/download_versions.py [configs/downloads.csv] [--only folder ...]
"""
import argparse
import csv
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DL = ROOT / "data" / "_zips"
KEY = os.environ.get("ROBOFLOW_API_KEY")


def export_link(ws, proj, ver, fmt):
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/{fmt}"
    for i in range(60):                                   # the export may need to be generated first
        r = requests.get(url, params={"api_key": KEY}, timeout=120)
        if r.status_code == 200:
            d = r.json()
            link = (d.get("export") or {}).get("link")
            if link:
                return link
        time.sleep(10)
    raise RuntimeError(f"no export link ({r.status_code}: {r.text[:200]})")


def fetch(link, dst):
    for i in range(30):
        have = dst.stat().st_size if dst.exists() else 0
        try:
            with requests.get(link, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
                if r.status_code == 416:                  # already complete
                    return
                if have and r.status_code == 200:         # server ignored Range: start over
                    have = 0
                r.raise_for_status()
                total = have + int(r.headers.get("Content-Length", 0))
                with open(dst, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if dst.stat().st_size >= total:
                return
        except (requests.RequestException, OSError):
            pass
        time.sleep(min(60, 5 * (i + 1)))
    raise RuntimeError("download kept failing")


def one(row):
    ws, proj, ver, folder = row["workspace"], row["project"], row["version"], row["folder"]
    out = RAW / folder
    if (out / "data.yaml").exists():
        return f"SKIP {folder}"
    t = time.time()
    try:
        link = export_link(ws, proj, ver, "yolo26")
        z = DL / f"{folder}.zip"
        fetch(link, z)
        tmp = RAW / f".{folder}_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        zipfile.ZipFile(z).extractall(tmp)
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        z.unlink()
        n = sum(1 for _ in out.glob("*/images/*"))
        return f"OK   {folder}: {n} images, {time.time() - t:.0f}s"
    except Exception as e:
        return f"FAIL {folder}: {str(e)[:200]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "configs" / "downloads.csv"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not KEY:
        raise SystemExit("set ROBOFLOW_API_KEY")
    RAW.mkdir(parents=True, exist_ok=True)
    DL.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf8")) if not a.only or r["folder"] in a.only]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(one, rows):
            print(msg, flush=True)
