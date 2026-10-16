"""Ask the Roboflow API about candidate datasets WITHOUT downloading images.

For every project: classes + counts, every version with its image count, splits, resize
(preprocessing) and augmentation. Raw answers -> data/survey/<ws>__<proj>.json,
summary -> data/survey/survey.csv. Needs ROBOFLOW_API_KEY.

  python tools/survey_datasets.py configs/candidates.txt
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "survey"
KEY = os.environ.get("ROBOFLOW_API_KEY")


def get(url):
    for i in range(4):
        try:
            r = requests.get(url, params={"api_key": KEY}, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 404):
                return {"error": f"HTTP {r.status_code}"}
        except requests.RequestException as e:
            err = str(e)
        time.sleep(2 * (i + 1))
    return {"error": "request failed"}


def describe_version(v):
    pre = v.get("preprocessing") or {}
    aug = v.get("augmentation") or {}
    resize = pre.get("resize") or {}
    rs = f"{resize.get('format', '')} {resize.get('width', '')}x{resize.get('height', '')}".strip() if resize else "none"
    gray = "grayscale" in pre
    return {
        "version": v.get("id", "").split("/")[-1],
        "images": v.get("images"),
        "splits": v.get("splits"),
        "resize": rs,
        "grayscale_pre": gray,
        "augmentation": ",".join(k for k in aug if k not in ("image",)) or "none",
        "aug_outputs": (aug.get("image") or {}).get("versions"),
    }


def survey(line):
    url = line.strip().rstrip("/")
    parts = url.split("universe.roboflow.com/")[-1].split("/")
    ws, proj = parts[0], parts[1]
    d = get(f"https://api.roboflow.com/{ws}/{proj}")
    (OUT / f"{ws}__{proj}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf8")
    if "error" in d or "project" not in d:
        return {"ws": ws, "proj": proj, "error": d.get("error", "no project")}
    p = d["project"]
    vers = [describe_version(v) for v in d.get("versions", [])]
    return {"ws": ws, "proj": proj, "name": p.get("name"), "type": p.get("type"),
            "images": p.get("images"), "classes": p.get("classes"), "license": p.get("license"),
            "versions": vers}


if __name__ == "__main__":
    if not KEY:
        sys.exit("set ROBOFLOW_API_KEY")
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [l for l in open(sys.argv[1], encoding="utf8") if "roboflow.com" in l]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(survey, lines))
    (OUT / "survey.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    with open(OUT / "survey.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["ws", "proj", "type", "project_images", "classes", "version", "v_images", "resize", "augmentation", "error"])
        for r in res:
            if "error" in r:
                w.writerow([r["ws"], r["proj"], "", "", "", "", "", "", "", r["error"]])
                continue
            cls = ";".join(f"{k}:{v}" for k, v in (r["classes"] or {}).items())
            for v in r["versions"] or [{}]:
                w.writerow([r["ws"], r["proj"], r["type"], r["images"], cls, v.get("version"), v.get("images"),
                            v.get("resize"), v.get("augmentation"), ""])
    print(f"{len(res)} projects, {sum('error' in r for r in res)} errors -> {OUT / 'survey.csv'}")
