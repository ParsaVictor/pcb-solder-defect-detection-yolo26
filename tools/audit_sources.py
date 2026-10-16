"""Audit downloaded datasets: what is really original and what is a copy of another source.

For every data/raw/<folder> (with data.yaml):
  files, originals after removing Roboflow augment copies (same base name), resolution,
  colour vs grey, classes + box counts, and - against a REFERENCE set of originals
  (default: the PKU full-resolution photos) - the share of a random sample that is a
  crop/resize/rotation of a reference photo (SIFT + RANSAC).
Writes data/audit/audit.json and audit.csv, plus a contact sheet per source
(data/audit/sheet_<folder>.jpg) to look at.

  python tools/audit_sources.py --ref pku_tedpublic
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from yololib import parse_line

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "audit"
RF_RE = re.compile(r"\.rf\.[0-9a-f]+$")
IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def base(stem):
    return RF_RE.sub("", stem)


def feats(path, side=800, n=1500):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    s = side / max(im.shape)
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    im = cv2.createCLAHE(2.0, (8, 8)).apply(im)
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(im, None)
    return (np.float32([k.pt for k in kp]), d) if d is not None and len(kp) > 10 else (None, None)


def match(fa, fb, bf=cv2.BFMatcher(cv2.NORM_L2)):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None:
        return 0
    m = bf.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 15:
        return 0
    H, inl = cv2.findHomography(pa[[g.queryIdx for g in good]], pb[[g.trainIdx for g in good]], cv2.RANSAC, 5.0)
    return int(inl.sum()) if inl is not None else 0


def audit(folder, ref_feats, sample, workers):
    d = RAW / folder
    cfg = yaml.safe_load(open(d / "data.yaml", encoding="utf8"))
    names = cfg.get("names")
    names = list(names.values()) if isinstance(names, dict) else list(names or [])
    files = sorted(p for p in d.glob("*/images/*") if p.suffix.lower() in IMG)
    groups = {}
    for p in files:
        groups.setdefault(base(p.stem), p)
    originals = list(groups.values())
    boxes, polys = Counter(), 0
    for p in files:
        lbl = p.parent.parent / "labels" / (p.stem + ".txt")
        for ln in (lbl.read_text(encoding="utf8").splitlines() if lbl.exists() else []):
            b = parse_line(ln)
            if b:
                boxes[names[b[0]] if b[0] < len(names) else str(b[0])] += 1
                polys += len(ln.split()) > 5
    rnd = random.Random(0)
    smp = rnd.sample(originals, min(sample, len(originals)))
    sizes, grey = Counter(), 0
    for p in smp:
        im = cv2.imread(str(p))
        sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        grey += float(hsv[..., 1].mean()) < 12
    derived = name_share = None
    if ref_feats:
        index, owner, refs, ref_names = ref_feats
        with ThreadPoolExecutor(workers) as ex:
            fs = list(ex.map(feats, smp))

        def best(f):
            if f[1] is None:
                return 0
            m = index.knnMatch(f[1], k=2)
            votes = Counter(a.imgIdx for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance)
            return max((match(f, refs[i]) for i, _ in votes.most_common(3)), default=0)
        scores = [best(f) for f in fs]                    # FLANN is not shared across threads
        derived = round(sum(s >= 25 for s in scores) / max(1, len(scores)), 3)
        name_share = round(sum(norm(base(p.stem)) in ref_names for p in originals) / max(1, len(originals)), 3)
    # contact sheet of originals with boxes
    tiles = []
    for p in smp[:24]:
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        lbl = p.parent.parent / "labels" / (p.stem + ".txt")
        for ln in (lbl.read_text(encoding="utf8").splitlines() if lbl.exists() else []):
            b = parse_line(ln)
            if b:
                c, cx, cy, bw, bh = b
                cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)),
                              [(0, 0, 255), (255, 128, 0), (0, 200, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)][c % 8],
                              max(2, w // 300))
        s = 260 / max(h, w)
        t = np.full((260, 260, 3), 30, np.uint8)
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        t[:im.shape[0], :im.shape[1]] = im
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.full((260, 260, 3), 30, np.uint8))
    cv2.imwrite(str(OUT / f"sheet_{folder}.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    fam = Counter(re.sub(r"\d+", "#", base(p.stem))[:40] for p in originals).most_common(4)
    return {"folder": folder, "files": len(files), "originals": len(originals), "classes": names,
            "boxes": dict(boxes), "polygon_labels": polys, "sizes": dict(sizes.most_common(4)),
            "grey_share": round(grey / max(1, len(smp)), 2), "copy_of_ref_share": derived,
            "name_in_ref_share": name_share, "name_patterns": fam}


def norm(b):
    """01_missing_hole_01_jpg / light_01_missing_hole_01_2_600_jpg -> comparable PKU photo key."""
    b = re.sub(r"_(jpg|jpeg|png|bmp)$", "", b.lower())
    m = re.search(r"(\d{2})_(missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)_(\d{2})", b)
    return m.group(0) if m else b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="pku_tedpublic")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = RAW / a.ref
    ref_files = sorted({base(p.stem): p for p in ref.glob("*/images/*")}.values()) if ref.exists() else []
    with ThreadPoolExecutor(a.workers) as ex:
        refs = [f for f in ex.map(lambda p: feats(p, n=800), ref_files) if f[1] is not None]
    ref_feats = None
    if refs:
        index = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
        index.add([r[1] for r in refs])
        index.train()
        owner = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(refs)])
        ref_feats = (index, owner, refs, {norm(base(p.stem)) for p in ref_files})
    print(f"reference {a.ref}: {len(refs)} originals", flush=True)
    folders = [p.name for p in sorted(RAW.iterdir()) if (p / "data.yaml").exists() and (not a.only or p.name in a.only)]
    res = []
    for f in folders:
        r = audit(f, ref_feats if f != a.ref else None, a.sample, a.workers)
        res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    old = OUT / "audit.json"
    prev = {r["folder"]: r for r in json.loads(old.read_text(encoding="utf8"))} if old.exists() else {}
    prev.update({r["folder"]: r for r in res})
    old.write_text(json.dumps(list(prev.values()), indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
