"""Pick the least-augmented copy of every source image from Roboflow exports.

Roboflow names every exported file `<name>_<ext>.rf.<hash>.<ext>`, so all augmented
copies of one source image share the part before `.rf.`. Within a group we keep the
copy with the least augmentation evidence:
  * black rotation fill (fraction of near-black pixels)
  * salt-and-pepper noise (difference to a 3x3 median)
  * label box area (rotation / shear inflates the boxes)
Exact pixel duplicates are dropped. Source folders are never modified.

Usage: python extract_originals.py <dataset_dir> <out_dir>
"""
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).parent))
from yololib import parse_line
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}
RF_RE = re.compile(r"\.rf\.[0-9a-f]+$")


def base_name(stem):
    return RF_RE.sub("", stem)


def box_stats(label_path):
    if not label_path.exists():
        return 0.0, 0
    areas = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        b = parse_line(ln)
        if b:
            areas.append(b[3] * b[4])
    return (float(np.mean(areas)) if areas else 0.0), len(areas)


def _feats(path, side=640):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    s = side / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    out = []
    for flip in (None, 1, 0):                      # original, mirrored left-right, up-down
        im = g if flip is None else cv2.flip(g, flip)
        kp, d = cv2.SIFT_create(nfeatures=800).detectAndCompute(im, None)
        out.append((np.float32([k.pt for k in kp]), d) if d is not None and len(kp) >= 8 else (None, None))
    return out


def _inliers(a, b, bf=cv2.BFMatcher(cv2.NORM_L2)):
    (pa, da), (pb, db) = a, b
    if da is None or db is None:
        return 0
    m = bf.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 10:
        return 0
    _, inl = cv2.estimateAffinePartial2D(pa[[g.queryIdx for g in good]], pb[[g.trainIdx for g in good]],
                                         method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return int(inl.sum()) if inl is not None else 0


def same_content_clusters(paths, min_inliers=25):
    """Group copies that show the same picture (any rotation, flip, crop-free). Returns lists of indices."""
    if len(paths) == 1:
        return [[0]]
    f = [_feats(p) for p in paths]
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if find(i) != find(j) and max(_inliers(f[i][0], f[j][k]) for k in range(3)) >= min_inliers:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(paths)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def metrics(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    black = float((img.max(axis=2) <= 3).mean())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = float(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)).mean())
    digest = hashlib.md5(img.tobytes()).hexdigest()
    area, n = box_stats(label_path)
    return {"black": black, "noise": noise, "area": area, "n": n, "md5": digest}


def ranks(values):
    order = np.argsort(values, kind="stable")
    r = np.empty(len(values))
    r[order] = np.arange(len(values))
    return r


def main(src, dst):
    src, dst = Path(src), Path(dst)
    out_img, out_lbl = dst / "images", dst / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                groups[base_name(p.stem)].append((split, p, src / split / "labels" / (p.stem + ".txt")))

    def handle(job):
        name, items = job
        rows = []
        seen, cands = set(), []
        for split, img, lbl in items:
            m = metrics(img, lbl)
            if m is None or m["md5"] in seen:
                continue
            seen.add(m["md5"])
            cands.append((split, img, lbl, m))
        if not cands:
            return rows
        # Same base name does NOT always mean augmented copies: some exports cut one photo into
        # tiles, or several different photos were uploaded with the same file name. Only copies
        # that show the same content (SIFT, rotation- and flip-tolerant) are merged.
        clusters = same_content_clusters([c[1] for c in cands])
        for k, members in enumerate(clusters):
            cl = [cands[i] for i in members]
            # least augmented: no rotation fill, no noise, tight boxes - but never a copy that lost its boxes
            score = sum(ranks([c[3][key] for c in cl]) for key in ("black", "noise", "area")) \
                + 3 * ranks([-c[3]["n"] for c in cl])
            split, img, lbl, m = cl[int(np.argmin(score))]
            out_name = name if len(clusters) == 1 else f"{name}__part{k}"
            shutil.copy2(img, out_img / (out_name + img.suffix))
            if lbl.exists():
                shutil.copy2(lbl, out_lbl / (out_name + ".txt"))
            rows.append({
                "source_name": out_name, "base_name": name, "kept_file": img.name, "original_split": split,
                "copies_in_group": len(items), "unique_copies": len(cl), "content_clusters": len(clusters),
                "splits_in_group": "+".join(sorted({c[0] for c in cl})),
                "black_frac": round(m["black"], 4), "noise": round(m["noise"], 3),
                "mean_box_area": round(m["area"], 5), "boxes": m["n"],
            })
        return rows

    with ThreadPoolExecutor(int(os.environ.get("EXTRACT_WORKERS", "8"))) as ex:
        rows = [r for rs in ex.map(handle, sorted(groups.items())) for r in rs]

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    yaml = src / "data.yaml"
    if yaml.exists():
        shutil.copy2(yaml, dst / "source_data.yaml")
    leaks = sum(1 for r in rows if "+" in r["splits_in_group"])
    print(f"{src.name}: {sum(len(v) for v in groups.values())} files -> {len(rows)} source images; "
          f"{leaks} groups were spread over several splits (leakage)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
