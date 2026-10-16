"""Board-level split + tiling -> final_dataset/ (YOLO format, ready for Colab).

* Every physical board (groups.csv) is in exactly ONE split: the test score is measured on boards
  the model has never seen. The assignment of boards to train/val/test is found by exhaustive
  search (3^boards) to hit the target shares of images and of boxes per class.
* Full-resolution originals (long side > --tile-above) are cut into --tile px tiles with overlap,
  so a missing hole stays ~40 px instead of ~10 px. Train also gets the whole photo (resized)
  for context, and only a share of the empty tiles (negatives). Val/test keep every tile.
* Low-resolution exports are copied as they are.

  python tools/build_dataset.py
"""
import argparse
import csv
import itertools
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "all_data"
SPLITS = ("train", "val", "test")


def read_boxes(p):
    out = []
    for ln in p.read_text(encoding="utf8").split("\n"):
        v = ln.split()
        if len(v) == 5:
            out.append((int(v[0]), *map(float, v[1:])))
    return out


def choose_split(boards, n_cls, target, seed_rules):
    """boards: {id: (images, Counter cls->boxes)}. Exhaustive search, returns {id: split}."""
    ids = sorted(boards)
    tot_img = sum(b[0] for b in boards.values())
    tot_cls = [sum(b[1][c] for b in boards.values()) for c in range(n_cls)]
    best, best_cost = None, 1e9
    for assign in itertools.product(range(3), repeat=len(ids)):
        if not all(rule(ids, assign) for rule in seed_rules):
            continue
        img = [0, 0, 0]
        cls = [[0] * n_cls for _ in range(3)]
        for i, s in zip(ids, assign):
            img[s] += boards[i][0]
            for c in range(n_cls):
                cls[s][c] += boards[i][1][c]
        if min(img) == 0:
            continue
        cost = sum((img[s] / tot_img - target[s]) ** 2 for s in range(3)) * 2
        for c in range(n_cls):
            if tot_cls[c]:
                cost += sum((cls[s][c] / tot_cls[c] - target[s]) ** 2 for s in range(3))
        if cost < best_cost:
            best, best_cost = dict(zip(ids, (SPLITS[s] for s in assign))), cost
    return best


def tiles(w, h, t, overlap):
    step = int(t * (1 - overlap))
    xs = list(range(0, max(1, w - t), step)) + [max(0, w - t)]
    ys = list(range(0, max(1, h - t), step)) + [max(0, h - t)]
    return sorted({(x, y) for x in xs for y in ys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_dataset")
    ap.add_argument("--target", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-above", type=int, default=2000, help="tile images whose long side is larger")
    ap.add_argument("--neg-ratio", type=float, default=0.35, help="train: empty tiles kept per positive tile")
    ap.add_argument("--min-visible", type=float, default=0.5, help="keep a cut box if this share is inside the tile")
    ap.add_argument("--full-side", type=int, default=1600, help="train: whole original resized to this long side")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    cfg = yaml.safe_load(open(ROOT / "configs" / "class_map.yaml", encoding="utf8"))
    names = cfg["final_classes"]
    rows = [r for r in csv.DictReader(open(A / "groups.csv", encoding="utf8")) if r["keep"] == "1"]

    # board statistics + split ---------------------------------------------------
    boards = defaultdict(lambda: [0, Counter()])
    for r in rows:
        r["boxes"] = read_boxes(A / "labels_final" / (r["file"][:-4] + ".txt"))
        b = boards[r["board"]]
        b[0] += 1
        b[1].update(c for c, *_ in r["boxes"])
    rules = []
    rare = [c for c in range(len(names)) if sum(1 for b in boards.values() if b[1][c]) >= 2]
    for c in rare:   # a class that lives on >= 2 boards must be in train AND test
        has = [bid for bid in sorted(boards) if boards[bid][1][c]]
        rules.append(lambda ids, asg, has=has: {asg[ids.index(h)] for h in has} >= {0, 2})
    split_of = choose_split({k: (v[0], v[1]) for k, v in boards.items()}, len(names), a.target, rules)

    out = ROOT / a.out
    shutil.rmtree(out, ignore_errors=True)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    man = []

    def write(split, stem, img, boxes_px, src, board, kind, x0=0, y0=0):
        h, w = img.shape[:2]
        cv2.imwrite(str(out / "images" / split / (stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [f"{c} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} {(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}"
                 for c, bx0, by0, bx1, by1 in boxes_px]
        (out / "labels" / split / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
        cnt = Counter(c for c, *_ in boxes_px)
        man.append({"file": stem + ".jpg", "split": split, "source_file": src, "board": board, "kind": kind,
                    "x0": x0, "y0": y0, "width": w, "height": h, "boxes": len(boxes_px),
                    **{f"n_{n}": cnt[i] for i, n in enumerate(names)}})

    for r in rows:
        split = split_of[r["board"]]
        img = cv2.imread(str(A / "images" / r["file"]))
        h, w = img.shape[:2]
        px = [(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h) for c, cx, cy, bw, bh in r["boxes"]]
        stem = r["file"][:-4]
        if max(h, w) <= a.tile_above:
            write(split, stem, img, px, r["file"], r["board"], "image")
            continue
        if split == "train":   # whole board, resized, for context
            s = a.full_side / max(h, w)
            write(split, stem + "__full", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                  [(c, x0 * s, y0 * s, x1 * s, y1 * s) for c, x0, y0, x1, y1 in px], r["file"], r["board"], "full")
        pos, neg = [], []
        for tx, ty in tiles(w, h, a.tile, a.overlap):
            tw, th = min(a.tile, w - tx), min(a.tile, h - ty)
            kept = []
            for c, x0, y0, x1, y1 in px:
                ix0, iy0, ix1, iy1 = max(x0, tx), max(y0, ty), min(x1, tx + tw), min(y1, ty + th)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                if (ix1 - ix0) * (iy1 - iy0) >= a.min_visible * (x1 - x0) * (y1 - y0):
                    kept.append((c, ix0 - tx, iy0 - ty, ix1 - tx, iy1 - ty))
            (pos if kept else neg).append((tx, ty, tw, th, kept))
        if split == "train":
            neg = rnd.sample(neg, min(len(neg), int(np.ceil(len(pos) * a.neg_ratio))))
        for tx, ty, tw, th, kept in pos + neg:
            write(split, f"{stem}__t{tx}_{ty}", img[ty:ty + th, tx:tx + tw], kept, r["file"], r["board"], "tile", tx, ty)

    with open(out / "split_manifest.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        wr.writeheader()
        wr.writerows(man)
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
                    "nc": len(names), "names": names}, open(out / "data.yaml", "w", encoding="utf8"), sort_keys=False)

    # report ------------------------------------------------------------------
    L = ["SOLDER/PCB-DEFECT DATASET REPORT", "=" * 40, f"classes: {names}", ""]
    L.append("board -> split (boards never cross splits):")
    for bid in sorted(boards, key=int):
        L.append(f"  board {bid:>2}: {split_of[bid]:5}  {boards[bid][0]:3} source images  "
                 + "  ".join(f"{n}={boards[bid][1][i]}" for i, n in enumerate(names)))
    L.append("")
    L.append(f"{'split':6} {'files':>6} {'tiles':>6} {'empty':>6} " + " ".join(f"{n:>13}" for n in names))
    for s in SPLITS:
        m = [x for x in man if x["split"] == s]
        L.append(f"{s:6} {len(m):6} {sum(x['kind'] == 'tile' for x in m):6} {sum(x['boxes'] == 0 for x in m):6} "
                 + " ".join(f"{sum(x['n_' + n] for x in m):13}" for n in names))
    src = Counter(r["source"] for r in rows)
    L += ["", "unique source images per dataset: " + ", ".join(f"{k}={v}" for k, v in src.most_common())]
    (out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
