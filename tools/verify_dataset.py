"""Check final_dataset before training (runs locally and on Colab, < 1 minute).

Hard errors (exit code 1): missing/invalid label files, values out of range, unknown class id,
unreadable image, a board in two splits, the same pixels in two splits, an empty split.
Warnings: classes with few boxes in val/test, tiny boxes.

  python verify_dataset.py --root final_dataset
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    cfg = yaml.safe_load(open(root / "data.yaml", encoding="utf8"))
    names = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else cfg["names"]
    err, warn, stats = [], [], {}
    seen_md5 = {}
    for sp in ("train", "val", "test"):
        imgs = sorted((root / "images" / sp).glob("*.jpg"))
        cnt, n_box, tiny, empty = Counter(), 0, 0, 0
        if not imgs:
            err.append(f"{sp}: no images")
        for im in imgs:
            lbl = root / "labels" / sp / (im.stem + ".txt")
            if not lbl.exists():
                err.append(f"{sp}: missing label for {im.name}")
                continue
            img = cv2.imread(str(im))
            if img is None:
                err.append(f"{sp}: unreadable {im.name}")
                continue
            h, w = img.shape[:2]
            md5 = hashlib.md5(img.tobytes()).hexdigest()
            if md5 in seen_md5 and seen_md5[md5][0] != sp:
                err.append(f"identical image in {seen_md5[md5][0]} and {sp}: {seen_md5[md5][1]} / {im.name}")
            seen_md5.setdefault(md5, (sp, im.name))
            lines = [l.split() for l in lbl.read_text(encoding="utf8").splitlines() if l.strip()]
            empty += not lines
            for p in lines:
                try:
                    c, x, y, bw, bh = int(p[0]), *map(float, p[1:5])
                    assert len(p) == 5
                except Exception:
                    err.append(f"{sp}: bad line in {lbl.name}: {p}")
                    continue
                if not (0 <= c < len(names)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    err.append(f"{sp}: out-of-range value in {lbl.name}: {p}")
                    continue
                cnt[names[c]] += 1
                n_box += 1
                tiny += min(bw * w, bh * h) < 6
        stats[sp] = {"files": len(imgs), "boxes": n_box, "empty_files": empty, "boxes_under_6px": tiny,
                     "per_class": {n: cnt[n] for n in names}}
        for n in names:
            if sp != "train" and cnt[n] < 15:
                warn.append(f"{sp}: class '{n}' has only {cnt[n]} boxes (its metric will be noisy or missing)")
        if tiny > 0.05 * max(1, n_box):
            warn.append(f"{sp}: {tiny} boxes are under 6 px")
    man = root / "split_manifest.csv"
    if man.exists():
        where = defaultdict(set)
        for r in csv.DictReader(open(man, encoding="utf8")):
            where[r.get("group") or r.get("board")].add(r["split"])
        for b, s in where.items():
            if len(s) > 1:
                err.append(f"board {b} is in several splits: {sorted(s)}")
    else:
        warn.append("split_manifest.csv missing: board leakage not checked")
    rep = {"ok": not err, "errors": err[:50], "n_errors": len(err), "warnings": warn, "splits": stats}
    (root / "qa_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf8")
    for sp, s in stats.items():
        print(f"{sp:5} files={s['files']:4} boxes={s['boxes']:5} empty={s['empty_files']:3} {s['per_class']}")
    for w_ in warn:
        print("WARNING:", w_)
    for e in err[:20]:
        print("ERROR:", e)
    print("DATASET OK" if not err else f"DATASET HAS {len(err)} ERRORS")
    sys.exit(0 if not err else 1)


if __name__ == "__main__":
    main()
