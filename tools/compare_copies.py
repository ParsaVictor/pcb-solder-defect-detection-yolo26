"""Side-by-side of every copy of one base name, boxes drawn (review/compare_<name>.jpg)."""
import sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "data/all_data"
name = sys.argv[1]
tiles = []
for p in sorted((A / "images").glob(f"*__{name}_*.jpg")):
    im = cv2.imread(str(p)); h, w = im.shape[:2]
    for ln in (A / "labels" / (p.stem + ".txt")).read_text().split("\n"):
        v = ln.split()
        if len(v) == 5:
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
            cv2.rectangle(im, (int((cx-bw/2)*w), int((cy-bh/2)*h)), (int((cx+bw/2)*w), int((cy+bh/2)*h)), (0,0,255), 3)
    im = cv2.resize(im, (int(w * 480 / h), 480))
    cv2.putText(im, p.stem.split("__")[0], (10, 30), 0, 1, (0,255,255), 2)
    tiles.append(im)
cv2.imwrite(str(ROOT / "review" / f"compare_{name}.jpg"), np.hstack(tiles))
