"""Contact sheets for visual label review of a built dataset:  review/<family>/<split>_NNN.jpg

Every box is drawn in its class colour with the class name; caption = source / file / box count.

  python tools/make_review.py bare
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
COL = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
       (128, 0, 255), (0, 128, 255), (180, 180, 180)]
CELL = 420


def tile(img_p, lbl_p, names):
    im = cv2.imread(str(img_p))
    h, w = im.shape[:2]
    th = max(2, round(max(h, w) / CELL * 2))
    n = 0
    for ln in (lbl_p.read_text(encoding="utf8").split("\n") if lbl_p.exists() else []):
        v = ln.split()
        if len(v) == 5:
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
            p0 = (int((cx - bw / 2) * w) - th, int((cy - bh / 2) * h) - th)
            cv2.rectangle(im, p0, (int((cx + bw / 2) * w) + th, int((cy + bh / 2) * h) + th), COL[c % len(COL)], th)
            cv2.putText(im, names[c][:12], (p0[0], max(10, p0[1] - th)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * th, COL[c % len(COL)], max(1, th // 2))
            n += 1
    s = CELL / max(h, w)
    im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((CELL + 30, CELL, 3), 30, np.uint8)
    canvas[30:30 + im.shape[0], :im.shape[1]] = im
    cv2.putText(canvas, f"{img_p.stem[:50]} [{n}]", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


def main(family, per=30, cols=6):
    F = ROOT / f"final_{family}"
    names = yaml.safe_load(open(F / "data.yaml", encoding="utf8"))["names"]
    out = ROOT / "review" / family
    shutil_rm = [p.unlink() for p in out.glob("*.jpg")] if out.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        ims = sorted((F / "images" / s).glob("*.jpg"))
        # every source spread over the sheets: interleave by source
        by_src = {}
        for p in ims:
            by_src.setdefault(p.stem.split("__")[0], []).append(p)
        order = [v[k] for k in range(max(len(v) for v in by_src.values())) for v in by_src.values() if k < len(v)]
        for k in range(0, len(order), per):
            cells = [tile(p, F / "labels" / s / (p.stem + ".txt"), names) for p in order[k:k + per]]
            while len(cells) % cols:
                cells.append(np.full_like(cells[0], 30))
            sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
            cv2.imwrite(str(out / f"{s}_{k // per + 1:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"review/{family} {s}: {len(ims)} files")


if __name__ == "__main__":
    for fam in sys.argv[1:] or ["bare", "solder"]:
        main(fam)
