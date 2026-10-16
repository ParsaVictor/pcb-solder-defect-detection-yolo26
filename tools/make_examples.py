"""Presentation figure: each defect shown on the whole board + zoomed in -> review/examples/defect_examples.jpg

  python tools/make_examples.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "all_data"
RED, WHITE, BG = (40, 40, 230), (255, 255, 255), (32, 32, 32)

# (file, box index, zoom context x box size, title, subtitle)
EXAMPLES = [
    ("thanakorn_missing_home__01_missing_hole_01_jpg.jpg", 0, 8,
     "1) MISSING HOLE", "One pad in the row was not drilled/finished - all its neighbours are complete"),
    ("nuu_rex_school__01_mouse_bite_10_jpg.jpg", 1, 5,
     "2) MOUSE BITE", "A small bite-shaped notch eaten into the edge of a copper trace"),
]


def box(f, k):
    ln = (A / "labels_final" / (f[:-4] + ".txt")).read_text().strip().split("\n")[k]
    return tuple(map(float, ln.split()[1:]))


def panel(f, k, ctx, title, sub, H=520):
    im = cv2.imread(str(A / "images" / f))
    h, w = im.shape[:2]
    cx, cy, bw, bh = box(f, k)
    x, y = int(cx * w), int(cy * h)
    half = int(max(bw * w, bh * h) * ctx / 2)
    x0, y0, x1, y1 = max(0, x - half), max(0, y - half), min(w, x + half), min(h, y + half)
    # whole board, zoom area marked
    s = H / h
    over = cv2.resize(im, (int(w * s), H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(over, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), RED, 3)
    # zoom, defect circled
    z = im[y0:y1, x0:x1]
    zs = H / z.shape[0]
    z = cv2.resize(z, (int(z.shape[1] * zs), H), interpolation=cv2.INTER_CUBIC)
    r = int(max(bw * w, bh * h) * zs * 0.85)
    cv2.circle(z, (int((x - x0) * zs), int((y - y0) * zs)), r, RED, 4)
    gap = np.full((H, 24, 3), BG, np.uint8)
    row = np.hstack([over, gap, z])
    head = np.full((96, row.shape[1], 3), BG, np.uint8)
    cv2.putText(head, title, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
    cv2.putText(head, sub, (8, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    foot = np.full((40, row.shape[1], 3), BG, np.uint8)
    cv2.putText(foot, "whole board (red box = zoom area)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    cv2.putText(foot, "zoomed in (red circle = defect)", (over.shape[1] + 32, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    return np.vstack([head, row, foot])


def main():
    rows = [panel(*e) for e in EXAMPLES]
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), BG, np.uint8)]) for r in rows]
    sep = np.full((30, W, 3), BG, np.uint8)
    out = ROOT / "review" / "examples" / "defect_examples.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([rows[0], sep, rows[1]]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(out)


if __name__ == "__main__":
    main()
