"""High-resolution version of the `aaa` source, without Roboflow's mirrored padding.

aaa exists as v4 (fit into 512, WHITE padding) and v1 (fit into 2048, REFLECT padding). v1 has real
extra detail, but its padding is a mirror image of the board: defects inside it carry no label and
would teach the model that a defect is background. Both versions fit the same photo the same way, so
the white padding of v4 tells exactly where the real photo is inside v1.

For every v1 file: find its v4 twin (same base name, closest content), take the v4 content box,
crop v1 to that box, move the labels into the crop (boxes less than half inside are dropped),
shrink so the long side is at most --max-side, write a normal Roboflow-style export:

  data/raw/aaa_hr/{train,valid,test}/{images,labels}, data.yaml, aaa_hr_log.csv

  python tools/aaa_hr.py
"""
import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from yololib import parse_line

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RF_RE = re.compile(r"\.rf\.[0-9a-f]+$")


def content_box(img512, white=246):
    """Normalised (x0, y0, x1, y1) of the photo inside its white fit-padding."""
    notwhite = (img512.min(axis=2) < white)
    rows, cols = np.where(notwhite.any(1))[0], np.where(notwhite.any(0))[0]
    if len(rows) == 0:
        return 0.0, 0.0, 1.0, 1.0
    h, w = notwhite.shape
    return cols[0] / w, rows[0] / h, (cols[-1] + 1) / w, (rows[-1] + 1) / h


def group(version):
    g = defaultdict(list)
    for split in ("train", "valid", "test"):
        d = RAW / version / split / "images"
        for p in sorted(d.glob("*")) if d.exists() else []:
            g[(split, RF_RE.sub("", p.stem))].append(p)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default="aaa_v1_2048")
    ap.add_argument("--lo", default="aaa")
    ap.add_argument("--out", default="aaa_hr")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    hi, lo = group(a.hi), group(a.lo)
    out = RAW / a.out
    shutil.rmtree(out, ignore_errors=True)
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    shutil.copy2(RAW / a.hi / "data.yaml", out / "data.yaml")

    def handle(key):
        split, base = key
        rows = []
        his, los = hi[key], lo.get(key, [])
        if not los:
            return [{"file": p.name, "status": "no 512 twin - skipped"} for p in his]
        lo_imgs = [cv2.imread(str(p)) for p in los]
        boxes = [content_box(im) for im in lo_imgs]
        used = set()
        for p in his:
            big = cv2.imread(str(p))
            H, W = big.shape[:2]
            small = cv2.resize(big, (lo_imgs[0].shape[1], lo_imgs[0].shape[0]), interpolation=cv2.INTER_AREA).astype(np.int16)
            best, bi = 1e9, -1
            for i, (im, (x0, y0, x1, y1)) in enumerate(zip(lo_imgs, boxes)):
                if i in used:
                    continue
                h, w = im.shape[:2]
                ya, yb, xa, xb = int(y0 * h) + 2, int(y1 * h) - 2, int(x0 * w) + 2, int(x1 * w) - 2
                if yb <= ya or xb <= xa:
                    continue
                d = float(np.abs(small[ya:yb, xa:xb] - im[ya:yb, xa:xb].astype(np.int16)).mean())
                if d < best:
                    best, bi = d, i
            if bi < 0:
                rows.append({"file": p.name, "status": "no match - skipped"})
                continue
            used.add(bi)
            x0, y0, x1, y1 = boxes[bi]
            cx0, cy0, cx1, cy1 = int(round(x0 * W)), int(round(y0 * H)), int(round(x1 * W)), int(round(y1 * H))
            crop = big[cy0:cy1, cx0:cx1]
            ch, cw = crop.shape[:2]
            lines, dropped = [], 0
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
            for ln in (lbl.read_text(encoding="utf8").splitlines() if lbl.exists() else []):
                b = parse_line(ln)
                if not b:
                    continue
                c, bx, by, bw, bh = b
                ax0, ay0, ax1, ay1 = (bx - bw / 2) * W - cx0, (by - bh / 2) * H - cy0, (bx + bw / 2) * W - cx0, (by + bh / 2) * H - cy0
                ix0, iy0, ix1, iy1 = max(0, ax0), max(0, ay0), min(cw, ax1), min(ch, ay1)
                area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
                if ix1 <= ix0 or iy1 <= iy0 or (ix1 - ix0) * (iy1 - iy0) < 0.5 * area:
                    dropped += 1
                    continue
                lines.append(f"{c} {(ix0 + ix1) / 2 / cw:.6f} {(iy0 + iy1) / 2 / ch:.6f} {(ix1 - ix0) / cw:.6f} {(iy1 - iy0) / ch:.6f}")
            s = a.max_side / max(ch, cw)
            if s < 1:
                crop = cv2.resize(crop, (round(cw * s), round(ch * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / split / "images" / p.name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / split / "labels" / (p.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
            rows.append({"file": p.name, "status": "ok", "match_diff": round(best, 2), "crop_w": cw, "crop_h": ch,
                         "saved_w": crop.shape[1], "saved_h": crop.shape[0], "boxes": len(lines), "boxes_dropped": dropped})
        return rows

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(handle, sorted(hi)) for r in rs]
    keys = ["file", "status", "match_diff", "crop_w", "crop_h", "saved_w", "saved_h", "boxes", "boxes_dropped"]
    with open(out / "aaa_hr_log.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    diffs = np.array([r["match_diff"] for r in ok])
    print(f"{len(ok)} of {len(rows)} converted; match difference median {np.median(diffs):.1f}, "
          f"worst {diffs.max():.1f}; saved size median {int(np.median([r['saved_w'] for r in ok]))}x"
          f"{int(np.median([r['saved_h'] for r in ok]))}; boxes {sum(r['boxes'] for r in ok)} kept, "
          f"{sum(r['boxes_dropped'] for r in ok)} dropped (outside the real photo)")


if __name__ == "__main__":
    main()
