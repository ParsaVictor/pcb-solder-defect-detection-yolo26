"""Remove cross-source copies and group images of the same physical board.

1. Duplicates: Roboflow keeps the original file name, so the same photo exported by several
   sources has the same base name (e.g. 01_missing_hole_01_jpg). One copy per name is kept:
   the copy whose boxes agree best with the other copies (consensus), then more boxes, then
   higher resolution. Groups whose copies disagree are written to dup_disagreements.csv.
2. Boards: SIFT + RANSAC between the kept images (any source, any lighting / crop / rotation).
   Connected images = same board -> build_dataset.py keeps every board in ONE split (no leakage).

Writes data/all_data/groups.csv (file, keep, dup_of, board).
"""
import argparse
import csv
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "all_data"
BOARD_RE = re.compile(r"^(?:light_)?(\d{2})_(?:missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)")


def norm_base(base):
    b = base.lower()
    b = re.sub(r"_(jpg|jpeg|png|bmp)$", "", b)
    return b


def boxes(file):
    p = A / "labels" / (Path(file).stem + ".txt")
    out = []
    for ln in p.read_text(encoding="utf8").split("\n"):
        v = ln.split()
        if len(v) == 5:
            c, cx, cy, w, h = int(v[0]), *map(float, v[1:])
            out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    u = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / u if u > 0 else 0


def f1(A_, B_, thr=0.4):
    if not A_ and not B_:
        return 1.0
    if not A_ or not B_:
        return 0.0
    used, tp = set(), 0
    for a in A_:
        best, bj = 0, -1
        for j, b in enumerate(B_):
            if j not in used and a[0] == b[0]:
                v = iou(a, b)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used.add(bj)
            tp += 1
    return 2 * tp / (len(A_) + len(B_))


def content_area(img):
    """Pixels that are real photo, not letterbox bars."""
    m = img.max(axis=2) > 12
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return 0 if len(rows) == 0 else (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)


def affine(fa, fb):
    """Full affine a->b (rotation, stretch, letterbox shift) or None."""
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 12:
        return None
    M, inl = cv2.estimateAffine2D(pa[[g.queryIdx for g in good]], pb[[g.trainIdx for g in good]],
                                  method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return M if inl is not None and inl.sum() >= 12 else None


def warp(bxs, M):
    out = []
    for c, x0, y0, x1, y1 in bxs:
        pts = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ M[:, :2].T + M[:, 2]
        out.append((c, *pts.min(0), *pts.max(0)))
    return out


def resolve_group(rs):
    """Pick the best copy of one photo and restore boxes that most other copies agree on.

    Returns (kept file, log entry or None). Repaired labels go to labels_final/.
    """
    if len(rs) == 1:
        f = rs[0]["file"]
        (A / "labels_final" / (Path(f).stem + ".txt")).write_text(
            (A / "labels" / (Path(f).stem + ".txt")).read_text(encoding="utf8"), encoding="utf8")
        return f, None
    imgs = [cv2.imread(str(A / "images" / r["file"])) for r in rs]
    px = []
    for r, im in zip(rs, imgs):
        h, w = im.shape[:2]
        px.append([(c, x0 * w, y0 * h, x1 * w, y1 * h) for c, x0, y0, x1, y1 in boxes(r["file"])])
    feats = []
    for im in imgs:
        s = min(1.0, 1600 / max(im.shape[:2]))           # full-resolution originals: detect on a smaller copy
        g = cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im, cv2.COLOR_BGR2GRAY)
        kp, d = cv2.SIFT_create(nfeatures=3000).detectAndCompute(cv2.createCLAHE(2.0, (8, 8)).apply(g), None)
        feats.append((np.float32([k.pt for k in kp]) / s, d) if d is not None else (None, None))
    area = [content_area(im) for im in imgs]
    ref = int(np.argmax(area))
    to_ref = [np.float32([[1, 0, 0], [0, 1, 0]]) if i == ref else affine(feats[i], feats[ref]) for i in range(len(rs))]
    ok = [i for i in range(len(rs)) if to_ref[i] is not None]
    in_ref = {i: warp(px[i], to_ref[i]) for i in ok}

    def cons(i):
        o = [f1(in_ref[i], in_ref[j], 0.3) for j in ok if j != i]
        return float(np.mean(o)) if o else 1.0
    # defects are ~10 px in the 640 px exports and ~40 px in the originals: resolution wins
    # whenever that copy's labels agree reasonably with the others
    win = max(ok, key=lambda i: (cons(i) >= 0.6, area[i], round(cons(i), 3), len(px[i])))
    before = cons(win)
    # voting: a box missing in the winner but present in >= half of the other aligned copies is restored
    others = [j for j in ok if j != win]
    need = max(1, int(np.ceil(len(others) / 2)))
    cand = [b for j in others for b in in_ref[j]]
    restored = []
    for b in cand:
        if any(b[0] == w[0] and iou(b, w) >= 0.3 for w in in_ref[win] + restored):
            continue
        votes = sum(any(b[0] == o[0] and iou(b, o) >= 0.3 for o in in_ref[j]) for j in others)
        if votes >= need:
            restored.append(b)
    h, w = imgs[win].shape[:2]
    final = list(px[win])
    if restored:
        inv = cv2.invertAffineTransform(to_ref[win])
        for c, x0, y0, x1, y1 in warp(restored, inv):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                final.append((c, x0, y0, x1, y1))
    lines = [f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
             for c, x0, y0, x1, y1 in final]
    (A / "labels_final" / (Path(rs[win]["file"]).stem + ".txt")).write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
    entry = (norm_base(rs[win]["base"]), len(rs), rs[win]["source"], round(before, 3), len(final) - len(px[win]),
             " | ".join(f"{r['source']}:{len(px[i])}{'' if i in ok else '(unaligned)'}" for i, r in enumerate(rs)))
    return rs[win]["file"], entry


def describe(path, side=512, n=600):
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA))
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(g, None)
    return (np.float32([k.pt for k in kp]), d) if d is not None and len(kp) >= 8 else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(A / "manifest.csv", encoding="utf8")) if r["excluded"] == "0"]
    (A / "labels_final").mkdir(exist_ok=True)
    for p in (A / "labels_final").glob("*.txt"):
        p.unlink()

    # 1 duplicates ---------------------------------------------------------------
    by_name = defaultdict(list)
    for r in rows:
        by_name[norm_base(r["base"])].append(r)
    groups = [rs for rs in by_name.values()]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(resolve_group, groups))
    keep, dup_of, log = {}, {}, []
    for rs, (best, entry) in zip(groups, results):
        for r in rs:
            keep[r["file"]] = r["file"] == best
            if r["file"] != best:
                dup_of[r["file"]] = best
        if entry:
            log.append(entry)
    kept = [r for r in rows if keep[r["file"]]]
    added = sum(e[4] for e in log)
    disagree = [e for e in log if e[3] < 0.8]
    print(f"{len(rows)} usable images -> {len(kept)} unique ({len(rows) - len(kept)} cross-source copies removed); "
          f"{added} boxes restored by voting; {len(disagree)} names still below 0.8 agreement after alignment")

    # 2 boards (SIFT) ------------------------------------------------------------
    with ThreadPoolExecutor(a.workers) as ex:
        feats = list(ex.map(lambda r: describe(A / "images" / r["file"]), kept))
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def compare(i):
        out = []
        pi, di = feats[i]
        if di is None:
            return out
        for j in range(i + 1, len(kept)):
            pj, dj = feats[j]
            if dj is None:
                continue
            m = bf.knnMatch(di, dj, k=2)
            good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
            if len(good) < a.min_inliers:
                continue
            M, inl = cv2.estimateAffine2D(pi[[g.queryIdx for g in good]], pj[[g.trainIdx for g in good]],
                                          method=cv2.RANSAC, ransacReprojThreshold=6.0)
            if inl is not None and int(inl.sum()) >= a.min_inliers:
                out.append((i, j, int(inl.sum())))
        return out

    with ThreadPoolExecutor(a.workers) as ex:
        pairs = [p for res in ex.map(compare, range(len(kept))) for p in res]
    parent = list(range(len(kept)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    # the file name also carries the board number (01_missing_hole_05, light_10_missing_hole_02_1)
    first = {}
    for i, r in enumerate(kept):
        m = BOARD_RE.match(norm_base(r["base"]))
        if m:
            parent[find(i)] = find(first.setdefault(m.group(1), i))
    roots = {}
    board = {r["file"]: roots.setdefault(find(i), len(roots)) for i, r in enumerate(kept)}
    sizes = defaultdict(int)
    for b in board.values():
        sizes[b] += 1
    print(f"{len(pairs)} same-board pairs -> {len(roots)} board groups; largest: "
          f"{sorted(sizes.values(), reverse=True)[:12]}")

    with open(A / "groups.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "keep", "dup_of", "board"])
        for r in rows:
            k = keep[r["file"]]
            w.writerow([r["file"], r["source"], int(k), dup_of.get(r["file"], ""),
                        board[r["file"]] if k else board[dup_of[r["file"]]]])
    with open(A / "duplicates_log.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "copies", "kept_source", "agreement_f1", "boxes_restored", "boxes_per_source"])
        w.writerows(sorted(log, key=lambda e: e[3]))
    with open(A / "board_pairs.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "file_b", "inliers"])
        w.writerows((kept[i]["file"], kept[j]["file"], n) for i, j, n in pairs)


if __name__ == "__main__":
    main()
