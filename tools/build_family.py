"""Build one training dataset (a "family" in configs/families.yaml) from the cleaned sources.

  data/clean/<source>/  ->  final_<family>/{images,labels}/{train,val,test}, data.yaml,
                            split_manifest.csv, excluded.csv, report.txt

Steps
1. collect: class mapping (unknown class name = error), polygon -> box, clip, drop degenerate /
   duplicate boxes, explicit exclusions (logged with their reason).
2. exact duplicates: identical pixels (any source) keep one copy (logged).
3. groups = one physical board / scene: file-name rule, or image content (SIFT + RANSAC through a
   FLANN index, also across sources). A group never crosses splits.
4. split: randomised greedy search over groups for the target share of images AND of boxes of every
   class; a class that lives in >= 3 groups must appear in val and in test.
5. "train_only" duplicates (mirrored / rotated copies) stay in train, are dropped from val/test.
6. photos larger than --tile-above: overlapping --tile px tiles (train: whole photo resized too,
   and only a share of empty tiles).

  python tools/build_family.py bare
"""
import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from yololib import parse_line

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- 1 collect
def collect(fam, log_excl):
    names = fam["classes"]
    items = []
    for src, sc in fam["sources"].items():
        d = CLEAN / src
        raw_names = yaml.safe_load(open(d / "source_data.yaml", encoding="utf8"))["names"]
        raw_names = list(raw_names.values()) if isinstance(raw_names, dict) else raw_names
        cmap = {str(k): v for k, v in sc["map"].items()}
        excl = sc.get("exclude") or {}
        for img in sorted((d / "images").iterdir()):
            reason = next((why for pat, why in excl.items() if re.search(pat, img.stem)), None)
            if reason:
                log_excl.append((src, img.name, reason))
                continue
            boxes, lbl = [], d / "labels" / (img.stem + ".txt")
            for ln in (lbl.read_text(encoding="utf8").splitlines() if lbl.exists() else []):
                b = parse_line(ln)
                if not b:
                    continue
                c, cx, cy, bw, bh = b
                raw = raw_names[c] if c < len(raw_names) else str(c)
                if raw not in cmap:
                    raise SystemExit(f"{src}: class '{raw}' is not in the map of configs/families.yaml")
                if cmap[raw] is None:
                    continue
                x0, y0, x1, y1 = max(0, cx - bw / 2), max(0, cy - bh / 2), min(1, cx + bw / 2), min(1, cy + bh / 2)
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                boxes.append((names.index(cmap[raw]), x0, y0, x1, y1))
            boxes = sorted(set((c, round(a, 5), round(b, 5), round(e, 5), round(f, 5)) for c, a, b, e, f in boxes))
            items.append({"src": src, "path": img, "stem": f"{src}__{img.stem}", "boxes": boxes,
                          "train_only": bool(sc.get("train_only") and re.search(sc["train_only"], img.stem))})
    return items


# ---------------------------------------------------------------- 3 groups
def feats(path, side=640, n=600):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    s = side / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(cv2.createCLAHE(2.0, (8, 8)).apply(g), None)
    return (np.float32([k.pt for k in kp]), d) if d is not None and len(kp) >= 15 else (None, None)


def sift_pairs(paths, min_inliers, workers):
    with ThreadPoolExecutor(workers) as ex:
        F = list(ex.map(feats, paths))
    ok = [i for i, f in enumerate(F) if f[1] is not None]
    if len(ok) < 2:
        return []
    index = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 48})
    index.add([F[i][1] for i in ok])
    index.train()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for qi, i in enumerate(ok):
        m = index.knnMatch(F[i][1], k=3)
        votes = Counter(x.imgIdx for ms in m for x in ms if x.imgIdx != qi)   # the image itself is in the index
        for cand, _ in votes.most_common(5):
            j = ok[cand]
            if j <= i:
                continue
            mm = bf.knnMatch(F[i][1], F[j][1], k=2)
            good = [x[0] for x in mm if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
            if len(good) < min_inliers:
                continue
            _, inl = cv2.estimateAffinePartial2D(F[i][0][[g.queryIdx for g in good]], F[j][0][[g.trainIdx for g in good]],
                                                 method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if inl is not None and int(inl.sum()) >= min_inliers:
                pairs.append((i, j, int(inl.sum())))
    return pairs


def make_groups(items, fam, min_inliers, workers):
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    sift_idx = []
    for i, it in enumerate(items):
        rule = fam["sources"][it["src"]]["group"]
        if rule == "sift":
            sift_idx.append(i)
            continue
        m = re.search(rule, it["path"].stem)
        key = (it["src"], m.group(1) if m else it["path"].stem)
        parent[find(i)] = find(first.setdefault(key, i))
    pairs = sift_pairs([items[i]["path"] for i in sift_idx], min_inliers, workers) if sift_idx else []
    for a, b, _ in pairs:
        parent[find(sift_idx[a])] = find(sift_idx[b])
    for i, it in enumerate(items):
        it["group"] = find(i)
    cross = sum(1 for a, b, _ in pairs if items[sift_idx[a]]["src"] != items[sift_idx[b]]["src"])
    return len(pairs), cross


# ---------------------------------------------------------------- 4 split
def choose_split(items, n_cls, target, seed, tries):
    groups = defaultdict(lambda: {"n": 0, "cls": np.zeros(n_cls), "src": Counter()})
    for it in items:
        g = groups[it["group"]]
        g["n"] += 1
        g["src"][it["src"]] += 1
        for c, *_ in it["boxes"]:
            g["cls"][c] += 1
    gids = list(groups)
    srcs = sorted({it["src"] for it in items})
    # one vector per group: [images, boxes per class..., images per source...]
    V = np.array([[groups[g]["n"], *groups[g]["cls"], *[groups[g]["src"][s] for s in srcs]] for g in gids], float)
    tot = V.sum(0)
    tot[tot == 0] = 1
    W = np.array([2.0] + [1.0 if V[:, 1 + c].sum() > 0 else 0.0 for c in range(n_cls)] + [0.5] * len(srcs))
    need = [1 + c for c in range(n_cls) if (V[:, 1 + c] > 0).sum() >= 3]
    T = np.array(target)[:, None]

    def cost(F):                                   # F: 3 x dims, the totals per split
        e = float(((F / tot - T) ** 2 * W).sum())
        return e + 10.0 * sum((F[1, c] == 0) + (F[2, c] == 0) for c in need)

    rnd = random.Random(seed)
    best, best_c = None, 1e18
    for t in range(tries):
        order = sorted(range(len(gids)), key=lambda i: -V[i, 0] * (1 + rnd.random()))
        F = np.zeros((3, V.shape[1]))
        assign = np.zeros(len(gids), int)
        for i in order:                            # greedy: the split that lowers the cost most
            costs = []
            for s in range(3):
                F[s] += V[i]
                costs.append(cost(F) + rnd.random() * 1e-4)
                F[s] -= V[i]
            s = int(np.argmin(costs))
            assign[i] = s
            F[s] += V[i]
        c0 = cost(F)
        for _ in range(4):                         # local moves while they help (totals updated in place)
            improved = False
            for i in rnd.sample(range(len(gids)), len(gids)):
                cur = assign[i]
                for s in range(3):
                    if s == cur:
                        continue
                    F[cur] -= V[i]
                    F[s] += V[i]
                    c1 = cost(F)
                    if c1 < c0 - 1e-12:
                        c0, cur, improved = c1, s, True
                        assign[i] = s
                    else:
                        F[s] -= V[i]
                        F[cur] += V[i]
            if not improved:
                break
        if c0 < best_c:
            best, best_c = assign.copy(), c0
    return {gid: SPLITS[s] for gid, s in zip(gids, best)}, best_c


# ---------------------------------------------------------------- 6 write
def tiles(w, h, t, overlap):
    step = int(t * (1 - overlap))
    xs = list(range(0, max(1, w - t), step)) + [max(0, w - t)]
    ys = list(range(0, max(1, h - t), step)) + [max(0, h - t)]
    return sorted({(x, y) for x in xs for y in ys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--target", nargs=3, type=float, default=[0.75, 0.125, 0.125])
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-above", type=int, default=2000)
    ap.add_argument("--neg-ratio", type=float, default=0.35)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--full-side", type=int, default=1600)
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fam = yaml.safe_load(open(ROOT / "configs" / "families.yaml", encoding="utf8"))[a.family]
    names = fam["classes"]
    out = ROOT / f"final_{a.family}"
    excluded = []

    items = collect(fam, excluded)
    print(f"[collect] {len(items)} images from {len(fam['sources'])} sources; {len(excluded)} excluded by rule")

    # 2 exact duplicates (same pixels)
    with ThreadPoolExecutor(a.workers) as ex:
        digests = list(ex.map(lambda it: hashlib.md5(cv2.imread(str(it["path"])).tobytes()).hexdigest(), items))
    same = defaultdict(list)
    for it, dg in zip(items, digests):
        same[dg].append(it)
    keep, disagree = [], 0
    for copies in same.values():
        best = max(copies, key=lambda it: len(it["boxes"]))      # never keep the copy that lost its labels
        disagree += len({tuple(it["boxes"]) for it in copies}) > 1
        keep.append(best)
        for it in copies:
            if it is not best:
                excluded.append((it["src"], it["path"].name, f"identical pixels to {best['src']}/{best['path'].name}"))
    items = keep
    print(f"[dedup] {len(items)} images after removing identical copies ({disagree} sets of identical copies had different labels)")

    n_pairs, cross = make_groups(items, fam, a.min_inliers, a.workers)
    n_groups = len({it["group"] for it in items})
    print(f"[groups] {n_groups} board/scene groups ({n_pairs} same-content pairs, {cross} across sources)")

    # groups made only of train-only images (augmented exports) are not part of the split search
    only_train = {g for g in {it["group"] for it in items} if all(it["train_only"] for it in items if it["group"] == g)}
    split_of, c = choose_split([it for it in items if it["group"] not in only_train], len(names), a.target, a.seed, a.tries)
    split_of.update({g: "train" for g in only_train})
    print(f"[split] cost {c:.4f}")

    rnd = random.Random(a.seed)
    shutil.rmtree(out, ignore_errors=True)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    man = []

    def write(split, stem, img, px, it, kind, x0=0, y0=0):
        h, w = img.shape[:2]
        cv2.imwrite(str(out / "images" / split / (stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [f"{c} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} {(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}"
                 for c, bx0, by0, bx1, by1 in px if bx1 - bx0 >= 1 and by1 - by0 >= 1]
        (out / "labels" / split / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
        cnt = Counter(int(l.split()[0]) for l in lines)
        man.append({"file": stem + ".jpg", "split": split, "source": it["src"], "source_file": it["path"].name,
                    "group": f"{it['src']}:{it['group']}", "kind": kind, "x0": x0, "y0": y0, "width": w, "height": h,
                    "boxes": len(lines), **{f"n_{n}": cnt[i] for i, n in enumerate(names)}})
        # more training data: an exact 90-degree copy (boxes move exactly, nothing is invented);
        # the random flips at training time then cover 180 / 270 as well. Train only - val/test stay real.
        if split == "train" and fam.get("rot90_train") and kind != "rot90":
            write(split, stem + "__r90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                  [(c, h - by1, bx0, h - by0, bx1) for c, bx0, by0, bx1, by1 in px], it, "rot90", x0, y0)

    def emit(it):
        split = split_of[it["group"]]
        if it["train_only"] and split != "train":
            excluded.append((it["src"], it["path"].name, f"mirrored/rotated duplicate, group is in {split}"))
            return
        img = cv2.imread(str(it["path"]))
        h, w = img.shape[:2]
        px = [(c, x0 * w, y0 * h, x1 * w, y1 * h) for c, x0, y0, x1, y1 in it["boxes"]]
        if max(h, w) <= a.tile_above:
            write(split, it["stem"], img, px, it, "image")
            return
        if split == "train":
            s = a.full_side / max(h, w)
            write(split, it["stem"] + "__full", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                  [(c, x0 * s, y0 * s, x1 * s, y1 * s) for c, x0, y0, x1, y1 in px], it, "full")
        pos, neg = [], []
        for tx, ty in tiles(w, h, a.tile, a.overlap):
            tw, th = min(a.tile, w - tx), min(a.tile, h - ty)
            kept = []
            for c, x0, y0, x1, y1 in px:
                ix0, iy0, ix1, iy1 = max(x0, tx), max(y0, ty), min(x1, tx + tw), min(y1, ty + th)
                if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) >= a.min_visible * (x1 - x0) * (y1 - y0):
                    kept.append((c, ix0 - tx, iy0 - ty, ix1 - tx, iy1 - ty))
            (pos if kept else neg).append((tx, ty, tw, th, kept))
        if split == "train":
            neg = rnd.sample(neg, min(len(neg), int(np.ceil(max(1, len(pos)) * a.neg_ratio))))
        for tx, ty, tw, th, kept in pos + neg:
            write(split, f"{it['stem']}__t{tx}_{ty}", img[ty:ty + th, tx:tx + tw], kept, it, "tile", tx, ty)

    # background (no-defect) images: all kept in val/test; in train at most --neg-ratio per labelled image, per source
    skip = set()
    for src in fam["sources"]:
        tr = [it for it in items if it["src"] == src and split_of[it["group"]] == "train"]
        pos = [it for it in tr if it["boxes"]]
        neg = [it for it in tr if not it["boxes"]]
        cap = int(np.ceil(len(pos) * a.neg_ratio))
        if len(neg) > cap:
            for it in rnd.sample(neg, len(neg) - cap):
                skip.add(id(it))
                excluded.append((it["src"], it["path"].name, "train: defect-free image beyond the background cap"))
    for it in items:
        if id(it) not in skip:
            emit(it)

    with open(out / "split_manifest.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        wr.writeheader()
        wr.writerows(man)
    with open(out / "excluded.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "file", "reason"])
        wr.writerows(excluded)
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
                    "nc": len(names), "names": names}, open(out / "data.yaml", "w", encoding="utf8"), sort_keys=False)

    # report
    L = [fam["title"], "=" * len(fam["title"]), f"classes: {names}", ""]
    L.append(f"{'source':18} {'images':>7} {'groups':>7}  train/val/test images")
    for src in fam["sources"]:
        its = [it for it in items if it["src"] == src]
        sp = Counter(split_of[it["group"]] for it in its)
        L.append(f"{src:18} {len(its):7} {len({it['group'] for it in its}):7}  "
                 f"{sp['train']}/{sp['val']}/{sp['test']}")
    L += ["", f"{'split':6} {'files':>6} {'empty':>6} " + " ".join(f"{n[:14]:>14}" for n in names)]
    for s in SPLITS:
        m = [x for x in man if x["split"] == s]
        L.append(f"{s:6} {len(m):6} {sum(x['boxes'] == 0 for x in m):6} " + " ".join(f"{sum(x['n_' + n] for x in m):14}" for n in names))
    why = Counter(r[2].split(",")[0].split(" to ")[0] for r in excluded)
    L += ["", "excluded (excluded.csv has every file and its reason):"] + [f"  {v:5}  {k}" for k, v in why.most_common()]
    (out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
