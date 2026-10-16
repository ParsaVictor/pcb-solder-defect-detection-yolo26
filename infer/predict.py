"""Find PCB defects on your own photos (an image or a folder).

Large photos are cut into overlapping tiles (as in training), every tile is predicted, and the
boxes are merged with NMS, so a 3000 px photo keeps its full detail. Outputs per image:
<name>_annotated.jpg and <name>.json (class, confidence, box in pixels); summary.csv for all.

Confidence: by default the per-class thresholds of thresholds.json next to the weights (tuned on val by
train_solder.py); --conf X uses one threshold X for every class instead.
--labels <folder of YOLO .txt>: also draws the true boxes (thin green) to compare with the predictions.

  python predict.py --weights best.pt --source my_board.jpg --out results
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

COL = [(0, 0, 255), (255, 128, 0), (255, 0, 255), (0, 215, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
       (180, 105, 255), (40, 180, 120)]
GT = (60, 220, 60)


def tiles(w, h, t, overlap):
    step = int(t * (1 - overlap))
    xs = list(range(0, max(1, w - t), step)) + [max(0, w - t)]
    ys = list(range(0, max(1, h - t), step)) + [max(0, h - t)]
    return sorted({(x, y) for x in xs for y in ys})


def predict_image(model, img, a, conf_min):
    h, w = img.shape[:2]
    jobs = [(0, 0, img)] if max(h, w) <= a.tile_above else \
        [(x, y, img[y:y + a.tile, x:x + a.tile]) for x, y in tiles(w, h, a.tile, a.overlap)]
    boxes, scores, classes = [], [], []
    for k in range(0, len(jobs), a.batch):
        chunk = jobs[k:k + a.batch]
        res = model.predict([c[2] for c in chunk], imgsz=a.imgsz, conf=conf_min, iou=a.iou, verbose=False,
                            device=a.device)
        for (x0, y0, _), r in zip(chunk, res):
            b = r.boxes
            if len(b):
                boxes.append(b.xyxy.cpu() + torch.tensor([x0, y0, x0, y0]))
                scores.append(b.conf.cpu())
                classes.append(b.cls.cpu())
    if not boxes:
        return []
    B, S, C = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep = torchvision.ops.batched_nms(B, S, C, a.iou) if len(jobs) > 1 else torch.arange(len(S))
    return [(int(C[i]), float(S[i]), *[float(v) for v in B[i]]) for i in keep]


def read_gt(lbl, w, h):
    out = []
    if lbl and lbl.exists():
        for ln in lbl.read_text(encoding="utf8").split("\n"):
            p = ln.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def draw(img, dets, gt, names):
    vis = img.copy()
    th = max(2, max(img.shape[:2]) // 500)
    for c, x0, y0, x1, y1 in gt:
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), GT, max(1, th // 2), cv2.LINE_AA)
    fs = 0.45 * th
    for c, s, x0, y0, x1, y1 in sorted(dets, key=lambda d: d[1]):
        col = COL[c % len(COL)]
        p0, p1 = (int(x0) - th, int(y0) - th), (int(x1) + th, int(y1) + th)
        cv2.rectangle(vis, p0, p1, col, th, cv2.LINE_AA)
        txt = f"{names[c]} {s:.2f}"
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th // 2))
        ty = p0[1] - 4 if p0[1] - tht - 8 > 0 else p1[1] + tht + 6
        cv2.rectangle(vis, (p0[0], ty - tht - 4), (p0[0] + tw + 4, ty + 3), col, -1)
        cv2.putText(vis, txt, (p0[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
    if gt:
        cv2.putText(vis, "green = true defect (label)", (8, 12 + 14 * th), cv2.FONT_HERSHEY_SIMPLEX, fs, GT,
                    max(1, th // 2), cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--conf", type=float, default=-1, help="one threshold for all classes (default: thresholds.json)")
    ap.add_argument("--thresholds", default=None, help="default: thresholds.json next to the weights")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=0, help="0 = the training image size")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-above", type=int, default=2000, help="tile photos whose long side is larger")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--labels", default=None, help="folder with YOLO label files: draw the true boxes too")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    model = YOLO(a.weights)
    names = model.names
    if not a.imgsz:
        a.imgsz = int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 800))
    thr_file = Path(a.thresholds) if a.thresholds else Path(a.weights).with_name("thresholds.json")
    if a.conf >= 0:
        thr = {i: a.conf for i in names}
    elif thr_file.exists():
        per = json.loads(thr_file.read_text(encoding="utf8"))["per_class"]
        thr = {i: float(per.get(n, 0.25)) for i, n in names.items()}
        print("per-class thresholds:", {names[i]: round(t, 2) for i, t in thr.items()})
    else:
        thr = {i: 0.25 for i in names}
    src = Path(a.source)
    files = sorted(p for p in (src.iterdir() if src.is_dir() else [src])
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print("cannot read", p)
            continue
        dets = [d for d in predict_image(model, img, a, min(thr.values())) if d[1] >= thr[d[0]]]
        gt = read_gt(Path(a.labels) / (p.stem + ".txt"), img.shape[1], img.shape[0]) if a.labels else []
        cv2.imwrite(str(out / f"{p.stem}_annotated.jpg"), draw(img, dets, gt, names))
        (out / f"{p.stem}.json").write_text(json.dumps(
            [{"class": names[c], "conf": round(s, 4), "box": [round(v, 1) for v in (x0, y0, x1, y1)]}
             for c, s, x0, y0, x1, y1 in dets], indent=1))
        cnt = {n: sum(1 for d in dets if d[0] == i) for i, n in names.items()}
        summary.append({"image": p.name, "defects": len(dets), **({"true_boxes": len(gt)} if a.labels else {}), **cnt})
        print(f"{p.name}: {len(dets)} defects" + (f" (true: {len(gt)})" if a.labels else "") +
              f" {({k: v for k, v in cnt.items() if v})}")
    if summary:
        with open(out / "summary.csv", "w", newline="", encoding="utf8") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            wr.writeheader()
            wr.writerows(summary)
    print("results ->", out)


if __name__ == "__main__":
    main()
