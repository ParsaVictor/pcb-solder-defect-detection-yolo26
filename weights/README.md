# Weights

| File | Content |
|---|---|
| `best.pt` | YOLO26m, 800 px, 9 classes, epoch 143 of 150 (best validation mAP@50-95) |
| `thresholds.json` | per-class confidence (F1 maximum on validation), read automatically by `infer/predict.py` |
| `summary.json` | chosen run and its validation / test mAP@50-95 |

Test set: mAP@50 0.895, mAP@50-95 0.657.
