# Training log — solder / assembly defect model

Every change to the training recipe, in order, with the reason.

## v1 — first notebook
- YOLO26s, 800 px, 80 epochs, AutoBatch.
- Progress was invisible in Colab: the log was read line by line and Ultralytics' `\r` progress bars never appeared, so the run was stopped after epoch 1.
- AutoBatch picked batch 7 and used only 3.8 of 15 GB on the T4; with 2 CPU cores, data loading (~55 ms / sample at 800 px) was the bottleneck.

## v2 — usable on free Colab
| Change | Why |
|---|---|
| Status cell + live HTML panel fed by a JSON status file | see epoch, losses, mAP, GPU/CPU, ETA and the last saved checkpoint |
| YOLO26m, 150 epochs, patience 50 | more capacity, better GPU use |
| Fixed batch by model size and GPU memory (m = 8 on T4) | AutoBatch was too conservative |
| Training images larger than `imgsz` pre-shrunk once (INTER_AREA) | faster loading, cleaner fine detail |
| Image cache: RAM if it fits, else disk, else none | loading speed |
| Train on local disk, copy to Drive every 2 epochs and on stop | speed, and Drive not filled with old checkpoints |
| Automatic resume from Drive (also from another account), save on manual stop | Colab session limits |
| Per-class confidence thresholds = best F1 on validation | better precision/recall per defect type |
| Model chosen on validation only; test only reported | honest evaluation |

## What went wrong with v2

| epoch | mAP@50 | mAP@50-95 | train cls loss |
|---|---|---|---|
| 1–3 | 0.23 → 0.34 | 0.11 → 0.17 | 5.8 → 3.5 (healthy) |
| **4 (best)** | **0.357** | **0.182** | 2.73 |
| 5 | 0.11 | 0.05 | **26.1 (explosion)** |
| 6–32 | ≈ 0.11 | ≈ 0.048 | ≈ 4.4 (stuck) |

The *training* loss exploded, so this was divergence — not over-fitting and not a lack of data (accuracy was rising until epoch 4). It happened exactly when warm-up ended. Two deviations from the Ultralytics defaults explain it:
1. `lr0 = 1e-3` instead of the automatic 7.7e-4 for 9 classes;
2. with an explicit `optimizer="AdamW"`, `warmup_bias_lr` stays at the SGD default **0.1** (Ultralytics only lowers it in `optimizer="auto"` mode) — far too high for AdamW.

A display bug was also found: the panel reported "mosaic off" from epoch 33 because the dataset object has no `mosaic` attribute. Fixed in v3.

## v3 — stable recipe
| Change | Detail |
|---|---|
| Learning rate | `lr0` 1e-3 → **5e-4**, `lrf` 0.02, cosine |
| Warm-up | `warmup_bias_lr = 0`, `warmup_epochs` 3 → **5** |
| **Guard** | after each epoch: if box/cls loss > 1.5 × the best of the previous 5 epochs, or mAP < 50 % of the best for 2 epochs → roll back to the best weights and optimizer state and **halve the LR**; logged in the panel; stops after 4 rollbacks. Replaying the real v2 numbers, it fires at epoch 5 (`26.1 > 1.5 × 2.73`) and never fires on a healthy run |
| LR multiplier persisted | stored in `progress.json`, survives Colab disconnects and account changes |
| Augmentation | GaussNoise, MotionBlur, JPEG compression (pixel-only), `scale` 0.5 → 0.6, `hsv_s` 0.6 → 0.5 |
| Old weights archived | the best weights of every earlier run are copied to `saved_weights/` on Drive, never overwritten |
| `close_mosaic` | adaptive: `min(20, max(5, epochs // 7))` |

**Result:** validation mAP@50 0.904 / mAP@50-95 0.648 (epoch 143 of 150); **test mAP@50 0.895 / mAP@50-95 0.657** (v2: 0.357 / 0.182 on validation).

## Data notes
- ~1,557 unique real photos (1,918 files after tiling large photos; 3,836 with the 90° copies).
- Only 9 files were removed from the solder data (5 mirrored duplicates, 4 surplus defect-free images).
- Offline augmented copies were rejected on purpose: online augmentation already produces a different version of every image in every epoch (mosaic, mixup, scale ±60 %, translation, rotation ±5°, flips, HSV, and in v3 noise/blur/JPEG); with 150 epochs each photo is seen in 150 different forms.
- `oxidation` has no validation/test examples, so its score cannot be measured yet.

## Next steps
1. If the small classes (`solder_void`, `excess_solder`, median ~20 px at 800) need more recall: `imgsz` 1024–1280 (1.6–2.5× training time).
2. 50–100 labelled photos from the target production line — the single most valuable improvement.
