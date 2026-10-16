"""Train YOLO26 on a PCB-defect dataset on Colab: live status, training on the local disk, checkpoints on Drive.

* Training runs on the fast local disk (<work>/<run>). That path is the same in every Colab session and in
  every Google account, so a resume never points at a folder of another session (Ultralytics stores the
  absolute save_dir inside last.pt).
* Drive (<drive>/<run>): results.csv and progress.json after every epoch; last.pt (the resume point, with
  optimizer) and best.pt (without optimizer, small) every --save-every epochs (and/or --sync-minutes), when
  training stops, and at the end. Each copy of a file is a new version on Google Drive and old versions use
  Drive space for up to 30 days.
* Speed: batch scaled to the GPU memory, images cached in RAM when they fit (else on the local disk),
  cudnn autotuning, all CPU cores as data workers.
* A new session (or the other account) copies <drive>/<run> back to the local disk and resumes from the
  next epoch. A finished run (DONE file) is skipped.
* --status FILE: a JSON file rewritten about once a second (phase, batch, losses, validation progress, every
  finished epoch, time per epoch). The notebook draws its live panel from it.
* Before training, training images larger than --imgsz are resized once to --imgsz with INTER_AREA
  (Ultralytics would shrink them on every load with plain bilinear: slower, and it aliases small defects).
  val/test images are not touched.
* After training best.pt is evaluated on val (-> a confidence threshold per class at its F1 maximum) and on
  test (boards never seen in training). <drive>/final/ gets the run with the best VAL score (test is only
  reported, never used to choose).

  python train_solder.py --data /content/work/final_solder --work /content/work/runs \
      --drive /content/drive/MyDrive/1_Solder_Model/runs/solder --model yolo26m.pt --imgsz 800 --epochs 150
"""
import argparse
import contextlib
import csv
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import smooth

RECIPE = "v3"   # part of the run folder name: a changed recipe never resumes an old run

# Small dataset (about 1,900 photos + their exact 90-degree copies), small defects that can appear anywhere
# and in any orientation, colour matters (copper / solder / mask), 2 CPU cores on Colab.
# v3: the v2 run (lr0=1e-3 with optimizer="AdamW") left warmup_bias_lr at its SGD default 0.1 - Ultralytics only
# zeroes it for optimizer="auto" - and at epoch 5 the loss exploded (cls 2.7 -> 26) and never recovered.
# Now: lower peak lr, warmup_bias_lr 0, longer warmup, and a divergence guard (Guard) that rolls back.
HYP = dict(
    optimizer="AdamW", lr0=0.0005, lrf=0.02, momentum=0.9, weight_decay=0.0005, cos_lr=True,
    warmup_epochs=5, warmup_bias_lr=0.0, warmup_momentum=0.8,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,          # lighting / camera differences (neazu photos are dark)
    degrees=5.0, translate=0.1, scale=0.6, shear=0.0, perspective=0.0,   # small rotation keeps boxes tight
    flipud=0.5, fliplr=0.5,                       # with the rot90 copies: all 8 orientations of a board
    mosaic=1.0, mixup=0.1, close_mosaic=20,       # the last epochs on whole, undistorted photos (see main)
)
# images per batch at imgsz 800 that fit a 15 GB T4 with room to spare (YOLO26s used 3.8 GB at batch 7)
BATCH_AT_800 = {"n": 32, "s": 16, "m": 8, "l": 6, "x": 4}


def auto_batch(model_name, imgsz):
    """Images per batch: the T4 value scaled by this GPU's memory (L4 24 GB, A100 40 GB get more)."""
    size = Path(model_name).stem[-1]
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 15
    return max(2, int(BATCH_AT_800.get(size, 8) * (800 / imgsz) ** 2 * max(1.0, gb / 15.5)))


def auto_cache(data_yaml, imgsz):
    """'ram' when the resized training images fit in free memory with a wide margin, else 'disk'."""
    try:
        import psutil
        cfg = yaml.safe_load(open(data_yaml, encoding="utf8"))
        need = 0
        for im in Path(cfg["train"]).iterdir():
            if im.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                with Image.open(im) as pil:
                    w, h = pil.size
                r = imgsz / max(w, h)
                need += w * r * h * r * 3
        free = psutil.virtual_memory().available
        disk = shutil.disk_usage(Path(cfg["train"])).free
        choice = "ram" if need * 1.6 < free - 3e9 else ("disk" if need * 1.6 < disk - 15e9 else False)
        print(f"[cache] training images need {need / 1e9:.1f} GB, free RAM {free / 1e9:.1f} GB, free disk "
              f"{disk / 1e9:.0f} GB -> {choice or 'no cache (read the JPEGs)'}", flush=True)
        return choice
    except Exception as e:
        print(f"[cache] {e}; using the local disk", flush=True)
        return "disk"


def extra_augmentations():
    """Camera-noise style augmentations on top of Ultralytics' defaults (sources differ in sensor, noise, JPEG).
    Pixel-level only, so the boxes stay valid. Works with albumentations 1.x and 2.x; whatever is missing is skipped."""
    try:
        import albumentations as A
    except ImportError:
        return None
    out = [A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01)]    # Ultralytics defaults
    tries = [lambda: A.GaussNoise(std_range=(0.02, 0.08), p=0.2), lambda: A.GaussNoise(var_limit=(5.0, 40.0), p=0.2),
             lambda: A.MotionBlur(blur_limit=(3, 7), p=0.1),
             lambda: A.ImageCompression(quality_range=(40, 95), p=0.2),
             lambda: A.ImageCompression(quality_lower=40, quality_upper=95, p=0.2)]
    got = set()
    for mk in tries:
        try:
            t = mk()
        except Exception:
            continue
        if type(t).__name__ not in got:
            got.add(type(t).__name__)
            out.append(t)
    print(f"[aug] extra augmentations: {sorted(got)}", flush=True)
    return out


def jdump(obj):
    """JSON with numpy numbers converted."""
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    return json.dumps(obj, default=conv, ensure_ascii=False, indent=1)


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------- data
def prepare_data(root, imgsz, resize=True):
    """data.yaml for this machine; train images larger than imgsz are shrunk once (INTER_AREA, JPEG q95).
    resize=False: the original folders (evaluation)."""
    root = Path(root).resolve()
    cfg = yaml.safe_load(open(root / "data.yaml", encoding="utf8"))
    cfg["path"] = str(root)
    if resize:
        dst = root.parent / f"{root.name}_train{imgsz}"
        if not (dst / ".ready").exists():
            shutil.rmtree(dst, ignore_errors=True)
            (dst / "images" / "train").mkdir(parents=True)
            (dst / "labels" / "train").mkdir(parents=True)
            n = 0
            t0 = time.time()
            for im in sorted((root / "images" / "train").iterdir()):
                out = dst / "images" / "train" / im.name
                with Image.open(im) as pil:
                    w, h = pil.size
                if max(w, h) > imgsz:
                    img = cv2.imread(str(im))
                    s = imgsz / max(w, h)
                    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n += 1
                else:
                    os.link(im, out)
                lbl = root / "labels" / "train" / (im.stem + ".txt")
                if lbl.exists():
                    os.link(lbl, dst / "labels" / "train" / lbl.name)
            (dst / ".ready").write_text("ok")
            print(f"[data] {n} training images larger than {imgsz}px shrunk once with INTER_AREA ({time.time() - t0:.0f}s)",
                  flush=True)
        cfg["train"] = str(dst / "images" / "train")
    out = root.parent / f"data_{root.name}_{imgsz if resize else 'orig'}.yaml"
    yaml.safe_dump(cfg, open(out, "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
    return str(out)


# ---------------------------------------------------------------------------------------- live status
class Live:
    """Everything the notebook shows; also saved as progress.json on Drive (without the per-batch fields)."""
    KEEP = ("run", "model", "imgsz", "batch", "epochs", "patience", "history", "synced_epoch", "synced_at",
            "drive_free_gb", "finished", "recipe", "hyp", "events", "lr_mult", "rollbacks")

    def __init__(self, path, **info):
        self.path = Path(path) if path else None
        self.s = {"phase": "setup", "msg": "", "history": [], "synced_epoch": 0, "synced_at": None,
                  "finished": False, "session_start": time.time(), "recipe": RECIPE, "hyp": HYP,
                  "events": [], "lr_mult": 1.0, "rollbacks": 0, **info}
        self.t = 0.0

    def set(self, force=False, **kw):
        self.s.update(kw)
        self.s["updated"] = time.time()
        if self.path and (force or self.s["updated"] - self.t >= 1.0):
            self.t = self.s["updated"]
            try:
                write_atomic(self.path, jdump(self.s))
            except OSError:
                pass

    def progress(self):
        return jdump({k: self.s.get(k) for k in self.KEEP} | {"updated": time.time()})


class DriveSync:
    """Copies the run to Drive: small files every epoch, the weights every `minutes` and at the end."""

    def __init__(self, local, drive, minutes, live, every=0):
        self.local, self.drive, self.minutes, self.live, self.every = local, drive, minutes, live, every
        self.t_weights = time.time()
        self.best_mtime = None

    def _put(self, src, rel):
        dst = self.drive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)          # in place: Drive keeps the old copy until this one is uploaded

    def small(self):
        try:
            for rel in ("results.csv", "args.yaml"):
                if (self.local / rel).exists():
                    self._put(self.local / rel, rel)
            self.drive.mkdir(parents=True, exist_ok=True)
            try:
                self.live.s["drive_free_gb"] = round(shutil.disk_usage(self.drive).free / 1e9, 1)
            except OSError:
                pass
            text = self.live.progress()
            (self.local / "progress.json").write_text(text, encoding="utf8")
            (self.drive / "progress.json").write_text(text, encoding="utf8")
        except OSError as e:
            print(f"WARNING: copy to Drive failed ({e}); training continues, will retry next epoch", flush=True)

    def weights(self, epoch, stripped_best=True):
        t0 = time.time()
        try:
            last = self.local / "weights" / "last.pt"
            if last.exists():
                self._put(last, "weights/last.pt")
            best = self.local / "weights" / "best.pt"
            if best.exists() and best.stat().st_mtime != self.best_mtime:
                self.best_mtime = best.stat().st_mtime
                if stripped_best:          # best.pt is only used for evaluation: no optimizer -> 3x smaller
                    ck = torch.load(best, map_location="cpu", weights_only=False)
                    ck["optimizer"] = None
                    tmp = self.local / "weights" / "best_small.pt"
                    torch.save(ck, tmp)
                    self._put(tmp, "weights/best.pt")
                    tmp.unlink()
                else:
                    self._put(best, "weights/best.pt")
            self.t_weights = time.time()
            self.live.set(force=True, synced_epoch=epoch, synced_at=time.time())
            self.small()
            print(f"[drive] epoch {epoch}: weights saved to Drive ({time.time() - t0:.0f}s)", flush=True)
        except OSError as e:
            print(f"WARNING: copying the weights to Drive failed ({e}); will retry later", flush=True)

    def after_epoch(self, epoch, last_epoch):
        due_epoch = self.every > 0 and epoch % self.every == 0
        due_time = self.minutes > 0 and time.time() - self.t_weights >= self.minutes * 60
        if last_epoch or due_epoch or due_time or (self.every <= 0 and self.minutes <= 0):
            self.weights(epoch)
        else:
            self.small()

    def everything(self):
        """End of training: the whole run folder (plots, stripped weights) to Drive."""
        try:
            shutil.copytree(self.local, self.drive, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.tmp", "best_small.pt", "progress.json"))
            self.live.set(force=True, synced_epoch=len(self.live.s["history"]), synced_at=time.time())
            (self.drive / "progress.json").write_text(self.live.progress(), encoding="utf8")
        except OSError as e:
            print(f"WARNING: final copy to Drive failed ({e})", flush=True)


class Guard:
    """Divergence guard. A training can blow up (loss x10 in one epoch) and then sit on a broken plateau for
    hundreds of epochs - exactly what the v2 run did at epoch 5. After every epoch:
      spike    train box/cls loss > 1.5 x the lowest of the previous 5 epochs
      collapse val mAP50-95 < 50% of the best (best >= 0.05) for 2 epochs in a row
    -> restore the best weights + optimizer, halve the learning rate for the rest of the run, note the event.
    After MAX_ROLLBACKS the training is stopped (best.pt is kept) instead of burning hours."""
    MAX_ROLLBACKS = 4

    def __init__(self, live):
        self.live, self.base_lf = live, None

    def detect(self, hist):
        if len(hist) < 3:
            return None
        cur, prev = hist[-1], hist[-6:-1]
        for k in ("cls", "box"):
            lo = min(h[k] for h in prev)
            if lo > 0 and cur[k] > 1.5 * lo:
                return f"loss {k} {cur[k]:.2f} > 1.5 x {lo:.2f} (the lowest of the previous epochs)"
        best = max(h["mAP"] for h in hist[:-1])
        if best >= 0.05 and cur["mAP"] < 0.5 * best and hist[-2]["mAP"] < 0.5 * best:
            return f"val mAP50-95 {cur['mAP']:.3f} and {hist[-2]['mAP']:.3f} < 50% of the best {best:.3f}"
        return None

    def apply_lr(self, tr):
        """(Re)apply the lr multiplier to the scheduler (also after a resume)."""
        mult = float(self.live.s.get("lr_mult", 1.0))
        if self.base_lf is None:
            self.base_lf = tr.lf
        if mult != 1.0:
            base = self.base_lf
            tr.lf = lambda x, f=base, m=mult: f(x) * m
            tr.scheduler.lr_lambdas = [tr.lf for _ in tr.scheduler.lr_lambdas]

    def rollback(self, tr, reason, hist):
        e = tr.epoch + 1
        s = self.live.s
        s["rollbacks"] = int(s.get("rollbacks", 0)) + 1
        if s["rollbacks"] > self.MAX_ROLLBACKS or not tr.best.exists():
            s["events"].append({"epoch": e, "text": f"epoch {e}: {reason} - unstable again after {self.MAX_ROLLBACKS} "
                                                    f"rollbacks: training stopped, best.pt is kept"})
            tr.stop = True
            return
        ck = torch.load(tr.best, map_location="cpu", weights_only=False)
        best_ep = int(ck.get("epoch", -1)) + 1
        state = ck["ema"].float().state_dict()
        from ultralytics.utils.torch_utils import unwrap_model
        unwrap_model(tr.model).load_state_dict(state)
        tr._load_checkpoint_state(ck)                 # optimizer, scaler, EMA, best_fitness of that epoch
        if ck.get("optimizer") is None:               # best.pt copied back from Drive has no optimizer state
            tr.optimizer.state.clear()
        s["lr_mult"] = float(s.get("lr_mult", 1.0)) * 0.5
        self.apply_lr(tr)
        shutil.copyfile(tr.best, tr.last)             # last.pt = the restored state: a disconnect now resumes from it
        s["events"].append({"epoch": e, "text": f"epoch {e}: {reason} -> back to the best weights (epoch {best_ep}), "
                                                f"learning rate x{s['lr_mult']:.3g} for the rest of the run"})
        print(f"[guard] {s['events'][-1]['text']}", flush=True)


class Monitor:
    """Ultralytics callbacks -> Live status + Drive sync + one log line per epoch."""

    def __init__(self, live, sync):
        self.live, self.sync = live, sync
        self.guard = Guard(live)
        self.in_epoch = self.seen = False
        self.i = self.vi = 0
        self.t_batch = 0.0

    def attach(self, model):
        for ev in ("on_train_start", "on_train_epoch_start", "on_train_batch_end", "on_val_start",
                   "on_val_batch_end", "on_fit_epoch_end", "on_train_end"):
            model.add_callback(ev, getattr(self, ev))
        return model

    def on_train_start(self, tr):
        self.guard.apply_lr(tr)                       # a resumed run keeps its reduced learning rate
        self.live.set(force=True, phase="train", epochs=tr.epochs, start_epoch=tr.start_epoch + 1,
                      batch=tr.batch_size, nb=len(tr.train_loader), session_start=time.time(), msg="")

    def on_train_epoch_start(self, tr):
        self.in_epoch, self.seen, self.i = True, True, 0
        now = time.time()
        self.live.set(force=True, phase="train", epoch=tr.epoch + 1, epochs=tr.epochs, ibatch=0,
                      nb=len(tr.train_loader), epoch_t0=now, phase_t0=now,
                      mosaic=tr.epoch < tr.epochs - int(tr.args.close_mosaic or 0), batch=tr.batch_size)

    def on_train_batch_end(self, tr):
        self.i += 1
        now = time.time()
        if now - self.t_batch < 1.0 and self.i != len(tr.train_loader):
            return
        self.t_batch = now
        loss = tr.tloss.tolist() if tr.tloss is not None and hasattr(tr.tloss, "tolist") else []
        self.live.set(ibatch=self.i, loss=[float(x) for x in np.atleast_1d(loss)],
                      gpu_mem=round(torch.cuda.memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0,
                      lr=float(tr.optimizer.param_groups[0]["lr"]))

    def on_val_start(self, v):
        self.vi = 0
        phase = "val" if self.in_epoch else ("final_val" if self.seen else self.live.s.get("eval_phase", "eval"))
        self.live.set(force=True, phase=phase, val_i=0, val_nb=len(v.dataloader), phase_t0=time.time())

    def on_val_batch_end(self, v):
        self.vi += 1
        self.live.set(val_i=self.vi)

    def on_fit_epoch_end(self, tr):
        if not self.in_epoch:              # Ultralytics calls this again after validating best.pt at the end
            return
        self.in_epoch = False
        m, e = tr.metrics or {}, tr.epoch + 1
        loss = [float(x) for x in np.atleast_1d(tr.tloss.tolist())] if tr.tloss is not None else [0, 0, 0]
        row = {"epoch": e, "sec": round(time.time() - self.live.s.get("epoch_t0", time.time()), 1),
               "box": loss[0], "cls": loss[1] if len(loss) > 1 else 0, "dfl": loss[2] if len(loss) > 2 else 0,
               "vbox": float(m.get("val/box_loss", 0)), "vcls": float(m.get("val/cls_loss", 0)),
               "P": float(m.get("metrics/precision(B)", 0)), "R": float(m.get("metrics/recall(B)", 0)),
               "mAP50": float(m.get("metrics/mAP50(B)", 0)), "mAP": float(m.get("metrics/mAP50-95(B)", 0)),
               "lr": float(tr.optimizer.param_groups[0]["lr"]), "mosaic": self.live.s.get("mosaic", True)}
        hist = [h for h in self.live.s["history"] if h["epoch"] < e] + [row]
        best = max(hist, key=lambda h: h["mAP"])
        self.live.set(force=True, history=hist, phase="saving", epochs=tr.epochs)
        times = [h["sec"] for h in hist[-5:]]
        left = (tr.epochs - e) * sum(times) / len(times)
        print(f">>> epoch {e}/{tr.epochs} | loss box {row['box']:.3f} cls {row['cls']:.3f} | P {row['P']:.3f} "
              f"R {row['R']:.3f} mAP50 {row['mAP50']:.3f} mAP50-95 {row['mAP']:.3f} | best {best['mAP']:.3f} "
              f"(epoch {best['epoch']}) | {row['sec'] / 60:.1f} min/epoch, ~{left / 3600:.1f} h left", flush=True)
        try:
            why = self.guard.detect(hist)
            if why and e < tr.epochs:
                self.guard.rollback(tr, why, hist)
                row["rolled_back"] = True
                self.live.set(force=True, history=hist)
        except Exception as err:                      # the guard must never kill a training
            print(f"[guard] WARNING: guard failed ({type(err).__name__}: {err}); training continues", flush=True)
        self.sync.after_epoch(e, e >= tr.epochs or tr.stop or bool(row.get("rolled_back")))
        self.live.set(force=True, phase="train")

    def on_train_end(self, tr):
        self.live.set(force=True, phase="finishing", msg="training finished")


# ---------------------------------------------------------------------------------------- resume
@contextlib.contextmanager
def saved_on_exit(sync, live):
    """Training stopped by an error, out of memory or the notebook's stop button: put the last finished
    epoch on Drive before exiting, so nothing since the last regular copy is lost."""
    try:
        yield
    except BaseException as e:
        done = len(live.s["history"])
        if done and live.s.get("synced_epoch", 0) < done:
            print(f"\n[stop] {type(e).__name__}: saving epoch {done} to Drive before exiting ...", flush=True)
            try:                           # the stop may have hit while last.pt was being written
                ckpt_epochs_done(sync.local / "weights" / "last.pt")
                sync.weights(done)
            except Exception as err:
                print(f"[stop] local last.pt is incomplete ({err}); Drive keeps epoch {live.s.get('synced_epoch')}",
                      flush=True)
        live.set(force=True, phase="stopped", msg=type(e).__name__)
        raise


def ckpt_epochs_done(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("epoch", -1) + 1, ck


def restore(local, drive):
    """Make sure the newest copy of the run is on the local disk. Returns (epochs done, from where)."""
    ll = local / "weights" / "last.pt"
    if ll.exists():
        try:
            return ckpt_epochs_done(ll)[0], "local disk"
        except Exception as e:             # cut off while being written: use the copy on Drive
            print(f"[resume] local last.pt is unreadable ({e}); using the copy on Drive", flush=True)
            ll.unlink()
    dl = drive / "weights" / "last.pt"
    if not dl.exists():
        return 0, None
    (local / "weights").mkdir(parents=True, exist_ok=True)
    for rel in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml", "progress.json"):
        if (drive / rel).exists():
            shutil.copyfile(drive / rel, local / rel)
    try:
        done, _ = ckpt_epochs_done(ll)
    except Exception as e:
        raise SystemExit(f"last.pt on Drive cannot be read ({e}). Delete the folder {drive} to start this run "
                         f"again, or rename it to keep it.")
    # results.csv / progress.json can be a few epochs ahead of last.pt (they are copied every epoch)
    rc = local / "results.csv"
    if rc.exists():
        rows = list(csv.reader(open(rc, encoding="utf8")))
        keep = [rows[0]] + [r for r in rows[1:] if r and float(r[0]) <= done]
        with open(rc, "w", newline="", encoding="utf8") as f:
            csv.writer(f).writerows(keep)
    return done, "Google Drive"


def retarget(last, local, data_yaml, epochs):
    """Point last.pt at this machine's folders; a different --epochs replaces the old target (if still ahead)."""
    done, ck = ckpt_epochs_done(last)
    ta = ck["train_args"]
    ta.update(save_dir=str(local), project=str(local.parent), name=local.name, data=data_yaml)
    if epochs and ta.get("epochs") != epochs and epochs > done:
        print(f"[resume] epochs {ta.get('epochs')} -> {epochs}", flush=True)
        ta["epochs"] = epochs
    torch.save(ck, last)
    return done, ta


# ---------------------------------------------------------------------------------------- evaluation
def evaluate(best, data_yaml, imgsz, out_dir, live, device):
    """val -> per-class thresholds; test -> honest scores (also precision / recall at those thresholds)."""
    res, thr = {}, None
    for split in ("val", "test"):
        live.set(force=True, eval_phase=f"eval_{split}", phase=f"eval_{split}")
        model = YOLO(str(best))
        mon = Monitor(live, None)
        model.add_callback("on_val_start", mon.on_val_start)
        model.add_callback("on_val_batch_end", mon.on_val_batch_end)
        m = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=16, conf=0.001, plots=True, device=device,
                      project=str(out_dir), name=f"{split}_eval", exist_ok=True)
        names = m.names
        idx = [int(c) for c in m.box.ap_class_index]
        px, f1 = np.asarray(m.box.px), np.asarray(m.box.f1_curve)
        pc, rc = np.asarray(m.box.p_curve), np.asarray(m.box.r_curve)
        nt = m.nt_per_class
        if split == "val":
            g = float(px[smooth(f1.mean(0), 0.1).argmax()]) if idx else 0.25
            per = {names[c]: (float(px[smooth(f1[i], 0.1).argmax()]) if nt[c] >= 15 else g) for i, c in enumerate(idx)}
            per = {n: round(min(0.8, max(0.05, per.get(n, g))), 3) for n in names.values()}
            thr = {"global": round(g, 3), "per_class": per,
                   "how": "confidence at the F1 maximum on val (classes with < 15 val boxes use the global one)"}
        classes = []
        for i, c in enumerate(idx):
            t = thr["per_class"][names[c]]
            j = min(len(px) - 1, int(np.searchsorted(px, t)))
            _, _, ap50, ap = m.box.class_result(i)
            classes.append({"class": names[c], "instances": int(nt[c]), "images": int(m.nt_per_image[c]),
                            "mAP50": float(ap50), "mAP50-95": float(ap), "conf": t,
                            "precision": float(pc[i, j]), "recall": float(rc[i, j])})
        res[split] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "precision": float(m.box.mp),
                      "recall": float(m.box.mr), "classes": classes,
                      "ms_per_image": float(sum(m.speed.values())) if m.speed else None}
        print(f"[{split}] mAP50 {res[split]['mAP50']:.3f}  mAP50-95 {res[split]['mAP50-95']:.3f}", flush=True)
    res["thresholds"] = thr
    return res


def choose_final(drive_project):
    """<project>/final: the run with the best val mAP50-95 (test is reported, never used to choose)."""
    runs = []
    for ev in sorted(Path(drive_project).glob("*/eval.json")):
        e = json.loads(ev.read_text(encoding="utf8"))
        if (ev.parent / "weights" / "best.pt").exists():
            runs.append((e["val"]["mAP50-95"], ev.parent, e))
    if not runs:
        return None
    _, run, e = max(runs, key=lambda r: r[0])
    final = Path(drive_project) / "final"
    final.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "weights" / "best.pt", final / "best.pt")
    (final / "thresholds.json").write_text(jdump(e["thresholds"]), encoding="utf8")
    summary = {"chosen": run.name, "chosen_by": "val mAP50-95",
               "runs": {r.name: {"val": x["val"]["mAP50-95"], "test": x["test"]["mAP50-95"]} for _, r, x in runs}}
    (final / "summary.json").write_text(jdump(summary), encoding="utf8")
    return run.name


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder (contains data.yaml)")
    ap.add_argument("--work", required=True, help="local fast folder for the runs")
    ap.add_argument("--drive", required=True, help="Google Drive folder for this model family (runs/solder)")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=0, help="0 = by model size (see BATCH_AT_800)")
    ap.add_argument("--patience", type=int, default=50, help="stop when val mAP50-95 has not improved for N epochs")
    ap.add_argument("--save-every", type=int, default=2, help="copy the weights to Drive every N epochs (0 = off)")
    ap.add_argument("--sync-minutes", type=float, default=0, help="also copy them when this many minutes passed (0 = off)")
    ap.add_argument("--cache", default="auto", help="auto (RAM if it fits, else disk) | ram | disk | False")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training set (quick test runs)")
    ap.add_argument("--no-resize", action="store_true", help="do not pre-shrink large training images")
    ap.add_argument("--device", default=None, help="0 or cpu (default: auto)")
    ap.add_argument("--status", default=None, help="live status JSON for the notebook")
    a = ap.parse_args()
    cache = False if a.cache in ("False", "false", "0", "") else a.cache
    batch = a.batch or (auto_batch(a.model, a.imgsz) if torch.cuda.is_available() else 4)

    run = f"{Path(a.model).stem}_{a.imgsz}_{RECIPE}"
    local, drive = Path(a.work) / run, Path(a.drive) / run
    local.mkdir(parents=True, exist_ok=True)
    live = Live(a.status, run=run, model=a.model, imgsz=a.imgsz, batch=batch, epochs=a.epochs, patience=a.patience)
    sync = DriveSync(local, drive, a.sync_minutes, live, a.save_every)
    torch.backends.cudnn.benchmark = True       # fixed 800x800 training batches: let cuDNN pick the fastest kernels

    if (drive / "DONE").exists():
        print(f"[{run}] training already finished - skipped", flush=True)
        if (drive / "progress.json").exists():
            live.s.update(json.loads((drive / "progress.json").read_text(encoding="utf8")))
        live.set(force=True, phase="done_before", finished=True)
    else:
        live.set(force=True, phase="setup", msg="data")
        data_yaml = prepare_data(a.data, a.imgsz, resize=not a.no_resize)
        if cache == "auto":
            cache = auto_cache(data_yaml, a.imgsz)
        done, where = restore(local, drive)
        if (local / "progress.json").exists():
            old = json.loads((local / "progress.json").read_text(encoding="utf8"))
            live.s["history"] = [h for h in old.get("history", []) if h["epoch"] <= done]
            live.s["synced_epoch"] = min(done, old.get("synced_epoch") or done)
            live.s["synced_at"] = old.get("synced_at")
            for k in ("events", "lr_mult", "rollbacks"):
                if k in old:
                    live.s[k] = old[k]
        if (local / "weights" / "best.pt").exists():
            sync.best_mtime = (local / "weights" / "best.pt").stat().st_mtime   # already on Drive
        if done > 0:
            done, ta = retarget(local / "weights" / "last.pt", local, data_yaml, a.epochs)
            print(f"[{run}] resuming after epoch {done}/{ta['epochs']} (from {where})", flush=True)
            live.set(force=True, phase="setup", msg=f"resume:{done}:{where}", epochs=ta["epochs"])
            model = Monitor(live, sync).attach(YOLO(str(local / "weights" / "last.pt")))
            try:
                with saved_on_exit(sync, live):
                    model.train(resume=True, batch=batch, workers=a.workers, cache=cache, patience=a.patience,
                                augmentations=extra_augmentations(), **({"device": a.device} if a.device else {}))
            except AssertionError as e:      # last.pt of a run that had already reached its final epoch
                print(f"[{run}] nothing to resume ({str(e).splitlines()[0]}); treating as finished", flush=True)
        elif done == 0 and (local / "weights" / "last.pt").exists():
            print(f"[{run}] training already finished on this machine", flush=True)
        else:
            print(f"[{run}] new run: {a.model}, imgsz {a.imgsz}, batch {batch}, {a.epochs} epochs", flush=True)
            live.set(force=True, phase="setup", msg="new")
            model = Monitor(live, sync).attach(YOLO(a.model))
            hyp = HYP | {"close_mosaic": min(20, max(5, a.epochs // 7))}   # 150 -> 20, 80 -> 11
            with saved_on_exit(sync, live):
                model.train(data=data_yaml, epochs=a.epochs, imgsz=a.imgsz, batch=batch, patience=a.patience,
                            project=str(local.parent), name=local.name, exist_ok=True, save=True, save_period=-1,
                            cache=cache, workers=a.workers, seed=0, deterministic=False, plots=True, amp=True,
                            fraction=a.fraction, device=a.device, augmentations=extra_augmentations(), **hyp)
        live.set(force=True, phase="finishing", finished=True)
        sync.everything()
        (drive / "DONE").write_text("ok")
        print(f"[{run}] training finished, everything is on Drive: {drive}", flush=True)

    # evaluation (skipped when it exists)
    if (drive / "eval.json").exists():
        ev = json.loads((drive / "eval.json").read_text(encoding="utf8"))
    else:
        best = local / "weights" / "best.pt"
        if not best.exists():
            best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(drive / "weights" / "best.pt", best)
        data_yaml = prepare_data(a.data, a.imgsz, resize=False)
        ev = evaluate(best, data_yaml, a.imgsz, local, live, a.device)
        ev |= {"run": run, "model": a.model, "imgsz": a.imgsz}
        (local / "eval.json").write_text(jdump(ev), encoding="utf8")
        try:
            for d in ("val_eval", "test_eval"):
                shutil.copytree(local / d, drive / d, dirs_exist_ok=True)
            shutil.copyfile(local / "eval.json", drive / "eval.json")
        except OSError as e:
            print(f"WARNING: copying the evaluation to Drive failed ({e})", flush=True)
    chosen = choose_final(a.drive)
    print(f"\nTEST (boards never seen in training): mAP50 {ev['test']['mAP50']:.3f}  mAP50-95 {ev['test']['mAP50-95']:.3f}"
          f"\nfinal model: {Path(a.drive) / 'final' / 'best.pt'} (run {chosen}, chosen by val)", flush=True)
    live.set(force=True, phase="done", finished=True, msg=f"final:{chosen}")


if __name__ == "__main__":
    main()
