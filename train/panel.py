"""HTML panels for the Colab notebook: training status (before training), live progress (during), results.

Everything is drawn from the files train_solder.py writes: the live status JSON (local disk, ~1 s) and
progress.json / results.csv / eval.json on Drive. No Ultralytics import, so it is fast and works on CPU.
"""
import base64
import html
import io
import json
import statistics
import time
from pathlib import Path

FA = {"insufficient_solder": "لحیم کم", "excess_solder": "لحیم زیاد", "solder_bridge": "پل لحیم",
      "solder_void": "حفره‌ی لحیم", "missing_component": "قطعه‌ی جاافتاده", "shifted_component": "قطعه‌ی جابه‌جا",
      "burn": "سوختگی", "physical_damage": "آسیب فیزیکی", "oxidation": "اکسید",
      "missing_hole": "سوراخ جاافتاده", "mouse_bite": "بریدگی لبه", "open_circuit": "قطعی مسیر", "short": "اتصال کوتاه",
      "spur": "زائده‌ی مسی", "spurious_copper": "مس اضافه", "pin_hole": "سوراخ ریز"}

PHASE = {"setup": ("آماده‌سازی", "داده، مدل و کش تصاویر آماده می‌شود (چند دقیقه، فقط اول هر جلسه)"),
         "train": ("آموزش", "مدل روی عکس‌های train یاد می‌گیرد"),
         "val": ("ارزیابی epoch", "سنجش این epoch روی عکس‌های val (بردهای دیده‌نشده)"),
         "saving": ("ذخیره", "ذخیره‌ی وزن‌ها و نتیجه‌ها"),
         "final_val": ("ارزیابی نهایی", "سنجش بهترین مدل (best.pt) روی val"),
         "finishing": ("کپی روی Drive", "همه‌ی نتیجه‌ها روی Google Drive کپی می‌شود"),
         "eval_val": ("تنظیم آستانه‌ها", "آستانه‌ی اطمینان هر کلاس روی val حساب می‌شود"),
         "eval_test": ("آزمون نهایی", "سنجش روی test: بردهایی که مدل هرگز ندیده"),
         "done": ("تمام شد", "آموزش و ارزیابی تمام شد"),
         "done_before": ("قبلاً تمام شده", "این آموزش قبلاً کامل شده است"),
         "stopped": ("متوقف شد", "آموزش متوقف شد؛ با اجرای دوباره از آخرین نسخه‌ی ذخیره‌شده ادامه می‌دهد"),
         "error": ("خطا", "آموزش با خطا متوقف شد؛ پایین پنل علت را ببین")}

CSS = """<style>
.pp{direction:rtl;font-family:Vazirmatn,Tahoma,'Segoe UI',sans-serif;background:#fff;color:#1f2328;border:1px solid #d0d7de;
 border-radius:10px;padding:14px 16px;max-width:1000px;line-height:1.75;font-size:14px}
.pp h3{margin:0 0 6px;font-size:17px}.pp .mut{color:#57606a;font-size:12.5px}
.pp .row{display:flex;gap:10px;align-items:center;margin:4px 0;flex-wrap:wrap}
.pp .lbl{min-width:92px;font-weight:600}
.pp .bar{flex:1;min-width:220px;height:16px;background:#eaeef2;border-radius:8px;overflow:hidden;direction:ltr}
.pp .bar>div{height:100%;background:linear-gradient(90deg,#2da44e,#3fb950)}
.pp .bar.b2>div{background:linear-gradient(90deg,#0969da,#54aeff)}
.pp .bar.b3>div{background:linear-gradient(90deg,#8250df,#c297ff)}
.pp .chip{display:inline-block;padding:1px 10px;border-radius:12px;background:#ddf4ff;color:#0969da;font-weight:600}
.pp .chip.ok{background:#dafbe1;color:#1a7f37}.pp .chip.warn{background:#fff8c5;color:#9a6700}.pp .chip.bad{background:#ffebe9;color:#cf222e}
.pp .kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:6px 14px;margin:8px 0}
.pp .kv div{background:#f6f8fa;border-radius:6px;padding:4px 10px}
.pp .kv b{font-family:Consolas,monospace}
.pp bdi{unicode-bidi:isolate}
.pp table{border-collapse:collapse;margin:8px 0;font-size:13px;direction:ltr}
.pp th,.pp td{border:1px solid #d0d7de;padding:2px 8px;text-align:center;font-family:Consolas,monospace}
.pp th{background:#f6f8fa;font-family:Tahoma,sans-serif}.pp tr.best td{background:#dafbe1;font-weight:700}
.pp td.fa{font-family:Tahoma,sans-serif;text-align:right;direction:rtl}
.pp pre{direction:ltr;text-align:left;background:#0d1117;color:#c9d1d9;padding:8px;border-radius:6px;font-size:11.5px;
 max-height:190px;overflow:auto;white-space:pre-wrap;margin:6px 0}
.pp img{max-width:100%;border:1px solid #eaeef2;border-radius:6px;margin-top:6px}
</style>"""


def esc(s):
    return html.escape(str(s))


def dur(sec):
    """seconds -> Persian 'x ساعت و y دقیقه'."""
    if sec is None or sec != sec:
        return "-"
    sec = max(0, int(sec))
    h, m = sec // 3600, (sec % 3600) // 60
    if h:
        return f"{h} ساعت و {m} دقیقه"
    if m:
        return f"{m} دقیقه"
    return f"{sec} ثانیه"


def ago(ts):
    return "-" if not ts else dur(time.time() - ts) + " پیش"


def bar(frac, cls=""):
    frac = max(0.0, min(1.0, frac or 0))
    return f'<div class="bar {cls}"><div style="width:{100 * frac:.1f}%"></div></div>'


def epoch_seconds(hist):
    t = [h["sec"] for h in hist[-6:] if h.get("sec")]
    return statistics.median(t) if t else None


def chart(hist, epochs=None, close_mosaic=None):
    """PNG (base64) of val mAP and losses per epoch."""
    if not hist:
        return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.1), dpi=90)
    ax[0].plot(e, [h["mAP50"] for h in hist], color="#0969da", label="mAP50 (val)")
    ax[0].plot(e, [h["mAP"] for h in hist], color="#2da44e", label="mAP50-95 (val)")
    b = max(hist, key=lambda h: h["mAP"])
    ax[0].scatter([b["epoch"]], [b["mAP"]], color="#cf222e", zorder=5, label=f"best: epoch {b['epoch']}")
    ax[0].set_ylim(0, max(0.05, max(h["mAP50"] for h in hist)) * 1.12)
    ax[1].plot(e, [h["box"] for h in hist], color="#8250df", label="train box")
    ax[1].plot(e, [h["cls"] for h in hist], color="#bf8700", label="train cls")
    if any(h.get("vbox") for h in hist):
        ax[1].plot(e, [h["vbox"] for h in hist], "--", color="#8250df", alpha=.6, label="val box")
        ax[1].plot(e, [h["vcls"] for h in hist], "--", color="#bf8700", alpha=.6, label="val cls")
    for a in ax:
        if epochs:
            a.set_xlim(0.5, epochs + 0.5)
            cm = close_mosaic or min(20, max(5, epochs // 7))      # same rule as train_solder.py
            if epochs > cm:
                a.axvline(epochs - cm + 0.5, color="#999", lw=.8, ls=":")
        a.grid(alpha=.25)
        a.legend(fontsize=7.5, loc="best")
        a.set_xlabel("epoch", fontsize=8)
        a.tick_params(labelsize=7.5)
    ax[0].set_title("accuracy on val (higher = better)", fontsize=9)
    ax[1].set_title("loss (lower = better)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def epoch_table(hist, n=10):
    if not hist:
        return ""
    best = max(hist, key=lambda h: h["mAP"])["epoch"]
    rows = hist[-n:]
    if best not in [h["epoch"] for h in rows]:
        rows = [h for h in hist if h["epoch"] == best] + rows
    out = ["<table><tr><th>epoch</th><th>زمان</th><th>loss box</th><th>loss cls</th><th>Precision</th>"
           "<th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr>"]
    for h in reversed(rows):
        out.append(f'<tr class="{"best" if h["epoch"] == best else ""}"><td>{h["epoch"]}{" ★" if h["epoch"] == best else ""}</td>'
                   f'<td>{h.get("sec", 0) / 60:.1f}m</td><td>{h["box"]:.3f}</td><td>{h["cls"]:.3f}</td><td>{h["P"]:.3f}</td>'
                   f'<td>{h["R"]:.3f}</td><td>{h["mAP50"]:.3f}</td><td>{h["mAP"]:.3f}</td></tr>')
    out.append("</table>")
    return "".join(out)


def eta(st, now=None):
    """(seconds left for the whole training, seconds per epoch) from the history and the running epoch."""
    now = now or time.time()
    hist, n = st.get("history") or [], st.get("epochs") or 0
    per = epoch_seconds(hist)
    done = len(hist)
    phase = st.get("phase")
    cur_left = 0.0
    if phase in ("train", "val", "saving") and st.get("epoch", 0) > done:
        nb, ib = st.get("nb") or 1, st.get("ibatch") or 0
        spent = now - (st.get("epoch_t0") or now)
        if per is None and phase == "train" and ib >= 10:
            per = spent / (ib / nb) * 1.15          # first epoch: extrapolate, +15% for validation
        if per:
            cur_left = max(0.0, per - spent)
        done += 1
    if per is None:
        return None, None
    finals = 2.5 * 60                               # final validation + test evaluation
    return cur_left + max(0, n - done) * per + (finals if phase not in ("done", "done_before") else 0), per


def summary_block(st):
    hist = st.get("history") or []
    n = st.get("epochs") or 0
    done = len(hist)
    out = []
    left, per = eta(st)
    out.append(f'<div class="row"><span class="lbl">کل آموزش</span>{bar(done / n if n else 0)}'
               f'<span><b>{done}</b> از <b>{n}</b> epoch تمام شده ({100 * done / n if n else 0:.0f}٪)</span></div>')
    kv = []
    if per:
        kv.append(f"زمان هر epoch: <b>{per / 60:.1f}</b> دقیقه")
    if left is not None and st.get("phase") not in ("done", "done_before"):
        kv.append(f"زمان باقی‌مانده: <b>{esc(dur(left))}</b>")
    if hist:
        b = max(hist, key=lambda h: h["mAP"])
        kv.append(f"بهترین تا الان: <bdi>epoch <b>{b['epoch']}</b> · mAP50 <b>{b['mAP50']:.3f}</b> · "
                  f"mAP50-95 <b>{b['mAP']:.3f}</b></bdi>")
        last = hist[-1]
        kv.append(f"آخرین epoch: <bdi>mAP50 <b>{last['mAP50']:.3f}</b> · P <b>{last['P']:.2f}</b> · "
                  f"R <b>{last['R']:.2f}</b></bdi>")
        pat = st.get("patience") or 0
        if pat and b["epoch"] + pat < n:
            kv.append(f"اگر تا <bdi>epoch <b>{b['epoch'] + pat}</b></bdi> بهتر نشود، زودتر تمام می‌شود")
    se = st.get("synced_epoch") or 0
    if done:
        kv.append(f"نسخه‌ی امن روی Drive: <bdi>epoch <b>{se}</b></bdi> ({esc(ago(st.get('synced_at')))})"
                  + (f" — اگر قطع شود از <bdi>epoch <b>{se + 1}</b></bdi> ادامه می‌دهد" if not st.get("finished") else ""))
    if st.get("drive_free_gb") is not None:
        free = st["drive_free_gb"]
        kv.append(f"فضای آزاد Drive: <bdi><b>{free:.1f}</b> GB</bdi>" + (" ⚠️ کم است" if free < 2 else ""))
    if kv:
        out.append('<div class="kv">' + "".join(f"<div>{k}</div>" for k in kv) + "</div>")
    for ev in (st.get("events") or [])[-4:]:                  # divergence guard: what happened and what it did
        out.append(f'<div class="row"><span class="chip warn">محافظ آموزش</span><span class="mut" style="direction:ltr;'
                   f'text-align:left">{esc(ev.get("text", ""))}</span></div>')
    if st.get("lr_mult") not in (None, 1, 1.0):
        out.append(f'<div class="mut">نرخ یادگیری به‌خاطر ناپایداری <bdi>×{st["lr_mult"]:.3g}</bdi> شده است '
                   f'({st.get("rollbacks", 0)} بار برگشت به بهترین وزن)</div>')
    return "".join(out)


def header(st, chip_cls="", chip=None):
    name, desc = PHASE.get(st.get("phase"), (st.get("phase") or "-", ""))
    chip = chip or name
    info = (f'{esc(st.get("model", ""))} · imgsz {st.get("imgsz", "")} · batch {st.get("batch", "")} · '
            f'{st.get("epochs", "")} epochs')
    return (f'<div class="row" style="justify-content:space-between"><h3>آموزش <bdi>{esc(st.get("run", ""))}</bdi></h3>'
            f'<span class="chip {chip_cls}">{esc(chip)}</span></div><div class="mut">{esc(desc)} — <bdi>{info}</bdi></div>')


def live_html(st, tail=(), sysinfo=None, chart_html="", error=None):
    """The panel shown while train_solder.py runs."""
    now = time.time()
    phase = st.get("phase", "setup")
    cls = {"done": "ok", "done_before": "ok", "stopped": "warn", "error": "bad"}.get(phase, "")
    out = [CSS, '<div class="pp">', header(st, cls)]
    out.append(summary_block(st))
    # the running step
    if phase == "train" and st.get("epoch"):
        nb, ib = st.get("nb") or 1, st.get("ibatch") or 0
        spent = max(1e-6, now - (st.get("phase_t0") or now))
        rate = ib / spent
        left = (nb - ib) / rate if rate > 0 else None
        loss = st.get("loss") or []
        ls = " · ".join(f"{k} {v:.3f}" for k, v in zip(("box", "cls", "dfl"), loss))
        mos = "" if st.get("mosaic", True) else " · بدون موزاییک (مرحله‌ی پایانی)"
        out.append(f'<div class="row"><span class="lbl"><bdi>epoch {st["epoch"]}</bdi></span>{bar(ib / nb, "b2")}'
                   f'<span><bdi>batch <b>{ib}</b>/{nb} · {rate:.2f} batch/s</bdi> · '
                   f'{("حدود " + esc(dur(left)) + " تا پایان این مرحله") if left else ""}</span></div>'
                   f'<div class="mut"><bdi>loss: {ls} · lr {st.get("lr", 0):.2e}</bdi>{mos}</div>')
    elif phase in ("val", "final_val", "eval_val", "eval_test", "eval"):
        vn, vi = st.get("val_nb") or 1, st.get("val_i") or 0
        out.append(f'<div class="row"><span class="lbl">{esc(PHASE.get(phase, (phase,))[0])}</span>{bar(vi / vn, "b3")}'
                   f'<span><bdi>{vi}/{vn}</bdi> · {esc(dur(now - (st.get("phase_t0") or now)))}</span></div>')
    elif phase == "setup":
        msg = st.get("msg") or ""
        if msg.startswith("resume:"):
            _, d, where = msg.split(":", 2)
            txt = f"ادامه‌ی آموزش بعد از epoch {d} (آخرین نسخه از {'Google Drive' if 'Drive' in where else 'همین دیسک'})"
        elif msg == "new":
            txt = "شروع آموزش جدید از وزن‌های پیش‌آموخته‌ی COCO"
        else:
            txt = "آماده‌سازی داده"
        out.append(f'<div class="row"><span class="lbl">مرحله</span><span>{esc(txt)} — {esc(dur(now - st.get("session_start", now)))}</span></div>')
    if sysinfo:
        out.append(f'<div class="mut"><bdi>{esc(sysinfo)}</bdi></div>')
    if error:
        out.append(f'<div class="row"><span class="chip bad">خطا</span><span>{esc(error)}</span></div>')
    out.append(chart_html)
    out.append(epoch_table(st.get("history") or []))
    if tail:
        out.append("<details" + (" open" if phase in ("setup", "error", "stopped") or error else "") +
                   "><summary class='mut'>آخرین خط‌های گزارش</summary><pre>" + esc("\n".join(tail)) + "</pre></details>")
    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------------------------------- status before training
def read_run(d):
    """Status of a run folder on Drive (progress.json + DONE + eval.json), or None."""
    d = Path(d)
    p = d / "progress.json"
    if not p.exists():
        return None
    try:
        st = json.loads(p.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    st["done_file"] = (d / "DONE").exists()
    st["eval"] = json.loads((d / "eval.json").read_text(encoding="utf8")) if (d / "eval.json").exists() else None
    st["has_last"] = (d / "weights" / "last.pt").exists()
    st["phase"] = "done_before" if st["done_file"] else ("stopped" if st.get("history") else "setup")
    return st


def status_html(runs_dir, run, cfg):
    """The 'training status' cell: what exists on Drive and what the training cell will do next."""
    runs_dir = Path(runs_dir)
    st = read_run(runs_dir / run)
    out = [CSS, '<div class="pp">']
    if st is None:
        out.append(f'<div class="row" style="justify-content:space-between"><h3>وضعیت آموزش: {esc(run)}</h3>'
                   f'<span class="chip">هنوز شروع نشده</span></div>')
        out.append(f'<div class="kv"><div>مدل: <b>{esc(cfg["model"])}</b></div><div>اندازه‌ی تصویر: <b>{cfg["imgsz"]}</b></div>'
                   f'<div>تعداد epoch: <b>{cfg["epochs"]}</b></div><div>patience: <b>{cfg["patience"]}</b></div></div>')
        out.append('<div>با اجرای بلوک بعد، آموزش <b>از epoch 1</b> شروع می‌شود. بعد از اولین epoch زمان هر epoch و '
                   'زمان باقی‌مانده دقیق نشان داده می‌شود.</div>')
    else:
        st.setdefault("run", run)
        hist = st.get("history") or []
        if st["done_file"]:
            chip, cls = "تمام شده", "ok"
            nxt = "آموزش این تنظیمات کامل شده؛ بلوک آموزش فقط نتیجه را نشان می‌دهد."
        else:
            se = st.get("synced_epoch") or 0
            chip, cls = f"نیمه‌کاره: {se}/{st.get('epochs')}", "warn"
            nxt = (f"با اجرای بلوک بعد، آموزش از <b>epoch {se + 1}</b> ادامه پیدا می‌کند (آخرین نسخه‌ی امن روی Drive)."
                   if st["has_last"] and se else "هنوز نسخه‌ای از وزن‌ها روی Drive نیست؛ آموزش از اول شروع می‌شود.")
        out.append(header(st, cls, chip))
        out.append(summary_block(st))
        out.append(f"<div>{nxt}</div>")
        if st.get("updated"):
            out.append(f'<div class="mut">آخرین به‌روزرسانی: {esc(ago(st["updated"]))}</div>')
        if st.get("epochs") and st["epochs"] != cfg["epochs"] and not st["done_file"]:
            out.append(f'<div class="mut">تعداد epoch در تنظیمات ({cfg["epochs"]}) با این آموزش ({st["epochs"]}) فرق دارد؛ '
                       f'هنگام ادامه، هدف به {cfg["epochs"]} تغییر می‌کند.</div>')
        out.append(chart(hist, st.get("epochs")))
        out.append(epoch_table(hist, 8))
        if st.get("eval"):
            out.append(eval_html(st["eval"], inner=True))
    others = []
    for d in sorted(p for p in runs_dir.glob("*") if p.is_dir() and p.name not in (run, "final")):
        o = read_run(d)
        if o:
            h = o.get("history") or []
            b = max(h, key=lambda x: x["mAP"]) if h else None
            others.append(f"<tr><td>{esc(d.name)}</td><td>{len(h)}/{o.get('epochs')}</td>"
                          f"<td>{'تمام' if o['done_file'] else 'کنار گذاشته (نسخه‌ی قدیمی)'}</td>"
                          f"<td>{b['mAP50']:.3f} / {b['mAP']:.3f}</td></tr>" if b else "")
        elif (d / "weights").exists() or (d / "results.csv").exists():
            others.append(f"<tr><td>{esc(d.name)}</td><td colspan=3>نسخه‌ی قدیمی (دستور قبلی) — استفاده نمی‌شود</td></tr>")
    if others:
        out.append("<details><summary class='mut'>آموزش‌های دیگر در همین پوشه</summary><table><tr><th>پوشه</th><th>epoch</th>"
                   "<th>وضعیت</th><th>بهترین mAP50 / mAP50-95</th></tr>" + "".join(others) + "</table></details>")
    out.append(f'<div class="mut" style="direction:ltr;text-align:right">{esc(runs_dir / run)}</div></div>')
    return "".join(out)


# ------------------------------------------------------------------------------------------- results
def eval_html(ev, inner=False):
    out = [] if inner else [CSS, '<div class="pp">']
    t, v = ev["test"], ev["val"]
    out.append(f'<h3>نتیجه روی test — بردهایی که مدل هرگز ندیده</h3>'
               f'<div class="kv"><div>mAP50: <b>{t["mAP50"]:.3f}</b></div><div>mAP50-95: <b>{t["mAP50-95"]:.3f}</b></div>'
               f'<div>(روی val: mAP50 <b>{v["mAP50"]:.3f}</b>)</div>'
               + (f'<div>سرعت: <b>{t["ms_per_image"]:.0f} ms</b> برای هر عکس</div>' if t.get("ms_per_image") else "")
               + "</div>")
    out.append("<table><tr><th>کلاس</th><th>کادر واقعی</th><th>mAP50</th><th>mAP50-95</th><th>آستانه</th>"
               "<th>Recall: از ۱۰۰ عیب چندتا پیدا شد</th><th>Precision: از ۱۰۰ هشدار چندتا درست بود</th></tr>")
    for c in sorted(t["classes"], key=lambda c: -c["instances"]):
        out.append(f'<tr><td class="fa">{esc(FA.get(c["class"], c["class"]))} <span class="mut">({esc(c["class"])})</span></td>'
                   f'<td>{c["instances"]}</td><td>{c["mAP50"]:.3f}</td><td>{c["mAP50-95"]:.3f}</td><td>{c["conf"]:.2f}</td>'
                   f'<td>{100 * c["recall"]:.0f}</td><td>{100 * c["precision"]:.0f}</td></tr>')
    out.append("</table><div class='mut'>mAP50 = دقت کلی (کادر پیدا‌شده حداقل ۵۰٪ روی کادر واقعی). آستانه‌ی هر کلاس روی val "
               "تنظیم شده (بیشترین F1) و Recall/Precision با همان آستانه روی test اندازه‌گیری شده است.</div>")
    if not inner:
        out.append("</div>")
    return "".join(out)
