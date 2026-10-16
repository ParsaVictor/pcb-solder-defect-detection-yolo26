"""Build one Colab notebook per model, each in its own folder with its data: upload_to_drive/1_Solder_Model, 2_Bare_Board_Model

All code (verify / train / predict / panel) is embedded base64-encoded and written to disk by the setup cell,
so the notebook never depends on cell order or on IPython magics. Every step skips itself when its work is
done: after a disconnect, Run all resumes. Training runs in a subprocess; the notebook redraws a live panel
from the status file the training script writes (train/panel.py).

  python tools/build_notebook.py solder
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT / "upload_to_drive"
SCRIPTS = {"verify_dataset.py": "tools/verify_dataset.py", "train_solder.py": "train/train_solder.py",
           "predict.py": "infer/predict.py", "panel.py": "train/panel.py"}
ULTRALYTICS = "8.4.14"      # the version every script was tested with


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def report(fam):
    return (ROOT / f"final_{fam}" / "report.txt").read_text(encoding="utf8")


blobs = {name: base64.b64encode((ROOT / src).read_bytes()).decode() for name, src in SCRIPTS.items()}

STORY_SOLDER = 'در آموزش قبلی (v2) بعد از epoch 4 دقت از ۰٫۱۸ (mAP50-95) به ۰٫۰۵ افتاد و دیگر بالا نیامد. علت **بیش\u200cبرازش یا کمبود داده نبود**:\nloss در epoch 5 ناگهان منفجر شد (cls از ۲٫۷ به ۲۶) و مدل هیچ\u200cوقت برنگشت. دو تنظیم v2 با پیش\u200cفرض Ultralytics فرق داشتند\n(نرخ یادگیری ۰٫۰۰۱ به\u200cجای ۰٫۰۰۰۷۷ و `warmup_bias_lr=0.1` که برای AdamW بسیار زیاد است). در v3:'
STORY_BARE = 'این مدل هنوز آموزش ندیده است؛ از آموزش لحیم یاد گرفتیم که تنظیم AdamW باید محتاط باشد. تنظیمات این نسخه:'

INFO = {
    "solder": dict(folder="1_Solder_Model", title="عیب‌های لحیم‌کاری و مونتاژ",
                   classes="لحیم کم، لحیم زیاد، پل لحیم، حفره‌ی لحیم، قطعه‌ی جاافتاده، قطعه‌ی جابه‌جا، سوختگی، آسیب فیزیکی، اکسید",
                   model="yolo26m.pt", imgsz=800, epochs=150, patience=50, v3_story=STORY_SOLDER,
                   model_note="- **مدل بزرگ‌تر: YOLO26m** (۲۰ میلیون پارامتر، به‌جای ۱۰ میلیون در s). Colab رایگان فقط ۲ هسته‌ی CPU دارد و "
                              "در آموزش قبلی GPU بخشی از وقت منتظر CPU بود؛ مدل بزرگ‌تر از همان GPU بیشتر استفاده می‌کند.",
                   resize_note="عکس‌های بزرگ train (برش‌های ۱۲۸۰ پیکسلی neazu)",
                   time_note="زمان تقریبی روی T4 رایگان: حدود ۴ تا ۵ دقیقه برای هر epoch با YOLO26m (چند جلسه‌ی Colab؛ ادامه خودکار است). "
                             "YOLO26s تقریباً دو برابر سریع‌تر ولی کمی کم‌دقت‌تر است."),
    "bare": dict(folder="2_Bare_Board_Model", title="عیب‌های برد خام (قبل از نصب قطعه)",
                 classes="سوراخ جاافتاده، بریدگی لبه، قطعی مسیر، اتصال کوتاه، زائده‌ی مسی، مس اضافه، سوراخ ریز",
                 model="yolo26s.pt", imgsz=1024, epochs=50, patience=20, v3_story=STORY_BARE,
                 model_note="- **YOLO26s با اندازه‌ی ۱۰۲۴**: این دیتاست بزرگ است (~۱۸ هزار فایل train)، پس هر epoch طولانی است؛ "
                            "s بهترین تعادل دقت و زمان است. برای دقت بیشتر (و حدود دو برابر زمان) MODEL_SIZE را yolo26m.pt کن.",
                 resize_note="عکس‌های بزرگ train (برش‌های ۱۲۸۰ و ۱۶۰۰ پیکسلی PKU)",
                 time_note="زمان تقریبی روی T4 رایگان: هر epoch چند ده دقیقه (۱۸ هزار فایل)؛ کل آموزش چند جلسه‌ی Colab طول می‌کشد و ادامه خودکار است."),
}


def head_cells(fam):
    i = INFO[fam]
    return [
        md(f"""
# مدل {i['title']} — YOLO26 روی Google Colab

**چه چیزی پیدا می‌کند:** {i['classes']}

## اجرا (فقط همین)
1. کل پوشه‌ی `{i['folder']}` را در Google Drive (داخل My Drive) آپلود کن: این نوت‌بوک + `{fam}_data.zip`.
2. نوت‌بوک را با Google Colaboratory باز کن ▸ `Runtime ▸ Change runtime type ▸ T4 GPU ▸ Save`
3. `Runtime ▸ Run all` و اجازه‌ی دسترسی به Google Drive را بده.
4. **قطع شد یا لیمیت خوردی؟** فقط دوباره `Run all` (با همین حساب یا حساب دیگری که پوشه را دارد): از آخرین نسخه‌ی ذخیره‌شده ادامه می‌دهد.

| بلوک | کار |
|---|---|
| ⚙️ تنظیمات | مدل، تعداد epoch، اندازه‌ی تصویر (پیش‌فرض‌ها برای بیشترین دقت انتخاب شده‌اند) |
| ۱، ۲ | آماده‌سازی Colab و داده (هر جلسه چند دقیقه) |
| **۳) وضعیت آموزش** | قبل از شروع: چند epoch انجام شده، بهترین دقت، زمان باقی‌مانده و این‌که آموزش از کجا ادامه پیدا می‌کند |
| **۴) آموزش** | پنل زنده که هر ۲ ثانیه به‌روز می‌شود: نوار پیشرفت کل و epoch جاری، loss، دقت هر epoch، نمودار، زمان باقی‌مانده، GPU |
| ۵، ۶ | نتیجه روی بردهای دیده‌نشده (هر کلاس جدا) و تست روی عکس خودت |

**توقف دستی:** دکمه‌ی ■ (Interrupt). آخرین epoch کامل‌شده روی Drive ذخیره می‌شود و دفعه‌ی بعد از همان‌جا ادامه می‌دهد.

<details open><summary>تنظیمات پایداری و دقت (نسخه‌ی v3)</summary>

{i['v3_story']}
- نرخ یادگیری **۰٫۰۰۰۵**، `warmup_bias_lr=0`، گرم‌کردن **۵ epoch**.
- **محافظ آموزش:** بعد از هر epoch چک می‌کند loss منفجر نشده و دقت نیفتاده. اگر شد، خودکار به **بهترین وزن** برمی‌گردد،
  نرخ یادگیری را نصف می‌کند و در پنل می‌نویسد چه شد. بعد از ۴ بار ناپایداری آموزش را متوقف می‌کند (ساعت‌ها هدر نمی‌رود).
- augmentation واقعی‌تر: نویز دوربین، محوشدگی حرکتی و فشرده‌سازی JPEG (روی تصویر، کادرها دست‌نخورده) + مقیاس بیشتر.
  کپی آفلاین از عکس‌ها فایده‌ای ندارد (همان augmentation تصادفی هر epoch را انجام می‌دهد)؛ تنوع واقعی فقط با عکس واقعی بیشتر می‌آید.
- وزن بهترین epoch آموزش‌های قبلی خودکار در `saved_weights` ذخیره می‌شود. آموزش v3 در پوشه‌ی جدید خودش است.

</details>

<details><summary>چه چیزهایی در نسخه‌ی v2 برای دقت و سرعت بهتر شده بود</summary>

{i['model_note']}
- **{i['epochs']} epoch** با برنامه‌ی کاهش نرم نرخ یادگیری (cosine) و AdamW؛ چند epoch آخر بدون موزاییک (روی عکس‌های کامل) برای دقت نهایی.
  هر epoch هر عکس train را دو بار می‌بیند (خود عکس + نسخه‌ی ۹۰ درجه).
  اگر {i['patience']} epoch پشت سر هم بهتر نشود، زودتر تمام می‌شود (patience).
- **batch ثابت و درست** به اندازه‌ی حافظه‌ی GPU (AutoBatch قبلی batch خیلی کوچکی انتخاب می‌کرد و GPU نیمه‌خالی می‌ماند).
- {i['resize_note']} **یک بار با کیفیت بالا (INTER_AREA)** به اندازه‌ی آموزش کوچک می‌شوند:
  بارگذاری سریع‌تر و جزئیات ریز تمیزتر.
- آموزش روی دیسک سریع Colab انجام می‌شود و نسخه‌ی امن **هر ۲ epoch**، هنگام توقف دستی و در پایان روی Drive کپی می‌شود.
- **حداکثر سرعت:** batch به اندازه‌ی حافظه‌ی GPU (روی GPUهای بزرگ‌تر خودکار بیشتر)، تصاویر در RAM کش می‌شوند اگر جا باشد
  (وگرنه روی دیسک سریع)، همه‌ی هسته‌های CPU برای بارگذاری داده، cuDNN autotune، AMP.
- **GPU نبود؟** خودکار روی CPU اجرا می‌شود (خیلی کندتر؛ فقط برای آزمایش مناسب است).
- **آستانه‌ی اطمینان جدا برای هر کلاس** روی val تنظیم می‌شود (بیشترین F1) و پیش‌بینی‌ها با همان آستانه‌ها انجام می‌شود.
- انتخاب بهترین مدل فقط با **val**؛ test فقط برای گزارش صادقانه است.

{i['time_note']} عدد دقیق بعد از اولین epoch در پنل نشان داده می‌شود.
</details>

<details><summary>گزارش دیتاست</summary>

```
{report(fam)}
```
</details>
"""),
        code(f"""
#@title ⚙️ تنظیمات (پیش‌فرض‌ها برای بیشترین دقت انتخاب شده‌اند)
MODEL        = "{fam}"
MODEL_SIZE   = "{i['model']}"   #@param ["yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]
EPOCHS       = {i['epochs']}   #@param {{type:"integer"}}
IMGSZ        = {i['imgsz']}   #@param {{type:"integer"}}
PATIENCE     = {i['patience']}   #@param {{type:"integer"}}
BATCH        = 0   #@param {{type:"integer"}}
SAVE_EVERY   = 2   #@param {{type:"integer"}}
QUICK_TEST   = False   #@param {{type:"boolean"}}
DRIVE_DIR    = "/content/drive/MyDrive/{i['folder']}"   #@param {{type:"string"}}
# MODEL_SIZE  : s سریع‌تر، m دقیق‌تر (پیشنهاد)، l کمی دقیق‌تر و خیلی کندتر
# EPOCHS      : وسط آموزش هم می‌شود عوضش کرد؛ دفعه‌ی بعد که ادامه پیدا کند هدف جدید اعمال می‌شود
# PATIENCE    : اگر دقت val این تعداد epoch بهتر نشود، آموزش زودتر تمام می‌شود
# BATCH       : 0 = خودکار بر اساس مدل و حافظه‌ی GPU (روی T4: s=16، m=8، l=6 در اندازه‌ی 800)
# SAVE_EVERY  : هر چند epoch مدل روی Drive ذخیره شود (قطع شدن = از دست رفتن حداکثر همین تعداد epoch)
# QUICK_TEST  : آزمایش چنددقیقه‌ای کل مسیر (2 epoch کوچک) در پوشه‌ی جدا؛ برای آموزش اصلی خاموش باشد
RECIPE = "v3"
RUN_IMGSZ, RUN_EPOCHS = (320, 2) if QUICK_TEST else (int(IMGSZ), int(EPOCHS))
RUN_NAME = f"{{MODEL_SIZE[:-3]}}_{{RUN_IMGSZ}}_{{RECIPE}}"
print("مدل:", MODEL_SIZE, "| imgsz", RUN_IMGSZ, "| epochs", RUN_EPOCHS, "| patience", PATIENCE,
      "| پوشه‌ی آموزش:", RUN_NAME, "| آزمایش سریع" if QUICK_TEST else "| آموزش کامل")
"""),
    ]


REST = [
    md("## 1) آماده‌سازی: Drive، GPU، کتابخانه‌ها، کدها"),
    code(f'''
import os, sys, subprocess, glob, json, shutil, zipfile, base64, time, re, signal, threading, collections, importlib
WORK = "/content/work"                        # local fast disk (wiped on disconnect; results live on Drive)
os.makedirs(WORK, exist_ok=True)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

def run_live(cmd):
    """Run a short command and stream its output into the cell (progress bars collapse to their last state)."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env={{**os.environ, "PYTHONUNBUFFERED": "1"}})
    try:
        for raw in p.stdout:                  # a progress bar redraws with \\r: keep only its final state
            parts = [s for s in raw.decode("utf-8", "replace").rstrip("\\r\\n").split("\\r") if s.strip()]
            if parts:
                print(parts[-1], flush=True)
    except KeyboardInterrupt:
        p.kill()
        raise
    if p.wait() != 0:
        raise RuntimeError(f"دستور با خطا تمام شد (کد {{p.returncode}}) - چند خط بالاتر علت را نوشته است")

# Python packages: the exact ultralytics version this code was tested with
try:
    import ultralytics
    have = ultralytics.__version__
except ImportError:
    have = None
if have != "{ULTRALYTICS}":
    run_live([sys.executable, "-m", "pip", "install", "-q", "ultralytics=={ULTRALYTICS}"])
import ultralytics, torch
importlib.reload(ultralytics)
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| python", sys.version.split()[0])

if torch.cuda.is_available():
    g = torch.cuda.get_device_properties(0)
    print(f"GPU: {{g.name}} ({{g.total_memory / 1e9:.0f}} GB) | CPU cores: {{os.cpu_count()}}")
else:
    print("⚠️ GPU پیدا نشد: آموزش خودکار روی CPU انجام می‌شود (خیلی کندتر). برای سرعت: "
          "Runtime ▸ Change runtime type ▸ T4 GPU ▸ Save و دوباره Run all")
print(f"فضای آزاد دیسک Colab: {{shutil.disk_usage(WORK).free / 1e9:.0f}} GB")

# the scripts (verify / train / predict / live panel), written next to each other on the local disk
SCRIPTS = {json.dumps(blobs)}
for name, blob in SCRIPTS.items():
    with open(f"{{WORK}}/{{name}}", "wb") as f:
        f.write(base64.b64decode(blob))
sys.path.insert(0, WORK)
import panel
importlib.reload(panel)
from IPython.display import display, HTML, Image
print("کدها آماده:", ", ".join(SCRIPTS))

ANSI = re.compile(r"\\x1b\\[[0-9;?]*[A-Za-z]")


def sysinfo():
    out = []
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        u, mu, mt, t = [x.strip() for x in q.strip().split(",")]
        out.append(f"GPU {{u}}% busy · {{float(mu) / 1024:.1f}}/{{float(mt) / 1024:.0f}} GB · {{t}}°C")
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.append(f"CPU {{psutil.cpu_percent():.0f}}% · RAM {{vm.used / 1e9:.1f}}/{{vm.total / 1e9:.1f}} GB")
    except Exception:
        pass
    return " | ".join(out)


def train_live(cmd, status_file, log_file):
    """Run train_solder.py and redraw the live panel every 2 s. Returns (exit code, last log lines, status)."""
    if os.path.exists(status_file):
        os.remove(status_file)
    tail = collections.deque(maxlen=14)
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("\\n$ " + " ".join(cmd) + "\\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env={{**os.environ, "PYTHONUNBUFFERED": "1"}})

    def reader():                              # reads the output as it comes: \\r progress bars included
        buf, t_bar = "", 0.0
        with open(log_file, "a", encoding="utf-8") as lf:
            while True:
                chunk = p.stdout.read1(65536)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", "replace")
                parts = re.split(r"(\\r|\\n)", buf)
                buf = parts.pop()
                for seg, sep in zip(parts[0::2], parts[1::2]):
                    s = ANSI.sub("", seg).strip()
                    if not s:
                        continue
                    is_bar = "━" in s or "─" in s
                    if is_bar and tail and ("━" in tail[-1] or "─" in tail[-1]):
                        tail[-1] = s           # the same progress bar, redrawn
                    else:
                        tail.append(s)
                    if not is_bar or sep == "\\n" or time.time() - t_bar > 30:
                        lf.write(s + "\\n")
                        lf.flush()
                        t_bar = time.time() if is_bar else t_bar

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    def status():
        try:
            return json.load(open(status_file, encoding="utf-8"))
        except Exception:
            return {{"phase": "setup", "run": RUN_NAME, "model": MODEL_SIZE, "imgsz": RUN_IMGSZ, "epochs": RUN_EPOCHS,
                    "session_start": t0}}

    t0 = time.time()
    h = display(HTML(panel.live_html(status(), list(tail))), display_id=True) or PanelFile(f"{{WORK}}/live_panel.html")
    n_hist, chart_html, info, t_info = -1, "", "", 0.0
    try:
        while True:
            rc = p.poll()
            st = status()
            hist = st.get("history") or []
            if len(hist) != n_hist:
                n_hist, chart_html = len(hist), panel.chart(hist, st.get("epochs"))
            if time.time() - t_info > 5:
                info, t_info = sysinfo(), time.time()
            if rc is not None:
                th.join(10)
                if rc != 0 and st.get("phase") not in ("stopped", "done"):
                    st["phase"] = "error"
            h.update(HTML(panel.live_html(st, list(tail), info, chart_html)))
            if rc is not None:
                return rc, list(tail), st
            time.sleep(2)
    except KeyboardInterrupt:
        print("در حال توقف ... آخرین epoch کامل روی Drive ذخیره می‌شود (حداکثر یک دقیقه)", flush=True)
        p.send_signal(signal.SIGINT)
        for _ in range(3):
            try:
                p.wait(timeout=90)
                break
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                p.kill()
        th.join(10)
        st = status()
        st["phase"] = "stopped"
        h.update(HTML(panel.live_html(st, list(tail), info, chart_html)))
        print("متوقف شد. برای ادامه: Run all (یا فقط بلوک‌ها را از اول اجرا کن).")
        raise
    except BaseException:                      # a problem in this loop must never leave the training running alone
        p.kill()
        raise


class PanelFile:
    """Outside a notebook display() returns nothing: the panel goes to an HTML file instead."""
    def __init__(self, path):
        self.path = path

    def update(self, obj):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write('<meta charset="utf-8">' + obj.data)
'''),
    md("## 2) داده: پیدا کردن zip در Drive، باز کردن روی دیسک سریع، بررسی"),
    code('''
DATA = f"{WORK}/final_{MODEL}"
if os.path.exists(f"{DATA}/data.yaml"):
    print("دیتاست از قبل باز شده:", DATA)
else:
    name = f"{MODEL}_data.zip"
    src = f"{DRIVE_DIR}/{name}"
    if not os.path.exists(src):                  # search the rest of the Drive
        # a folder shared with you lives under "Shared with me": add it to My Drive as a shortcut, then
        # Colab sees it in MyDrive or under .shortcut-targets-by-id
        roots = [DRIVE_DIR, "/content/drive/MyDrive", "/content/drive/.shortcut-targets-by-id",
                 "/content/drive/Shareddrives"] if IN_COLAB else [DRIVE_DIR]
        hits = []
        for r in roots:                           # look up to 6 folders deep, nearest first
            for d in range(7):
                hits += glob.glob(r + "/*" * d + "/" + name)
            if hits:
                break
        if not hits:
            raise SystemExit(f"فایل {name} در Google Drive پیدا نشد. اگر پوشه برایت شیر شده: در Drive روی پوشه "
                             f"راست‌کلیک ▸ Organize ▸ Add shortcut ▸ My Drive، بعد دوباره Run all.")
        src = hits[0]
    size = os.path.getsize(src) / 1e9
    if shutil.disk_usage(WORK).free / 1e9 < size * 2.2 + 10:
        raise SystemExit(f"فضای دیسک Colab کم است: {size * 2.2 + 10:.0f} GB لازم است")
    print("داده پیدا شد:", src)
    open(f"{WORK}/.drive_folder", "w").write(os.path.dirname(src))   # results are saved next to the data
    print(f"کپی ({size:.1f} GB) روی دیسک سریع ...", flush=True)
    local = f"{WORK}/{name}"
    shutil.copyfile(src, local)
    print("باز کردن ...", flush=True)
    with zipfile.ZipFile(local) as z:
        z.extractall(WORK)
    os.remove(local)
    assert os.path.exists(f"{DATA}/data.yaml"), f"zip خراب است یا مال مدل دیگری است: {src}"
    print("باز شد:", DATA)

if not os.path.exists(f"{DATA}/.verified"):
    run_live([sys.executable, f"{WORK}/verify_dataset.py", "--root", DATA])
    open(f"{DATA}/.verified", "w").write("ok")
else:
    print("دیتاست قبلاً بررسی شده")

# results go next to the data; if that shared folder is read-only for you, into your own Drive
base = open(f"{WORK}/.drive_folder").read().strip() if os.path.exists(f"{WORK}/.drive_folder") else DRIVE_DIR
try:
    os.makedirs(f"{base}/runs", exist_ok=True)
    open(f"{base}/runs/.write_test", "w").write("ok")
except OSError:
    base = "/content/drive/MyDrive/PCB_Defect_Detection_results" if IN_COLAB else f"{WORK}/results"
    print("پوشه‌ی داده فقط-خواندنی است؛ نتیجه‌ها اینجا ذخیره می‌شوند:", base)
RUNS = f"{base}/runs_test" if QUICK_TEST else f"{base}/runs"
PROJECT = f"{RUNS}/{MODEL}"
os.makedirs(PROJECT, exist_ok=True)
print("نتیجه‌ها روی Drive:", PROJECT)

# keep the best weights of every OLDER run in saved_weights/ (never overwritten, never deleted)
SAVED = f"{base}/saved_weights"
for old in sorted(glob.glob(f"{PROJECT}/*/weights/best.pt")):
    run = os.path.basename(os.path.dirname(os.path.dirname(old)))
    if run == RUN_NAME:
        continue
    dst = f"{SAVED}/{MODEL}_{run}_best.pt"
    if not os.path.exists(dst):
        os.makedirs(SAVED, exist_ok=True)
        shutil.copyfile(old, dst)
        try:
            ep = torch.load(old, map_location="cpu", weights_only=False).get("epoch", -1) + 1
        except Exception:
            ep = "?"
        print(f"وزن بهترین epoch آموزش قبلی ({run}, epoch {ep}) ذخیره شد: {dst}")
    else:
        print("وزن آموزش قبلی از قبل ذخیره است:", dst)
'''),
    md("""## 3) وضعیت آموزش
قبل از شروع نشان می‌دهد چه چیزی روی Drive ذخیره شده: چند epoch انجام شده، بهترین دقت تا الان، زمان هر epoch،
زمان باقی‌مانده، و این‌که بلوک بعد آموزش را **از کدام epoch** ادامه می‌دهد. هر وقت خواستی فقط همین بلوک را اجرا کن
(بعد از بلوک‌های ۱ و ۲)."""),
    code('''
display(HTML(panel.status_html(PROJECT, RUN_NAME, {"model": MODEL_SIZE, "imgsz": RUN_IMGSZ, "epochs": RUN_EPOCHS,
                                                   "patience": PATIENCE})))
'''),
    md("""## 4) آموزش — پنل زنده
پنل هر ۲ ثانیه به‌روز می‌شود: **نوار کل آموزش** (epoch چندم از چند، زمان باقی‌مانده)، **نوار epoch جاری**
(batch چندم، سرعت، loss)، مرحله‌ی ارزیابی روی val، جدول و نمودار دقت هر epoch، آخرین نسخه‌ی امن روی Drive، و وضعیت GPU.

اولین بار در هر جلسه چند دقیقه «آماده‌سازی» طول می‌کشد (کوچک کردن عکس‌های بزرگ و کش کردن تصاویر روی دیسک).
گزارش کامل: `/content/work/train_log.txt` (در پنل: «آخرین خط‌های گزارش»)."""),
    code('''
status_file, log_file = f"{WORK}/live_status.json", f"{WORK}/train_log.txt"
try:                                           # a training left over from an earlier stopped cell: stop it cleanly
    if subprocess.run(["pkill", "-INT", "-f", "train_solder.py"], capture_output=True).returncode == 0:
        print("یک آموزش قبلی هنوز در حال اجرا بود؛ متوقفش می‌کنم ...", flush=True)
        for _ in range(60):
            time.sleep(2)
            if subprocess.run(["pgrep", "-f", "train_solder.py"], capture_output=True).returncode != 0:
                break
        subprocess.run(["pkill", "-KILL", "-f", "train_solder.py"], capture_output=True)
except FileNotFoundError:
    pass
cmd =[sys.executable, f"{WORK}/train_solder.py", "--data", DATA, "--work", f"{WORK}/runs", "--drive", PROJECT,
       "--model", MODEL_SIZE, "--imgsz", str(RUN_IMGSZ), "--epochs", str(RUN_EPOCHS), "--patience", str(PATIENCE),
       "--status", status_file, "--save-every", "1" if QUICK_TEST else str(SAVE_EVERY)]
if QUICK_TEST:
    cmd += ["--fraction", "0.03", "--cache", "False"]
cmd += ["--workers", str(max(2, min(16, os.cpu_count() or 2)))]           # every CPU core loads data
if not torch.cuda.is_available():                                           # no GPU: train on the CPU
    cmd += ["--device", "cpu"]
batch = int(BATCH)
for attempt in range(3):
    log_start = os.path.getsize(log_file) if os.path.exists(log_file) else 0
    rc, tail, st = train_live(cmd + (["--batch", str(batch)] if batch else []), status_file, log_file)
    if rc == 0:
        break
    with open(log_file, encoding="utf-8", errors="replace") as f:
        f.seek(log_start)
        log = f.read()
    if "out of memory" in log.lower() and attempt < 2:      # resume with a smaller batch
        batch = max(2, int(st.get("batch") or 8) // 2)
        print(f"حافظه‌ی GPU کم آمد؛ ادامه با batch {batch} ...", flush=True)
        continue
    print("\\n".join(log.strip().splitlines()[-25:]))
    raise RuntimeError(f"آموزش با خطا تمام شد (کد {rc}) - خط‌های بالا علت را نوشته‌اند")
if QUICK_TEST:
    print("\\nآزمایش سریع بی‌خطا تمام شد. در تنظیمات QUICK_TEST را خاموش کن و دوباره Run all بزن تا آموزش اصلی شروع شود.")
'''),
    md("""## 5) نتیجه
متریک‌ها روی **بردهایی که مدل هرگز ندیده** (test) حساب شده‌اند؛ برای هر کلاس جدا.
عکس‌های نمونه: کادر سبز نازک = عیب واقعی (برچسب)، کادر رنگی = پیش‌بینی مدل."""),
    code('''
FINAL = f"{PROJECT}/final"
if not os.path.exists(f"{FINAL}/summary.json"):
    print("هنوز آموزش تمام نشده")
else:
    s = json.load(open(f"{FINAL}/summary.json"))
    run = s["chosen"]
    ev = json.load(open(f"{PROJECT}/{run}/eval.json"))
    display(HTML(panel.eval_html(ev)))
    if len(s["runs"]) > 1:
        print("همه‌ی آموزش‌ها (mAP50-95):", json.dumps(s["runs"]), "| انتخاب‌شده با val:", run)
    for pat, w in [("results.png", 950), ("test_eval/confusion_matrix_normalized.png", 720), ("test_eval/*PR_curve.png", 720)]:
        for f in sorted(glob.glob(f"{PROJECT}/{run}/{pat}"))[:1]:
            display(Image(f, width=w))
    out = f"{WORK}/test_preview_{run}"
    if not glob.glob(out + "/*_annotated.jpg"):
        by_src = collections.defaultdict(list)       # test photos with defects, a few from every source
        for p in sorted(glob.glob(f"{DATA}/images/test/*.jpg")):
            lbl = p.replace("/images/", "/labels/")[:-4] + ".txt"
            if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                by_src[os.path.basename(p).split("__")[0]].append(p)
        picks = []
        while len(picks) < 8 and any(by_src.values()):
            for k in list(by_src):
                if by_src[k] and len(picks) < 8:
                    lst = by_src[k]
                    picks.append(lst.pop(len(lst) // 2))
        prev = f"{WORK}/preview_src"
        shutil.rmtree(prev, ignore_errors=True)
        os.makedirs(prev)
        for p in picks:
            shutil.copy2(p, prev)
        run_live([sys.executable, f"{WORK}/predict.py", "--weights", f"{FINAL}/best.pt", "--source", prev,
                  "--out", out, "--labels", f"{DATA}/labels/test"])
    for p in sorted(glob.glob(out + "/*_annotated.jpg")):
        display(Image(p, width=620))
    print("بهترین مدل:", f"{FINAL}/best.pt", "| آستانه‌ها:", f"{FINAL}/thresholds.json")
'''),
    md("## 6) تست روی عکس خودت\n`IMAGE` خالی بماند تا دکمه‌ی آپلود بیاید، یا مسیر یک عکس/پوشه در Drive را بنویس.\n"
       "`CONF` = 0 یعنی آستانه‌ی تنظیم‌شده‌ی هر کلاس؛ عدد بین 0 و 1 یعنی یک آستانه برای همه (کمتر = عیب‌های بیشتر و هشدار اشتباه بیشتر)."),
    code('''
IMAGE = ""   #@param {type:"string"}
CONF  = 0   #@param {type:"number"}
weights = f"{PROJECT}/final/best.pt"
if not os.path.exists(weights):
    print("اول آموزش باید تمام شود:", weights)
else:
    if not IMAGE and IN_COLAB:
        from google.colab import files
        up = files.upload(); IMAGE = f"{WORK}/upload"; os.makedirs(IMAGE, exist_ok=True)
        for n, b in up.items():
            open(f"{IMAGE}/{n}", "wb").write(b)
    if IMAGE:
        out = f"{WORK}/my_test_{MODEL}"
        run_live([sys.executable, f"{WORK}/predict.py", "--weights", weights, "--source", IMAGE, "--out", out]
                 + (["--conf", str(CONF)] if CONF else []))
        for p in sorted(glob.glob(out + "/*_annotated.jpg")):
            display(Image(p, width=1100))
'''),
]


def make(fam):
    folder = SHARE / INFO[fam]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{fam.capitalize()}_Model_Colab.ipynb"
    nb = {"cells": head_cells(fam) + REST,
          "metadata": {"accelerator": "GPU", "colab": {"name": out.name, "provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("notebook ->", out)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["solder"]:
        make(f)
