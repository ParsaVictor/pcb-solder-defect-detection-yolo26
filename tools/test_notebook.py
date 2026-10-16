"""Run PCB_Defects_Colab.ipynb end to end on this computer, the way Colab would (cell by cell, in order).

Colab paths are mapped to local folders, the GPU check is relaxed to CPU and QUICK_TEST is switched on.
Every model is run twice: the second "Run all" must skip finished work (resume path).

  python tools/test_notebook.py <empty test folder> [solder bare]
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"solder": ("1_Solder_Model", "Solder_Model_Colab.ipynb"), "bare": ("2_Bare_Board_Model", "Bare_Model_Colab.ipynb")}


def cells_of(nb, model, drive, work):
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = src.replace(f'"/content/drive/MyDrive/{FOLDERS[model][0]}"', repr(str(drive)))
        src = src.replace('"/content/work"', repr(str(work)))
        src = src.replace("QUICK_TEST = False", "QUICK_TEST = True")
        out.append(src)
    return out


def main():
    base = Path(sys.argv[1]).resolve()
    models = sys.argv[2:] or ["solder", "bare"]
    drive, work = base / "drive", base / "work"
    drive.mkdir(parents=True, exist_ok=True)
    for m in models:                       # the zips, "uploaded" to the fake Drive (hard links, no copy)
        z = drive / f"{m}_data.zip"
        if not z.exists():
            os.link(ROOT / "upload_to_drive" / FOLDERS[m][0] / f"{m}_data.zip", z)
    os.environ["PCB_ALLOW_CPU"] = "1"
    for m in models:
        nb = json.load(open(ROOT / "upload_to_drive" / FOLDERS[m][0] / FOLDERS[m][1], encoding="utf8"))
        for attempt in (1, 2):
            print(f"\n################ {m} - Run all #{attempt} ################", flush=True)
            ns = {"__name__": "__main__"}
            for i, src in enumerate(cells_of(nb, m, drive, work), 1):
                t = time.time()
                print(f"\n----- cell {i} -----", flush=True)
                exec(compile(src, f"<cell {i}>", "exec"), ns)
                print(f"----- cell {i} ok ({time.time() - t:.0f}s)", flush=True)
        print(f"\n######## {m}: NOTEBOOK OK ########", flush=True)


if __name__ == "__main__":
    main()
