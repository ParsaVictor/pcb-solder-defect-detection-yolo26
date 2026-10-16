"""Put every cleaned source image in ONE folder with unified class ids and sanitised labels.

  data/clean/<source>/{images,labels}  ->  data/all_data/{images,labels}/<source>__<name>.*
                                           data/all_data/manifest.csv
                                           data/all_data/label_issues.csv

Per box: polygon -> tight box, class remapped by configs/class_map.yaml, clipped to the image,
boxes that are degenerate (< 2 px) or exact duplicates are dropped and logged.
"""
import csv
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from yololib import parse_line

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "data" / "all_data"


def family(name):
    """Image family = the original dataset the photo came from (helps grouping in reports)."""
    if name.startswith("light_"):
        return "light"
    if name[:2].isdigit() and "_" in name:
        return "pku"
    return "other"


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "class_map.yaml", encoding="utf8"))
    final = cfg["final_classes"]
    cmap = {str(k): v for k, v in cfg["map"].items()}
    excl = set(cfg.get("exclude_sources") or [])
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    rows, issues = [], []
    for src in sorted(p for p in CLEAN.iterdir() if (p / "manifest.csv").exists()):
        names = yaml.safe_load(open(src / "source_data.yaml", encoding="utf8"))["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        for img_p in sorted((src / "images").iterdir()):
            im = cv2.imread(str(img_p))
            if im is None:
                issues.append((src.name, img_p.name, "unreadable image"))
                continue
            h, w = im.shape[:2]
            out_name = f"{src.name}__{img_p.stem}"
            lines, seen = [], set()
            lbl = src / "labels" / (img_p.stem + ".txt")
            for ln in (lbl.read_text(encoding="utf8").splitlines() if lbl.exists() else []):
                b = parse_line(ln)
                if b is None:
                    if ln.strip():
                        issues.append((src.name, img_p.name, f"bad line: {ln[:60]}"))
                    continue
                c, cx, cy, bw, bh = b
                raw = names[c] if c < len(names) else str(c)
                if raw not in cmap:
                    issues.append((src.name, img_p.name, f"unmapped class '{raw}'"))
                    continue
                if cmap[raw] is None:
                    continue
                x0, y0 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
                x1, y1 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
                if (x1 - x0) * w < 2 or (y1 - y0) * h < 2:
                    issues.append((src.name, img_p.name, "degenerate box dropped"))
                    continue
                key = (final.index(cmap[raw]), round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
                if key in seen:
                    issues.append((src.name, img_p.name, "duplicate box dropped"))
                    continue
                seen.add(key)
                lines.append(f"{key[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
            cv2.imwrite(str(OUT / "images" / (out_name + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 97]) \
                if img_p.suffix.lower() != ".jpg" else (OUT / "images" / (out_name + ".jpg")).write_bytes(img_p.read_bytes())
            (OUT / "labels" / (out_name + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
            counts = [sum(1 for l in lines if l.startswith(f"{i} ")) for i in range(len(final))]
            rows.append({"file": out_name + ".jpg", "source": src.name, "base": img_p.stem, "family": family(img_p.stem),
                         "width": w, "height": h, "boxes": len(lines),
                         **{f"n_{c}": n for c, n in zip(final, counts)},
                         "excluded": int(src.name in excl)})
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(OUT / "label_issues.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "file", "issue"])
        wr.writerows(issues)
    used = [r for r in rows if not r["excluded"]]
    print(f"collected {len(rows)} images ({len(used)} usable, {len(rows) - len(used)} from excluded sources), "
          f"{sum(r['boxes'] for r in used)} boxes; {len(issues)} label issues logged")


if __name__ == "__main__":
    main()
