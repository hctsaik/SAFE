"""
yolo_quickstart.py — 給 YOLO 使用者「最方便」的零手動策展路徑。

你已經有：YOLO 權重(.pt) + 你的標註資料(images + YOLO txt) + data.yaml(類別名)。
本工具不訓練、不動核心程式，自動完成：
  ① 把 config.yaml 的 yolo.weights 指向你的 .pt
  ② 依你的 txt 標註，把每類物件自動裁出來、用「和推論同一個 SAM」去背 → 存成 Golden 樣本
  ③ 把 config.yaml 的 classes 設為你的類別名
之後只要：  python src/build_bank.py   再   python src/pipeline.py --run

用法：
  python src/yolo_quickstart.py --data path/to/data.yaml --weights path/to/best.pt
  python src/yolo_quickstart.py --images imgs/ --labels labels/ --names a,b,c --weights best.pt
選項： --per-class N(每類取幾張,預設20)  --min-area-frac(濾掉太小的框)  --no-sam(不去背,較快)
       --train-dir / --config (測試用,改輸出位置)
"""
from __future__ import annotations
import sys, argparse, random
from pathlib import Path
import numpy as np
import cv2
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, ROOT, SamSegmenter

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_names(spec):
    """data.yaml 的 names 可能是 list 或 dict{0:'a'}；統一成 list。"""
    if isinstance(spec, dict):
        return [spec[k] for k in sorted(spec, key=lambda x: int(x))]
    return list(spec)


def gather_from_data_yaml(data_yaml):
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    names = parse_names(d["names"])
    base = Path(d.get("path") or Path(data_yaml).parent)
    if not base.is_absolute():
        base = (Path(data_yaml).parent / base).resolve()
    img_dirs = []
    for key in ("train", "val"):
        v = d.get(key)
        if not v:
            continue
        p = (base / v) if not Path(v).is_absolute() else Path(v)
        img_dirs.append(p)
    return names, img_dirs


def find_label(img_path: Path) -> Path | None:
    """YOLO 慣例：images/ 旁有平行的 labels/，同檔名 .txt；否則找同目錄 .txt。"""
    s = str(img_path)
    if "images" in s.replace("\\", "/"):
        cand = Path(s.replace("\\", "/").replace("/images/", "/labels/")).with_suffix(".txt")
        if cand.exists():
            return cand
    cand = img_path.with_suffix(".txt")
    return cand if cand.exists() else None


def iter_images(img_dirs):
    for d in img_dirs:
        d = Path(d)
        if d.is_file() and d.suffix.lower() in IMG_EXT:
            yield d
        elif d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in IMG_EXT:
                    yield p


def collect_boxes(img_dirs, n_classes):
    """回傳 {cls_id: [(img_path, (x1,y1,x2,y2), area_frac), ...]}。"""
    out = {c: [] for c in range(n_classes)}
    n_imgs = n_lbls = 0
    for img in iter_images(img_dirs):
        n_imgs += 1
        lbl = find_label(img)
        if not lbl:
            continue
        n_lbls += 1
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        for line in lbl.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c = int(float(t[0])); cx, cy, w, h = map(float, t[1:5])
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 4 or y2 - y1 < 4 or c not in out:
                continue
            out[c].append((img, (x1, y1, x2, y2), (w * h)))
    return out, n_imgs, n_lbls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="YOLO data.yaml（含 names 與 train/val 影像路徑）")
    ap.add_argument("--images", help="影像資料夾（不用 data.yaml 時）")
    ap.add_argument("--labels", help="YOLO txt 標註資料夾（預設自動找平行 labels/）")
    ap.add_argument("--names", help="類別名,逗號分隔（搭配 --images）")
    ap.add_argument("--weights", required=True, help="你的 YOLO .pt 路徑")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--min-area-frac", type=float, default=0.0005, help="濾掉小於此面積佔比的框")
    ap.add_argument("--no-sam", action="store_true", help="不去背(較快;但需推論也關SAM才一致)")
    ap.add_argument("--train-dir", default=str(ROOT / "Workspace" / "Training"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args()

    # 取得 names 與影像來源
    if a.data:
        names, img_dirs = gather_from_data_yaml(a.data)
    else:
        if not (a.images and a.names):
            ap.error("未用 --data 時，需提供 --images 與 --names")
        names = [s.strip() for s in a.names.split(",")]
        img_dirs = [Path(a.images)]
    print(f"[quickstart] 類別: {names}")
    print(f"[quickstart] 影像來源: {[str(p) for p in img_dirs]}")

    boxes, n_imgs, n_lbls = collect_boxes(img_dirs, len(names))
    print(f"[quickstart] 掃描 {n_imgs} 張影像、{n_lbls} 份標註")

    cfg = load_config()
    sam = None if a.no_sam else SamSegmenter(*[resolve_env(cfg)[k] for k in ("sam", "device")], cfg)
    train_dir = Path(a.train_dir)
    rng = random.Random(0)
    total = 0
    for c, nm in enumerate(names):
        items = [it for it in boxes[c] if it[2] >= a.min_area_frac]
        items.sort(key=lambda it: it[2], reverse=True)          # 大框優先(較清楚)
        items = items[:max(a.per_class * 3, a.per_class)]
        rng.shuffle(items); items = items[:a.per_class]
        d = train_dir / nm; d.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for img, box, _ in items:
            im = cv2.imread(str(img))
            if im is None:
                continue
            if sam is not None:
                crop, _ = sam.segment_crop(im, box)             # 與推論同一 SAM 去背
            else:
                x1, y1, x2, y2 = box; crop = im[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cv2.imwrite(str(d / f"{nm}_{cnt:03d}.png"), crop); cnt += 1
        print(f"[quickstart] {nm}: 取得 {cnt} 張 Golden -> {d}")
        total += cnt

    # 寫回 config：classes + yolo.weights
    c = yaml.safe_load(open(a.config, encoding="utf-8"))
    c["classes"] = names
    c.setdefault("yolo", {})["weights"] = str(Path(a.weights).resolve())
    yaml.safe_dump(c, open(a.config, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"[quickstart] 已更新 {a.config}: classes={names}, yolo.weights={a.weights}")
    print(f"\n[完成] 共 {total} 張 Golden。接著執行：")
    print("   python src/build_bank.py        # 建特徵庫(嵌入,非訓練)")
    print("   python src/pipeline.py --run    # 對 Workspace/Check/ 推論分發")


if __name__ == "__main__":
    main()
