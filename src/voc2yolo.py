"""
voc2yolo.py — 通用 Pascal VOC(XML) -> YOLO(txt) 轉換器 + train/val/holdout 切分。

把外部 VOC 偵測資料集接進安全網/飛輪：解析 <object><name>/<bndbox> -> 正規化 YOLO 標註，
依比例切 train/val/holdout（holdout 當飛輪/eval_real 的真實裁判，與訓練不重疊）。

輸出：
  out/data.yaml                    # ultralytics 訓練用（train/val）
  out/classes.txt                  # 類別名（eval_real/distill 自動讀）
  out/{train,val,holdout}/images   # 影像
  out/{train,val,holdout}/labels   # YOLO txt（cls cx cy w h 正規化）

用法：
  python src/voc2yolo.py --voc DIR --out Workspace/pothole --train 300 --val 80 --holdout 150
  python src/voc2yolo.py --voc DIR --out OUT --images_sub images --anno_sub annotations
"""
from __future__ import annotations
import sys, shutil, random, argparse
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def find_pairs(voc, images_sub, anno_sub):
    """配對 (影像, xml)。以 annotations 為主，依檔名（無副檔名）找對應影像。"""
    img_dir = Path(voc) / images_sub; anno_dir = Path(voc) / anno_sub
    imgs = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT}
    pairs = []
    for x in sorted(anno_dir.glob("*.xml")):
        ip = imgs.get(x.stem)
        if ip is not None:
            pairs.append((ip, x))
    return pairs


def parse_voc(xml_path):
    """回傳 (W, H, [(name, x1,y1,x2,y2), ...])。"""
    r = ET.parse(xml_path).getroot()
    sz = r.find("size")
    W = int(float(sz.find("width").text)); H = int(float(sz.find("height").text))
    objs = []
    for o in r.findall("object"):
        nm = o.find("name").text.strip()
        b = o.find("bndbox")
        x1 = float(b.find("xmin").text); y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text); y2 = float(b.find("ymax").text)
        objs.append((nm, x1, y1, x2, y2))
    return W, H, objs


def collect_classes(pairs):
    names = set()
    for _, x in pairs:
        for nm, *_ in parse_voc(x)[2]:
            names.add(nm)
    return sorted(names)


def write_split(pairs, names, out_split):
    cls_idx = {n: i for i, n in enumerate(names)}
    img_o = out_split / "images"; lbl_o = out_split / "labels"
    img_o.mkdir(parents=True, exist_ok=True); lbl_o.mkdir(parents=True, exist_ok=True)
    n_box = 0
    for ip, xp in pairs:
        W, H, objs = parse_voc(xp)
        lines = []
        for nm, x1, y1, x2, y2 in objs:
            if nm not in cls_idx or x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2) / W; cy = ((y1 + y2) / 2) / H
            lines.append(f"{cls_idx[nm]} {cx:.6f} {cy:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            n_box += 1
        shutil.copy(ip, img_o / ip.name)
        (lbl_o / f"{ip.stem}.txt").write_text("\n".join(lines))
    return len(pairs), n_box


def convert(voc, out, images_sub="images", anno_sub="annotations",
            train=300, val=80, holdout=150, seed=42):
    """VOC -> YOLO 切分（可被 onboard 呼叫）。回傳 (out_dir, names, stats{split:(imgs,boxes)})。"""
    pairs = find_pairs(voc, images_sub, anno_sub)
    if not pairs:
        raise FileNotFoundError(f"找不到 (影像,xml) 配對於 {voc}")
    names = collect_classes(pairs)
    rng = random.Random(seed); rng.shuffle(pairs)
    ho = pairs[:holdout]
    va = pairs[holdout:holdout + val]
    rest = pairs[holdout + val:]
    tr = rest[:train] if train else rest
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, sp in [("train", tr), ("val", va), ("holdout", ho)]:
        stats[split] = write_split(sp, names, out / split)
        (out / split / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return out, names, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", required=True, help="VOC 根目錄（含 images/ 與 annotations/）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images_sub", default="images")
    ap.add_argument("--anno_sub", default="annotations")
    ap.add_argument("--train", type=int, default=300, help="train 上限（0=用剩餘全部）")
    ap.add_argument("--val", type=int, default=80)
    ap.add_argument("--holdout", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    try:
        out, names, stats = convert(a.voc, a.out, a.images_sub, a.anno_sub,
                                    a.train, a.val, a.holdout, a.seed)
    except FileNotFoundError as e:
        print(f"[voc2yolo] {e}"); sys.exit(2)
    print(f"[voc2yolo] 類別 {names}")
    for s, (ni, nb) in stats.items():
        print(f"   {s:>8}: {ni} 圖 / {nb} 框")
    print(f"[voc2yolo] -> {out}  (data.yaml, classes.txt, train/val/holdout)")


if __name__ == "__main__":
    main()
