"""
refbank.py — 建立「多參考庫嵌入審查」用的輔助參考庫：Reject(reflection) 與 Normal。

安全網原本只有單一 Defect 庫（Workspace/vector_bank.npz）。多庫審查再加兩個：
  - Reject 庫 (reflection_bank)：已確認的誤報 crop（路面裂縫/反光/OOD）。靠近它 -> 攔截。
  - Normal 庫 (normal_bank)       ：乾淨背景 crop。靠近它 -> 視為背景/非目標。
有了這兩庫，match_audit 才能用「相對距離」分開細粒度誤報（坑洞 vs 裂縫、bubble vs reflection）。

來源（任選）：
  - 既有 crop 目錄（每張即一個 crop；遞迴）
  - hardneg 挖出的硬負樣本（Workspace/HardNegatives，當 Reject 來源）
  - 自 YOLO 切分自動抽「不重疊 GT 的背景 patch」（當 Normal 來源，零標註成本）

用法：
  python src/refbank.py --reject_from_hardneg                 # HardNegatives -> reject_bank.npz
  python src/refbank.py --reject_dir path/to/reflection_crops
  python src/refbank.py --normal_from Workspace/<name>/train --n 200   # 自抽背景 -> normal_bank.npz
  python src/refbank.py --normal_dir path/to/normal_crops
"""
from __future__ import annotations
import sys, argparse, random
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, imread, ROOT, iou_xyxy
from audit import get_dino, _imgs_in
from setup_build import read_yolo_txt
from yolo_quickstart import find_label, iter_images

WS = ROOT / "Workspace"
HARDNEG = WS / "HardNegatives"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def embed_crop_dir(crop_dir, dino, progress=None):
    """遞迴嵌入一個目錄下所有 crop 影像。回傳 (vecs, paths)。"""
    paths = sorted([p for p in Path(crop_dir).rglob("*") if p.suffix.lower() in IMG_EXT])
    vecs = []
    for i, p in enumerate(paths):
        try:
            vecs.append(dino.embed(imread(p)))
        except Exception:
            continue
        if progress:
            progress(i + 1, len(paths))
    return (np.stack(vecs).astype(np.float32) if vecs else
            np.zeros((0, dino.dim), np.float32)), paths


def sample_background(split_dir, dino, n=200, seed=0, max_per_img=4):
    """從 YOLO 切分自動抽「不與任何 GT 重疊」的背景 patch（patch 尺寸≈該圖 GT 中位邊長）。"""
    rng = random.Random(seed)
    imgs = list(iter_images([Path(split_dir) / "images"
                             if (Path(split_dir) / "images").is_dir() else Path(split_dir)]))
    rng.shuffle(imgs)
    vecs = []
    for ip in imgs:
        if len(vecs) >= n:
            break
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        lbl = find_label(ip)
        gt = [b for _, b, _ in read_yolo_txt(lbl, W, H)] if lbl else []
        side = int(np.median([min(b[2] - b[0], b[3] - b[1]) for b in gt])) if gt else int(0.12 * min(W, H))
        side = max(16, min(side, min(W, H) // 2))
        got = 0
        for _ in range(30):
            if got >= max_per_img or len(vecs) >= n:
                break
            x = rng.randint(0, max(1, W - side)); y = rng.randint(0, max(1, H - side))
            box = (x, y, x + side, y + side)
            if any(iou_xyxy(box, g) > 0.05 for g in gt):
                continue
            crop = im[y:y + side, x:x + side]
            if crop.size:
                vecs.append(dino.embed(crop)); got += 1
    return np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, dino.dim), np.float32)


def save_bank(vecs, path, kind):
    path = Path(path)
    np.savez(path, vecs=vecs)
    print(f"[refbank] {kind} 庫 -> {path}  ({len(vecs)} 向量)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reject_dir", default=None, help="reflection/誤報 crop 目錄")
    ap.add_argument("--reject_from_hardneg", action="store_true", help="用 Workspace/HardNegatives")
    ap.add_argument("--normal_dir", default=None, help="背景/正常 crop 目錄")
    ap.add_argument("--normal_from", default=None, help="自 YOLO 切分自動抽背景 patch")
    ap.add_argument("--n", type=int, default=200, help="背景自抽樣數量上限")
    ap.add_argument("--out_reject", default=str(WS / "reject_bank.npz"))
    ap.add_argument("--out_normal", default=str(WS / "normal_bank.npz"))
    a = ap.parse_args()
    if not (a.reject_dir or a.reject_from_hardneg or a.normal_dir or a.normal_from):
        ap.error("至少給一個來源：--reject_dir/--reject_from_hardneg/--normal_dir/--normal_from")
    dino, sel = get_dino(load_config())
    print(f"[refbank] device={sel['device']} DINO={sel['dino']}")

    rsrc = a.reject_dir or (str(HARDNEG) if a.reject_from_hardneg else None)
    if rsrc:
        if not Path(rsrc).exists():
            print(f"[refbank] reject 來源不存在: {rsrc}")
        else:
            vecs, _ = embed_crop_dir(rsrc, dino,
                progress=lambda i, n: print(f"[refbank] reject embed {i}/{n}") if i % 20 == 0 else None)
            save_bank(vecs, a.out_reject, "Reject(reflection)")

    if a.normal_dir:
        vecs, _ = embed_crop_dir(a.normal_dir, dino)
        save_bank(vecs, a.out_normal, "Normal")
    elif a.normal_from:
        vecs = sample_background(a.normal_from, dino, a.n)
        save_bank(vecs, a.out_normal, "Normal(自抽背景)")


if __name__ == "__main__":
    main()
