"""
build_bank.py — Phase 2b：讀取 Workspace/Training/[Class]/ 的 Golden Samples，
用 DINOv2 抽特徵，建立「標準特徵庫 (Vector Bank)」。
存：每張樣本向量(供 kNN) + 每類 prototype(均值向量)。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, DinoEmbedder, ROOT, imread, SamSegmenter


def _photo(img, gain=1.0, bias=0.0):
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def augment_views(crop_bgr, fill, n_rot=8, photometric=True):
    """對 Golden cut-out 產生「旋轉 × 光度」增強視角 -> 旋轉/光照不變的特徵庫。
    這讓嚴苛情境(反光/明暗/模糊/小物件)下劣化的目標仍能匹配到某個 golden 視角，
    大幅提升 retention，同時不會把語意不同的 OOD 拉近(interception 不受損)。"""
    h, w = crop_bgr.shape[:2]
    fv = (fill, fill, fill) if isinstance(fill, int) else fill
    rots = []
    for ang in np.linspace(0, 360, n_rot, endpoint=False):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang), 1.0)
        rots.append(cv2.warpAffine(crop_bgr, M, (w, h), borderValue=fv))
    if not photometric:
        return rots
    views = []
    for r in rots:
        views.append(r)                                  # 原始
        views.append(_photo(r, 0.6))                     # 偏暗
        views.append(_photo(r, 1.4))                     # 偏亮(近反光)
        views.append(_photo(r, 0.7, 40))                 # 低對比
        views.append(cv2.GaussianBlur(r, (0, 0), 1.6))   # 模糊(近小物件/低解析)
    return views

TRAIN_DIR = ROOT / "Workspace" / "Training"
BANK_PATH = ROOT / "Workspace" / "vector_bank.npz"
HUB = str(ROOT / ".cache" / "torchhub")


def build(cfg=None):
    cfg = cfg or load_config()
    sel = resolve_env(cfg)
    print(f"[bank] device={sel['device']} DINO={sel['dino']}")
    dino = DinoEmbedder(sel["dino"], sel["device"], hub_dir=HUB)
    use_sam = cfg["matching"].get("use_sam_for_bank", False)
    sam = SamSegmenter(sel["sam"], sel["device"], cfg) if use_sam else None
    do_aug = cfg["matching"].get("bank_augment", True)
    n_rot = int(cfg["matching"].get("aug_rotations", 8))
    photo = cfg["matching"].get("aug_photometric", True)
    fill = {"black": 0, "white": 255, "gray": 114}.get(cfg["sam"].get("bg_fill"), 0)

    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    vecs, labels = [], []
    for ci, cls in enumerate(classes):
        imgs = sorted(list((TRAIN_DIR / cls).glob("*.png")) +
                      list((TRAIN_DIR / cls).glob("*.jpg")))
        cnt = 0
        for p in imgs:
            img = imread(p)
            crop = img
            if sam is not None:
                h, w = img.shape[:2]
                crop, _ = sam.segment_crop(img, (0, 0, w, h))
            views = augment_views(crop, fill, n_rot, photo) if do_aug else [crop]
            for v in views:
                vecs.append(dino.embed(v)); labels.append(ci); cnt += 1
        print(f"[bank] {cls}: {cnt} vectors ({len(imgs)} golden x "
              f"{len(views) if do_aug else 1} views)")

    vecs = np.stack(vecs).astype(np.float32)
    labels = np.array(labels, np.int64)
    protos = np.stack([vecs[labels == i].mean(0) for i in range(len(classes))])
    protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9)
    np.savez(BANK_PATH, vecs=vecs, labels=labels,
             classes=np.array(classes), protos=protos.astype(np.float32),
             dim=vecs.shape[1])
    print(f"[bank] saved {BANK_PATH}  vecs={vecs.shape} classes={classes}")
    return BANK_PATH


if __name__ == "__main__":
    build()
