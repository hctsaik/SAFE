"""
common.py — 共用核心：設定載入、環境自適配、DINO 嵌入器、SAM 去背器、IO 工具。
所有 Phase 腳本都從這裡取得模型與設定，確保 GPU/CPU 自動適配一致。
"""
from __future__ import annotations
import os, sys, random, warnings, functools
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_VERBOSE", "False")
# polars 自帶的 CPU 旗標檢查在此機器上有 bug（誤報 'sse3'）；CPU 實際支援，故略過檢查。
# ultralytics 訓練時會延遲 import polars 讀 results.csv，必須在其之前設定。
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------- 設定 / 環境 -----------------------------
def load_config(path: str | Path = None) -> dict:
    import yaml
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except Exception:
        pass


def resolve_env(cfg: dict) -> dict:
    """偵測 CUDA 並依設定挑選模型規格（GPU 大模型 / CPU 輕量）。"""
    import torch
    want = cfg.get("device", "auto")
    if want == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = want
    m = cfg["models"]
    sel = {
        "device": device,
        "yolo": m["yolo"],
        "sam": m["sam_gpu"] if device == "cuda" else m["sam_cpu"],
        "dino": m["dino_gpu"] if device == "cuda" else m["dino_cpu"],
    }
    return sel


def print_env(sel: dict):
    print(f"[env] device={sel['device']} | YOLO={sel['yolo']} | "
          f"SAM={sel['sam']} | DINO={sel['dino']}")


# ----------------------------- 影像 IO -----------------------------
def imread(path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img  # BGR


def imwrite(path, img):
    import cv2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1)); x2 = max(0, min(int(round(x2)), w))
    y1 = max(0, min(int(round(y1)), h - 1)); y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def classify_box(box, targets, oods, hi=0.5, lo=0.12):
    """把一個 YOLO 偵測框依與 GT 的 IoU 分類（評測/校準共用，定義一致）：
      target    : IoU>hi 命中真目標物件 -> 應通過(True)。回傳 (kind, 該目標類別)
      ood       : IoU>hi 命中 OOD 硬負樣本 -> 應攔截(False)
      bg        : 與所有物件 IoU<lo -> 純背景誤報 -> 應攔截(False)
      ambiguous : 部分重疊(lo~hi) -> YOLO 定位瑕疵，超出安全網職責 -> 評分時排除
    """
    bt, bcls = 0.0, None
    for b, c in targets:
        v = iou_xyxy(box, b)
        if v > bt: bt, bcls = v, c
    bo = max((iou_xyxy(box, b) for b in oods), default=0.0)
    if bt > hi: return ("target", bcls)
    if bo > hi: return ("ood", None)
    if bt < lo and bo < lo: return ("bg", None)
    return ("ambiguous", None)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1); area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


# ----------------------------- SAM 去背器 -----------------------------
class SamSegmenter:
    def __init__(self, weights: str, device: str, cfg: dict):
        from ultralytics import SAM
        self.model = SAM(weights)
        self.device = device
        self.bg_fill = cfg["sam"].get("bg_fill", "black")
        self.min_frac = cfg["sam"].get("min_mask_area_frac", 0.02)
        self.pad = cfg["sam"].get("pad", 0.04)

    def _fill_value(self, crop):
        if self.bg_fill == "white": return 255
        if self.bg_fill == "gray": return 114
        return 0  # black (default)

    def _apply_mask(self, img, box, m):
        """給定原圖、原始 box、整圖 mask(0/1)，回傳 (去背 tight crop, ok)。"""
        import cv2
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bx1, by1, bx2, by2 = clip_box(x1, y1, x2, y2, w, h)
        raw_crop = img[by1:by2, bx1:bx2].copy()
        if m is None:
            return raw_crop, False
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        box_area = max(1, (bx2 - bx1) * (by2 - by1))
        if int(m[by1:by2, bx1:bx2].sum()) < self.min_frac * box_area:
            return raw_crop, False  # fallback：中空/反光導致 mask 過小
        out = img.copy()
        if self.bg_fill == "blur":
            blurred = cv2.GaussianBlur(out, (0, 0), 8)
            out = np.where(m[..., None].astype(bool), out, blurred)
        else:
            out[m == 0] = self._fill_value(out)
        ys, xs = np.where(m[by1:by2, bx1:bx2] > 0)
        if len(xs) == 0:
            return raw_crop, False
        cx1, cy1 = bx1 + xs.min(), by1 + ys.min()
        cx2, cy2 = bx1 + xs.max() + 1, by1 + ys.max() + 1
        return out[cy1:cy2, cx1:cx2].copy(), True

    def _pad_box(self, box, w, h):
        x1, y1, x2, y2 = box
        pw, ph = (x2 - x1) * self.pad, (y2 - y1) * self.pad
        return clip_box(x1 - pw, y1 - ph, x2 + pw, y2 + ph, w, h)

    def segment_crops(self, img: np.ndarray, boxes: list) -> list:
        """批次：一次 SAM 編碼處理多個 box（CPU 大幅加速）。回傳 [(crop, ok), ...]。"""
        if not boxes:
            return []
        h, w = img.shape[:2]
        padded = [list(self._pad_box(b, w, h)) for b in boxes]
        masks = None
        try:
            res = self.model.predict(img, bboxes=padded, device=self.device,
                                     verbose=False)
            if res[0].masks is not None:
                masks = res[0].masks.data.cpu().numpy().astype(np.uint8)
        except Exception:
            masks = None
        out = []
        for i, b in enumerate(boxes):
            m = masks[i] if (masks is not None and i < len(masks)) else None
            out.append(self._apply_mask(img, b, m))
        return out

    def segment_crop(self, img: np.ndarray, box) -> tuple[np.ndarray, bool]:
        """單框版（內部呼叫批次版）。mask 空/過小 -> fallback 原始 bbox crop。"""
        return self.segment_crops(img, [box])[0]


# ----------------------------- DINO 嵌入器 -----------------------------
class DinoEmbedder:
    def __init__(self, name: str, device: str, hub_dir: str = None):
        import torch
        if hub_dir:
            torch.hub.set_dir(hub_dir)
        self.torch = torch
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        self.model.eval().to(device)
        self.dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                    "dinov2_vitl14": 1024}.get(name, 384)

    def _preprocess(self, crop_bgr: np.ndarray):
        import cv2
        if crop_bgr.size == 0:
            crop_bgr = np.zeros((14, 14, 3), np.uint8)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None]
        return self.torch.from_numpy(x).to(self.device)

    @functools.cached_property
    def _noop(self):
        return None

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(crop_bgr)
        with self.torch.no_grad():
            f = self.model(x)
        v = f[0].float().cpu().numpy()
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed(c) for c in crops]) if crops else \
            np.zeros((0, self.dim), np.float32)
