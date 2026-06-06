"""
setup_build.py — 「上傳資料 → 一鍵建置」後端（不依賴 Streamlit，可獨立 E2E 測試）。

run_build() 編排（全程零訓練）：
  1. 安全解壓 GT/Check ZIP（淨化路徑 + 大小上限）
  2. GT 影像「按影像切兩半」：半邊裁 Golden(同一 SAM 去背)、半邊當 hold-out 校準集
  3. build_bank 建特徵庫（嵌入，非訓練）
  4. 閾值：有權重→在 hold-out 跑 YOLO、IoU 標正/負(硬負樣本)→校準；無→預設
  5. 用「上傳的預測框」預算 Check 快取（不再跑 YOLO）
寫到 Workspace 標準路徑，既有 Inspector/Dashboard 直接沿用。
"""
from __future__ import annotations
import sys, zipfile, shutil, random
from pathlib import Path
import numpy as np
import cv2
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, ROOT, imread, iou_xyxy, SamSegmenter
from yolo_quickstart import find_label, iter_images, parse_names

WS = ROOT / "Workspace"
UP = WS / "uploads"
TRAIN_DIR = WS / "Training"
CHECK_DIR = WS / "Check"
CONFIG = ROOT / "config.yaml"
MAX_TOTAL = 4 * 1024**3   # 解壓總量上限 4GB（防 zip bomb）


# ----------------------------- 安全解壓 -----------------------------
def safe_extract(zip_path, dest: Path) -> Path:
    """淨化路徑(防 ../ 穿越) + 解壓總量上限。清空 dest 後解壓。"""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            # 淨化：丟掉絕對路徑與 .. 片段
            parts = [p for p in Path(info.filename.replace("\\", "/")).parts
                     if p not in ("", ".", "..") and ":" not in p]
            if not parts:
                continue
            total += info.file_size
            if total > MAX_TOTAL:
                raise ValueError("ZIP 解壓總量超過上限（疑似異常檔）")
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return dest


# ----------------------------- 類別名 / 標註 -----------------------------
def resolve_class_names(root: Path):
    """從 data.yaml / classes.txt 推類別名；找不到回傳 None。"""
    for y in list(Path(root).rglob("*.yaml")) + list(Path(root).rglob("*.yml")):
        try:
            d = yaml.safe_load(open(y, encoding="utf-8"))
            if isinstance(d, dict) and "names" in d:
                return parse_names(d["names"])
        except Exception:
            pass
    for t in Path(root).rglob("classes.txt"):
        names = [l.strip() for l in t.read_text(encoding="utf-8").splitlines() if l.strip()]
        if names:
            return names
    return None


def list_pairs(root: Path):
    """回傳 [(image_path, label_path_or_None)]。"""
    return [(img, find_label(img)) for img in iter_images([root])]


def class_ids_present(pairs):
    ids = set()
    for _, lbl in pairs:
        if not lbl:
            continue
        for line in Path(lbl).read_text().splitlines():
            t = line.split()
            if t:
                ids.add(int(float(t[0])))
    return sorted(ids)


def read_yolo_txt(txt, W, H):
    """解析 YOLO txt（容忍 cls cx cy w h [conf]）→ [(cls, (x1,y1,x2,y2), conf)]。"""
    rows = []
    for line in Path(txt).read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0])); cx, cy, w, h = map(float, t[1:5])
        conf = float(t[5]) if len(t) >= 6 else 1.0
        x1 = int((cx - w/2)*W); y1 = int((cy - h/2)*H)
        x2 = int((cx + w/2)*W); y2 = int((cy + h/2)*H)
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            rows.append((c, (x1, y1, x2, y2), conf))
    return rows


def _safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s)).strip("_") or "cls"


# ----------------------------- Golden 抽取 / 切分 -----------------------------
def split_images(pairs, seed=0):
    """只取有標註者，按『影像』切兩半（Golden / 校準）。資料太少時全給 Golden。"""
    labeled = [p for p in pairs if p[1]]
    rng = random.Random(seed); rng.shuffle(labeled)
    if len(labeled) < 4:
        return labeled, []
    h = len(labeled) // 2
    return labeled[:h], labeled[h:]


def extract_golden(pairs, names, sam, per_class=20, min_area_frac=0.0005, progress=None):
    """依 GT 框裁每類物件 → 同一 SAM 去背 → Workspace/Training/<類別>/。回傳每類張數。"""
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    buckets = {i: [] for i in range(len(names))}
    for img, lbl in pairs:
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        for c, box, _ in read_yolo_txt(lbl, W, H):
            if 0 <= c < len(names) and (box[2]-box[0])*(box[3]-box[1]) >= min_area_frac*W*H:
                buckets[c].append((str(img), box))
    counts = {}
    rng = random.Random(0)
    for i, nm in enumerate(names):
        items = buckets[i]; items.sort(key=lambda it: -(it[1][2]-it[1][0])*(it[1][3]-it[1][1]))
        items = items[:max(per_class*3, per_class)]; rng.shuffle(items); items = items[:per_class]
        d = TRAIN_DIR / _safe_name(nm); d.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for img, box in items:
            im = cv2.imread(img)
            if im is None:
                continue
            crop, _ = sam.segment_crop(im, box) if sam else (im[box[1]:box[3], box[0]:box[2]], False)
            if crop.size == 0:
                continue
            cv2.imwrite(str(d / f"{_safe_name(nm)}_{cnt:03d}.png"), crop); cnt += 1
        counts[nm] = cnt
        if progress:
            progress("golden", i+1, len(names), f"{nm}: {cnt}")
    return counts


# ----------------------------- 閾值校準 (hold-out) -----------------------------
def calibrate_from_gt(net, calib_pairs, progress=None):
    """在 hold-out GT 影像跑 YOLO；預測框 IoU>0.5 命中任一 GT=正、否則=負(誤報)。
    掃 cosine 閾值取最大化 F1。資料不足回傳 None。"""
    scores, pos = [], []
    for k, (img, lbl) in enumerate(calib_pairs):
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        gt = [b for _, b, _ in read_yolo_txt(lbl, W, H)]
        dets = net.detect(im)
        crops = net.sam.segment_crops(im, [d[0] for d in dets])
        for (xyxy, _, _ycls), (crop, _) in zip(dets, crops):
            if crop.size == 0:
                continue
            _, s, _ = net.match(net.dino.embed(crop))
            scores.append(s); pos.append(any(iou_xyxy(xyxy, g) > 0.5 for g in gt))
        if progress:
            progress("calib", k+1, len(calib_pairs), Path(img).name)
    scores = np.array(scores); pos = np.array(pos, bool)
    if len(scores) < 6 or pos.sum() < 2 or (~pos).sum() < 2:
        return None
    best_t, best_f1 = None, -1
    for t in np.linspace(scores.min(), scores.max(), 80):
        pr = scores >= t
        tp = int((pr & pos).sum()); fp = int((pr & ~pos).sum()); fn = int((~pr & pos).sum())
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); f1 = 2*p*r/(p+r+1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ----------------------------- Check 快取（用上傳的框）-----------------------------
def build_cache_from_boxes(net, check_pairs, progress=None):
    """用『上傳的 YOLO 預測框』(非跑 YOLO) 預算 Check 快取，存到 GUI_CACHE。"""
    from pipeline import _golden_gallery, GUI_CACHE
    import pickle
    gp, gc, gv = _golden_gallery(net)
    images = {}
    for k, (img, lbl) in enumerate(check_pairs):
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = [b for _, b, _ in read_yolo_txt(lbl, W, H)] if lbl else []
        confs = [c for _, _, c in read_yolo_txt(lbl, W, H)] if lbl else []
        recs = []
        crops = net.sam.segment_crops(im, boxes)
        for (box, (crop, ok)), conf in zip(zip(boxes, crops), confs):
            if crop.size == 0:
                continue
            vec = net.dino.embed(crop); cls, score, score2 = net.match(vec)
            golden = []
            if len(gv):
                sims = gv @ vec
                golden = [(gp[i], gc[i], float(sims[i])) for i in np.argsort(sims)[::-1][:3]]
            x1, y1, x2, y2 = box
            recs.append(dict(box=[float(v) for v in box], conf=float(conf), score=float(score),
                             score2=float(score2), margin=float(score - score2) if score2 >= 0 else 1.0,
                             pred_class=cls, sam_ok=bool(ok), vec=vec.astype(np.float32),
                             raw=im[y1:y2, x1:x2].copy(), sam=crop, golden=golden))
        images[Path(img).name] = recs
        if progress:
            progress("cache", k+1, len(check_pairs), Path(img).name)
    cache = dict(classes=net.classes, threshold=net.threshold, dino=net.sel["dino"],
                 device=net.sel["device"], images=images, source="upload")
    GUI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, open(GUI_CACHE, "wb"))
    return cache


# ----------------------------- 編排 -----------------------------
def write_config(names, dino_variant, weights, threshold):
    c = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    c["classes"] = list(names)
    c.setdefault("models", {})["dino_cpu"] = dino_variant
    c["models"]["dino_gpu"] = dino_variant
    c.setdefault("yolo", {})["weights"] = (str(Path(weights).resolve()) if weights else None)
    c.setdefault("matching", {})["threshold"] = round(float(threshold), 4)
    yaml.safe_dump(c, open(CONFIG, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return c


def run_build(gt_dir, check_dir, names, weights=None, dino_variant="dinov2_vits14",
              per_class=20, use_sam=True, progress=None):
    """一鍵建置主流程。回傳 summary。progress(stage, i, n, msg)。"""
    def pg(*a):
        if progress:
            progress(*a)
    names = [str(n) for n in names]
    # 預先把 dino 寫進 config，讓 build_bank / SafetyNet 用對的規格
    write_config(names, dino_variant, weights, threshold=load_config()["matching"]["threshold"])
    cfg = load_config(); sel = resolve_env(cfg)
    sam = SamSegmenter(sel["sam"], sel["device"], cfg) if use_sam else None

    pg("split", 0, 1, "切分 GT 影像")
    gt_pairs = list_pairs(Path(gt_dir))
    golden_half, calib_half = split_images(gt_pairs)

    pg("golden", 0, len(names), "裁切 Golden")
    counts = extract_golden(golden_half or gt_pairs, names, sam, per_class, progress=progress)
    if sum(counts.values()) == 0:
        raise ValueError("沒有取得任何 Golden（檢查標註與類別對應）")

    pg("bank", 0, 1, "建特徵庫（嵌入，非訓練）")
    import build_bank
    build_bank.build(cfg)

    from pipeline import SafetyNet
    net = SafetyNet(cfg, weights)

    threshold = net.threshold; calibrated = False
    if weights and calib_half:
        pg("calib", 0, len(calib_half), "自動校準閾值")
        t = calibrate_from_gt(net, calib_half, progress=progress)
        if t is not None:
            threshold = t; net.threshold = t; calibrated = True
    write_config(names, dino_variant, weights, threshold)

    # 影像落地：Check 影像複製到 Workspace/Check（既有頁用 imread(CHECK_DIR/name)）
    if CHECK_DIR.exists():
        shutil.rmtree(CHECK_DIR)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    check_pairs = list_pairs(Path(check_dir))
    for img, lbl in check_pairs:
        shutil.copy(img, CHECK_DIR / Path(img).name)
    check_pairs = [(CHECK_DIR / Path(i).name, l) for i, l in check_pairs]

    pg("cache", 0, len(check_pairs), "預算 Check 快取（用上傳的框）")
    cache = build_cache_from_boxes(net, check_pairs, progress=progress)

    n_box = sum(len(v) for v in cache["images"].values())
    summary = dict(classes=names, golden=counts, threshold=round(float(threshold), 4),
                   calibrated=calibrated, dino=dino_variant, n_check_imgs=len(check_pairs),
                   n_check_boxes=n_box, weights=bool(weights))
    pg("done", 1, 1, "完成")
    return summary
