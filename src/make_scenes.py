"""
make_scenes.py — Phase 1b：合成場景產生器 (Composite Scene Generator)。

流程：
 1) extract_cutouts: 用 SAM 把 raw 物件圖去背成 RGBA cut-out（含 Otsu fallback），快取。
 2) build_golden:     從 cut-out 取每類 N 張 -> Workspace/Training/[Class]/ (Golden Samples)。
 3) make_split:       把物件 cut-out 貼到紋理背景大圖，撒入紋理/反光誘餌，
                      產生 Check/ 大圖 + YOLO 標註(單類 object) + 評測用 GT(含語意類別)。

設計重點：物件=合法目標(有 GT)；紋理/反光誘餌=False Alarm 來源(無物件 GT)。
場景由我們合成 => 擁有精確 bbox GT => 可算真實 Precision/Recall。
"""
from __future__ import annotations
import sys, json, random, math
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (load_config, set_seed, ROOT, SamSegmenter, resolve_env,
                    clip_box, iou_xyxy)

RAW = ROOT / "Workspace" / "raw"
CUTOUT = ROOT / "Workspace" / "cutouts"
CUTOUT_OOD = ROOT / "Workspace" / "cutouts_ood"
TRAIN_DIR = ROOT / "Workspace" / "Training"
CHECK_DIR = ROOT / "Workspace" / "Check"
SCENES = ROOT / "Workspace" / "scenes"


# ----------------------------- cut-out 抽取 -----------------------------
def _otsu_mask(gray):
    """fallback：Otsu 分割 + 取最大連通元件。自動判斷物件較亮或較暗。"""
    _, m1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 物件通常佔較小面積；選前景佔比較合理(10~70%)的極性
    cand = []
    for m in (m1, 255 - m1):
        frac = m.mean() / 255.0
        cand.append((abs(frac - 0.3), m))
    m = min(cand, key=lambda t: t[0])[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == idx).astype(np.uint8) * 255)


def _extract(classes, src_sub, dst_root, sam, per_class, overwrite=False):
    out = {}
    for cls in classes:
        dst = dst_root / cls
        dst.mkdir(parents=True, exist_ok=True)
        srcs = sorted((RAW / src_sub / cls).glob("*.png"))[:per_class]
        paths = []
        for i, sp in enumerate(srcs):
            op = dst / f"{cls}_{i:03d}.png"
            if op.exists() and not overwrite:
                paths.append(op); continue
            img = cv2.imread(str(sp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            # 物件置中 -> 用中央 ~62% 框當 SAM prompt
            cx1, cy1, cx2, cy2 = w*0.19, h*0.19, w*0.81, h*0.81
            mask = None
            try:
                res = sam.model.predict(img, bboxes=[[cx1, cy1, cx2, cy2]],
                                        device=sam.device, verbose=False)
                if res[0].masks is not None and len(res[0].masks.data):
                    m = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                    frac = m.mean()
                    if 0.03 < frac < 0.85:
                        mask = (m * 255).astype(np.uint8)
            except Exception:
                mask = None
            if mask is None:
                mask = _otsu_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # 形態學清理
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            ys, xs = np.where(mask > 0)
            if len(xs) < 50:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
            rgba = np.dstack([img, mask])[y1:y2, x1:x2]
            cv2.imwrite(str(op), rgba)
            paths.append(op)
        out[cls] = paths
        print(f"[cutout] {src_sub}/{cls}: {len(paths)} cut-outs")
    return out


def extract_cutouts(cfg, sam, per_class=40, overwrite=False):
    return _extract(cfg["classes"], "objects", CUTOUT, sam, per_class, overwrite)


def extract_ood(cfg, sam, per_class=20, overwrite=False):
    return _extract(cfg.get("ood_classes", []), "ood", CUTOUT_OOD, sam,
                    per_class, overwrite)


# ----------------------------- Golden Samples -----------------------------
def build_golden(cfg, cutouts, bg_fill="black"):
    n = cfg["data"]["golden_per_class"]
    fill = {"black": 0, "white": 255, "gray": 114}.get(bg_fill, 0)
    for cls, paths in cutouts.items():
        d = TRAIN_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        for p in paths[:n]:
            rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                continue
            bgr, a = rgba[..., :3], rgba[..., 3]
            comp = bgr.copy(); comp[a == 0] = fill
            cv2.imwrite(str(d / p.name), comp)
        print(f"[golden] {cls}: {min(n, len(paths))} samples -> {d}")


# ----------------------------- 場景合成工具 -----------------------------
def load_pools(cfg):
    """從快取 cutouts 目錄重建 golden / pool 不相交切分 + OOD 池（供評測重現場景）。"""
    G, P = cfg["data"]["golden_per_class"], cfg["data"]["pool_per_class"]
    golden, pool = {}, {}
    for c in cfg["classes"]:
        ps = sorted((CUTOUT / c).glob("*.png"))
        golden[c] = ps[:G]; pool[c] = ps[G:G + P]
    ood = {c: sorted((CUTOUT_OOD / c).glob("*.png")) for c in cfg.get("ood_classes", [])}
    return golden, pool, ood


def _load_textures(cfg):
    tex = {}
    for t in cfg["textures"]:
        imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR)
                for p in sorted((RAW / "textures" / t).glob("*.png"))]
        tex[t] = [i for i in imgs if i is not None]
    return tex


def _bg_from_texture(tex, size, rng):
    t = rng.choice(list(tex.keys()))
    img = rng.choice(tex[t])
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5: img = cv2.flip(img, 1)
    if rng.random() < 0.5: img = cv2.flip(img, 0)
    return img, t


def _rotate_rgba(rgba, angle):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h*sin + w*cos), int(h*cos + w*sin)
    M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def _paste(bg, rgba, x, y, alpha_scale=1.0):
    H, W = bg.shape[:2]; h, w = rgba.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x+w), min(H, y+h)
    if x2 <= x1 or y2 <= y1:
        return bg, None
    sx1, sy1 = x1-x, y1-y; sx2, sy2 = sx1+(x2-x1), sy1+(y2-y1)
    patch = rgba[sy1:sy2, sx1:sx2]
    a = (patch[..., 3:4].astype(np.float32)/255.0) * alpha_scale
    bg[y1:y2, x1:x2] = (patch[..., :3]*a + bg[y1:y2, x1:x2]*(1-a)).astype(np.uint8)
    return bg, (x1, y1, x2, y2)


def _color_jitter(bgr, rng, strength=0.3):
    b = 1 + rng.uniform(-strength, strength)
    c = 1 + rng.uniform(-strength, strength)
    out = bgr.astype(np.float32)*c + (b-1)*60
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_specular(bg, box, rng, intensity=0.8):
    """在指定框內加入高亮反光（白色高斯橢圓）-> 模擬金屬眩光誘餌。
    強度上限 0.5：真實反光是局部高光而非把物件整片洗白（避免不合理地完全抹除目標）。"""
    intensity = min(intensity, 0.5)
    x1, y1, x2, y2 = box; cx, cy = (x1+x2)//2, (y1+y2)//2
    ax, ay = max(4, (x2-x1)//2), max(4, (y2-y1)//2)
    overlay = bg.copy()
    cv2.ellipse(overlay, (cx, cy), (ax, ay), rng.uniform(0, 180), 0, 360,
                (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), max(3, ax//2))
    cv2.addWeighted(overlay, intensity, bg, 1-intensity, 0, bg)


# ----------------------------- 單一場景 -----------------------------
def make_one(cfg, cutouts, tex, rng, knobs, ood_pool=None):
    S = cfg["data"]["scene_size"]
    bg, bg_tex = _bg_from_texture(tex, S, rng)
    objs, distractors, ood_objs = [], [], []
    placed_boxes = []
    scale_lo, scale_hi = knobs["scale_range"]

    def overlaps(b, thr):
        return any(iou_xyxy(b, pb) > thr for pb in placed_boxes)

    def place(rgba):
        """縮放/旋轉/抖動 + 找不重疊位置 + alpha 貼上，回傳框或 None。"""
        nonlocal bg
        target = int(S * rng.uniform(scale_lo, scale_hi))
        h, w = rgba.shape[:2]; sc = target/max(h, w)
        r = cv2.resize(rgba, (max(8, int(w*sc)), max(8, int(h*sc))))
        r = _rotate_rgba(r, rng.uniform(0, 360))
        r[..., :3] = _color_jitter(r[..., :3], rng, knobs["color_jitter"])
        ow, oh = r.shape[1], r.shape[0]
        x = y = 0
        for _try in range(12):
            if knobs.get("out_of_frame"):  # 最多 ~30% 出框（邊緣截斷但主體仍可見，公平）
                mx, my = int(ow * 0.3), int(oh * 0.3)
                x = rng.randint(-mx, max(1, S - ow + mx))
                y = rng.randint(-my, max(1, S - oh + my))
            else:
                x = rng.randint(0, max(1, S - ow)); y = rng.randint(0, max(1, S - oh))
            if not overlaps((x, y, x+ow, y+oh), knobs["max_overlap"]):
                break
        bg, box = _paste(bg, r, x, y, alpha_scale=knobs.get("obj_alpha", 1.0))
        if box is None or (box[2]-box[0]) < 8 or (box[3]-box[1]) < 8:
            return None
        placed_boxes.append(box)
        return list(box)

    def _rng_count(spec):
        return rng.randint(*spec) if isinstance(spec, (list, tuple)) else int(spec)

    # --- 貼真實目標物件（有 GT，應通過）---
    for _ in range(rng.randint(*cfg["data"]["objects_per_scene"])):
        cls = rng.choice(cfg["classes"])
        if not cutouts.get(cls):
            continue
        rgba = cv2.imread(str(rng.choice(cutouts[cls])), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        box = place(rgba)
        if box:
            objs.append({"cls": cls, "box": box})

    # --- 貼 OOD 硬負樣本物件（YOLO 會誤觸發，DINO 應攔截）---
    n_ood = _rng_count(knobs.get("n_ood", 0))
    ood_classes = [c for c in (ood_pool or {}) if ood_pool[c]]
    for _ in range(n_ood if ood_classes else 0):
        oc = rng.choice(ood_classes)
        rgba = cv2.imread(str(rng.choice(ood_pool[oc])), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        box = place(rgba)
        if box:
            ood_objs.append({"cls": oc, "box": box})

    # --- 撒入誘餌（紋理斑塊 / 反光）-> 純背景 False Alarm 來源 ---
    nd = rng.randint(*cfg["data"]["distractors_per_scene"])
    for _ in range(nd):
        dsize = int(S * rng.uniform(0.05, 0.16))
        x = rng.randint(0, max(1, S-dsize)); y = rng.randint(0, max(1, S-dsize))
        box = (x, y, x+dsize, y+dsize)
        # 偏好擬真誘餌（紋理斑塊 / 反光），少量彩色斑塊
        kind = rng.choice(["texture_blob", "specular", "texture_blob",
                           "specular", "color_blob"])
        if kind == "specular" or (knobs.get("force_specular") and rng.random() < 0.5):
            _add_specular(bg, box, rng, intensity=knobs.get("specular", 0.85))
        elif kind == "texture_blob":
            # 從另一張紋理切一塊不規則斑塊貼上
            ot, oimg = _bg_from_texture(tex, dsize, rng)
            mask = np.zeros((dsize, dsize), np.uint8)
            cv2.circle(mask, (dsize//2, dsize//2), dsize//2, 255, -1)
            rgba = np.dstack([ot, mask])
            bg, _ = _paste(bg, rgba, x, y)
        else:
            col = tuple(rng.randint(0, 255) for _ in range(3))
            cv2.circle(bg, (x+dsize//2, y+dsize//2), dsize//2, col, -1)
        distractors.append({"kind": kind, "box": list(box)})

    # --- 全域光影 ---
    if knobs.get("global_light", 0):
        g = knobs["global_light"]
        bg = np.clip(bg.astype(np.float32)*rng.uniform(1-g, 1+g) +
                     rng.uniform(-g, g)*50, 0, 255).astype(np.uint8)
    return bg, objs, distractors, ood_objs, bg_tex


DEFAULT_KNOBS = dict(scale_range=(0.08, 0.20), color_jitter=0.25,
                     max_overlap=0.05, global_light=0.15, specular=0.85,
                     obj_alpha=1.0, force_specular=False, n_ood=0)


def make_split(cfg, cutouts, tex, split, n, seed, knobs=None,
               write_yolo=True, dst_images=None, ood_pool=None):
    rng = random.Random(seed)
    # knobs 可為 dict(固定) 或 list(每張場景隨機取一種難度 -> 代表性校準集)
    (SCENES/split).mkdir(parents=True, exist_ok=True)  # 確保 gt.json 目錄存在
    img_dir = Path(dst_images) if dst_images else (SCENES/split/"images")
    lbl_dir = SCENES/split/"labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    if write_yolo: lbl_dir.mkdir(parents=True, exist_ok=True)
    gt = {"split": split, "scene_size": cfg["data"]["scene_size"], "scenes": []}
    for i in range(n):
        r = random.Random(seed*10000 + i)
        sel = r.choice(knobs) if isinstance(knobs, list) else (knobs or {})
        k = dict(DEFAULT_KNOBS); k.update(sel)
        bg, objs, dist, ood_objs, bgtex = make_one(cfg, cutouts, tex, r, k, ood_pool)
        name = f"{split}_{i:04d}.jpg"
        cv2.imwrite(str(img_dir/name), bg)
        if write_yolo:
            S = bg.shape[0]
            lines = []
            for o in objs:
                x1, y1, x2, y2 = o["box"]
                cxn = ((x1+x2)/2)/S; cyn = ((y1+y2)/2)/S
                wn = (x2-x1)/S; hn = (y2-y1)/S
                lines.append(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")
            (lbl_dir/f"{split}_{i:04d}.txt").write_text("\n".join(lines))
        gt["scenes"].append({"image": name, "bg_texture": bgtex,
                             "objects": objs, "ood": ood_objs, "distractors": dist})
    (SCENES/split/"gt.json").write_text(json.dumps(gt, indent=1))
    print(f"[scenes] {split}: {n} images -> {img_dir}  (gt.json written)")
    return gt


# ----------------------------- main -----------------------------
def main():
    cfg = load_config(); set_seed(cfg["seed"])
    sel = resolve_env(cfg)
    print(f"[scenes] device={sel['device']} SAM={sel['sam']}")
    sam = SamSegmenter(sel["sam"], sel["device"], cfg)
    d = cfg["data"]
    G, P = d["golden_per_class"], d["pool_per_class"]
    all_cut = extract_cutouts(cfg, sam, per_class=G + P + 4)
    ood_pool = extract_ood(cfg, sam, per_class=20)
    # Golden(建向量庫) 與 場景貼圖池 使用「不相交」的 cut-out 實例 -> 評測誠實，非記憶
    golden = {c: all_cut[c][:G] for c in all_cut}
    pool = {c: all_cut[c][G:G + P] for c in all_cut}
    build_golden(cfg, golden, bg_fill=cfg["sam"]["bg_fill"])
    tex = _load_textures(cfg)
    no = tuple(d.get("ood_per_scene", [1, 3]))
    ood_knob = {"n_ood": no}
    # 代表性校準集：混合各種難度 -> 校準閾值能轉移到嚴苛情境（解決「校準集太簡單」）
    calib_mix = [
        {"n_ood": no},
        {"n_ood": no, "force_specular": True, "specular": 0.5},
        {"n_ood": no, "global_light": 0.45, "color_jitter": 0.45},
        {"n_ood": no, "scale_range": (0.06, 0.10)},
        {"n_ood": no, "max_overlap": 0.55, "scale_range": (0.10, 0.22)},
        {"n_ood": no, "color_jitter": 0.05, "obj_alpha": 0.9},
    ]
    # 訓練場景：純目標(不含 OOD)，與既有 YOLO 權重一致
    make_split(cfg, pool, tex, "train", d["n_train_scenes"], seed=100)
    # 校準場景：混合難度 + OOD 硬負樣本
    make_split(cfg, pool, tex, "calib", d["n_calib_scenes"], seed=200,
               knobs=calib_mix, ood_pool=ood_pool)
    # Check 場景：含 OOD 硬負樣本（YOLO 會誤觸發 -> DINO 攔截）
    make_split(cfg, pool, tex, "check", d["n_check_scenes"], seed=300,
               knobs=ood_knob, ood_pool=ood_pool, write_yolo=False, dst_images=CHECK_DIR)
    print("[scenes] DONE.")


if __name__ == "__main__":
    main()
