"""
pipeline.py — Phase 2c：The Safety Net 推論核心。
YOLO(高靈敏觸發) -> SAM(去背) -> DINOv2(特徵) -> Vector Matching(裁決) -> 分發。

提供：
  SafetyNet 類別            : 封裝四級管線
  run_on_dir()             : 批次推論 Check/ 並分發到 Result/{True,False,No_Detection}
  calibrate()              : 用 calib 場景的 GT 自動校準 cosine 閾值(最大化 F1)
輸出模式 output_mode:
  cropped_roi (預設)        : 存 SAM 去背小圖（供 Hard Negative 再訓練）
  annotated_full           : 存畫框大圖（綠=通過 / 紅=誤報）
"""
from __future__ import annotations
import sys, json, shutil, argparse
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (load_config, resolve_env, print_env, DinoEmbedder,
                    SamSegmenter, ROOT, imread, imwrite, clip_box, iou_xyxy,
                    classify_box)

WS = ROOT / "Workspace"
BANK_PATH = WS / "vector_bank.npz"
HUB = str(ROOT / ".cache" / "torchhub")
DEFAULT_YOLO = WS / "runs" / "yolo" / "weights" / "best.pt"


class SafetyNet:
    def __init__(self, cfg, yolo_weights=None):
        from ultralytics import YOLO
        self.cfg = cfg
        self.sel = resolve_env(cfg); print_env(self.sel)
        self.yolo = YOLO(str(yolo_weights or DEFAULT_YOLO))
        self.sam = SamSegmenter(self.sel["sam"], self.sel["device"], cfg)
        self.dino = DinoEmbedder(self.sel["dino"], self.sel["device"], hub_dir=HUB)
        b = np.load(BANK_PATH, allow_pickle=True)
        self.bank_vecs = b["vecs"]; self.bank_labels = b["labels"]
        self.classes = [str(c) for c in b["classes"]]; self.protos = b["protos"]
        self.threshold = float(cfg["matching"]["threshold"])
        self.topk = int(cfg["matching"]["topk"])
        self.agg = cfg["matching"]["agg"]
        self.conf = float(cfg["yolo"]["conf"])

    # ---- YOLO 觸發 ----
    def detect(self, img):
        r = self.yolo.predict(img, conf=self.conf, iou=0.5,
                              device=self.sel["device"], verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                out.append((xyxy, float(b.conf[0])))
        return out

    # ---- DINO 比對裁決 ----
    def match(self, vec):
        sims = self.bank_vecs @ vec  # cosine（皆已 L2 normalize）
        best_cls, best_score = -1, -1.0
        for ci in range(len(self.classes)):
            cl = sims[self.bank_labels == ci]
            if len(cl) == 0:
                continue
            if self.agg == "prototype":
                s = float(self.protos[ci] @ vec)
            else:  # knn: 前 k 高相似度平均
                k = min(self.topk, len(cl))
                s = float(np.sort(cl)[-k:].mean())
            if s > best_score:
                best_score, best_cls = s, ci
        return self.classes[best_cls], best_score

    # ---- 單圖完整管線 ----
    def process(self, img, thr=None):
        thr = self.threshold if thr is None else thr
        dets = self.detect(img)
        crops = self.sam.segment_crops(img, [d[0] for d in dets])  # 批次去背
        recs = []
        for (xyxy, conf), (crop, sam_ok) in zip(dets, crops):
            if crop.size == 0:
                continue
            vec = self.dino.embed(crop)
            cls, score = self.match(vec)
            decision = "True" if score >= thr else "False"
            recs.append(dict(box=[float(v) for v in xyxy], conf=conf,
                             pred_class=cls, score=score, sam_ok=sam_ok,
                             decision=decision, crop=crop))
        return recs


# ----------------------------- 分發 / 輸出 -----------------------------
def _annotate(img, recs):
    out = img.copy()
    for r in recs:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        green = (0, 200, 0); red = (0, 0, 255)
        col = green if r["decision"] == "True" else red
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
        tag = f'{r["pred_class"]} {r["score"]:.2f}' if r["decision"] == "True" \
            else f'FALSE {r["score"]:.2f}'
        cv2.putText(out, tag, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


def run_on_dir(net, in_dir, out_dir, output_mode=None):
    output_mode = output_mode or net.cfg["pipeline"]["output_mode"]
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    imgs = sorted([p for p in in_dir.glob("*")
                   if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    summary = {"images": 0, "True": 0, "False": 0, "No_Detection": 0,
               "output_mode": output_mode, "per_image": []}
    for p in imgs:
        img = imread(p); recs = net.process(img)
        summary["images"] += 1
        if not recs:
            imwrite(out_dir / "No_Detection" / p.name, img)
            summary["No_Detection"] += 1
        else:
            if output_mode == "annotated_full":
                anno = _annotate(img, recs)
                dec = "True" if any(r["decision"] == "True" for r in recs) else "False"
                # 大圖整張各放一份到 True / False（依是否至少一個通過）
                imwrite(out_dir / dec / "_full" / p.name, anno)
            for j, r in enumerate(recs):
                summary[r["decision"]] += 1
                if output_mode == "cropped_roi":
                    sub = out_dir / r["decision"] / r["pred_class"]
                    imwrite(sub / f"{p.stem}_roi{j}_{r['score']:.2f}.png", r["crop"])
        summary["per_image"].append({
            "image": p.name,
            "dets": [{k: v for k, v in r.items() if k != "crop"} for r in recs]})
    (out_dir / "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k != "per_image"}, indent=2))
    print(f"[run] {out_dir.name}: imgs={summary['images']} "
          f"True={summary['True']} False={summary['False']} "
          f"No_Detection={summary['No_Detection']} mode={output_mode}")
    return summary


# ----------------------------- 閾值校準 -----------------------------
def calibrate(net, calib_split="calib", write_back=True):
    """用 calib 場景 GT 校準 cosine 閾值（最大化 F1）。
    正樣本=命中真目標(target)；負樣本=OOD 硬負樣本 + 純背景；部分重疊(ambiguous)排除。"""
    gt = json.loads((WS / "scenes" / calib_split / "gt.json").read_text())
    img_dir = WS / "scenes" / calib_split / "images"
    scores, is_real = [], []
    for sc in gt["scenes"]:
        img = imread(img_dir / sc["image"])
        targets = [(o["box"], o["cls"]) for o in sc["objects"]]
        oods = [o["box"] for o in sc.get("ood", [])]
        dets = net.detect(img)
        crops = net.sam.segment_crops(img, [d[0] for d in dets])
        for (xyxy, conf), (crop, _) in zip(dets, crops):
            if crop.size == 0:
                continue
            kind, _ = classify_box(xyxy, targets, oods)
            if kind == "ambiguous":
                continue
            _, score = net.match(net.dino.embed(crop))
            scores.append(score); is_real.append(kind == "target")
    scores = np.array(scores); is_real = np.array(is_real, bool)
    if len(scores) == 0 or is_real.sum() == 0:
        print("[calib] insufficient detections; keep default threshold")
        return net.threshold
    best_t, best_f1 = net.threshold, -1
    for t in np.linspace(scores.min(), scores.max(), 80):
        pred = scores >= t
        tp = int((pred & is_real).sum()); fp = int((pred & ~is_real).sum())
        fn = int((~pred & is_real).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"[calib] dets={len(scores)} real={int(is_real.sum())} "
          f"-> threshold={best_t:.4f} (F1={best_f1:.3f})")
    net.threshold = best_t
    if write_back:
        import yaml
        cfgp = ROOT / "config.yaml"
        c = yaml.safe_load(open(cfgp, encoding="utf-8"))
        c["matching"]["threshold"] = round(best_t, 4)
        yaml.safe_dump(c, open(cfgp, "w", encoding="utf-8"),
                       allow_unicode=True, sort_keys=False)
        print(f"[calib] threshold written back to config.yaml")
    return best_t


# ----------------------------- GUI 預算快取 -----------------------------
GUI_CACHE = WS / "gui_cache" / "check_records.pkl"


def _golden_gallery(net, per_class=8):
    """載入未增強 golden 影像供「最相似樣本」展示比對。回傳 (paths, classes, vecs)。"""
    from common import imread as _imread
    paths, clss, vecs = [], [], []
    for cls in net.classes:
        for p in sorted((WS / "Training" / cls).glob("*.png"))[:per_class]:
            paths.append(str(p)); clss.append(cls); vecs.append(net.dino.embed(_imread(p)))
    return paths, clss, (np.stack(vecs) if vecs else np.zeros((0, net.dino.dim), np.float32))


def process_for_gui(net, img, gal):
    """跑 YOLO→SAM→DINO，回傳每框完整紀錄（含 crop/向量/top-3 golden），與閾值無關。"""
    gp, gc, gv = gal
    dets = net.detect(img); crops = net.sam.segment_crops(img, [d[0] for d in dets])
    h, w = img.shape[:2]; out = []
    for (xyxy, conf), (crop, ok) in zip(dets, crops):
        if crop.size == 0:
            continue
        vec = net.dino.embed(crop); cls, score = net.match(vec)
        golden = []
        if len(gv):
            sims = gv @ vec
            golden = [(gp[i], gc[i], float(sims[i])) for i in np.argsort(sims)[::-1][:3]]
        x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
        out.append(dict(box=[float(v) for v in xyxy], conf=float(conf), score=float(score),
                        pred_class=cls, sam_ok=bool(ok), vec=vec.astype(np.float32),
                        raw=img[y1:min(h, y2), x1:min(w, x2)].copy(), sam=crop, golden=golden))
    return out


def build_gui_cache(net=None, check_dir=None, progress=None):
    """預先計算 Check/ 全部偵測紀錄並存成 pickle（GUI 之後瞬間載入）。"""
    import pickle
    cfg = load_config()
    net = net or SafetyNet(cfg)
    gal = _golden_gallery(net)
    check_dir = Path(check_dir or (WS / "Check"))
    imgs = sorted([p for p in check_dir.glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    images = {}
    for i, p in enumerate(imgs):
        if progress:
            progress(i, len(imgs), p.name)
        images[p.name] = process_for_gui(net, imread(p), gal)
    cache = dict(classes=net.classes, threshold=net.threshold, dino=net.sel["dino"],
                 device=net.sel["device"], images=images)
    GUI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(GUI_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[gui-cache] saved {GUI_CACHE}  ({len(imgs)} images, "
          f"{sum(len(v) for v in images.values())} detections)")
    return cache


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gui_cache", action="store_true", help="預算 GUI 快取")
    ap.add_argument("--check_dir", default=str(WS / "Check"))
    ap.add_argument("--out_dir", default=str(WS / "Result"))
    ap.add_argument("--output_mode", default=None)
    args = ap.parse_args()
    cfg = load_config()
    net = SafetyNet(cfg, args.weights)
    if args.calibrate:
        calibrate(net)
    if args.run:
        run_on_dir(net, args.check_dir, args.out_dir, args.output_mode)
    if args.gui_cache:
        build_gui_cache(net, args.check_dir,
                        progress=lambda i, n, nm: print(f"[gui-cache] {i+1}/{n} {nm}"))
    if not (args.calibrate or args.run or args.gui_cache):
        calibrate(net); run_on_dir(net, args.check_dir, args.out_dir, args.output_mode)


if __name__ == "__main__":
    main()
