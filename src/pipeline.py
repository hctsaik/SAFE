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
        # 偵測器權重優先序：CLI --weights > config 的 yolo.weights > 預設 best.pt
        # 想用「你自己的 YOLO .pt」只要在 config.yaml 設 yolo.weights 路徑，免動程式。
        self.yolo = YOLO(str(yolo_weights or cfg["yolo"].get("weights") or DEFAULT_YOLO))
        self.sam = SamSegmenter(self.sel["sam"], self.sel["device"], cfg)
        self.dino = DinoEmbedder(self.sel["dino"], self.sel["device"], hub_dir=HUB)
        b = np.load(BANK_PATH, allow_pickle=True)
        self.bank_vecs = b["vecs"]; self.bank_labels = b["labels"]
        self.classes = [str(c) for c in b["classes"]]; self.protos = b["protos"]
        self.threshold = float(cfg["matching"]["threshold"])
        # per-class 閾值（#5）：難分類別(zipper≈screw)可獨立收緊；缺項退回全域 threshold。
        self.thresholds = {str(k): float(v)
                           for k, v in (cfg["matching"].get("thresholds") or {}).items()}
        self.topk = int(cfg["matching"]["topk"])
        self.agg = cfg["matching"]["agg"]
        self.conf = float(cfg["yolo"]["conf"])
        # R2 雙軌畢業：YOLO 是否多類別、哪些類別「已畢業」(交給 YOLO 自己分，安全網不再否決)。
        self.yolo_names = dict(getattr(self.yolo, "names", {}) or {})
        self.multiclass = len(self.yolo_names) > 1
        self.graduated = set(str(c) for c in (cfg["matching"].get("graduated_classes") or []))
        self.grad_conf = float(cfg["matching"].get("graduate_conf", 0.5))

    def _thr(self, cls):
        """取某類別的判決閾值：有 per-class 設定用之，否則用全域 threshold。"""
        return self.thresholds.get(str(cls), self.threshold)

    # ---- YOLO 觸發 ----
    def detect(self, img):
        """回傳 [(xyxy, conf, ycls_name), ...]；單類觸發器 ycls 恆為該單一類別名。"""
        r = self.yolo.predict(img, conf=self.conf, iou=0.5,
                              device=self.sel["device"], verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cid = int(b.cls[0]) if b.cls is not None else 0
                out.append((xyxy, float(b.conf[0]), self.yolo_names.get(cid, str(cid))))
        return out

    # ---- DINO 比對裁決 ----
    def match_scores(self, vec):
        """回傳「每類相似度」陣列（依 self.classes 順序；空類別為 -1）。
        是 match() 的底層；多回傳全類別分數讓上層算 margin / 第二名 / 裁決衝突。"""
        sims = self.bank_vecs @ vec  # cosine（皆已 L2 normalize）
        out = np.full(len(self.classes), -1.0, np.float32)
        for ci in range(len(self.classes)):
            cl = sims[self.bank_labels == ci]
            if len(cl) == 0:
                continue
            if self.agg == "prototype":
                out[ci] = float(self.protos[ci] @ vec)
            else:  # knn: 前 k 高相似度平均
                k = min(self.topk, len(cl))
                out[ci] = float(np.sort(cl)[-k:].mean())
        return out

    def match(self, vec):
        """回傳 (最近類別, 最近分數 s1, 次近分數 s2)。
        margin = s1 - s2 量「類別間分得多開」，是 Active Learning / 偽標 gating /
        Golden 策展共用的不確定性訊號（s2<0 表示只有單一類別，無從比較）。"""
        s = self.match_scores(vec)
        order = np.argsort(s)[::-1]
        best = int(order[0]); s1 = float(s[best])
        s2 = float(s[int(order[1])]) if len(order) > 1 else -1.0
        return self.classes[best], s1, s2

    def _is_graduated(self, ycls, conf):
        """R2 雙軌：多類別 YOLO 對『已畢業類別』且夠有把握 -> 直接信任，安全網不再否決。"""
        return self.multiclass and ycls in self.graduated and conf >= self.grad_conf

    # ---- 單圖完整管線（雙軌：畢業類別走 YOLO，其餘走安全網）----
    def process(self, img, thr=None):
        dets = self.detect(img)
        crops = self.sam.segment_crops(img, [d[0] for d in dets])  # 批次去背
        recs = []
        for (xyxy, conf, ycls), (crop, sam_ok) in zip(dets, crops):
            if crop.size == 0:
                continue
            if self._is_graduated(ycls, conf):
                # 已畢業：YOLO 自己分類，安全網不跑 DINO 否決（蒸餾完成的證據）
                recs.append(dict(box=[float(v) for v in xyxy], conf=conf,
                                 pred_class=ycls, score=float(conf), score2=-1.0,
                                 margin=1.0, thr=self.grad_conf, sam_ok=sam_ok,
                                 decision="True", via="yolo", crop=crop))
                continue
            vec = self.dino.embed(crop)
            cls, score, score2 = self.match(vec)
            thr_c = self._thr(cls)
            decision = "True" if score >= thr_c else "False"
            recs.append(dict(box=[float(v) for v in xyxy], conf=conf,
                             pred_class=cls, score=score, score2=score2,
                             margin=round(score - score2, 4) if score2 >= 0 else 1.0,
                             thr=thr_c, sam_ok=sam_ok,
                             decision=decision, via="safetynet", crop=crop))
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
def _best_f1_threshold(scores, is_real, default):
    """掃描閾值取最大化 F1 的值。回傳 (threshold, f1)。"""
    scores = np.asarray(scores); is_real = np.asarray(is_real, bool)
    if len(scores) == 0 or is_real.sum() == 0 or (~is_real).sum() == 0:
        return float(default), -1.0
    best_t, best_f1 = float(default), -1.0
    for t in np.linspace(scores.min(), scores.max(), 80):
        pred = scores >= t
        tp = int((pred & is_real).sum()); fp = int((pred & ~is_real).sum())
        fn = int((~pred & is_real).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def calibrate(net, calib_split="calib", write_back=True, per_class=True, min_per_class=8):
    """用 calib 場景 GT 校準 cosine 閾值（最大化 F1）。
    正樣本=命中真目標(target)；負樣本=OOD 硬負樣本 + 純背景；部分重疊(ambiguous)排除。
    per_class=True 時，對每個「被判為該類」的框群獨立校準閾值（#5）：難分類別(zipper≈screw)
    可收得更緊、好分類別放得更鬆；某類樣本 < min_per_class 則退回全域閾值。"""
    gt = json.loads((WS / "scenes" / calib_split / "gt.json").read_text())
    img_dir = WS / "scenes" / calib_split / "images"
    scores, is_real, pred_cls = [], [], []
    for sc in gt["scenes"]:
        img = imread(img_dir / sc["image"])
        targets = [(o["box"], o["cls"]) for o in sc["objects"]]
        oods = [o["box"] for o in sc.get("ood", [])]
        dets = net.detect(img)
        crops = net.sam.segment_crops(img, [d[0] for d in dets])
        for (xyxy, conf, _ycls), (crop, _) in zip(dets, crops):
            if crop.size == 0:
                continue
            kind, _ = classify_box(xyxy, targets, oods)
            if kind == "ambiguous":
                continue
            cls, score, _ = net.match(net.dino.embed(crop))
            scores.append(score); is_real.append(kind == "target"); pred_cls.append(cls)
    scores = np.array(scores); is_real = np.array(is_real, bool); pred_cls = np.array(pred_cls)
    if len(scores) == 0 or is_real.sum() == 0:
        print("[calib] insufficient detections; keep default threshold")
        return net.threshold
    best_t, best_f1 = _best_f1_threshold(scores, is_real, net.threshold)
    print(f"[calib] global dets={len(scores)} real={int(is_real.sum())} "
          f"-> threshold={best_t:.4f} (F1={best_f1:.3f})")
    net.threshold = best_t
    # --- per-class 閾值 ---
    thresholds = {}
    if per_class:
        for c in sorted(set(pred_cls)):
            sel = pred_cls == c
            n = int(sel.sum()); npos = int(is_real[sel].sum()); nneg = n - npos
            if n < min_per_class or npos < 2 or nneg < 2:
                print(f"[calib]   {c}: n={n}(pos={npos}) < 門檻 -> 用全域 {best_t:.4f}")
                continue
            t, f1 = _best_f1_threshold(scores[sel], is_real[sel], best_t)
            thresholds[str(c)] = round(t, 4)
            print(f"[calib]   {c}: n={n}(pos={npos}) -> thr={t:.4f} (F1={f1:.3f})")
        net.thresholds = {k: float(v) for k, v in thresholds.items()}
    if write_back:
        import yaml
        cfgp = ROOT / "config.yaml"
        c = yaml.safe_load(open(cfgp, encoding="utf-8"))
        c["matching"]["threshold"] = round(best_t, 4)
        if per_class:
            c["matching"]["thresholds"] = thresholds  # 空 dict 也寫回（明確表示已校準）
        yaml.safe_dump(c, open(cfgp, "w", encoding="utf-8"),
                       allow_unicode=True, sort_keys=False)
        print(f"[calib] written back: threshold + {len(thresholds)} per-class thresholds")
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
    for (xyxy, conf, _ycls), (crop, ok) in zip(dets, crops):
        if crop.size == 0:
            continue
        vec = net.dino.embed(crop); cls, score, score2 = net.match(vec)
        golden = []
        if len(gv):
            sims = gv @ vec
            golden = [(gp[i], gc[i], float(sims[i])) for i in np.argsort(sims)[::-1][:3]]
        x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
        out.append(dict(box=[float(v) for v in xyxy], conf=float(conf), score=float(score),
                        score2=float(score2), margin=float(score - score2) if score2 >= 0 else 1.0,
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
