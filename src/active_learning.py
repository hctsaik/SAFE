"""
active_learning.py — #2 Active Learning：把待標影像「依資訊量排序」，先標最划算的。

小資料下，隨機標註浪費預算；最該標的是「模型最沒把握」的框。本工具對每個偵測框算
三種不確定性訊號並加權成 uncertainty，排序後輸出人工標註佇列 + 縮圖，餵養 Data Flywheel。

不確定性訊號（皆 0~1，越大越該標）：
  1. near_threshold : |score - 該類閾值| 越小 -> 越在決策邊界（最可能標錯/翻盤）。
  2. low_margin     : (最近-次近類別) 分差越小 -> 類別歸屬越模糊。
  3. disagreement   : knn 與 prototype 兩種裁決的最佳類別不一致 -> 方法本身打架。

效能：優先吃 gui_cache（已含 vec/score/margin，零模型載入）；無快取才載模型重算。
為何有效：每張標註換到的 YOLO mAP 增益最大化 -> 用最少人力把資料集補在刀口上。

用法：
  python src/active_learning.py                      # 讀 gui_cache，輸出 Workspace/ActiveQueue
  python src/active_learning.py --top 50 --rebuild   # 強制重算快取
  python src/active_learning.py --w_thr 0.5 --w_margin 0.35 --w_disagree 0.15
"""
from __future__ import annotations
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT

WS = ROOT / "Workspace"
GUI_CACHE = WS / "gui_cache" / "check_records.pkl"
BANK = WS / "vector_bank.npz"


def _ensure_cache(rebuild=False, check_dir=None):
    if rebuild or not GUI_CACHE.exists():
        from pipeline import build_gui_cache
        print("[active] 無快取或 --rebuild -> 載入模型預算 gui_cache…")
        build_gui_cache(check_dir=check_dir,
                        progress=lambda i, n, nm: print(f"[active] cache {i+1}/{n} {nm}"))
    with open(GUI_CACHE, "rb") as f:
        return pickle.load(f)


def _agg_full(vec, bank_vecs, labels, protos, n_cls, topk):
    """回傳 (knn 分數陣列, knn 最佳類別, prototype 最佳類別)。
    讓 active_learning 可自行從 vec 算 margin，不依賴 cache 是否含 margin（相容舊快取）。"""
    sims = bank_vecs @ vec
    knn = np.full(n_cls, -1.0, np.float32); proto = np.full(n_cls, -1.0, np.float32)
    for ci in range(n_cls):
        cl = sims[labels == ci]
        if len(cl) == 0:
            continue
        knn[ci] = float(np.sort(cl)[-min(topk, len(cl)):].mean())
        proto[ci] = float(protos[ci] @ vec)
    return knn, int(np.argmax(knn)), int(np.argmax(proto))


def rank(cache, cfg, top=50, thr_band=0.10, margin_band=0.15,
         w_thr=0.5, w_margin=0.35, w_disagree=0.15):
    classes = [str(c) for c in cache["classes"]]
    # 操作閾值以「算出這些分數時所用的閾值」為準（cache 自帶），config 為後備
    g_thr = float(cache.get("threshold") or cfg["matching"]["threshold"])
    per_cls_thr = {str(k): float(v) for k, v in (cfg["matching"].get("thresholds") or {}).items()}
    topk = int(cfg["matching"].get("topk", 8))
    bank = np.load(BANK, allow_pickle=True) if BANK.exists() else None
    bvecs = blabels = protos = None
    if bank is not None:
        bvecs, blabels, protos = bank["vecs"], bank["labels"], bank["protos"]

    rows = []
    for img_name, recs in cache["images"].items():
        for j, r in enumerate(recs):
            score = float(r["score"]); cls = r["pred_class"]
            thr = per_cls_thr.get(str(cls), g_thr)
            margin = r.get("margin")
            disagree = 0
            if bvecs is not None and "vec" in r:
                knn, kb, pb = _agg_full(np.asarray(r["vec"], np.float32), bvecs, blabels,
                                        protos, len(classes), topk)
                disagree = int(kb != pb)
                if margin is None:   # 舊快取無 margin -> 自 vec 算 (最近-次近)
                    srt = np.sort(knn)[::-1]
                    margin = float(srt[0] - srt[1]) if len(srt) > 1 and srt[1] >= 0 else 1.0
            margin = float(margin if margin is not None else 1.0)
            u_thr = max(0.0, 1.0 - abs(score - thr) / max(1e-6, thr_band))
            u_margin = max(0.0, 1.0 - max(0.0, margin) / max(1e-6, margin_band))
            u = w_thr * u_thr + w_margin * u_margin + w_disagree * disagree
            reason = max([("near_threshold", u_thr * w_thr),
                          ("low_margin", u_margin * w_margin),
                          ("disagreement", disagree * w_disagree)],
                         key=lambda t: t[1])[0]
            rows.append(dict(image=img_name, det=j, pred_class=cls, score=round(score, 4),
                             thr=round(thr, 4), margin=round(margin, 4),
                             u_thr=round(u_thr, 3), u_margin=round(u_margin, 3),
                             disagree=disagree, uncertainty=round(float(u), 4),
                             top_reason=reason, _crop=r.get("sam")))
    rows.sort(key=lambda x: x["uncertainty"], reverse=True)
    return rows[:top] if top else rows


def export_queue(rows, out_dir, save_crops=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    review = out_dir / "review"
    if save_crops:
        review.mkdir(parents=True, exist_ok=True)
    clean = []
    for rank_i, r in enumerate(rows):
        crop = r.pop("_crop", None)
        if save_crops and crop is not None and getattr(crop, "size", 0):
            cv2.imwrite(str(review / f"{rank_i:03d}_{Path(r['image']).stem}_d{r['det']}"
                                     f"_u{r['uncertainty']:.2f}.png"), crop)
        clean.append(dict(rank=rank_i, **r))
    # CSV
    cols = ["rank", "image", "det", "pred_class", "score", "thr", "margin",
            "u_thr", "u_margin", "disagree", "uncertainty", "top_reason"]
    lines = [",".join(cols)]
    for r in clean:
        lines.append(",".join(str(r[c]) for c in cols))
    (out_dir / "active_queue.csv").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "active_queue.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[active] 佇列 {len(clean)} 筆 -> {out_dir}/active_queue.csv (+ review/ 縮圖)")
    if clean:
        print("[active] 最該標 Top5：")
        for r in clean[:5]:
            print(f"   #{r['rank']:>2} {r['image']} det{r['det']} "
                  f"u={r['uncertainty']:.3f} ({r['top_reason']}) "
                  f"pred={r['pred_class']} score={r['score']} margin={r['margin']}")
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(WS / "ActiveQueue"))
    ap.add_argument("--check_dir", default=str(WS / "Check"))
    ap.add_argument("--top", type=int, default=50, help="輸出前 N 筆（0=全部）")
    ap.add_argument("--rebuild", action="store_true", help="強制重算 gui_cache")
    ap.add_argument("--thr_band", type=float, default=0.10)
    ap.add_argument("--margin_band", type=float, default=0.15)
    ap.add_argument("--w_thr", type=float, default=0.5)
    ap.add_argument("--w_margin", type=float, default=0.35)
    ap.add_argument("--w_disagree", type=float, default=0.15)
    ap.add_argument("--no_crops", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cache = _ensure_cache(a.rebuild, a.check_dir)
    rows = rank(cache, cfg, a.top, a.thr_band, a.margin_band,
                a.w_thr, a.w_margin, a.w_disagree)
    export_queue(rows, a.out_dir, save_crops=not a.no_crops)


if __name__ == "__main__":
    main()
