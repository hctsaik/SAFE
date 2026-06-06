"""
hardpos.py — Hard POSITIVE 探勘（hardneg 的鏡像）：撈回「YOLO 信心不足、但嵌入空間像 defect」的漏檢。

Recall 瓶頸常在 YOLO 第一階段：真 defect 可能因小/透明/低對比/反光而 conf 偏低、被 threshold 砍掉。
本工具用「低 conf 偵測 + DINO 嵌入審查」主動撈回這些案例：
  低 conf 候選框 + match_audit 判 defect_like（離 Defect 庫近、且明顯比 Reject/Normal 近）
  -> 高風險漏檢 -> 進「recall 撈回 / hard-positive 池」給工程師確認 -> 成 YOLO 重訓正樣本。

排序：recall_score = 與 Defect 庫的 Top-K 相似度（越像越優先）。有 Reject/Normal 庫時，僅收 defect_like。

用法（建議搭配低 conf 與已建好的 reject/normal 庫）：
  python src/hardpos.py --conf 0.03                 # 掃 Workspace/Check，低 conf 撈回
  python src/hardpos.py --in_dir DIR --conf 0.03 --conf_high 0.25 --top 50
"""
from __future__ import annotations
import sys, json, shutil, argparse
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, imread, ROOT
from pipeline import SafetyNet

WS = ROOT / "Workspace"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def mine(net, in_dir, conf_high, progress=None):
    """回傳依 recall_score 由大到小排序的 hard-positive 候選（低 conf + 嵌入像 defect）。"""
    imgs = sorted([p for p in Path(in_dir).glob("*") if p.suffix.lower() in IMG_EXT])
    rows = []
    for k, p in enumerate(imgs):
        if progress:
            progress(k, len(imgs), p.name)
        img = imread(p)
        dets = net.detect(img)
        crops = net.sam.segment_crops(img, [d[0] for d in dets])
        for (xyxy, conf, _y), (crop, _ok) in zip(dets, crops):
            if crop.size == 0 or conf >= conf_high:
                continue                       # 只看低信心候選
            a = net.match_audit(net.dino.embed(crop))
            # 有 reject/normal 庫 -> 只收 defect_like；無 aux 庫 -> 用 defect 相似度直接排序
            if net.multibank and a["verdict"] != "defect_like":
                continue
            rows.append(dict(image=p.name, box=[float(v) for v in xyxy],
                             yolo_conf=round(float(conf), 4), s_defect=a["s_defect"],
                             s_reject=a["s_reject"], s_normal=a["s_normal"],
                             verdict=a["verdict"], pred_class=a["pred_class"],
                             recall_score=a["s_defect"], _crop=crop))
    rows.sort(key=lambda x: -x["recall_score"])
    return rows


def export(rows, out_dir, top=50):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = []
    for i, r in enumerate(rows[:top] if top else rows):
        crop = r.pop("_crop", None)
        if crop is not None and getattr(crop, "size", 0):
            d = out_dir / r["pred_class"]; d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{i:03d}_{Path(r['image']).stem}_s{r['recall_score']:.2f}.png"), crop)
        clean.append(dict(rank=i, **r))
    (out_dir / "hardpos_manifest.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[hardpos] 撈回 {len(rows)} 個低信心 defect-like 候選 -> 匯出前 {len(clean)} 至 {out_dir}")
    for r in clean[:5]:
        print(f"   #{r['rank']:>2} {r['image']} YOLO conf={r['yolo_conf']} 但 defect 相似={r['s_defect']}"
              f" (reject={r['s_reject']} normal={r['s_normal']} → {r['verdict']})")
    print("[hardpos] 工程師確認為真 defect 者，複製去 Workspace/Training/<類別>/ 或 YOLO 正樣本池後重訓。")
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(WS / "Check"))
    ap.add_argument("--out_dir", default=str(WS / "HardPositives"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None, help="YOLO 偵測 conf（建議調低如 0.03 以撈回）")
    ap.add_argument("--conf_high", type=float, default=None, help="低/高信心分界（預設讀 config.audit）")
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args()
    cfg = load_config()
    net = SafetyNet(cfg, a.weights)
    if a.conf is not None:
        net.conf = a.conf
    conf_high = a.conf_high if a.conf_high is not None else net.conf_high
    print(f"[hardpos] YOLO conf={net.conf} 低/高分界={conf_high} 多庫={net.multibank}")
    rows = mine(net, a.in_dir, conf_high,
                progress=lambda i, n, nm: print(f"[hardpos] {i+1}/{n} {nm}") if i % 10 == 0 else None)
    export(rows, a.out_dir, a.top)


if __name__ == "__main__":
    main()
