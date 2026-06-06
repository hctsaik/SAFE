"""
hardneg.py — R7 硬負樣本挖掘：專挑「YOLO 最自信、卻被安全網攔下」的框 = YOLO 的系統性盲點。

不是所有被攔截的框都一樣有價值。最該回灌訓練的，是 YOLO **高 conf 觸發、DINO 卻判遠低於閾值**
的那些——代表 YOLO 把某種背景/紋理/非目標物件當成目標，且很有把握。把這些當「重點背景負樣本」
並在重訓時 over-weight，能最快教 YOLO 別再誤報。

wrongness = yolo_conf × max(0, 閾值 - dino_score)   # 越大 = YOLO 越自信地犯錯
輸出：Workspace/HardNegatives/ 代表縮圖 + manifest（provenance：圖/框/conf/score/最近類別/wrongness）。
（assemble_dataset 另以 wrongness 對「純負樣本影像」加權重取樣，見 retrain_loop。）

用法：
  python src/hardneg.py                       # 掃 Workspace/Check
  python src/hardneg.py --in_dir DIR --top 50
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


def mine(net, in_dir, progress=None):
    """回傳依 wrongness 由大到小排序的硬負樣本紀錄（含 crop）。"""
    imgs = sorted([p for p in Path(in_dir).glob("*") if p.suffix.lower() in IMG_EXT])
    rows = []
    for k, p in enumerate(imgs):
        if progress:
            progress(k, len(imgs), p.name)
        for r in net.process(imread(p)):
            if r["decision"] != "False":        # 只要被攔截的
                continue
            wrong = float(r["conf"]) * max(0.0, float(r["thr"]) - float(r["score"]))
            rows.append(dict(image=p.name, box=r["box"], yolo_conf=round(float(r["conf"]), 4),
                             dino_score=round(float(r["score"]), 4), thr=round(float(r["thr"]), 4),
                             nearest_class=r["pred_class"], wrongness=round(wrong, 4),
                             _crop=r["crop"]))
    rows.sort(key=lambda x: -x["wrongness"])
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
            d = out_dir / r["nearest_class"]; d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{i:03d}_{Path(r['image']).stem}_w{r['wrongness']:.2f}.png"), crop)
        clean.append(dict(rank=i, **r))
    (out_dir / "hardneg_manifest.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[hardneg] 挖到 {len(rows)} 個硬負樣本 -> 匯出前 {len(clean)} 個至 {out_dir}")
    for r in clean[:5]:
        print(f"   #{r['rank']:>2} {r['image']} wrongness={r['wrongness']} "
              f"(YOLO conf={r['yolo_conf']} 但 DINO={r['dino_score']}<{r['thr']}, 最近 {r['nearest_class']})")
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(WS / "Check"))
    ap.add_argument("--out_dir", default=str(WS / "HardNegatives"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--top", type=int, default=50, help="匯出前 N 個（0=全部）")
    a = ap.parse_args()
    net = SafetyNet(load_config(), a.weights)
    rows = mine(net, a.in_dir, progress=lambda i, n, nm: print(f"[hardneg] {i+1}/{n} {nm}"))
    export(rows, a.out_dir, a.top)


if __name__ == "__main__":
    main()
