"""
autolabel.py — #1 Auto-Label：把「安全網的判決」自動匯出成 YOLO 訓練標註。

Data Flywheel 的起點：YOLO(觸發) -> SAM(去背收緊框) -> DINO(類別+分數) 跑完後，
把高信度的 True 框寫成 YOLO txt 偽標（class=DINO 類別、box=SAM 收緊框），
被攔截的誤報則讓該影像成為「純背景場景」(只標真目標) -> 教 YOLO 自己別再框那些東西。

為何有效：
  - 資料集管理：人從「畫框」降級為「打勾驗證」，標註成本砍 ~80%。
  - YOLO 準度  ：SAM 收緊框比 YOLO 原框更貼合 -> box 品質↑ -> mAP↑；
                 攔截框變隱性負樣本 -> YOLO 學會少誤報。
gating（避免偽標噪音毒化訓練，呼應閉環 confirmation-bias 風險）：
  只匯出 decision=True 且 score>=閾值+pad 且 margin>=min_margin 的框為偽正樣本；
  灰帶（貼近閾值/低 margin）不自動標 -> 交給 active_learning.py 找人。

輸出（標準 YOLO 目錄，可直接餵 train_yolo / ultralytics）：
  out_dir/images/*.jpg
  out_dir/labels/*.txt        # cls cx cy w h  (normalized)
  out_dir/data.yaml           # names（多類）或 [object]（--single-class）
  out_dir/manifest.json       # 每框 audit：box/類別/score/margin/decision/exported/reason

用法：
  python src/autolabel.py                         # 對 Workspace/Check 自動標 -> Workspace/AutoLabel
  python src/autolabel.py --in_dir path --out_dir path --single-class
  python src/autolabel.py --min_score_pad 0.05 --min_margin 0.05 --positives_only
"""
from __future__ import annotations
import sys, json, shutil, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, imread, ROOT, iou_xyxy
from pipeline import SafetyNet

WS = ROOT / "Workspace"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _to_yolo(box, W, H):
    """(x1,y1,x2,y2) 絕對座標 -> (cx,cy,w,h) 正規化。"""
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / W; cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W; h = (y2 - y1) / H
    return cx, cy, w, h


def _mask_to_polygon(mask, W, H, min_area_frac=0.0005):
    """R10：SAM 二值遮罩 -> 最大輪廓的正規化多邊形 [(x,y),...]（YOLO-seg 標籤）。失敗回 None。"""
    import cv2
    if mask is None:
        return None
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * W * H:
        return None
    eps = 0.01 * cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return [(float(x) / W, float(y) / H) for x, y in ap]


def _box_polygon(box, W, H):
    """退路：用 bbox 四角當多邊形（SAM 遮罩不可用時）。"""
    x1, y1, x2, y2 = box
    return [(x1 / W, y1 / H), (x2 / W, y1 / H), (x2 / W, y2 / H), (x1 / W, y2 / H)]


def label_image(net, img, min_score_pad, min_margin, tta=False, min_box_iou=0.0,
                cotrain=False, min_consistency=0.5, seg=False):
    """跑完整管線並對每框下「是否可作偽標」的判定。回傳 list[dict]。
    R8：tta=True 用多視角平均嵌入(更穩)+回傳 consistency；min_box_iou 要求 YOLO 框與 SAM 框夠一致。
    R9：cotrain=True 要求兩視角一致（多類別下 YOLO 類別==DINO 類別）且 TTA 夠穩定，才當偽正樣本。
    R10：seg=True 另用 SAM 遮罩產多邊形（rec['polygon']），供分割蒸餾 YOLO-seg。"""
    H, W = img.shape[:2]
    dets = net.detect(img)
    if seg:
        segd = net.sam.segment_full(img, [d[0] for d in dets])
    else:
        segd = [(c, ok, t, None) for (c, ok, t) in
                net.sam.segment_crops_boxed(img, [d[0] for d in dets])]
    recs = []
    for (xyxy, conf, ycls), (crop, sam_ok, tight, mask) in zip(dets, segd):
        if crop.size == 0:
            continue
        if tta:
            vec, cons = net.dino.embed_tta(crop)
        else:
            vec, cons = net.dino.embed(crop), 1.0
        cls, score, score2 = net.match(vec)
        margin = (score - score2) if score2 >= 0 else 1.0
        thr = net._thr(cls)
        decision = "True" if score >= thr else "False"
        box = list(tight) if sam_ok else [int(v) for v in xyxy]  # 收緊框優先
        box_iou = iou_xyxy(xyxy, tight) if sam_ok else 1.0       # R8 框一致性
        # gating：高信度 True 才當偽正樣本
        exported = (decision == "True" and score >= thr + min_score_pad
                    and margin >= min_margin)
        reason = "confident_true" if exported else (
            "uncertain_true" if decision == "True" else "intercepted")
        if exported and sam_ok and box_iou < min_box_iou:        # R8：框不一致 -> 定位可疑
            exported = False; reason = "box_disagree"
        if exported and cotrain:                                 # R9：兩視角須一致
            if net.multiclass and ycls != cls:
                exported = False; reason = "class_disagree"
            elif tta and cons < min_consistency:
                exported = False; reason = "unstable"
        polygon = (_mask_to_polygon(mask, W, H) if (seg and sam_ok) else None) if seg else None
        recs.append(dict(box=box, yolo_box=[int(v) for v in xyxy], pred_class=cls,
                         yolo_class=ycls, score=round(float(score), 4),
                         margin=round(float(margin), 4), box_iou=round(float(box_iou), 4),
                         consistency=round(float(cons), 4), thr=round(float(thr), 4),
                         conf=round(float(conf), 4), sam_ok=bool(sam_ok), polygon=polygon,
                         decision=decision, exported=bool(exported), reason=reason))
    return recs


def autolabel_dir(net, in_dir, out_dir, min_score_pad=0.05, min_margin=0.05,
                  single_class=False, positives_only=False, tta=False,
                  min_box_iou=0.0, cotrain=False, min_consistency=0.5, seg=False,
                  progress=None):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out = out_dir / "images"; lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
    cls_to_idx = {c: i for i, c in enumerate(net.classes)}
    imgs = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in IMG_EXT])
    summ = dict(images=0, dets=0, exported=0, uncertain=0, intercepted=0,
                empty_images=0, classes=list(net.classes), task="segment" if seg else "detect",
                single_class=single_class, per_image=[])
    for k, p in enumerate(imgs):
        if progress:
            progress(k, len(imgs), p.name)
        img = imread(p); H, W = img.shape[:2]
        recs = label_image(net, img, min_score_pad, min_margin, tta=tta,
                           min_box_iou=min_box_iou, cotrain=cotrain,
                           min_consistency=min_consistency, seg=seg)
        pos = [r for r in recs if r["exported"]]
        summ["dets"] += len(recs)
        summ["exported"] += len(pos)
        summ["uncertain"] += sum(r["reason"] == "uncertain_true" for r in recs)
        summ["intercepted"] += sum(r["reason"] == "intercepted" for r in recs)
        if positives_only and not pos:
            continue  # 只要含偽正樣本的圖
        summ["images"] += 1
        if not pos:
            summ["empty_images"] += 1   # 純負樣本場景（教 YOLO 別誤報）
        shutil.copy(p, img_out / p.name)
        lines = []
        for r in pos:
            idx = 0 if single_class else cls_to_idx[r["pred_class"]]
            if seg:   # R10：多邊形標籤（無遮罩時退回 bbox 矩形）
                poly = r.get("polygon") or _box_polygon(r["box"], W, H)
                lines.append(f"{idx} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
            else:
                cx, cy, w, h = _to_yolo(r["box"], W, H)
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_out / f"{p.stem}.txt").write_text("\n".join(lines))
        summ["per_image"].append(dict(image=p.name, exported=len(pos),
                                      dets=[{k: v for k, v in r.items()} for r in recs]))
    # data.yaml
    names = ["object"] if single_class else list(net.classes)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\ntrain: images\nval: images\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(
        {k: v for k, v in summ.items() if k != "per_image"}
        | {"per_image": summ["per_image"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[autolabel] imgs={summ['images']} dets={summ['dets']} "
          f"exported_pos={summ['exported']} uncertain={summ['uncertain']} "
          f"intercepted={summ['intercepted']} empty(neg-only)={summ['empty_images']}")
    print(f"[autolabel] -> {out_dir}  (images/ labels/ data.yaml manifest.json)")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(WS / "Check"))
    ap.add_argument("--out_dir", default=str(WS / "AutoLabel"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--min_score_pad", type=float, default=0.05,
                    help="只匯出 score>=閾值+此值 的框（避免貼近閾值的偽標噪音）")
    ap.add_argument("--min_margin", type=float, default=0.05,
                    help="只匯出 (最近-次近) 類別分差>=此值 的框（類別要分得開）")
    ap.add_argument("--single-class", dest="single_class", action="store_true",
                    help="所有偽正樣本標為單類 'object'（餵回觸發器 YOLO）；預設多類")
    ap.add_argument("--positives_only", action="store_true",
                    help="只保留含偽正樣本的影像（不輸出純負樣本場景）")
    ap.add_argument("--tta", action="store_true",
                    help="R8：多視角平均嵌入(更穩)+回傳一致性")
    ap.add_argument("--min_box_iou", type=float, default=0.0,
                    help="R8：要求 YOLO 框與 SAM 框 IoU>=此值才當偽正樣本（定位可信）")
    ap.add_argument("--cotrain", action="store_true",
                    help="R9：兩視角一致才匯出（多類別下 YOLO 類別==DINO 類別 + TTA 穩定）")
    ap.add_argument("--min_consistency", type=float, default=0.5,
                    help="R9：搭配 --tta，視角一致性下限")
    ap.add_argument("--seg", action="store_true",
                    help="R10：用 SAM 遮罩輸出 YOLO-seg 多邊形標籤（分割蒸餾）")
    a = ap.parse_args()
    net = SafetyNet(load_config(), a.weights)
    autolabel_dir(net, a.in_dir, a.out_dir, a.min_score_pad, a.min_margin,
                  a.single_class, a.positives_only, tta=a.tta, min_box_iou=a.min_box_iou,
                  cotrain=a.cotrain, min_consistency=a.min_consistency, seg=a.seg,
                  progress=lambda i, n, nm: print(f"[autolabel] {i+1}/{n} {nm}"))


if __name__ == "__main__":
    main()
