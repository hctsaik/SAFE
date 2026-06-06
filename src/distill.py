"""
distill.py — R1 蒸餾 KPI：量測「YOLO 到底吸收了多少安全網」。

蒸餾的目標不是「跑得動」，而是讓 YOLO 自己越來越強、安全網介入越來越少。本工具在真實
hold-out 上把兩個系統擺在同一批 YOLO 偵測框上對比：
  - YOLO-alone：把 YOLO 的每個偵測都當數（觸發器無法分類時，類別正確率不計）。
  - Safety-net：YOLO + SAM + DINO 的最終裁決（True/False + 類別）。

核心 KPI：
  - intervention_rate 介入率：安全網推翻 YOLO 的比例（攔截 False，或多類別下改類別）。
        蒸餾有效 -> 隨輪次「下降」（YOLO 自己少犯錯，不需安全網兜）。
  - precision_gap     精準度落差：net_precision - yolo_alone_precision。
        蒸餾有效 -> 「縮小」（YOLO 自己的精準度逼近安全網）。
  - class_acc_gap     類別落差（多類別 YOLO 才有）：net 與 YOLO-alone 的類別正確率差。

用法：
  python src/distill.py --data path/to/holdout
  python src/distill.py --data holdout --weights cand.pt   # 量某候選權重的吸收程度
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT, iou_xyxy, classify_box
from pipeline import SafetyNet
from eval_real import _gather
from setup_build import read_yolo_txt

WS = ROOT / "Workspace"


def detect_with_cls(net, img):
    """YOLO 偵測並保留『YOLO 自己的類別』。回傳 (dets[(xyxy,conf,ycls)], n_classes)。
    直接複用 SafetyNet.detect()（已回傳 ycls）。"""
    return net.detect(img), (len(net.yolo_names) if net.yolo_names else 1)


def distill_metrics(net, pairs, names, hi=0.5, lo=0.1):
    """在同一批 YOLO 框上對比 YOLO-alone vs 安全網，算介入率與各落差。"""
    name_of = {i: n for i, n in enumerate(names)}
    n_dets = 0; yolo_tp = 0; net_true = 0; net_tp = 0
    overturn = 0; relabel = 0; n_targets = 0; rec_yolo = 0; rec_net = 0
    multiclass = False
    cy_correct = cy_total = cn_correct = cn_total = 0
    # 逐類（依真目標類別 tcls）統計，供 R2 逐類畢業判定
    pc = {n: dict(support=0, yolo_correct=0, overrides=0) for n in names}
    for img_path, lbl in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_yolo_txt(lbl, W, H)
        targets = [(b, name_of.get(c, str(c))) for c, b, _ in gt]
        n_targets += len(targets)
        dets, ncl = detect_with_cls(net, img)
        if ncl > 1:
            multiclass = True
        crops = net.sam.segment_crops(img, [d[0] for d in dets])
        hit_yolo = [False] * len(targets); hit_net = [False] * len(targets)
        for (xyxy, conf, ycls), (crop, ok) in zip(dets, crops):
            if crop.size == 0:
                continue
            n_dets += 1
            kind, tcls = classify_box(xyxy, targets, [], hi=hi, lo=lo)
            ncls, score, _ = net.match(net.dino.embed(crop))
            ndec = score >= net._thr(ncls)
            j = int(np.argmax([iou_xyxy(xyxy, b) for b, _ in targets])) if targets else -1
            this_overturn = (not ndec) or (multiclass and ndec and ncls != ycls)
            if kind == "target":
                yolo_tp += 1
                if j >= 0:
                    hit_yolo[j] = True
                if multiclass:
                    cy_total += 1; cy_correct += int(ycls == tcls)
                if ndec:
                    net_tp += 1
                    if j >= 0:
                        hit_net[j] = True
                    cn_total += 1; cn_correct += int(ncls == tcls)
                if tcls in pc:                       # 逐類：該真類別的 YOLO 表現與被介入程度
                    pc[tcls]["support"] += 1
                    pc[tcls]["yolo_correct"] += int(ycls == tcls)
                    pc[tcls]["overrides"] += int(this_overturn)
            if ndec:
                net_true += 1
            if not ndec:
                overturn += 1                       # 安全網攔掉 YOLO 的框
            elif multiclass and ncls != ycls:
                overturn += 1; relabel += 1         # 安全網替 YOLO 改類別
        rec_yolo += sum(hit_yolo); rec_net += sum(hit_net)
    yolo_prec = yolo_tp / n_dets if n_dets else 1.0
    net_prec = net_tp / net_true if net_true else 1.0
    out = dict(
        n_yolo_dets=n_dets, n_targets=n_targets,
        yolo_alone_precision=round(yolo_prec, 4), net_precision=round(net_prec, 4),
        precision_gap=round(net_prec - yolo_prec, 4),
        intervention_rate=round(overturn / n_dets, 4) if n_dets else 0.0,
        relabel_rate=round(relabel / n_dets, 4) if n_dets else 0.0,
        recall_yolo=round(rec_yolo / n_targets, 4) if n_targets else 1.0,
        recall_net=round(rec_net / n_targets, 4) if n_targets else 1.0,
        multiclass=multiclass)
    if multiclass:
        out["class_acc_yolo"] = round(cy_correct / cy_total, 4) if cy_total else 1.0
        out["class_acc_net"] = round(cn_correct / cn_total, 4) if cn_total else 1.0
        out["class_acc_gap"] = round(out["class_acc_net"] - out["class_acc_yolo"], 4)
    out["per_class"] = {c: dict(
        support=v["support"],
        yolo_class_acc=round(v["yolo_correct"] / v["support"], 4) if v["support"] else 0.0,
        override_rate=round(v["overrides"] / v["support"], 4) if v["support"] else 0.0)
        for c, v in pc.items()}
    return out


# ----------------------------- R2 逐類畢業 -----------------------------
def graduation(metrics, min_support=10, min_yolo_acc=0.9, max_override=0.1):
    """依逐類 KPI 決定哪些類別可『畢業』(交給 YOLO 自己分，安全網不再否決)：
    YOLO 自分夠準(yolo_class_acc≥min_yolo_acc) 且 安全網介入夠低(override_rate≤max_override)
    且 樣本夠多(support≥min_support)。難分類別(zipper≈screw)會留在安全網兜底。"""
    grad, stay = [], []
    for c, v in metrics.get("per_class", {}).items():
        ok = (v["support"] >= min_support and v["yolo_class_acc"] >= min_yolo_acc
              and v["override_rate"] <= max_override)
        row = dict(cls=c, **v, graduate=ok)
        (grad if ok else stay).append(row)
    if not metrics.get("multiclass"):
        return dict(ready=False, reason="YOLO 為單類觸發器；需先 flywheel --multiclass 訓練多類別 YOLO",
                    graduated=[], stay=[])
    return dict(ready=True, criteria=dict(min_support=min_support, min_yolo_acc=min_yolo_acc,
                max_override=max_override),
                graduated=[r["cls"] for r in grad], detail_graduated=grad, stay=stay)


def write_graduated(classes):
    import yaml
    cfgp = ROOT / "config.yaml"
    c = yaml.safe_load(open(cfgp, encoding="utf-8"))
    c.setdefault("matching", {})["graduated_classes"] = list(classes)
    yaml.safe_dump(c, open(cfgp, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"[graduate] 已寫回 config.matching.graduated_classes = {list(classes)}")


def print_kpi(m):
    print("\n========== 蒸餾 KPI（YOLO 吸收了多少安全網）==========")
    print(f"YOLO 偵測框 {m['n_yolo_dets']}　真目標 {m['n_targets']}")
    print(f"介入率 intervention_rate = {m['intervention_rate']}   ← 蒸餾有效應隨輪次下降")
    print(f"精準度  YOLO-alone {m['yolo_alone_precision']}  vs  安全網 {m['net_precision']}"
          f"  (落差 {m['precision_gap']} ← 應縮小)")
    print(f"召回    YOLO {m['recall_yolo']}  vs  安全網 {m['recall_net']}")
    if m.get("multiclass"):
        print(f"類別正確 YOLO-alone {m['class_acc_yolo']} vs 安全網 {m['class_acc_net']}"
              f" (落差 {m['class_acc_gap']} ← 應縮小)")
    else:
        print("（觸發器為單類，類別正確率需多類別蒸餾 R2 才可量）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="YOLO 目錄或 data.yaml（hold-out）")
    ap.add_argument("--images", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--names", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--out", default=str(WS / "distill_kpi.json"))
    ap.add_argument("--graduate", action="store_true",
                    help="依逐類 KPI 判定可畢業類別，並寫回 config.matching.graduated_classes")
    ap.add_argument("--min_support", type=int, default=10)
    ap.add_argument("--min_yolo_acc", type=float, default=0.9)
    ap.add_argument("--max_override", type=float, default=0.1)
    ap.add_argument("--write", action="store_true", help="搭配 --graduate：把結果寫回 config")
    a = ap.parse_args()
    names = [s.strip() for s in a.names.split(",")] if a.names else None
    names, pairs = _gather(a.data, a.images, a.labels, names)
    if not pairs:
        print("[distill] 找不到含標註的 hold-out 影像"); return
    net = SafetyNet(load_config(), a.weights)
    if names is None:
        names = list(net.classes)
    print(f"[distill] {len(pairs)} 張 hold-out　類別={names}")
    m = distill_metrics(net, pairs, names, a.hi, a.lo)
    Path(a.out).write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print_kpi(m)
    if a.graduate:
        g = graduation(m, a.min_support, a.min_yolo_acc, a.max_override)
        print("\n========== 逐類畢業評估 ==========")
        if not g["ready"]:
            print(f"[graduate] 尚不可評估：{g['reason']}")
        else:
            print(f"條件：support≥{a.min_support} 且 YOLO自分≥{a.min_yolo_acc} 且 介入率≤{a.max_override}")
            for r in g["detail_graduated"] + g["stay"]:
                tag = "🎓畢業(交YOLO)" if r["graduate"] else "🛡️留安全網"
                print(f"   {r['cls']:>12}: support={r['support']} YOLO自分={r['yolo_class_acc']}"
                      f" 介入率={r['override_rate']}  {tag}")
            print(f"可畢業：{g['graduated']}")
            if a.write:
                write_graduated(g["graduated"])
            else:
                print("[graduate] （唯讀；加 --write 才寫回 config）")
    print(f"[distill] KPI -> {a.out}")
    return m


if __name__ == "__main__":
    main()
