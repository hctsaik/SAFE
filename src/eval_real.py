"""
eval_real.py — #6 真實資料 Hold-out 評測：在「不是合成產生器生的」真實標註資料上量測安全網。

為何缺它不行（QA Reviewer 的核心批評）：evaluate.py 的高分是「同一個合成產生器自考自」，
無法證明真實泛化。本工具用真實 YOLO 標註的 hold-out 跑同一張評分卡，給出可信的真實指標，
並作為 #7 半自動閉環重訓的「裁判」——重訓後 mAP/分數有沒有在真實資料上真的變好。

輸入：標準 YOLO 目錄（images/ + labels/ + data.yaml 或 classes.txt）。
  GT 框 = 真目標（含類別）；YOLO 偵測到但不命中任何 GT 的框 = 真實 False Alarm（應被攔截）。
評分卡（與 evaluate.py 一致，WEIGHTS 直接沿用）：
  攔截率(35) + 保留率(30) + 類別正確率(25) + SAM 去背成功率(10)。

用法：
  python src/eval_real.py --data path/to/holdout            # images/labels/data.yaml
  python src/eval_real.py --images imgs/ --labels lbls/ --names screw,metal_nut
  python src/eval_real.py --data holdout --hi 0.5 --lo 0.1  # IoU 命中/背景門檻
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT, iou_xyxy, classify_box
from pipeline import SafetyNet
from evaluate import WEIGHTS
from setup_build import read_yolo_txt, resolve_class_names
from yolo_quickstart import find_label, iter_images, parse_names
import yaml

WS = ROOT / "Workspace"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _gather(data=None, images=None, labels=None, names=None):
    """回傳 (names, [(img_path, label_path)])。支援 data.yaml 或 images+labels。"""
    if data:
        d = Path(data)
        if d.is_dir():
            ny = resolve_class_names(d)
            names = names or ny
            img_root = d / "images" if (d / "images").is_dir() else d
            pairs = [(p, find_label(p)) for p in iter_images([img_root])]
        else:  # data.yaml
            y = yaml.safe_load(open(d, encoding="utf-8"))
            names = names or parse_names(y["names"])
            base = Path(y.get("path") or d.parent)
            if not base.is_absolute():
                base = (d.parent / base).resolve()
            pairs = []
            for k in ("val", "train"):
                if y.get(k):
                    pairs += [(p, find_label(p)) for p in iter_images([base / y[k]])]
    else:
        pairs = [(p, find_label(p)) for p in iter_images([Path(images)])]
    return names, [(i, l) for i, l in pairs if l]


def evaluate_real(net, pairs, names, hi=0.5, lo=0.1):
    """對真實 hold-out 跑安全網並量測評分卡。GT 類別索引以 names 映射到類別名。"""
    name_of = {i: n for i, n in enumerate(names)}
    TP = FP = FN = TN = 0
    class_correct = class_total = ambiguous = 0
    sam_ok = sam_total = 0; recovered = 0; n_objects = 0
    per_class = {n: dict(tp=0, fn=0, correct=0) for n in names}
    for img_path, lbl in pairs:
        img = __import__("cv2").imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_yolo_txt(lbl, W, H)  # [(cls_idx, box, conf)]
        targets = [(b, name_of.get(c, str(c))) for c, b, _ in gt]
        n_objects += len(targets)
        obj_hit = [False] * len(targets)
        for r in net.process(img):
            kind, tcls = classify_box(r["box"], targets, [], hi=hi, lo=lo)
            if kind == "ambiguous":
                ambiguous += 1; continue
            sam_total += 1; sam_ok += int(r["sam_ok"])
            passed = r["decision"] == "True"
            if kind == "target":
                if passed:
                    TP += 1; class_total += 1
                    ok = (r["pred_class"] == tcls)
                    class_correct += int(ok)
                    if tcls in per_class:
                        per_class[tcls]["tp"] += 1; per_class[tcls]["correct"] += int(ok)
                    if ok:
                        j = int(np.argmax([iou_xyxy(r["box"], b) for b, _ in targets]))
                        obj_hit[j] = True
                else:
                    FN += 1
                    if tcls in per_class:
                        per_class[tcls]["fn"] += 1
            else:  # bg（不命中任何 GT）-> 真實誤報，應攔截
                if passed: FP += 1
                else:      TN += 1
        recovered += sum(obj_hit)
    interception = TN / (TN + FP) if (TN + FP) else 1.0
    retention = TP / (TP + FN) if (TP + FN) else 1.0
    class_acc = class_correct / class_total if class_total else 1.0
    sam_succ = sam_ok / sam_total if sam_total else 1.0
    precision = TP / (TP + FP) if (TP + FP) else 1.0
    e2e_recall = recovered / n_objects if n_objects else 1.0
    score = (WEIGHTS["interception"]*interception + WEIGHTS["retention"]*retention +
             WEIGHTS["class_acc"]*class_acc + WEIGHTS["sam"]*sam_succ)
    return dict(score=round(score, 2), interception=round(interception, 3),
                retention=round(retention, 3), class_acc=round(class_acc, 3),
                sam=round(sam_succ, 3), precision=round(precision, 3),
                e2e_recall=round(e2e_recall, 3),
                counts=dict(TP=TP, FP=FP, FN=FN, TN=TN, ambig=ambiguous,
                            dets=sam_total, objects=n_objects),
                per_class=per_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="YOLO 目錄或 data.yaml")
    ap.add_argument("--images", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--names", default=None, help="逗號分隔；缺則讀 data.yaml/classes.txt")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--out", default=str(WS / "eval_real_report.json"))
    a = ap.parse_args()
    names = [s.strip() for s in a.names.split(",")] if a.names else None
    names, pairs = _gather(a.data, a.images, a.labels, names)
    if not pairs:
        print("[eval-real] 找不到含標註的影像（需 images + 平行 labels/）"); return
    net = SafetyNet(load_config(), a.weights)
    if names is None:
        names = list(net.classes)
    print(f"[eval-real] {len(pairs)} 張標註影像　類別={names}")
    m = evaluate_real(net, pairs, names, a.hi, a.lo)
    Path(a.out).write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n========== Real Hold-out 評測 ==========")
    print(f"分數 {m['score']}/100　攔截率 {m['interception']}　保留率 {m['retention']}　"
          f"類別正確 {m['class_acc']}　SAM {m['sam']}")
    print(f"Precision {m['precision']}　端到端 Recall {m['e2e_recall']}　{m['counts']}")
    for n, pc in m["per_class"].items():
        print(f"    {n:>12}: TP={pc['tp']} FN={pc['fn']} 類別正確={pc['correct']}")
    print(f"[eval-real] 報告 -> {a.out}")
    return m


if __name__ == "__main__":
    main()
