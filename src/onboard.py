"""
onboard.py — 一鍵把「你自己的偵測資料集」接上安全網（通用 Bring-Your-Own-Dataset 入口）。

把原本散落的步驟（轉檔 / 切分 / 裁 Golden / 建特徵庫 / 校準 / 可選訓練）收成一條流程：
  1. 標準化成 YOLO 切分 train/val/holdout（VOC 自動轉；YOLO 直接切）。
  2. 寫 config（類別、DINO 規格、權重）。
  3. 從 train GT 裁每類 Golden（SAM 去背）-> 建 DINO 特徵庫。
  4.（可選）沒權重時訓練 YOLO11n；或直接用你提供的 .pt。
  5. 有權重 -> 在 val 上自動校準 cosine 閾值。

用法：
  # VOC 資料（含既有 YOLO 權重，免訓練，最快）
  python src/onboard.py --voc DIR --name mydata --weights best.pt
  # VOC 資料，順便訓練一個觸發器（CPU 慢）
  python src/onboard.py --voc DIR --name mydata --train --epochs 20
  # 已是 YOLO 格式（images/ + labels/ + classes.txt 或 data.yaml）
  python src/onboard.py --yolo DIR --name mydata --weights best.pt
  完成後：python src/eval_real.py --data Workspace/<name>/holdout   # 真實評測
          python src/distill.py   --data Workspace/<name>/holdout   # 蒸餾 KPI
"""
from __future__ import annotations
import sys, argparse, random, shutil
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, SamSegmenter, ROOT

WS = ROOT / "Workspace"
CONFIG = ROOT / "config.yaml"


def _write_config(names, dino, weights):
    c = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    c["classes"] = list(names)
    c.setdefault("models", {})["dino_cpu"] = dino; c["models"]["dino_gpu"] = dino
    c.setdefault("yolo", {})["weights"] = (str(Path(weights).resolve()) if weights else None)
    c.setdefault("matching", {}).setdefault("threshold", 0.35)
    c["matching"].pop("thresholds", None); c["matching"].pop("graduated_classes", None)
    yaml.safe_dump(c, open(CONFIG, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)


def _split_yolo(src, out, split, seed=42):
    """已是 YOLO 格式的目錄 -> 切 train/val/holdout。回傳 (names, stats)。"""
    from setup_build import list_pairs, resolve_class_names
    src = Path(src); out = Path(out)
    names = resolve_class_names(src)
    if not names:
        raise ValueError(f"在 {src} 找不到 data.yaml / classes.txt（無法得知類別名）")
    pairs = [(i, l) for i, l in list_pairs(src) if l]
    if not pairs:
        raise ValueError(f"在 {src} 找不到 (影像, YOLO 標註) 配對")
    tr, va, ho = [int(x) for x in split.split(",")]
    rng = random.Random(seed); rng.shuffle(pairs)
    parts = {"holdout": pairs[:ho], "val": pairs[ho:ho + va],
             "train": (pairs[ho + va:][:tr] if tr else pairs[ho + va:])}
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for sp, items in parts.items():
        io = out / sp / "images"; lo = out / sp / "labels"
        io.mkdir(parents=True, exist_ok=True); lo.mkdir(parents=True, exist_ok=True)
        nb = 0
        for img, lbl in items:
            shutil.copy(img, io / Path(img).name)
            shutil.copy(lbl, lo / f"{Path(img).stem}.txt")
            nb += len([ln for ln in Path(lbl).read_text().splitlines() if ln.strip()])
        (out / sp / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        stats[sp] = (len(items), nb)
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return names, stats


def onboard(voc=None, yolo=None, name="dataset", weights=None, do_train=False,
            epochs=20, imgsz=512, dino="dinov2_vits14", per_class=20, use_sam=True,
            split="300,80,150"):
    out = WS / name
    # 1) 標準化成 YOLO 切分
    if voc:
        from voc2yolo import convert
        tr, va, ho = [int(x) for x in split.split(",")]
        out, names, stats = convert(voc, out, train=tr, val=va, holdout=ho)
        stats = {k: v for k, v in stats.items()}
    else:
        names, stats = _split_yolo(yolo, out, split)
    print(f"[onboard] 類別 {names}")
    for s, v in stats.items():
        print(f"   {s:>8}: {v[0]} 圖 / {v[1]} 框")

    # 2) config
    _write_config(names, dino, weights)
    cfg = load_config(); sel = resolve_env(cfg)

    # 3) Golden + bank
    from setup_build import extract_golden, list_pairs
    sam = SamSegmenter(sel["sam"], sel["device"], cfg) if use_sam else None
    counts = extract_golden(list_pairs(out / "train"), names, sam, per_class=per_class,
                            progress=lambda *x: None)
    print(f"[onboard] Golden 每類張數: {counts}")
    if sum(counts.values()) == 0:
        raise ValueError("沒有取得任何 Golden（檢查標註與類別）")
    import build_bank
    build_bank.build(cfg)

    # 4) 可選訓練
    if not weights and do_train:
        from retrain_loop import retrain
        print(f"[onboard] 無權重 -> 訓練 YOLO11n（CPU 慢；epochs={epochs} imgsz={imgsz}）")
        best = retrain("yolo11n.pt", str(out / "data.yaml"), epochs, imgsz,
                       cfg["yolo"]["batch"], sel["device"], cfg["seed"], name)
        weights = str(best); _write_config(names, dino, weights); cfg = load_config()
        print(f"[onboard] 訓練完成 -> {best}")

    # 5) 校準（有權重才有意義）
    if weights:
        from pipeline import SafetyNet
        from setup_build import calibrate_from_gt
        net = SafetyNet(cfg, weights)
        t = calibrate_from_gt(net, list_pairs(out / "val"), progress=lambda *x: None)
        if t is not None:
            c = yaml.safe_load(open(CONFIG, encoding="utf-8"))
            c["matching"]["threshold"] = round(float(t), 4)
            yaml.safe_dump(c, open(CONFIG, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
            print(f"[onboard] 已校準閾值 -> {round(float(t), 4)}")
        else:
            print("[onboard] 校準資料不足，保持預設閾值（可日後 src/pipeline.py 校準）")
    else:
        print("[onboard] 未提供權重、也未 --train：略過校準。")
        print("          請設定 config.yaml 的 yolo.weights 指向你的 .pt，或重跑加 --train。")

    print("\n[onboard] ✅ 完成！接著可以：")
    print(f"   python src/eval_real.py --data {out.as_posix()}/holdout   # 真實 hold-out 評測")
    print(f"   python src/distill.py   --data {out.as_posix()}/holdout   # 蒸餾 KPI(介入率)")
    print(f"   python run.py gui                                          # 視覺化 Inspector")
    return out, names


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--voc", help="Pascal VOC 目錄（images/ + annotations/）")
    g.add_argument("--yolo", help="YOLO 目錄（images/ + labels/ + classes.txt 或 data.yaml）")
    ap.add_argument("--name", default="dataset", help="輸出資料集名 -> Workspace/<name>")
    ap.add_argument("--weights", default=None, help="你的 YOLO .pt（有就校準）")
    ap.add_argument("--train", dest="do_train", action="store_true", help="沒權重時訓練 YOLO11n")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--per-class", dest="per_class", type=int, default=20)
    ap.add_argument("--no-sam", dest="use_sam", action="store_false", help="Golden 不去背(較快)")
    ap.add_argument("--split", default="300,80,150", help="train,val,holdout 數量")
    a = ap.parse_args()
    onboard(a.voc, a.yolo, a.name, a.weights, a.do_train, a.epochs, a.imgsz,
            a.dino, a.per_class, a.use_sam, a.split)


if __name__ == "__main__":
    main()
