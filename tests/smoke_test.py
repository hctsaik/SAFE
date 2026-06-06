"""
smoke_test.py — 免模型、零外部相依的純邏輯煙霧測試（CI 友善）。

只測「不需載 YOLO/SAM/DINO」的核心邏輯（閾值校準、margin、三面向煞車、信度/硬負加權、
逐類畢業、多邊形轉換、類別重映射、雙軌畢業判定…），秒級跑完，證明關鍵決策邏輯正確。
模型相關功能請用 README 的實跑指令（onboard / eval_real / distill）驗證。

執行：  python tests/smoke_test.py
"""
from __future__ import annotations
import sys, tempfile, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: F401 — 觸發 UTF-8 stdout 設定（Windows cp950 相容）

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}")


# ---- pipeline: 閾值校準 + per-class _thr ----
def t_threshold():
    from pipeline import _best_f1_threshold, SafetyNet
    sc = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1]); real = np.array([1, 1, 1, 0, 0, 0], bool)
    t, f1 = _best_f1_threshold(sc, real, 0.5)
    check("calib: 可分集 F1≈1 且閾值居中", f1 > 0.99 and 0.2 < t <= 0.8)
    check("calib: 全負回退預設", _best_f1_threshold(np.array([.1, .2]), np.array([0, 0], bool), 0.42)[0] == 0.42)

    class Fake:
        threshold = 0.40; thresholds = {"screw": 0.55}; multiclass = True
        graduated = {"screw"}; grad_conf = 0.5
        _thr = SafetyNet._thr; _is_graduated = SafetyNet._is_graduated
    f = Fake()
    check("per-class _thr 命中/回退", f._thr("screw") == 0.55 and f._thr("x") == 0.40)
    check("R2 _is_graduated: 高conf畢業類", f._is_graduated("screw", 0.9))
    check("R2 _is_graduated: 低conf不畢業", not f._is_graduated("screw", 0.3))
    check("R2 _is_graduated: 非畢業類", not f._is_graduated("hazelnut", 0.9))


# ---- retrain_loop: gate / plateau / 加權 / remap ----
def t_flywheel():
    from retrain_loop import gate, _plateau, _neg_weight, _img_confidence, _copy_pairs_remap
    base = dict(map50=0.80, per_class={"object": 0.6}, anchor_map50=0.75,
                net_precision=0.9, recall_net=0.8)
    check("gate: 全進步->升級", gate(base, dict(map50=.85, per_class={"object": .65},
          anchor_map50=.76, net_precision=.92, recall_net=.82))[0] is True)
    check("gate: 錨點遺忘->擋", gate(base, dict(map50=.85, per_class={"object": .65},
          anchor_map50=.60, net_precision=.92, recall_net=.82))[0] is False)
    check("gate: per-class退步->擋", gate(base, dict(map50=.85, per_class={"object": .50},
          anchor_map50=.76, net_precision=.92, recall_net=.82))[0] is False)
    check("gate: 端到端掉precision->擋", gate(base, dict(map50=.85, per_class={"object": .65},
          anchor_map50=.76, net_precision=.80, recall_net=.82))[0] is False)
    check("plateau: 仍進步->False", _plateau([.8, .81, .83], 2, .005) is False)
    check("plateau: 平台->True", _plateau([.8, .85, .851, .852], 2, .005) is True)
    check("R7 neg_weight: 自信誤報高", _neg_weight([dict(reason="intercepted", conf=.9, thr=.4, score=.1)]) > 0.8)
    check("R7 neg_weight: 非攔截為0", _neg_weight([dict(reason="confident_true", conf=.9, thr=.4, score=.9)]) == 0.0)
    check("R3 img_confidence: 高信度→1", _img_confidence([dict(exported=True, score=.9, thr=.5, margin=.4)]) > 0.9)
    check("R3 img_confidence: 無偽正→0", _img_confidence([]) == 0.0)
    # remap（需 cv2 寫暫存）
    import cv2
    tmp = Path(tempfile.mkdtemp())
    try:
        img = tmp / "a.jpg"; cv2.imwrite(str(img), np.zeros((100, 100, 3), np.uint8))
        lbl = tmp / "a.txt"; lbl.write_text("0 .5 .5 .2 .2\n1 .3 .3 .1 .1\n2 .7 .7 .1 .1\n")
        _copy_pairs_remap([(img, lbl)], tmp / "io", tmp / "lo", remap={0: 1, 1: 0, 2: None})
        out = [l.split()[0] for l in (tmp / "lo" / "a.txt").read_text().splitlines()]
        check("R2a remap: 映射+None略過", out == ["1", "0"])
        _copy_pairs_remap([(img, lbl)], tmp / "io2", tmp / "lo2", remap=None)
        out2 = [l.split()[0] for l in (tmp / "lo2" / "a.txt").read_text().splitlines()]
        check("單類塌縮: 全為0", all(c == "0" for c in out2) and len(out2) == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- distill: 逐類畢業 ----
def t_graduation():
    from distill import graduation
    m = dict(multiclass=True, per_class={
        "screw": dict(support=40, yolo_class_acc=0.95, override_rate=0.05),
        "zipper": dict(support=40, yolo_class_acc=0.70, override_rate=0.30),
        "rare": dict(support=3, yolo_class_acc=1.0, override_rate=0.0)})
    g = graduation(m, 10, 0.9, 0.1)
    check("graduate: 只有 screw 畢業", g["graduated"] == ["screw"])
    check("graduate: 單類觸發器未就緒", graduation(dict(multiclass=False, per_class={}))["ready"] is False)


# ---- autolabel: 多邊形 ----
def t_polygon():
    from autolabel import _mask_to_polygon, _box_polygon
    m = np.zeros((100, 100), np.uint8); m[20:70, 30:80] = 1
    poly = _mask_to_polygon(m, 100, 100)
    xs = [x for x, y in poly]; ys = [y for x, y in poly]
    check("R10 mask->polygon: 界內且框住物件",
          poly and all(0 <= v <= 1 for v in xs + ys) and min(xs) < .4 and max(xs) > .7)
    check("R10 box_polygon 四角", _box_polygon((10, 10, 90, 90), 100, 100)[2] == (0.9, 0.9))
    check("R10 mask=None->None", _mask_to_polygon(None, 100, 100) is None)


def main():
    print("== SAFE 純邏輯 smoke test ==")
    for fn in (t_threshold, t_flywheel, t_graduation, t_polygon):
        print(f"[{fn.__name__}]")
        try:
            fn()
        except Exception as e:
            global FAIL; FAIL += 1; print(f"  ❌ {fn.__name__} 例外: {e}")
    print(f"\n結果：PASS={PASS}  FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
