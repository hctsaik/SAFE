"""
retrain_loop.py — #7 半自動閉環重訓（Data Flywheel 的引擎）：把安全網的判決蒸餾回 YOLO。

唯一真正「動到 YOLO 權重」的功能。流程：
  1. autolabel：對未標註池(Check/真實圖) 產生「高信度偽正樣本」(SAM 收緊框) + 「攔截→純負樣本圖」。
  2. 組訓練集：偽標(+可選合成 train) 當 train；真實 hold-out 當 val（類別塌縮成單類 'object' 量觸發器 mAP）。
  3. 重訓：自現有 best.pt 暖啟動，訓練候選權重。
  4. 裁判：在 hold-out 比 mAP50 —— 候選 vs 現役。
  5. 煞車：候選 mAP 未超過現役 + min_gain → 不換（rollback）；超過才升級 best.pt（舊權重備份）。

為何這樣設計（呼應辯論的 confirmation-bias 風險）：
  - 只用高信度偽標(autolabel 的 gating)；灰帶交給 active_learning 找人，不進訓練。
  - 每輪必過 hold-out mAP 煞車，沒進步就不換 -> 飛輪不會把錯誤越滾越大。

用法：
  python src/retrain_loop.py --holdout path/to/holdout --epochs 25
  python src/retrain_loop.py --pool_dir Workspace/Check --holdout HO --include_synthetic
  python src/retrain_loop.py --holdout HO --dry_run     # 只組資料集+比基準，不重訓
"""
from __future__ import annotations
import sys, shutil, argparse, json, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, ROOT
from pipeline import SafetyNet, DEFAULT_YOLO
from autolabel import autolabel_dir
from eval_real import _gather
from setup_build import read_yolo_txt

WS = ROOT / "Workspace"
RUNS = WS / "runs"
FLY = WS / "flywheel"
SCENES_TRAIN = WS / "scenes" / "train"


def _copy_pairs_remap(pairs, img_out, lbl_out, remap=None):
    """把 (img,label) 複製到目標並重映射類別索引。
    remap=None -> 全塌縮為 0（單類觸發器，mAP 只看定位）；
    remap=dict{src:dst或None} -> 多類別：依名稱映射到資料集類別序，None 表略過該框。"""
    img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img, lbl in pairs:
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        rows = read_yolo_txt(lbl, W, H) if lbl else []
        lines = []
        for c, (x1, y1, x2, y2), _ in rows:
            dst = 0 if remap is None else remap.get(c)
            if dst is None:
                continue
            cx = ((x1 + x2) / 2) / W; cy = ((y1 + y2) / 2) / H
            lines.append(f"{dst} {cx:.6f} {cy:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        shutil.copy(img, img_out / Path(img).name)
        (lbl_out / f"{Path(img).stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


def _neg_weight(dets):
    """R7：由該圖『被攔截框』的 wrongness=conf×(thr-score) 算 0~1 負樣本重要度（YOLO 越自信地犯錯越大）。"""
    w = [d["conf"] * max(0.0, d.get("thr", 0.0) - d["score"])
         for d in dets if d.get("reason") == "intercepted"]
    return min(1.0, (max(w) if w else 0.0) / 0.30)


def _img_confidence(dets):
    """R3：由該圖『被匯出的偽正樣本框』的 (score-thr) 與 margin 算 0~1 信度。"""
    exp = [d for d in dets if d.get("exported")]
    if not exp:
        return 0.0
    vals = []
    for d in exp:
        s_pad = min(1.0, max(0.0, d["score"] - d.get("thr", 0.0)) / 0.15)
        m = min(1.0, max(0.0, d.get("margin", 1.0)) / 0.20)
        vals.append(0.5 * s_pad + 0.5 * m)
    return sum(vals) / len(vals)


def assemble_dataset(net, pool_dir, holdout_pairs, out_dir, include_synthetic=False,
                     min_score_pad=0.05, min_margin=0.05, max_repeat=3,
                     multiclass=False, holdout_names=None, hardneg_weight=1,
                     tta=False, min_box_iou=0.0, cotrain=False, progress=None):
    """組資料集：train=偽標(+合成)、val=hold-out。回傳 (data_yaml, stats)。
    R3 信度加權課程：高信度偽標的影像在 train 重複放更多份（max_repeat）-> 軟監督(重取樣)。
    R7 硬負樣本加權：含「YOLO 自信誤報」的純負樣本影像放更多份（hardneg_weight）-> 修正盲點。
    R8/R9：tta/min_box_iou/cotrain 透傳 autolabel，產更乾淨的偽標。
    multiclass=True（R2a）：偽標用 DINO 類別(net.classes)，val 依名稱映射到 net.classes。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_tr = out_dir / "images" / "train"; lbl_tr = out_dir / "labels" / "train"
    img_va = out_dir / "images" / "val"; lbl_va = out_dir / "labels" / "val"
    classes = list(net.classes)

    # 1) 偽標 -> 暫存，再依信度加權併入 train（多類別用 DINO 類別索引）
    tmp_auto = out_dir / "_auto"
    summ = autolabel_dir(net, pool_dir, tmp_auto, min_score_pad, min_margin,
                         single_class=not multiclass, positives_only=False,
                         tta=tta, min_box_iou=min_box_iou, cotrain=cotrain, progress=progress)
    conf_of = {pi["image"]: _img_confidence(pi["dets"]) for pi in summ["per_image"]}
    neg_of = {pi["image"]: _neg_weight(pi["dets"]) for pi in summ["per_image"]}
    img_tr.mkdir(parents=True, exist_ok=True); lbl_tr.mkdir(parents=True, exist_ok=True)
    n_auto = 0; n_copies = 0
    for p in (tmp_auto / "images").glob("*"):
        lp = tmp_auto / "labels" / f"{p.stem}.txt"
        label = lp.read_text() if lp.exists() else ""
        if label.strip():
            copies = 1 + round((max_repeat - 1) * conf_of.get(p.name, 0.0))   # R3 正樣本信度加權
        else:
            copies = 1 + round((hardneg_weight - 1) * neg_of.get(p.name, 0.0))  # R7 硬負樣本加權
        for k in range(copies):
            stem = p.stem if k == 0 else f"{p.stem}_r{k}"
            shutil.copy(p, img_tr / f"{stem}{p.suffix}")
            (lbl_tr / f"{stem}.txt").write_text(label)
            n_copies += 1
        n_auto += 1
    shutil.rmtree(tmp_auto)

    # 2) 可選：併入既有合成 train（單類定位）。多類別時合成標註為單類，無法對應 -> 略過。
    n_syn = 0
    if include_synthetic and not multiclass and (SCENES_TRAIN / "images").is_dir():
        for p in (SCENES_TRAIN / "images").glob("*.jpg"):
            shutil.copy(p, img_tr / p.name)
            lp = SCENES_TRAIN / "labels" / f"{p.stem}.txt"
            if lp.exists():
                shutil.copy(lp, lbl_tr / lp.name); n_syn += 1
    elif include_synthetic and multiclass:
        print("[flywheel] 多類別模式：合成 train 為單類標註無法對應類別 -> 略過併入")

    # 3) val = hold-out：單類塌縮 或 多類別依名稱映射到 net.classes
    if multiclass:
        hn = holdout_names or classes
        remap = {i: (classes.index(nm) if nm in classes else None) for i, nm in enumerate(hn)}
        n_val = _copy_pairs_remap(holdout_pairs, img_va, lbl_va, remap)
        names_block = "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
    else:
        n_val = _copy_pairs_remap(holdout_pairs, img_va, lbl_va, None)
        names_block = "  0: object\n"

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n{names_block}", encoding="utf-8")
    stats = dict(train_auto=n_auto, train_copies=n_copies, train_synthetic=n_syn, val=n_val,
                 exported_pos=summ["exported"], intercepted=summ["intercepted"],
                 uncertain=summ["uncertain"], multiclass=multiclass)
    return data_yaml, stats


def yolo_val(weights, data_yaml, device):
    """在 data_yaml 的 val 上量 mAP，回傳 (map50, per_class{name:map50-95})。"""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(data_yaml), device=device, verbose=False, plots=False)
    names = getattr(r, "names", None) or getattr(m, "names", {}) or {}
    per_class = {}
    try:
        for i, ap in enumerate(r.box.maps):           # 每類 mAP50-95
            per_class[str(names.get(i, i))] = round(float(ap), 4)
    except Exception:
        pass
    return float(r.box.map50), per_class


def yolo_map(weights, data_yaml, device):
    return yolo_val(weights, data_yaml, device)[0]


# ----------------------------- R5 三面向升級閘 -----------------------------
def gate(base, cand, min_gain=0.0, tol=0.02):
    """三面向都不退步才升級（任一惡化即 rollback）。base/cand 為各自的指標 dict：
      {map50, per_class, anchor_map50, net_precision, recall_net}
    回傳 (promote: bool, axes: dict 逐項通過與否+原因)。"""
    axes = {}
    # 軸1 hold-out 整體 + per-class（任一類退步 > tol 即擋）
    a1 = cand["map50"] >= base["map50"] + min_gain
    regressed = [c for c in base.get("per_class", {})
                 if cand.get("per_class", {}).get(c, 0) < base["per_class"][c] - tol]
    a1 = a1 and not regressed
    axes["holdout_map"] = dict(ok=bool(a1), base=round(base["map50"], 4),
                               cand=round(cand["map50"], 4), regressed_classes=regressed)
    # 軸2 錨點集（防災難性遺忘）
    if base.get("anchor_map50") is not None and cand.get("anchor_map50") is not None:
        a2 = cand["anchor_map50"] >= base["anchor_map50"] - tol
        axes["anchor"] = dict(ok=bool(a2), base=round(base["anchor_map50"], 4),
                              cand=round(cand["anchor_map50"], 4))
    else:
        a2 = True; axes["anchor"] = dict(ok=True, note="無錨點集，略過")
    # 軸3 端到端安全網（介入後 precision/recall 不退步）
    if base.get("net_precision") is not None and cand.get("net_precision") is not None:
        a3 = (cand["net_precision"] >= base["net_precision"] - tol and
              cand["recall_net"] >= base["recall_net"] - tol)
        axes["end_to_end"] = dict(ok=bool(a3),
                                  base=(base["net_precision"], base["recall_net"]),
                                  cand=(cand["net_precision"], cand["recall_net"]))
    else:
        a3 = True; axes["end_to_end"] = dict(ok=True, note="未量端到端，略過")
    return bool(a1 and a2 and a3), axes


def retrain(base, data_yaml, epochs, imgsz, batch, device, seed, name):
    from ultralytics import YOLO
    m = YOLO(str(base))
    m.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
            project=str(RUNS), name=name, exist_ok=True, verbose=False, plots=False,
            seed=seed, patience=0)
    return RUNS / name / "weights" / "best.pt"


def _measure(weights, cfg, sel, data_yaml, holdout_pairs, names, anchor_yaml):
    """量一組權重的三面向指標：hold-out mAP(+per-class)、錨點 mAP、端到端安全網(R1)。"""
    from distill import distill_metrics
    map50, per_class = yolo_val(weights, data_yaml, sel["device"])
    anchor_map = yolo_map(weights, anchor_yaml, sel["device"]) if anchor_yaml else None
    net = SafetyNet(cfg, str(weights))
    dm = distill_metrics(net, holdout_pairs, names)
    return dict(map50=map50, per_class=per_class, anchor_map50=anchor_map,
                net_precision=dm["net_precision"], recall_net=dm["recall_net"],
                distill=dm)


# ----------------------------- R6 凍結教師 + 飛輪帳本 -----------------------------
LEDGER = FLY / "flywheel_ledger.json"


def teacher_fingerprint(cfg):
    """教師指紋（golden 庫 + DINO + 閾值）。多輪迴圈中教師必須凍結，靠此偵測漂移。"""
    import hashlib
    bank = WS / "vector_bank.npz"
    sha = hashlib.sha1(bank.read_bytes()).hexdigest()[:12] if bank.exists() else None
    return dict(bank_sha=sha, dino=cfg["models"]["dino_cpu"],
                threshold=cfg["matching"]["threshold"],
                thresholds=cfg["matching"].get("thresholds") or {})


def append_ledger(rec):
    """逐輪追加血緣紀錄（可審計、可 rollback 到任一輪的權重）。"""
    data = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    data.append(rec)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _flywheel_round(cfg, sel, pool_dir, holdout_pairs, names, anchor_yaml, current_w,
                    current_m, epochs, include_synthetic, min_gain, tol, dry_run,
                    round_idx, teacher_fp, multiclass=False, hardneg_weight=1,
                    tta=False, min_box_iou=0.0, cotrain=False):
    """單輪：以 current_w 再挖掘偽標 -> 重訓 -> 三面向煞車 -> promote/rollback。
    回傳 (record, new_current_w, new_current_m, promoted)。"""
    print(f"\n{'='*64}\n=== 飛輪 Round {round_idx} | 現役 {Path(current_w).name}"
          f"{' | 多類別軌' if multiclass else ''} ===")
    net = SafetyNet(cfg, str(current_w))          # 再挖掘：用「當前最佳 YOLO」產偽標
    data_yaml, stats = assemble_dataset(net, pool_dir, holdout_pairs, FLY / "dataset",
        include_synthetic=include_synthetic, multiclass=multiclass, holdout_names=names,
        hardneg_weight=hardneg_weight, tta=tta, min_box_iou=min_box_iou, cotrain=cotrain,
        progress=lambda i, n, nm: print(f"[flywheel] autolabel {i+1}/{n} {nm}"))
    print(f"[flywheel] 資料集：train(偽標 {stats['train_auto']}->重取樣 {stats['train_copies']} "
          f"+ 合成 {stats['train_synthetic']}) val {stats['val']}　偽正 {stats['exported_pos']} "
          f"攔截 {stats['intercepted']} 灰帶 {stats['uncertain']}(->active learning)")
    if current_m is None:                          # 第一輪才量現役基準（之後沿用，省算力）
        current_m = _measure(current_w, cfg, sel, data_yaml, holdout_pairs, names, anchor_yaml)
    print(f"[flywheel] 現役：mAP50={current_m['map50']:.4f} 錨點={current_m['anchor_map50']} "
          f"介入率={current_m['distill']['intervention_rate']} "
          f"端到端P/R={current_m['net_precision']}/{current_m['recall_net']}")
    rec = dict(round=round_idx, time=time.strftime("%Y-%m-%d %H:%M"), teacher=teacher_fp,
               current_weights=str(current_w), epochs=epochs, stats=stats, base=current_m)
    if dry_run:
        print("[flywheel] --dry_run：只組資料集+量基準三面向，不重訓。")
        rec["action"] = "dry_run"; append_ledger(rec)
        return rec, current_w, current_m, False

    cand = retrain(current_w, data_yaml, epochs, cfg["yolo"]["imgsz"], cfg["yolo"]["batch"],
                   sel["device"], cfg["seed"], name="yolo_flywheel")
    cand_m = _measure(cand, cfg, sel, data_yaml, holdout_pairs, names, anchor_yaml)
    promote, axes = gate(current_m, cand_m, min_gain, tol)
    rec["candidate"] = cand_m; rec["gate"] = axes
    print(f"[flywheel] 候選：mAP50={cand_m['map50']:.4f} 錨點={cand_m['anchor_map50']} "
          f"介入率={cand_m['distill']['intervention_rate']} "
          f"端到端P/R={cand_m['net_precision']}/{cand_m['recall_net']}")
    for ax, v in axes.items():
        print(f"   閘[{ax}] {'✅通過' if v['ok'] else '❌退步'}: "
              f"{ {k: val for k, val in v.items() if k != 'ok'} }")
    if promote:
        vdir = DEFAULT_YOLO.parent; vdir.mkdir(parents=True, exist_ok=True)
        saved = vdir / f"flywheel_r{round_idx}_{int(time.time())}.pt"
        shutil.copy(cand, saved)                   # 版本化備份（可 rollback 到本輪）
        shutil.copy(cand, DEFAULT_YOLO)            # 升級為現役
        print(f"[flywheel][PROMOTE] 三面向皆不退步 -> 升級；版本備份 {saved.name}")
        rec["action"] = "promote"; rec["promoted_to"] = str(DEFAULT_YOLO)
        rec["versioned"] = str(saved)
        append_ledger(rec)
        return rec, DEFAULT_YOLO, cand_m, True
    print("[flywheel][ROLLBACK] 至少一面向退步 -> 保留現役（煞車生效）")
    rec["action"] = "rollback"; append_ledger(rec)
    return rec, current_w, current_m, False


def _plateau(history, patience, min_delta):
    """近 patience 輪的最佳 mAP 沒再進步 min_delta -> 平台期。"""
    if len(history) <= patience:
        return False
    return max(history[-patience:]) < max(history[:-patience]) + min_delta


def run_flywheel(pool_dir, holdout, epochs=25, include_synthetic=False, min_gain=0.0,
                 tol=0.02, anchor=None, dry_run=False, weights=None,
                 rounds=1, patience=2, min_delta=0.005, multiclass=False,
                 hardneg_weight=1, tta=False, min_box_iou=0.0, cotrain=False):
    """R4 多輪飛輪：每輪用『當前最佳 YOLO』再挖掘偽標 -> 重訓 -> 三面向煞車 -> 升級/回退。
    停止條件：達 rounds、連續 patience 輪未升級、或 mAP 進入平台期。
    multiclass=True（R2a）：訓練「會分類的多類別 YOLO」，蒸餾後可搭配 distill --graduate 逐類畢業。"""
    cfg = load_config(); sel = resolve_env(cfg)
    names, holdout_pairs = _gather(data=holdout) if Path(holdout).exists() else (None, [])
    if not holdout_pairs:
        print("[flywheel][FATAL] hold-out 無標註影像；閉環需要真實裁判 -> 中止"); return None
    if names is None:
        names = list(SafetyNet(cfg, weights).classes)
    current_w = Path(weights or cfg["yolo"].get("weights") or DEFAULT_YOLO)
    anchor_yaml = anchor or (str(WS / "scenes" / "safetynet.yaml")
                             if (WS / "scenes" / "safetynet.yaml").exists() else None)
    teacher_fp = teacher_fingerprint(cfg)
    if dry_run:
        rounds = 1
    print(f"[flywheel] 起始權重 {current_w}　hold-out {len(holdout_pairs)} 張　錨點={anchor_yaml}")
    print(f"[flywheel] 教師指紋(凍結) {teacher_fp['bank_sha']} dino={teacher_fp['dino']}　"
          f"rounds={rounds} patience={patience}")

    current_m = None; history = []; rollbacks = 0; rounds_done = 0
    for r in range(1, rounds + 1):
        # 教師漂移檢查（多輪中 golden 庫不應改變）
        fp_now = teacher_fingerprint(cfg)
        if fp_now["bank_sha"] != teacher_fp["bank_sha"]:
            print("[flywheel][WARN] 偵測到教師(golden 庫)漂移！蒸餾失去固定錨 -> 中止。"); break
        rec, current_w, current_m, promoted = _flywheel_round(
            cfg, sel, pool_dir, holdout_pairs, names, anchor_yaml, current_w, current_m,
            epochs, include_synthetic, min_gain, tol, dry_run, r, teacher_fp, multiclass,
            hardneg_weight, tta, min_box_iou, cotrain)
        rounds_done = r
        history.append(current_m["map50"])
        if dry_run:
            break
        rollbacks = 0 if promoted else rollbacks + 1
        if rollbacks >= patience:
            print(f"[flywheel][STOP] 連續 {rollbacks} 輪未升級 -> 停止。"); break
        if _plateau(history, patience, min_delta):
            print(f"[flywheel][STOP] mAP 進入平台期(近 {patience} 輪 < +{min_delta}) -> 停止。"); break

    summary = dict(rounds_done=rounds_done, final_weights=str(current_w),
                   map50_history=[round(h, 4) for h in history],
                   final_intervention=current_m["distill"]["intervention_rate"] if current_m else None,
                   ledger=str(LEDGER))
    (FLY / "flywheel_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[flywheel] 共 {rounds_done} 輪　mAP50 軌跡 {summary['map50_history']}　"
          f"最終介入率 {summary['final_intervention']}")
    print(f"[flywheel] 帳本 -> {LEDGER}　摘要 -> {FLY / 'flywheel_report.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default=str(WS / "Check"), help="未標註池（產偽標）")
    ap.add_argument("--holdout", required=True, help="真實 hold-out（裁判，需 YOLO 標註）")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--include_synthetic", action="store_true", help="併入既有合成 train")
    ap.add_argument("--min_gain", type=float, default=0.0, help="候選需超過現役 mAP 此幅度才升級")
    ap.add_argument("--tol", type=float, default=0.02, help="各面向容許的退步幅度（超過即 rollback）")
    ap.add_argument("--anchor", default=None, help="錨點集 data.yaml（防遺忘；預設用合成 train）")
    ap.add_argument("--rounds", type=int, default=1, help="R4 迭代輪數（每輪用最新權重再挖掘）")
    ap.add_argument("--patience", type=int, default=2, help="連續幾輪未升級就停止")
    ap.add_argument("--min_delta", type=float, default=0.005, help="平台期判定：mAP 最小進步幅度")
    ap.add_argument("--multiclass", action="store_true",
                    help="R2a：訓練會分類的多類別 YOLO（用 DINO 類別當偽標）")
    ap.add_argument("--hardneg_weight", type=int, default=1,
                    help="R7：含『YOLO 自信誤報』的純負樣本影像最多放幾份（1=不加權）")
    ap.add_argument("--tta", action="store_true", help="R8：教師多視角平均嵌入(更穩)")
    ap.add_argument("--min_box_iou", type=float, default=0.0,
                    help="R8：偽標需 YOLO 框與 SAM 框 IoU>=此值（定位可信）")
    ap.add_argument("--cotrain", action="store_true",
                    help="R9：兩視角一致(類別+穩定)才當偽標（最乾淨）")
    ap.add_argument("--dry_run", action="store_true", help="只組資料集+量基準三面向，不重訓")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    run_flywheel(a.pool_dir, a.holdout, a.epochs, a.include_synthetic, a.min_gain,
                 a.tol, a.anchor, a.dry_run, a.weights, a.rounds, a.patience,
                 a.min_delta, a.multiclass, a.hardneg_weight, a.tta, a.min_box_iou, a.cotrain)


if __name__ == "__main__":
    main()
