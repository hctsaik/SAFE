"""
audit.py — #3 Dataset Auditing：用 DINOv2 嵌入體檢資料集，找出「會悄悄拖垮準度」的病。

Data Flywheel 會把資料越滾越多，髒資料滾得更快。進補前先體檢，揪出四種常見病：
  1. near-duplicate  近重複影像：灌水指標、訓練/驗證重疊造成假高分。
  2. leakage         訓練↔驗證洩漏：Golden 與 hold-out 幾乎同張 -> 評測樂觀失真。
  3. label-error     疑似標錯：某 Golden 離自己類別中心很遠、反而更像別類。
  4. balance/coverage 類別不均與覆蓋：樣本數懸殊、類內過度集中(視角單一)。

只載 DINOv2（不載 YOLO/SAM），對 Workspace/Training/[類別]/ 的 Golden 影像做未增強嵌入。
為何有效：去重/洩漏防止「假準」、標錯偵測防止特徵庫被毒化、覆蓋分析指出該補哪種樣本。

用法：
  python src/audit.py                              # 體檢 Workspace/Training
  python src/audit.py --train_dir path --holdout_dir path   # 另查 train↔holdout 洩漏
  python src/audit.py --dup_thr 0.97 --label_margin 0.0
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, DinoEmbedder, ROOT, imread

WS = ROOT / "Workspace"
TRAIN_DIR = WS / "Training"
HUB = str(ROOT / ".cache" / "torchhub")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _imgs_in(d):
    return sorted([p for p in Path(d).glob("*") if p.suffix.lower() in IMG_EXT])


def get_dino(cfg=None):
    cfg = cfg or load_config()
    sel = resolve_env(cfg)
    return DinoEmbedder(sel["dino"], sel["device"], hub_dir=HUB), sel


def embed_goldens(train_dir, dino, progress=None):
    """對每類 Golden 影像做未增強 DINO 嵌入。回傳 (V[N,D], labels, classes, paths)。"""
    train_dir = Path(train_dir)
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    vecs, labels, paths = [], [], []
    for ci, c in enumerate(classes):
        for p in _imgs_in(train_dir / c):
            vecs.append(dino.embed(imread(p))); labels.append(ci); paths.append(str(p))
        if progress:
            progress(ci + 1, len(classes), c)
    if not vecs:
        return np.zeros((0, dino.dim), np.float32), np.array([], int), classes, []
    return np.stack(vecs).astype(np.float32), np.array(labels, int), classes, paths


def _prototypes(V, labels, n_cls):
    P = np.zeros((n_cls, V.shape[1]), np.float32)
    for ci in range(n_cls):
        m = V[labels == ci]
        if len(m):
            v = m.mean(0); P[ci] = v / (np.linalg.norm(v) + 1e-9)
    return P


# ----------------------------- 四項檢查 -----------------------------
def find_near_duplicates(V, labels, classes, paths, dup_thr=0.97, top=30):
    S = V @ V.T
    n = len(V); iu = np.triu_indices(n, k=1)
    pairs = [(int(i), int(j), float(S[i, j])) for i, j in zip(*iu) if S[i, j] >= dup_thr]
    pairs.sort(key=lambda t: -t[2])
    out = []
    for i, j, s in pairs[:top]:
        out.append(dict(a=paths[i], b=paths[j], sim=round(s, 4),
                        cross_class=bool(labels[i] != labels[j]),
                        cls_a=classes[labels[i]], cls_b=classes[labels[j]]))
    cross = sum(1 for p in out if p["cross_class"])
    return dict(threshold=dup_thr, n_pairs=len(pairs), cross_class_pairs=cross, pairs=out)


def find_label_errors(V, labels, classes, paths, label_margin=0.0):
    """每個 Golden：比它對『自己類別質心』與『最近他類質心』的相似度。
    若更像他類（other - own > label_margin）-> 疑似標錯/壞樣本。"""
    P = _prototypes(V, labels, len(classes))
    suspects = []
    for i in range(len(V)):
        c = labels[i]
        sims = V[i] @ P.T
        own = float(sims[c])
        others = sims.copy(); others[c] = -1
        oc = int(np.argmax(others)); osim = float(others[oc])
        if osim - own > label_margin:
            suspects.append(dict(path=paths[i], labeled=classes[c], looks_like=classes[oc],
                                 own_sim=round(own, 4), other_sim=round(osim, 4),
                                 gap=round(osim - own, 4)))
    suspects.sort(key=lambda t: -t["gap"])
    return dict(label_margin=label_margin, n_suspects=len(suspects), suspects=suspects[:30])


def balance_and_coverage(V, labels, classes):
    rows = []
    for ci, c in enumerate(classes):
        m = V[labels == ci]; n = len(m)
        if n >= 2:
            S = m @ m.T; iu = np.triu_indices(n, k=1)
            compact = float(S[iu].mean())          # 類內平均 cosine：高=視角單一、低=覆蓋廣/可能含雜
        else:
            compact = 1.0
        rows.append(dict(cls=c, n=n, intra_compactness=round(compact, 4)))
    counts = [r["n"] for r in rows] or [0]
    imbalance = round(max(counts) / max(1, min(counts)), 2)
    return dict(imbalance_ratio=imbalance, n_total=int(sum(counts)), per_class=rows)


def find_leakage(Vg, classes_g, paths_g, holdout_dir, dino, leak_thr=0.97):
    himgs = _imgs_in(holdout_dir)
    if not himgs:
        return dict(holdout=str(holdout_dir), n_holdout=0, leaks=[])
    leaks = []
    for p in himgs:
        v = dino.embed(imread(p))
        sims = Vg @ v
        j = int(np.argmax(sims)); s = float(sims[j])
        if s >= leak_thr:
            leaks.append(dict(holdout=str(p), nearest_golden=paths_g[j], sim=round(s, 4)))
    leaks.sort(key=lambda t: -t["sim"])
    return dict(holdout=str(holdout_dir), n_holdout=len(himgs), leak_thr=leak_thr,
                n_leaks=len(leaks), leaks=leaks[:30])


# ----------------------------- 編排 -----------------------------
def run_audit(train_dir=TRAIN_DIR, holdout_dir=None, dup_thr=0.97,
              label_margin=0.0, leak_thr=0.97, out_path=None, cfg=None):
    dino, sel = get_dino(cfg)
    print(f"[audit] device={sel['device']} DINO={sel['dino']}  embedding goldens…")
    V, labels, classes, paths = embed_goldens(train_dir, dino,
        progress=lambda i, n, c: print(f"[audit]   embed {i}/{n} {c}"))
    if len(V) == 0:
        print("[audit] 找不到 Golden 影像"); return {}
    report = dict(train_dir=str(train_dir), n_goldens=len(V), classes=classes)
    report["duplicates"] = find_near_duplicates(V, labels, classes, paths, dup_thr)
    report["label_errors"] = find_label_errors(V, labels, classes, paths, label_margin)
    report["balance"] = balance_and_coverage(V, labels, classes)
    if holdout_dir:
        print(f"[audit] 檢查 train↔holdout 洩漏: {holdout_dir}")
        report["leakage"] = find_leakage(V, classes, paths, holdout_dir, dino, leak_thr)
    out_path = Path(out_path or (WS / "audit_report.json"))
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 摘要
    d = report["duplicates"]; le = report["label_errors"]; b = report["balance"]
    print("\n========== Dataset Audit ==========")
    print(f"Golden 總數 {report['n_goldens']}　類別 {classes}")
    print(f"⚠ 近重複對：{d['n_pairs']}（跨類={d['cross_class_pairs']}，sim≥{dup_thr}）")
    print(f"⚠ 疑似標錯：{le['n_suspects']}（更像他類，margin>{label_margin}）")
    print(f"⚠ 類別不均比：{b['imbalance_ratio']}（max/min 樣本數）")
    for r in b["per_class"]:
        flag = "← 樣本過少" if r["n"] < 8 else ("← 視角單一" if r["intra_compactness"] > 0.92 else "")
        print(f"    {r['cls']:>12}: n={r['n']:>3} 緊緻={r['intra_compactness']:.3f} {flag}")
    if holdout_dir:
        print(f"⚠ 洩漏：{report['leakage']['n_leaks']} / {report['leakage']['n_holdout']} hold-out 幾乎同張")
    if le["n_suspects"]:
        print("  疑似標錯 Top3：")
        for s in le["suspects"][:3]:
            print(f"    {Path(s['path']).name}: 標={s['labeled']} 但更像={s['looks_like']} (gap {s['gap']})")
    print(f"[audit] 完整報告 -> {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=str(TRAIN_DIR))
    ap.add_argument("--holdout_dir", default=None)
    ap.add_argument("--dup_thr", type=float, default=0.97)
    ap.add_argument("--label_margin", type=float, default=0.0,
                    help="他類-自類相似度差 > 此值才報標錯（調高=只報最明顯的）")
    ap.add_argument("--leak_thr", type=float, default=0.97)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_audit(a.train_dir, a.holdout_dir, a.dup_thr, a.label_margin, a.leak_thr, a.out)


if __name__ == "__main__":
    main()
