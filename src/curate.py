"""
curate.py — #4 Golden 庫策展：量化「你的 Golden 特徵庫哪裡弱」，並給出可執行的補強方向。

安全網的辨別力 = Golden 庫的品質。本工具用 DINO 嵌入算三件事：
  1. 類間分離度 (separation)：每對類別質心 cosine -> 找出最易混淆的類別對(如 zipper≈screw)。
  2. 類內冗餘   (redundancy) ：同類中彼此過度相似的樣本 -> 可剪除而不損覆蓋（省算力、去灌水）。
  3. 補強優先序 (priority)  ：綜合「樣本太少 + 與鄰類太近」-> 排出該優先補哪一類 Golden。
  （可選）--pool_dir：對候選影像依「離現有 Golden 最遠」排序 -> 最能補覆蓋缺口的優先收。

為何有效：直接指出資料集的薄弱類別與冗餘，讓你用最少的新樣本把「最易誤判的類別」補強，
          同時提升安全網對難分類別的 retention/interception 與 YOLO 後續訓練的類別品質。

用法：
  python src/curate.py                       # 策展 Workspace/Training
  python src/curate.py --redundancy_thr 0.96
  python src/curate.py --pool_dir path/to/candidate_crops   # 另排候選樣本補強順序
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT, imread
from audit import get_dino, embed_goldens, _prototypes, _imgs_in, balance_and_coverage

WS = ROOT / "Workspace"
TRAIN_DIR = WS / "Training"


def separation(V, labels, classes):
    """類間質心相似度矩陣 + 每類最近鄰類別（越近=越易混淆）。"""
    P = _prototypes(V, labels, len(classes))
    M = P @ P.T
    np.fill_diagonal(M, -1)
    rows, pairs = [], []
    for ci, c in enumerate(classes):
        nj = int(np.argmax(M[ci])); ns = float(M[ci][nj])
        rows.append(dict(cls=c, nearest=classes[nj], nearest_sim=round(ns, 4)))
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(dict(a=classes[i], b=classes[j], sim=round(float(M[i][j]), 4)))
    pairs.sort(key=lambda t: -t["sim"])
    return dict(per_class=rows, most_confusable=pairs[:10])


def redundancy(V, labels, classes, paths, thr=0.96):
    """每類內貪婪去重：與已保留樣本相似度>=thr 者標為冗餘（可剪）。"""
    out = []; total_red = 0
    for ci, c in enumerate(classes):
        idx = np.where(labels == ci)[0]
        if len(idx) < 2:
            out.append(dict(cls=c, n=len(idx), redundant=0, drop=[])); continue
        sub = V[idx]; S = sub @ sub.T
        kept, drop = [], []
        for a in range(len(idx)):
            if any(S[a, b] >= thr for b in kept):
                drop.append(paths[idx[a]])
            else:
                kept.append(a)
        total_red += len(drop)
        out.append(dict(cls=c, n=len(idx), kept=len(kept), redundant=len(drop),
                        drop=[Path(p).name for p in drop[:20]]))
    return dict(threshold=thr, total_redundant=total_red, per_class=out)


def priority(balance_rows, sep_rows, min_n=12, w_few=0.5, w_conf=0.5):
    """補強優先序：樣本越少 + 與鄰類越近 -> 越該優先補該類 Golden。"""
    maxn = max((r["n"] for r in balance_rows), default=1)
    sep = {r["cls"]: r["nearest_sim"] for r in sep_rows}
    ranked = []
    for r in balance_rows:
        few = 1.0 - r["n"] / max(1, maxn)          # 0~1，越少越大
        conf = max(0.0, sep.get(r["cls"], 0.0))    # 與鄰類相似度，越大越混淆
        urgency = w_few * few + w_conf * conf
        reason = []
        if r["n"] < min_n:
            reason.append(f"樣本少({r['n']})")
        if conf > 0.55:
            reason.append(f"與[{next((s['nearest'] for s in sep_rows if s['cls']==r['cls']),'?')}]易混淆({conf:.2f})")
        ranked.append(dict(cls=r["cls"], n=r["n"], nearest_sim=round(conf, 4),
                           urgency=round(float(urgency), 4),
                           reason="；".join(reason) or "尚可"))
    ranked.sort(key=lambda t: -t["urgency"])
    return ranked


def rank_pool(pool_dir, Vg, paths_g, dino, top=50):
    """對候選影像依『離最近 Golden 最遠』排序 -> 最能補覆蓋缺口的優先收。"""
    imgs = _imgs_in(pool_dir)
    rows = []
    for p in imgs:
        v = dino.embed(imread(p)); sims = Vg @ v
        j = int(np.argmax(sims))
        rows.append(dict(path=str(p), nearest_golden=Path(paths_g[j]).name,
                         nearest_sim=round(float(sims[j]), 4),
                         novelty=round(float(1 - sims[j]), 4)))
    rows.sort(key=lambda t: -t["novelty"])
    return rows[:top]


def run_curate(train_dir=TRAIN_DIR, redundancy_thr=0.96, pool_dir=None,
               out_path=None, cfg=None):
    dino, sel = get_dino(cfg)
    print(f"[curate] device={sel['device']} DINO={sel['dino']}  embedding goldens…")
    V, labels, classes, paths = embed_goldens(train_dir, dino,
        progress=lambda i, n, c: print(f"[curate]   embed {i}/{n} {c}"))
    if len(V) == 0:
        print("[curate] 找不到 Golden 影像"); return {}
    bal = balance_and_coverage(V, labels, classes)
    sep = separation(V, labels, classes)
    red = redundancy(V, labels, classes, paths, redundancy_thr)
    pri = priority(bal["per_class"], sep["per_class"])
    report = dict(train_dir=str(train_dir), n_goldens=len(V), classes=classes,
                  separation=sep, redundancy=red, balance=bal, priority=pri)
    if pool_dir:
        print(f"[curate] 評估候選池: {pool_dir}")
        report["pool_ranking"] = rank_pool(pool_dir, V, paths, dino)
    out_path = Path(out_path or (WS / "curation_report.json"))
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 摘要
    print("\n========== Golden Curation ==========")
    print(f"Golden 總數 {len(V)}　類別 {classes}")
    print("最易混淆類別對 Top3：")
    for p in sep["most_confusable"][:3]:
        print(f"    {p['a']} ≈ {p['b']}  (質心 cosine {p['sim']})")
    print(f"類內冗餘可剪：{red['total_redundant']} 張（thr={redundancy_thr}）")
    for r in red["per_class"]:
        if r["redundant"]:
            print(f"    {r['cls']}: {r['n']}→保留{r.get('kept','?')}（冗餘 {r['redundant']}）")
    print("補強優先序（該優先收哪類 Golden）：")
    for r in pri:
        print(f"    {r['cls']:>12}: urgency={r['urgency']:.3f}  n={r['n']}  → {r['reason']}")
    if pool_dir and report.get("pool_ranking"):
        print("候選池最該收 Top3（覆蓋缺口最大）：")
        for r in report["pool_ranking"][:3]:
            print(f"    {Path(r['path']).name}: novelty={r['novelty']} (最近 {r['nearest_golden']} {r['nearest_sim']})")
    print(f"[curate] 完整報告 -> {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=str(TRAIN_DIR))
    ap.add_argument("--redundancy_thr", type=float, default=0.96)
    ap.add_argument("--pool_dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_curate(a.train_dir, a.redundancy_thr, a.pool_dir, a.out)


if __name__ == "__main__":
    main()
