"""
novelty.py — #8 Novel-class Discovery：把「開集」變成「資料集該長出哪個新類別」的訊號。

安全網把「離所有 Golden 都遠」的框一律攔截成 False。但其中常混著「不是誤報、而是你還沒
收進資料集的新料件/新缺陷」。若一群被攔截的框彼此很像（緊密群聚），那多半是一個**反覆出現
的新類別**，值得建成新 Golden 類別，而不是當雜訊丟掉。

做法（模型免載，吃 gui_cache 的 vec/crop）：
  1. 取所有「被攔截」(score < 該類閾值) 的偵測框。
  2. 對其 DINO 向量做貪婪 cosine 群聚（同群 = 彼此夠像）。
  3. 群夠大且夠緊密(cohesion 高) -> 列為「建議新類別」，存代表縮圖供你命名後加入 Training/。

為何有效：主動告訴你資料集的「盲區」往哪補；把不斷攔截的同類物件升級成正式類別，
          一次同時改善資料集涵蓋度與後續 YOLO/安全網對該類的辨識。

用法：
  python src/novelty.py                          # 讀 gui_cache，輸出 Workspace/NovelCandidates
  python src/novelty.py --cluster_thr 0.6 --min_cluster 3 --min_cohesion 0.55
"""
from __future__ import annotations
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT

WS = ROOT / "Workspace"
GUI_CACHE = WS / "gui_cache" / "check_records.pkl"


def _collect_intercepted(cache, cfg):
    """回傳被攔截框的 (vecs, metas, crops)。攔截=score < 該類閾值。"""
    g_thr = float(cache.get("threshold") or cfg["matching"]["threshold"])
    per_cls = {str(k): float(v) for k, v in (cfg["matching"].get("thresholds") or {}).items()}
    vecs, metas, crops = [], [], []
    for img, recs in cache["images"].items():
        for j, r in enumerate(recs):
            if "vec" not in r:
                continue
            thr = per_cls.get(str(r["pred_class"]), g_thr)
            if float(r["score"]) < thr:   # 被攔截 = 離所有 golden 都不夠近
                vecs.append(np.asarray(r["vec"], np.float32))
                metas.append(dict(image=img, det=j, score=round(float(r["score"]), 4),
                                  nearest_class=r["pred_class"]))
                crops.append(r.get("sam"))
    V = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 1), np.float32)
    return V, metas, crops


def greedy_cluster(V, thr):
    """貪婪 cosine 群聚（向量已 L2-normalize）：以首個未分配點為種子，吸收相似度>=thr 者。"""
    n = len(V); assigned = -np.ones(n, int); clusters = []
    S = V @ V.T
    for i in range(n):
        if assigned[i] >= 0:
            continue
        cid = len(clusters); members = [i]; assigned[i] = cid
        for j in range(i + 1, n):
            if assigned[j] < 0 and S[i, j] >= thr:
                assigned[j] = cid; members.append(j)
        clusters.append(members)
    return clusters, S


def discover(cache, cfg, cluster_thr=0.6, min_cluster=3, min_cohesion=0.55):
    V, metas, crops = _collect_intercepted(cache, cfg)
    if len(V) == 0:
        return dict(n_intercepted=0, clusters=[]), [], []
    clusters, S = greedy_cluster(V, cluster_thr)
    out = []
    for cid, mem in enumerate(clusters):
        if len(mem) < min_cluster:
            continue
        sub = S[np.ix_(mem, mem)]; iu = np.triu_indices(len(mem), k=1)
        cohesion = float(sub[iu].mean()) if len(mem) > 1 else 1.0
        if cohesion < min_cohesion:
            continue
        near = {}
        for m in mem:
            near[metas[m]["nearest_class"]] = near.get(metas[m]["nearest_class"], 0) + 1
        out.append(dict(cluster=cid, size=len(mem), cohesion=round(cohesion, 4),
                        members=[metas[m] for m in mem],
                        nearest_class_hist=near, member_idx=mem))
    out.sort(key=lambda c: (-c["size"], -c["cohesion"]))
    return dict(n_intercepted=len(V), n_clusters_found=len(out), clusters=out), metas, crops


def export(report, crops, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        import shutil; shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, c in enumerate(report["clusters"]):
        d = out_dir / f"cluster_{rank:02d}_size{c['size']}_coh{c['cohesion']:.2f}"
        d.mkdir(parents=True, exist_ok=True)
        for m in c["member_idx"]:
            crop = crops[m]
            if crop is not None and getattr(crop, "size", 0):
                cv2.imwrite(str(d / f"{Path(report['_meta'][m]['image']).stem}_d{report['_meta'][m]['det']}.png"), crop)
    clean = dict(report); clean.pop("_meta", None)
    for c in clean["clusters"]:
        c.pop("member_idx", None)
    (out_dir / "novelty_report.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(WS / "NovelCandidates"))
    ap.add_argument("--cluster_thr", type=float, default=0.6,
                    help="同群的 cosine 門檻（越高=群越純越小）")
    ap.add_argument("--min_cluster", type=int, default=3, help="成為候選新類別的最小群大小")
    ap.add_argument("--min_cohesion", type=float, default=0.55, help="群內平均相似度下限")
    a = ap.parse_args()
    if not GUI_CACHE.exists():
        print("[novelty] 無 gui_cache；先跑 python src/pipeline.py --gui_cache"); return
    cfg = load_config()
    cache = pickle.load(open(GUI_CACHE, "rb"))
    report, metas, crops = discover(cache, cfg, a.cluster_thr, a.min_cluster, a.min_cohesion)
    report["_meta"] = metas
    print(f"\n========== Novel-class Discovery ==========")
    print(f"被攔截框 {report['n_intercepted']} 個 -> 找到 {report.get('n_clusters_found',0)} 個候選新類別群")
    for rank, c in enumerate(report["clusters"]):
        print(f"  候選#{rank}: size={c['size']} cohesion={c['cohesion']} "
              f"(最接近現有類別分佈 {c['nearest_class_hist']})")
    if report["clusters"]:
        out = export(report, crops, a.out_dir)
        print(f"[novelty] 代表縮圖 + 報告 -> {out}")
        print("[novelty] 檢視各 cluster 縮圖；若是有意義的新類別，命名後複製到 Workspace/Training/<新類別>/ 再 build_bank。")
    else:
        print("[novelty] 沒有夠大夠緊密的群（攔截框多為零散雜訊，符合預期）。")


if __name__ == "__main__":
    main()
