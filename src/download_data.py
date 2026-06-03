"""
download_data.py — Phase 1a：自動下載 MVTec AD 子集（僅所需類別）。
主來源：HuggingFace 鏡像 TheoM55/mvtec_anomaly_detection（含 metadata.csv，可逐檔下載）。
fallback：鏡像失效時提示改用 Roboflow（保留介面，主來源穩定時不觸發）。
"""
from __future__ import annotations
import csv, io, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT

REPO = "TheoM55/mvtec_anomaly_detection"
META_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/metadata.csv"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/images/"

N_OBJECT = 60   # 每個物件類別下載的 good 圖數（供 cut-out 池 + golden + 校驗）
N_TEXTURE = 18  # 每個紋理類別下載數（供背景 + 誘餌）


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_metadata() -> list[dict]:
    raw = _get(META_URL).read().decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(raw)))


def plan_downloads(rows, classes, textures, ood=()):
    """挑選每類別 train/good 影像，回傳 [(remote_path, local_path)]。"""
    raw_dir = ROOT / "Workspace" / "raw"
    jobs = []
    for cat, limit, sub in ([(c, N_OBJECT, "objects") for c in classes] +
                            [(o, 24, "ood") for o in ood] +
                            [(t, N_TEXTURE, "textures") for t in textures]):
        picked = [r for r in rows
                  if r["object"] == cat and r["split"] == "train" and r["defect"] == "good"]
        picked = sorted(picked, key=lambda r: r["path"])[:limit]
        for r in picked:
            fn = Path(r["path"]).name
            jobs.append((r["path"], raw_dir / sub / cat / fn))
    return jobs


def download_one(job, overwrite=False):
    remote, local = job
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and not overwrite and local.stat().st_size > 0:
        return ("skip", local)
    for attempt in range(3):
        try:
            data = _get(FILE_URL + remote).read()
            local.write_bytes(data)
            return ("ok", local)
        except Exception as e:
            if attempt == 2:
                return ("err", f"{local}: {e}")
    return ("err", str(local))


def main():
    cfg = load_config()
    classes, textures = cfg["classes"], cfg["textures"]
    ood = cfg.get("ood_classes", [])
    print(f"[download] classes={classes} ood={ood} textures={textures}")
    print(f"[download] fetching metadata from HF mirror: {REPO}")
    rows = fetch_metadata()
    print(f"[download] metadata rows: {len(rows)}")

    # 驗證所需類別都存在
    avail = sorted({r["object"] for r in rows})
    missing = [c for c in classes + textures + ood if c not in avail]
    if missing:
        print(f"[download][FATAL] categories missing in mirror: {missing}")
        print(f"[download] available: {avail}")
        sys.exit(2)

    jobs = plan_downloads(rows, classes, textures, ood)
    print(f"[download] scheduling {len(jobs)} files...")
    ok = skip = err = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for status, info in ex.map(download_one, jobs):
            if status == "ok": ok += 1
            elif status == "skip": skip += 1
            else:
                err += 1; print("  [err]", info)
    print(f"[download] done. ok={ok} skip={skip} err={err}")

    # 摘要
    raw = ROOT / "Workspace" / "raw"
    for sub in ["objects", "ood", "textures"]:
        for d in sorted((raw / sub).glob("*")):
            n = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
            print(f"   {sub}/{d.name}: {n}")
    if err > len(jobs) * 0.3:
        print("[download][WARN] high error rate — mirror may be unstable.")
        sys.exit(3)


if __name__ == "__main__":
    main()
