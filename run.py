"""
run.py — The Safety Net 端到端協調器 (YOLO -> SAM -> DINO)。

用法：
  python run.py all                  # 端到端：資料 -> 訓練 -> 建庫 -> 校準 -> 推論 -> 驗證迴圈
  python run.py data                 # 下載 + 合成場景
  python run.py train                # 訓練 YOLO 觸發器
  python run.py bank                 # 建 DINO 特徵庫
  python run.py infer [--mode M]     # 校準 + 對 Check/ 推論分發
  python run.py eval                 # Phase 3 95 分驗證迴圈
  python run.py infer --mode annotated_full   # 切換輸出模式
  python run.py gui                  # 啟動 Safety Net Inspector 視覺化 GUI
"""
from __future__ import annotations
import sys, subprocess, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def sh(mod, *args):
    cmd = [sys.executable, str(SRC / mod), *args]
    print(f"\n$ python src/{mod} {' '.join(args)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[run] {mod} failed (exit {r.returncode})"); sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "data", "train", "bank", "infer", "eval", "gui"])
    ap.add_argument("--mode", default=None, help="output_mode: cropped_roi|annotated_full")
    a = ap.parse_args()

    if a.stage in ("all", "data"):
        sh("download_data.py")
        sh("make_scenes.py")
    if a.stage in ("all", "train"):
        sh("train_yolo.py")
    if a.stage in ("all", "bank"):
        sh("build_bank.py")
    if a.stage in ("all", "infer"):
        args = ["--calibrate", "--run"]
        if a.mode: args += ["--output_mode", a.mode]
        sh("pipeline.py", *args)
    if a.stage in ("all", "eval"):
        sh("evaluate.py")
    if a.stage == "gui":
        import subprocess as sp
        # 上傳頁可直接在瀏覽器內建置資料 → 不強制預先計算；加大上傳上限給 ZIP/權重
        sp.run([sys.executable, "-m", "streamlit", "run", str(SRC / "gui_app.py"),
                "--server.maxUploadSize", "2048"])


if __name__ == "__main__":
    main()
