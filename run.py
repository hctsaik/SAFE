"""
run.py — The Safety Net 端到端協調器 (YOLO -> SAM -> DINO)。

核心流程：
  python run.py all                  # 端到端：資料 -> 訓練 -> 建庫 -> 校準 -> 推論 -> 驗證迴圈
  python run.py data                 # 下載 + 合成場景
  python run.py train                # 訓練 YOLO 觸發器
  python run.py bank                 # 建 DINO 特徵庫
  python run.py infer [--mode M]     # 校準 + 對 Check/ 推論分發
  python run.py eval                 # Phase 3 95 分驗證迴圈
  python run.py gui                  # 啟動 Safety Net Inspector 視覺化 GUI

Bring Your Own Dataset（一鍵把你的資料接上安全網）：
  python run.py onboard --voc DIR --name mydata --weights best.pt   # VOC，用既有權重
  python run.py onboard --voc DIR --name mydata --train             # VOC，順便訓練觸發器
  python run.py onboard --yolo DIR --name mydata --weights best.pt  # 已是 YOLO 格式

Data Flywheel / 資料集管理（新增；額外參數會原樣轉給對應腳本）：
  python run.py autolabel            # 安全網判決 -> YOLO 偽標（Workspace/AutoLabel）
  python run.py active               # 依資訊量排序待標佇列（Workspace/ActiveQueue）
  python run.py audit                # 資料集健檢：去重/洩漏/標籤錯誤/覆蓋
  python run.py curate               # Golden 庫策展：分離度/冗餘/補強優先序
  python run.py eval_real --data DIR # 真實 hold-out 評測（可信指標）
  python run.py distill --data DIR   # 蒸餾 KPI：安全網介入率 / YOLO 吸收程度
  python run.py distill --data DIR --graduate --write  # 逐類畢業評估(寫 graduated_classes)
  python run.py hardneg              # R7 硬負樣本挖掘：YOLO 最自信卻被攔的盲點
  python run.py autolabel --tta --cotrain --min_box_iou 0.5  # R8/R9 更乾淨的偽標
  python run.py autolabel --seg      # R10 SAM 遮罩 -> YOLO-seg 多邊形標籤
  python run.py flywheel --holdout DIR [--epochs N] [--rounds N] [--multiclass]
                         [--hardneg_weight 3] [--tta] [--cotrain] [--min_box_iou 0.5]
                                     # 多輪偽標+重訓->三面向煞車->過了才換權重
  python run.py novelty              # 攔截框群聚 -> 建議新類別
"""
from __future__ import annotations
import sys, subprocess, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# stage -> 對應腳本（單純轉呼叫，額外 CLI 參數原樣傳遞）
SCRIPTS = {
    "onboard": "onboard.py", "autolabel": "autolabel.py", "active": "active_learning.py",
    "audit": "audit.py", "curate": "curate.py", "eval_real": "eval_real.py",
    "distill": "distill.py", "flywheel": "retrain_loop.py", "novelty": "novelty.py",
    "hardneg": "hardneg.py",
}


def sh(mod, *args):
    cmd = [sys.executable, str(SRC / mod), *args]
    print(f"\n$ python src/{mod} {' '.join(args)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[run] {mod} failed (exit {r.returncode})"); sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "data", "train", "bank", "infer", "eval",
                                      "gui", *SCRIPTS.keys()])
    ap.add_argument("--mode", default=None, help="output_mode: cropped_roi|annotated_full")
    a, extra = ap.parse_known_args()   # 未知參數（如 --holdout/--data/--epochs）轉給子腳本

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
    if a.stage in SCRIPTS:                # 飛輪/資料集管理工具
        sh(SCRIPTS[a.stage], *extra)


if __name__ == "__main__":
    main()
