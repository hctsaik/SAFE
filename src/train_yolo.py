"""
train_yolo.py — Phase 2a：訓練輕量級 YOLO11n 作為「高靈敏度觸發器」。
單一類別 'object'（類別判斷交給後級 DINO）。小資料 + 低 conf -> 刻意過敏感、容許誤報。
資料：scenes/train (訓練) + scenes/calib (驗證)，皆含 YOLO 格式標註。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_env, ROOT

SCENES = ROOT / "Workspace" / "scenes"
RUNS = ROOT / "Workspace" / "runs"
DATA_YAML = SCENES / "safetynet.yaml"


def write_data_yaml():
    txt = (f"path: {SCENES.as_posix()}\n"
           f"train: train/images\n"
           f"val: calib/images\n"
           f"names:\n  0: object\n")
    DATA_YAML.write_text(txt)
    return DATA_YAML


def train(cfg=None):
    from ultralytics import YOLO
    cfg = cfg or load_config()
    sel = resolve_env(cfg)
    y = cfg["yolo"]
    write_data_yaml()
    print(f"[yolo] training {sel['yolo']} device={sel['device']} "
          f"epochs={y['epochs']} imgsz={y['imgsz']}")
    model = YOLO(sel["yolo"])
    model.train(
        data=str(DATA_YAML), epochs=y["epochs"], imgsz=y["imgsz"],
        batch=y["batch"], device=sel["device"], project=str(RUNS), name="yolo",
        exist_ok=True, verbose=False, plots=False, seed=cfg["seed"],
        # 輕量小資料：適度增強，避免過擬合，但保持觸發器特性
        degrees=10, translate=0.1, scale=0.3, fliplr=0.5, mosaic=0.5,
        patience=0,
    )
    best = RUNS / "yolo" / "weights" / "best.pt"
    print(f"[yolo] done -> {best} (exists={best.exists()})")
    return best


if __name__ == "__main__":
    train()
