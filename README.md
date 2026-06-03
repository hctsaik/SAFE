# The Safety Net — Two-Stage 視覺特徵驗證管線 (YOLO → SAM → DINO)

> 把 YOLO 當「高靈敏度觸發器」（高 Recall、容許誤報），在它後面架設
> **SAM 去背 + DINOv2 語意特徵比對** 的安全網，攔截 YOLO 的 False Alarm。
> SAM / DINO 皆為 foundation model，**免重訓**即可在 Small Data 下大幅提升 Precision。

落地情境：**工業 AOI 光學檢測**（金屬反光、複雜紋理，YOLO 在小資料下極易誤判）。

---

## 架構

```
原始大圖 → YOLO11n(觸發,低conf) → bbox → SAM(去背) → DINOv2(特徵) → 向量比對 → 裁決
                                          │(mask空/過小 fallback)        │
                                          └→ 原始bbox crop          近=True / 遠=False(攔截)
```

| 級 | 模型 | 角色 |
|---|---|---|
| 1 粗定位 | YOLO11n | 高 Recall 觸發器，容許誤報 |
| 2 去背 | MobileSAM(CPU)/SAM2(GPU) | bbox 當 prompt 取 mask，剔除背景雜訊 |
| 3 特徵 | DINOv2 ViT-S/14(CPU)/ViT-B/14(GPU) | 抽語意特徵向量（免重訓） |
| 4 裁決 | Vector Matching | 與 Golden 特徵庫比 cosine，過閾值=True |

環境自適配：腳本偵測 CUDA，GPU 用大模型、CPU 用輕量模型。

---

## 安裝

```bash
pip install -r requirements.txt
# torch/torchvision 請依平台安裝（CPU 或對應 CUDA 版本，見 https://pytorch.org）
```
DINOv2 權重於首次執行時自動由 torch.hub 下載。

## 一鍵執行

```bash
python run.py all        # 資料→訓練→建庫→校準→推論→95分驗證迴圈
```

分階段：
```bash
python run.py data       # 下載 MVTec 子集 + 合成場景（產生 Workspace/Training, Check）
python run.py train      # 訓練 YOLO11n 觸發器
python run.py bank       # 建 DINO 特徵庫（含旋轉增強）
python run.py infer                         # 校準閾值 + 對 Check/ 推論分發
python run.py infer --mode annotated_full   # 切換輸出模式
python run.py eval       # Phase 3：10 情境壓測 + 95 分迴圈
```

## 視覺化 GUI — Safety Net Inspector

```bash
python run.py gui            # 或 streamlit run src/gui_app.py
```
通用視覺化工具（Streamlit + Plotly），全程可解釋地呈現管線與判決：
- **🔍 Inspector**：原圖(綠=通過/紅=攔截框) + 管線瀑布（YOLO 觸發 → SAM 去背 → DINO 比對 → 裁決）+ **top-3 最相似 Golden 並排** + 分數溫度計；可即時上傳新圖推論；👍/👎 人在迴路（👎 一鍵匯出 Hard Negative）。
- **📊 Dashboard**：誤報率 / 攔截率 / 保留率 / Precision 指標卡 + 分數直方圖（通過/攔截兩色 + 閾值線）+ 分數排序縮圖牆（篩選/分頁）+ No_Detection 專區。
- **⚙️ Advanced**：Golden 庫總覽 + 特徵空間 2D（PCA）投影。
- **側邊閾值滑桿即時重判**整批結果（precision/recall 取捨變成手感）；有 GT 時可疊真值並算 TP/FP/FN。

> 完全由 `config.yaml` + `vector_bank.npz` + `Workspace/Check` 驅動 → 換成你自己的專案目錄即可重用。

## 輸出

```
Workspace/Result/
├── True/[Class]/         # 通過驗證（真目標）
├── False/[Class]/        # 被攔截的 False Alarm
└── No_Detection/         # YOLO 無偵測
```
- `output_mode=cropped_roi`（預設）：存 SAM 去背小圖（供 Hard Negative 再訓練）。
- `output_mode=annotated_full`：存畫框大圖（綠=通過 / 紅=誤報，供 BI 審查）。

## 目錄結構

```
config.yaml              # 中央設定（device/模型/閾值/輸出模式…）
run.py                   # 端到端協調器
src/
  common.py              # 環境自適配 + SAM 去背器 + DINO 嵌入器
  download_data.py       # MVTec 子集自動下載（HF 鏡像）
  make_scenes.py         # 合成場景產生器（cut-out + Golden + Check + GT）
  train_yolo.py          # YOLO11n 觸發器訓練
  build_bank.py          # DINO 特徵庫（旋轉增強）
  pipeline.py            # 四級管線 + 分發 + 閾值校準
  evaluate.py            # 10 情境壓測 + 95 分驗證迴圈
  gui_app.py             # Safety Net Inspector 視覺化 GUI（Streamlit）
Workspace/               # 資料、權重、特徵庫、結果（執行後生成）
```

## 驗證結果 (Phase 3)

**最終配置**：YOLO11n (conf 0.12) → MobileSAM(批次去背) → DINOv2 **ViT-B/14** → cosine kNN(topk 8, 旋轉增強庫 960 向量) → 閾值 **0.355**(自動校準)。

QA Reviewer 的 10 種嚴苛情境壓測（指標皆量測自具 GT 的合成場景，非自評）：

| 指標 | 結果 |
|---|---|
| **平均分數** | **97.56 / 100**（第 1 輪即 ≥ 95 達標）|
| False Alarm 攔截率 | **1.00（全部情境）** |
| 真目標保留率 | 0.64–1.00（最低為「中空小物件+反光」極端情境）|
| 類別正確率 | 0.93–1.00 |
| SAM 去背成功率 | 1.00 |

10 情境分數：baseline 98.3、heavy_occlusion 100、same_color_bg 100、drastic_light 98.4、
hollow_sam_fail 89.1、many_distractors 97.0、specular_glare 96.8、nearmiss_decoy 98.2、
small_far 97.7、out_of_frame 100。完整紀錄見 `Workspace/validation_report.json`。

**Check/ → Result/ 實跑**：30 張大圖 → True 102 / False 60 / No_Detection 0（非目標物件與紋理/反光誘餌全部被攔截）。

### 關鍵工程決策（迭代得出）
1. **非目標物件（硬負樣本）**是本架構價值的核心展示：YOLO 對「物件」過度觸發，DINO 做「類別歸屬」驗證並攔截非目標物件（cable/toothbrush/pill）。
2. **旋轉增強特徵庫**：DINOv2 對旋轉敏感（capsule 90° 0.88→0.53）；旋轉增強後各角度穩定 0.85+。
3. **DINOv2 ViT-B/14 > ViT-S/14**：非目標物件與目標的語意分離更佳（NEG 0.45→0.29）。
4. **已知限制**：與目標形狀本質相近的物件（zipper≈screw、bottle≈metal_nut）zero-shot 難以區分；正式環境需為該目標類別擴充 golden 樣本或微調。

