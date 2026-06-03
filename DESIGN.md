# DESIGN — Two-Stage 視覺特徵驗證管線 (YOLO → SAM → DINO)
> The Safety Net：把 YOLO 當高靈敏觸發器，用 SAM 去背 + DINO 語意比對攔截 False Alarm。
> 狀態：**設計已定案（待最終開工指令）**。本文件為無人值守開發階段要遵循的規格書。

---

## 0. 定案決策摘要 (Locked Decisions)

| 項目 | 決策 |
|---|---|
| 落地情境 | 工業 AOI 光學檢測（金屬反光、複雜紋理） |
| 資料集 | **MVTec AD + 合成場景產生器 (Composite Scene Generator)** |
| True/False 語意 | DINO 比對上**任一真物件類別（含瑕疵件）= True**；只有紋理/反光/雜訊 = **False Alarm 攔截** |
| 運算環境 | 不確定 → 腳本內**偵測 CUDA 自動適配**（GPU 大模型 / CPU 輕量模型） |
| 推進方式 | **先定案設計，再開發**；定案後轉**無人值守**，自主解環境/依賴/下載/程式錯誤，直到平均 95 分才回報 |

---

## 1. 核心架構 (Pipeline)

```
                 [高 Recall, 容許誤報]        [去背, 剔雜訊]         [zero-shot 語意]      [距離裁決]
原始大圖 ──▶ YOLO11n ──bbox──▶ SAM(box prompt) ──去背crop──▶ DINOv2 ──向量──▶ Vector Matching
                                     │                                              │
                                     └─ fallback: mask 空/過小 → 退回原始 bbox crop  │
                                                                                     ▼
                                          cosine ≥ 閾值 → True/[Class]  |  < 閾值 → False/[Class]
                                          無 YOLO 偵測 → No_Detection
```

**設計理念**：不花海量標註把 YOLO 訓到完美；改用 SAM + DINO（皆為 foundation model，免重訓）做後級驗證，在 Small Data 下同時保 Recall（YOLO 寬鬆）與拉 Precision（後級嚴格）。

---

## 2. 資料策略 (Phase 1)

### 2.1 來源：MVTec AD
- **物件類**（screw / metal_nut / hazelnut / capsule / transistor …）→ 合法目標類別，建 Golden Samples。
- **紋理類**（carpet / grid / leather / tile / wood）→ False Alarm 誘餌來源。
- 下載：以 HuggingFace/Kaggle 鏡像自動下載為主；**fallback**：鏡像失效時改用 Roboflow 工業資料集，確保無人值守不卡關。

### 2.2 合成場景產生器 (`make_scenes.py`)
MVTec 原生無 bbox、物件置中於乾淨背景 → 自行合成「複雜大圖」：
- 把物件 crop 隨機貼到紋理背景，套用旋轉/縮放/光影增強 → 取得**精確 bbox Ground Truth**。
- 撒入「純紋理 / 反光亮點」干擾框 → 製造**真實 False Alarm**。
- 好處：可算真實 Precision/Recall（95 分有客觀依據）、可控生成 QA 的 10 情境。

### 2.3 目錄結構
```
Workspace/
├── Training/[Class]/            # Golden Samples（每類少量標準樣本）
├── Check/                       # 待驗證的未分類原始大圖
└── Result/
    ├── True/[Class]/            # 通過驗證（真目標）
    ├── False/[Class]/           # 被攔截的 False Alarm（標最近類別）
    └── No_Detection/            # YOLO 無偵測
```

---

## 3. 模型與環境自適配 (Phase 0)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
| 模組 | GPU | CPU |
|---|---|---|
| YOLO | YOLO11n（兩者皆用 nano，輕量觸發器） | YOLO11n |
| SAM | SAM / SAM2 (ViT-B) | MobileSAM / FastSAM-s |
| DINO | DINOv2 ViT-B/14（768 維） | DINOv2 ViT-S/14（384 維） |

- 以 **Ultralytics** 單套件涵蓋 YOLO + SAM/SAM2/MobileSAM/FastSAM，降低依賴衝突；DINOv2 走 `torch.hub` 或 HF。
- 設定集中於 `config.yaml`：`device / yolo_conf / sam_variant / dino_variant / match_threshold / output_mode / topk`。

---

## 4. 開發細節 (Phase 2)

### 4.1 YOLO 訓練 (`train_yolo.py`)
- 小資料訓練 YOLO11n；`conf` 調低以維持高 Recall（刻意過敏感，靠後級攔誤報）。

### 4.2 SAM 去背
- 用 YOLO bbox 當 **box prompt** 取 mask → 去背 crop。
- **Fallback**：mask 為空 / 面積過小（中空物件、強反光失效）→ 退回原始 bbox crop，避免漏判。

### 4.3 Vector Bank (`build_bank.py`)
- 讀 `Training/[Class]/`，每張（可選經 SAM 去背）→ DINOv2 embed → **L2-normalize**。
- 同時保留 **全樣本 kNN** 與 **類別 prototype（均值）** 兩種比對，取較佳。
- 存成 `.npy`（小資料免 FAISS）。

### 4.4 比對裁決
- 對每個 YOLO box：去背 → DINO 向量 → 與 Bank 算 **cosine 相似度** → 取最近類別。
- `sim ≥ match_threshold` → `True/[class]`；否則 → `False/[nearest class]`。

### 4.5 閾值校準（成敗核心）
- 切一小塊有標籤校準集，掃描閾值，選 **最大化 F1**（或在「目標 Precision」下的最高 Recall）的值，寫回 `config.yaml`，**不可拍腦袋**。

### 4.6 批次推論與分發 (`pipeline.py`)
- 讀 `Check/` → 逐圖推論 → 分發至 `Result/{True,False,No_Detection}`。
- **`output_mode`**：
  - `cropped_roi`（預設）：存 SAM 去背的局部目標小圖，供 Hard Negative 再訓練。
  - `annotated_full`：存原始大圖畫框（**綠框=通過 / 紅框=誤報**），供 BI 視覺化與人工審查。

---

## 5. 驗證與重構迴圈 (Phase 3)

### 5.1 評分綁真實指標（反對純自評）
合成場景有 GT → 量測 **Precision / Recall / F1 + False Alarm 攔截率 + SAM 去背成功率**，分數有客觀依據。

### 5.2 首輪 10 種嚴苛情境
1. 嚴重遮蔽 2. 物件與背景同色 3. 光影/Gamma 劇變 4. 中空物件(metal_nut 孔/grid)讓 SAM 失效 5. 一對多干擾 6. 極小/遠距物件 7. 金屬反光眩光 8. 近似誘餌(紋理像物件) 9. 物件出框/截斷 10. 瑕疵件 vs 良品（依語意：瑕疵件仍屬該類 = True）。

### 5.3 終止條件
- 均分 **< 95** → 診斷失分原因 → 改架構/閾值/去背餵法 → **重生全新 10 情境** → 再測。
- 反覆直到 **平均 ≥ 95**，才輸出最終總結與啟動指令。

---

## 6. 交付物與檔案

| Phase | 腳本 | 交付 |
|---|---|---|
| 0 環境 | — | `requirements.txt`、`config.yaml` |
| 1 資料 | `download_data.py`、`make_scenes.py` | MVTec 下載 + 合成場景 + 目錄樹 |
| 2 開發 | `train_yolo.py`、`build_bank.py`、`pipeline.py` | YOLO 權重、Vector Bank、推論分發 |
| 3 驗證 | `evaluate.py` | 指標報告、10 情境壓測、95 分迴圈、最終總結 |

---

## 7. 已知風險 (Risks)
1. MVTec 自動下載穩定性 → 備 Roboflow fallback。
2. 合成場景「真實度」需貼近真實 AOI 反光/紋理。
3. 去背後餵 DINO 的方式（黑底貼上 / 緊裁 / alpha）會影響特徵 → Phase 2 實驗定案。
4. 中空/強反光物件 SAM 失效 → 已設 bbox crop fallback。

---

## 8. 強制日誌格式（每輪迭代必記）
`📌 [目前討論項目]` / `✅ [達成共識]` / `⚠️ [潛在爭議與風險]` / `🚀 [下一步行動]`
