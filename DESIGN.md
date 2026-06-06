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

---

## 9. Data Flywheel 擴充（資料集管理 + 提升 YOLO 準度）

### 9.1 核心洞察
原系統是**推論期過濾器**（攔誤報），不會讓 YOLO *本身*變準。但安全網每跑一張圖就產出
「已分類、SAM 去背、含 score/margin」的高品質判決——回收這些即可同時**自動標註**＋**回頭練 YOLO**。
缺口的兩面（管理資料集、提升 YOLO 準度）其實是同一個飛輪。

### 9.2 模組對應
| # | 模組 | 服務目標 | 機制 |
|---|---|---|---|
| 地基 | `pipeline.match()` | 共用訊號 | 回傳 (cls, s1, s2)；margin=s1-s2 點亮 #2/#4/#7/#8 |
| 1 | `autolabel.py` | 資料集管理 | 判決->YOLO txt；SAM 收緊框；高信度才匯出、灰帶留給 #2 |
| 2 | `active_learning.py` | YOLO 準度 | margin/近閾值/裁決衝突 -> 待標佇列排序（模型免載，吃 cache） |
| 3 | `audit.py` | 資料集管理 | DINO 嵌入：去重/洩漏/標籤錯誤/覆蓋 |
| 4 | `curate.py` | 兩者 | 類間分離度/類內冗餘/補強優先序 |
| 5 | per-class 閾值 | YOLO 準度 | `calibrate()` 逐類校準、`SafetyNet._thr()` 套用 |
| 6 | `eval_real.py` | 裁判 | 真實 hold-out 跑同一張評分卡（複用 `evaluate.WEIGHTS`） |
| 7 | `retrain_loop.py` | YOLO 準度 | 偽標+HardNeg->暖啟動重訓->hold-out 比 mAP->過了才換權重 |
| 8 | `novelty.py` | 資料集管理 | 攔截框貪婪 cosine 群聚 -> 建議新類別 |

### 9.3 閉環風險與煞車（confirmation bias）
偽標噪音回灌可能讓 YOLO 越練越爛。對策（已落地於 #1/#2/#7）：
1. **高信度才自動標**：`autolabel` 只匯出 `score≥閾值+pad 且 margin≥min_margin` 的框。
2. **灰帶交給人**：貼近閾值/低 margin 的框不進訓練，改進 `active_learning` 佇列。
3. **mAP 煞車**：`flywheel` 每輪在真實 hold-out 比 mAP50，未超過現役就 rollback、不換權重。

### 9.4 已知限制
- 飛輪 val 以「單類觸發器 mAP」量定位能力；類別品質由安全網/`eval_real` 把關。
- `eval_real`/`flywheel` 的類別正確率需 hold-out 類別名與 Golden 類別名對齊（依名稱映射）。
- per-class 閾值需該類在校準集有足量樣本（< `min_per_class` 退回全域）。

### 9.5 強制日誌（本次擴充）
- 📌 缺什麼能幫到資料集管理 + 提升 YOLO 準度。
- ✅ 核心缺口＝Data Flywheel；最高 ROI＝autolabel+active；先體檢(audit/eval_real)再進補(flywheel)。
- ⚠️ 閉環 confirmation bias、合成自評侷限 -> 以 gating + 真實 hold-out + mAP 煞車緩解。
- 🚀 八項全數實作並逐一 smoke test 通過；指令併入 `run.py`。

### 9.6 強化「判決蒸餾回 YOLO」（共識起手三件套：R1 + R3 + R5）
原 #7 僅兌現蒸餾最弱版本（單類硬偽標 + 一刀切 + 單一全域 mAP + 一次性）。本次補上三條：

| 代號 | 模組/變更 | 強化的蒸餾通道 | 機制 |
|---|---|---|---|
| **R1** | `distill.py` | 量測（成敗 KPI） | 同一批 YOLO 框上對比 YOLO-alone vs 安全網：**介入率**(應↓)、**precision_gap**(應縮)、(多類別)**class_acc_gap** |
| **R3** | `retrain_loop.assemble_dataset` | 多信任（軟監督） | 由 (score-thr)×margin 算每圖信度 -> **高信度偽標重複放更多份**(重取樣)；純負樣本不放大；不改 ultralytics 內核 |
| **R5** | `retrain_loop.gate()` | 煞車 | **三面向皆不退步才升級**：① hold-out per-class mAP ② 錨點集(防災難性遺忘) ③ 端到端安全網 P/R；任一退步即 rollback |

驗證：R1 在合成 hold-out 上正確抓到「2 類 bank 缺類別 -> 安全網過度攔截真目標(recall 0.89->0.56)」；
R3 信度->份數單調(高=3 份/低=1 份)；R5 anchor 遺忘 / per-class 退步 / 端到端掉 precision 任一情況皆正確擋下。
KPI 指令：`python run.py distill --data <holdout>`。

### 9.8 多類別逐類畢業（R2，雙軌蒸餾）
把 DINO 的**分類知識**真正搬進 YOLO 的終局。雙軌：單類觸發器永遠是安全預設，多類別 opt-in，
且**逐類畢業**——某類在 hold-out 上 YOLO 自分夠準且安全網介入夠低，那一類才交給 YOLO。

| 子項 | 模組/變更 | 機制 |
|---|---|---|
| **R2a 多類別軌** | `flywheel --multiclass` / `assemble_dataset(multiclass)` | 偽標用 DINO 類別(net.classes)；val 依**名稱映射**到 net.classes(`_copy_pairs_remap`)；data.yaml 多類別 -> 訓「會分類的 YOLO」 |
| **R2b 逐類畢業** | `distill.graduation()` + `distill --graduate [--write]` | 逐類算 YOLO 自分準確率/介入率/樣本數；達標(預設 acc≥0.9, 介入≤0.1, support≥10)才畢業，寫 `config.matching.graduated_classes` |
| **R2c 雙軌推論** | `SafetyNet.detect()`(回傳 ycls) + `process()` + `_is_graduated()` | 多類別 YOLO 對**已畢業且高 conf** 的框直接信任(via=yolo，不跑 DINO 否決)；其餘照走安全網(via=safetynet) |

安全設計：難分類別(zipper≈screw)因介入率高/自分低**永不畢業**，留安全網兜底；單類觸發器或
`graduated_classes` 空時 `_is_graduated` 恆 False -> **完全向後相容**（行為與 R2 前一致）。
驗證：`graduation()`(screw 畢業/zipper/rare 留)、`_is_graduated`(conf 門檻)、`_copy_pairs_remap`
(映射/塌縮/None 略過)、多類別 data.yaml+偽標類別 {0,1}、`process()` 單類回歸(全 via=safetynet) 皆通過。

### 9.9 進階標籤品質（R7–R10）
讓蒸餾的偽標更乾淨、更聚焦 YOLO 盲點 —— 偽標品質直接決定學生上限。

| 代號 | 模組/變更 | 機制 |
|---|---|---|
| **R7 硬負樣本挖掘** | `hardneg.py` + `assemble_dataset(hardneg_weight)` | 依 **wrongness=conf×(thr-score)** 排序「YOLO 最自信卻被攔」的盲點，匯出 provenance；含此類框的純負樣本影像**重取樣加權**回灌 -> 最快修正誤報 |
| **R8 強化教師** | `DinoEmbedder.embed_tta()` + autolabel `--tta`/`--min_box_iou` | 多視角平均嵌入(更穩)+回傳一致性；要求 YOLO 框與 SAM 框 **IoU≥門檻**(定位可信)才當偽標 |
| **R9 co-training** | autolabel `--cotrain` | 兩視角一致才匯出：多類別下 **YOLO 類別==DINO 類別** + TTA **一致性≥門檻**；分歧者落 active learning |
| **R10 分割蒸餾** | `SamSegmenter.segment_full()` + autolabel `--seg` | SAM 遮罩 -> 最大輪廓 **多邊形** YOLO-seg 標籤（遮罩失敗退回 bbox 矩形）；可訓 `yolo11n-seg` 把 SAM 定位精度蒸進 YOLO |

旗標可組合，並透傳 `flywheel`（`--hardneg_weight/--tta/--cotrain/--min_box_iou`）。
驗證：`_neg_weight`(自信誤報高/正樣本 0)、`_mask_to_polygon`(方塊→4 點且界內)、`_box_polygon` 單測通過；
模型 smoke：嚴格門檻觸發 box_disagree/unstable、hardneg 依 wrongness 排序、seg 產合法多邊形(task=segment)。
備註：TTA 會改變匹配分數（多視角平均更貼近旋轉增強庫），可能升或降匯出量，屬預期。

至此 R1–R10 全數落地；後續可往「真實產線資料 + 多輪實跑」收斂。

### 9.7 多輪飛輪 + 血緣（R4 + R6）
| 代號 | 模組/變更 | 機制 |
|---|---|---|
| **R4** | `run_flywheel(rounds, patience, min_delta)` + `_flywheel_round` | 每輪用**當前最佳 YOLO 再挖掘**偽標（YOLO 變強 -> 觸發先前漏抓 -> 偽標更多更好）；停止條件：達 rounds / 連續 `patience` 輪未升級 / mAP 平台期(`_plateau`) |
| **R6** | `teacher_fingerprint()` + `append_ledger()` | 教師(golden 庫)每輪**凍結 + sha 指紋漂移檢查**（漂移即中止）；逐輪寫 `flywheel_ledger.json`(教師指紋/權重/三面向指標/gate/動作)；promote 時版本化備份權重 `flywheel_rN_*.pt` 可回溯任一輪 |

效率：現役指標跨輪沿用（hold-out/錨點固定，僅升級時更新），避免重複量測。
驗證：`_plateau`(改善中/平台期/樣本不足)、`teacher_fingerprint`(12 碼 sha)、`append_ledger`(append 正確) 單測通過；
多輪 `--dry_run` 整合跑通（R3 重取樣 6->18、教師指紋入帳本、停止控制正確）。
指令：`python run.py flywheel --holdout <dir> --rounds 5`。
