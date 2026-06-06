# The Safety Net — Two-Stage 視覺特徵驗證管線 (YOLO → SAM → DINO)

> 把 YOLO 當「高靈敏度觸發器」（高 Recall、容許誤報），在它後面架設
> **SAM 去背 + DINOv2 語意特徵比對** 的安全網，攔截 YOLO 的 False Alarm。
> SAM / DINO 皆為 foundation model，**免重訓**即可在 Small Data 下大幅提升 Precision。

落地情境：**工業 AOI 光學檢測**（金屬反光、複雜紋理，YOLO 在小資料下極易誤判）。

> **這是一套通用、可重用的框架**：換成你自己的偵測資料集（VOC 或 YOLO 格式）即可用——見
> [換成你自己的資料](#換成你自己的資料-bring-your-own-dataset)。
> 想知道**它何時有效、何時失效**（在合成 vs 真實資料上的誠實壓測）→ 見 **[FINDINGS.md](FINDINGS.md)**。

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

## 換成你自己的資料 (Bring Your Own Dataset)

一條命令把**你自己的偵測資料集**接上安全網（自動轉檔/切分 train·val·holdout、裁 Golden、
建特徵庫、校準閾值）：

```bash
# Pascal VOC（images/ + annotations/），且你已有 YOLO 權重 → 最快，免訓練
python run.py onboard --voc path/to/VOC --name mydata --weights best.pt

# VOC，沒有權重 → 順便訓練一個 YOLO11n 觸發器（CPU 慢）
python run.py onboard --voc path/to/VOC --name mydata --train --epochs 20

# 已是 YOLO 格式（images/ + labels/ + classes.txt 或 data.yaml）
python run.py onboard --yolo path/to/YOLO --name mydata --weights best.pt
```
產出 `Workspace/<name>/{train,val,holdout}` + 設定好的 `config.yaml` + `vector_bank.npz`。接著：
```bash
python src/eval_real.py --data Workspace/mydata/holdout   # 真實 hold-out 評測（可信指標）
python src/distill.py   --data Workspace/mydata/holdout   # 蒸餾 KPI（安全網介入率）
python run.py gui                                          # 視覺化 Inspector
```
> 選項：`--per-class N`（每類 Golden 數）、`--dino dinov2_vitb14`（更強特徵）、
> `--no-sam`（Golden 不去背較快）、`--split 300,80,150`（train,val,holdout 數量）。
> 真實道路坑洞資料的完整案例（含誠實結果）見 [FINDINGS.md](FINDINGS.md)。

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

## Data Flywheel — 讓 YOLO 越用越準、資料集越管越乾淨

安全網原本只是「推論期過濾器」（攔誤報），但它每跑一張圖就產出**已分類、SAM 去背、含信度
與 margin 的高品質判決**——把這些回收起來，就能同時**自動標註資料**＋**回頭把 YOLO 練得更準**。

```bash
# ── 資料集管理 ──
python run.py audit                          # 健檢：近重複/訓練驗證洩漏/標籤錯誤/類別不均
python run.py curate                         # Golden 策展：最易混淆類別對、冗餘可剪、該優先補哪類
python run.py active --top 50                # 待標佇列：先標「模型最沒把握」的(資訊量最大)
python run.py novelty                        # 一直被攔截又彼此很像的物件 -> 建議建成新類別

# ── 提升 YOLO 準度（飛輪）──
python run.py autolabel                      # 判決 -> YOLO 偽標(收緊框)；灰帶自動留給 active learning
python run.py eval_real --data path/holdout  # 真實 hold-out 上量可信指標（不是自評）
python run.py distill   --data path/holdout  # 蒸餾 KPI：安全網介入率 / YOLO 吸收程度（應隨輪次↓）
python run.py flywheel --holdout path/holdout --epochs 25 --rounds 5
#   多輪飛輪：每輪用「當前最佳 YOLO」再挖掘偽標(信度加權重取樣)+Hard Negative -> 暖啟動重訓
#   三面向升級閘：① hold-out per-class mAP ② 錨點集(防遺忘) ③ 端到端安全網 P/R
#   三者皆不退步才升級；任一退步 rollback。連續未升級/mAP 平台期 -> 自動停止
#   教師(golden 庫)每輪凍結+指紋檢查；逐輪寫血緣帳本(flywheel_ledger.json)可審計/回溯

# ── 終局：讓 YOLO 自己學會分類（多類別逐類畢業，雙軌蒸餾）──
python run.py flywheel --holdout path/holdout --multiclass --rounds 5  # 訓「會分類的 YOLO」
python run.py distill  --data path/holdout --graduate --write
#   逐類評估：某類 YOLO 自分夠準+安全網介入夠低+樣本夠多 -> 該類「畢業」交給 YOLO 自己分
#   難分類別(zipper≈screw)永不畢業、留安全網兜底；畢業類別推論時 YOLO 說了算(不再跑 DINO 否決)

# ── 進階：更乾淨、更聚焦盲點的偽標（R7–R10）──
python run.py hardneg                         # R7 挖「YOLO 最自信卻被攔」的盲點(wrongness 排序)
python run.py autolabel --tta --cotrain --min_box_iou 0.5   # R8 教師TTA+框一致 / R9 兩視角一致才收
python run.py autolabel --seg                 # R10 SAM 遮罩 -> YOLO-seg 多邊形(分割蒸餾)
python run.py flywheel --holdout HO --hardneg_weight 3 --tta --cotrain --min_box_iou 0.5
#   旗標可組合進飛輪：硬負樣本加權 + 更乾淨偽標 -> 更快修正誤報、更高偽標可信度
```

| 缺口 | 工具 | 為何幫到你 |
|---|---|---|
| 標註成本高 | `autolabel` | 人從「畫框」降級為「打勾」；SAM 收緊框比 YOLO 原框更貼合 -> box 品質↑ |
| 標哪張最划算 | `active` | 用 margin/近閾值/裁決衝突排序，最少人力換最大 mAP 增益 |
| 資料髒拖垮準度 | `audit` | DINO 嵌入揪出去重/洩漏/標錯，防「假高分」與特徵庫被毒化 |
| 不知資料集哪裡弱 | `curate` | 指出最易混淆類別、冗餘樣本、該優先補強的類別 |
| 分數可信嗎 | `eval_real` | 在真實資料而非合成自評上量指標 |
| YOLO 不會變強 | `flywheel` | 唯一真正動到 YOLO 權重；蒸餾安全網知識，三面向煞車 |
| 蒸餾有沒有效 | `distill` | 量「安全網介入率」隨輪次下降＝YOLO 真的在吸收安全網 |
| 出現沒見過的東西 | `novelty` | 把反覆攔截的同類物件升級成「待建新類別」訊號 |

> 設計上呼應閉環風險：偽標只取高信度框、灰帶交給人、每輪重訓都過 hold-out mAP 煞車，
> 飛輪不會把錯誤越滾越大（見 DESIGN §9）。

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
config.yaml              # 中央設定（device/模型/閾值[含per-class]/輸出模式…）
run.py                   # 端到端協調器
FINDINGS.md              # 誠實案例研究：安全網何時有效/失效（MVTec vs 真實坑洞）
src/
  common.py              # 環境自適配 + SAM 去背器(可回收緊框) + DINO 嵌入器
  onboard.py             # 換你自己的資料：VOC/YOLO → golden → bank → [train] → calibrate
  voc2yolo.py            # 通用 Pascal VOC(XML) → YOLO(txt) 轉換器 + 切分
  download_data.py       # MVTec 子集自動下載（HF 鏡像）
  make_scenes.py         # 合成場景產生器（cut-out + Golden + Check + GT）
  train_yolo.py          # YOLO11n 觸發器訓練
  build_bank.py          # DINO 特徵庫（旋轉增強）
  pipeline.py            # 四級管線 + 分發 + 閾值校準（含 per-class + margin）
  evaluate.py            # 10 情境壓測 + 95 分驗證迴圈
  gui_app.py             # Safety Net Inspector 視覺化 GUI（Streamlit）
  # ── Data Flywheel / 資料集管理（把安全網的判決回收成價值）──
  autolabel.py           # #1 判決 -> YOLO 偽標（SAM 收緊框 + 信度/margin gating）
  active_learning.py     # #2 依資訊量(margin/近閾值/裁決衝突)排序待標佇列
  audit.py               # #3 資料集健檢：去重 / 洩漏 / 標籤錯誤 / 覆蓋
  curate.py              # #4 Golden 庫策展：分離度 / 冗餘剪除 / 補強優先序
  eval_real.py           # #6 真實 hold-out 評測（可信指標，非自評）
  distill.py             # R1 蒸餾 KPI(介入率) + R2 逐類畢業評估
  retrain_loop.py        # #7/R4 多輪閉環重訓（信度加權偽標 -> 三面向煞車 -> R6 帳本）
  hardneg.py             # R7 硬負樣本挖掘（YOLO 自信誤報的盲點）
  novelty.py             # #8 攔截框群聚 -> 建議新類別
tests/smoke_test.py      # 免模型純邏輯煙測（23 項，秒級）：python tests/smoke_test.py
Workspace/               # 資料、權重、特徵庫、結果（執行後生成）
```

## 測試

```bash
python tests/smoke_test.py   # 免模型純邏輯煙測（閾值校準/煞車/畢業/多邊形/重映射… 23 項）
```
模型相關（YOLO/SAM/DINO）功能以實跑驗證：`onboard` → `eval_real` / `distill`。

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

