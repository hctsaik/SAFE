/goal 啟動 Multi-Agent 協作模式，為「Two-Stage 視覺特徵驗證管線 (YOLO -> SAM -> DINO)」設計、尋找資料集並開發出一個完整的可運作系統。先不要開發,請一直討論, 直到有結論為止

在你的 Production 環境中，主力軍依然是 YOLO，因為它輕量、速度快，適合做第一線的即時掃描。
但問題在於你的訓練資料量太小 (Small Data)。在資料量不足的情況下，YOLO 無法學到足夠強健、泛化的分類特徵。為了確保不漏抓（維持高 Recall），YOLO 往往會變得過於敏感，把很多背景雜訊或長得稍微有點像的物件都框出來，導致大量的 False Alarm (誤報)。

你的解決方案：The Safety Net (安全網)
你不想（或無法）花費大量時間去標註幾萬張照片來硬把 YOLO 訓練到完美。相反地，你把 YOLO 降級為一個「高靈敏度的觸發器 (Trigger)」，並在它背後架設了一套基於 SAM + DINO 的 Double Check 機制。

實際運作流程如下：

第一線快速掃描 (YOLO)： YOLO 在畫面上掃描，只要覺得「疑似」有目標，就立刻框出 Bounding Box。這時我們允許它犯錯。

啟動去背 (SAM)： 針對 YOLO 框出來的可疑區域，交給 SAM 去做精細的 Segmentation，把背景的雜訊（草地、機台反光、複雜紋理）全部剔除，只留下物件本體。

降維打擊 (DINO)： 將去背後的乾淨影像餵給 DINO。因為 DINO 是 Foundation Model，它對物件語意的理解極度深刻，完全不需要你用小資料重新訓練。DINO 會直接吐出這個物件的高維度特徵向量。

最終裁決 (Vector Matching)： 拿這個向量去跟你手邊「少量的標準樣本 (Golden Samples)」所建立的 DINO 特徵向量庫進行距離比對。

距離近 ＝ 真的是目標物件 (True Positive)。

距離遠 ＝ YOLO 判斷錯誤，攔截這個 False Alarm。

### 核心運作規則 (Directive)
1. **無人值守模式 (Unattended Execution)：** 在完成最終 95 分的驗證之前，請自主解決所有遇到的環境、依賴套件、資料下載與程式碼錯誤問題，**請不要中斷並詢問我**。
2. **多智能體協作 (Multi-Agent Debate)：** 系統內必須模擬三個角色進行對話與決策：
   - [Domain Expert]：負責定義這個架構最能發揮價值的落地情境（例如：高精度的工業自動化光學檢測 AOI，或極易產生 False Alarm 的複雜背景物件定位），並主導資料集的挑選。
   - [AI Engineer]：負責實作 YOLO (粗定位) -> SAM (去背) -> DINO (特徵對比) 的 Pipeline，處理資料夾流轉，以及輕量級 YOLO 的訓練腳本。
   - [QA Reviewer]：負責發掘極端案例 (Edge cases)，並對最終系統進行嚴格評分。
3. **強制紀錄日誌 (Continuous Logging)：** 每一輪迭代都必須在輸出中明確記錄以下區塊，達成共識後才能進入開發：
   - 📌 [目前討論項目]
   - ✅ [達成共識]
   - ⚠️ [潛在爭議與風險]
   - 🚀 [下一步行動]

### 任務執行階段 (Phases)

**Phase 1: 價值定義與資料準備**
由 Domain Expert 主導，思考這個雙重驗證架構到底解決什麼痛點。請自主上網搜尋或利用開源平台找尋一個適合的資料集。該資料集必須具備「容易讓 YOLO 產生誤判 (False Alarm)」的特性。確定後，撰寫腳本自動下載並建立以下目錄結構：
- `Workspace/Training/` (依子資料夾名稱作為分類，放置標準樣本)
- `Workspace/Check/` (放置等待驗證的未分類原始大圖)

**Phase 2: 模型建構與開發**
由 AI Engineer 實作：
1. **建構基礎 Pipeline：** 建立 YOLO 訓練腳本 (訓練輕量級模型)；串接 SAM (建議 MobileSAM/FastSAM) 透過 YOLO Bounding Box 進行去背；串接 DINOv2 提取特徵，並讀取 `Training/` 資料夾建立「標準特徵庫 (Vector Bank)」。
2. **批次推論與分發 (Batch Inference & Routing)：** 讀取 `Check/` 內的影像進行推論，並將結果輸出至 `Workspace/Result/`。
   - 命中特徵庫：分發至 `Result/True/[Class]/`
   - 被判定為誤報：分發至 `Result/False/[Class]/`
   - 無物件偵測：分發至 `Result/No_Detection/`
3. **實作輸出配置 (Output Configuration)：** 系統必須支援 `output_mode` 參數切換：
   - `cropped_roi` (預設值)：在 True/False 內僅儲存被 YOLO 裁切且經 SAM 去背的局部目標小圖，便於後續 Hard Negative 模型再訓練。
   - `annotated_full`：在 True/False 內儲存畫上 Bounding Box (通過為綠框，False Alarm 為紅框) 的原始完整大圖，用於 BI 視覺化與人工全局審查。




**Phase 3: 嚴格驗證與重構迴圈 (Validation Loop)**
完成系統開發後，進入評分迴圈。
1. QA Reviewer 必須定義 **10 種嚴苛的真實環境情境**（例如：物件被嚴重遮蔽、背景與物件顏色極度相似、光影變化劇烈、物件呈現中空姿態導致 SAM 提取失敗、一對多目標干擾等）。
2. 使用這 10 個情境對開發出的 Pipeline 進行壓力測試與理論打分（總分 100 分）。
3. **終止條件：** 如果 Multi-Agent 綜合評分沒有達到 **平均 95 分**，請「不要停下來」。Agent 必須自行檢討丟失分數的原因，修改架構設計或程式碼，接著 **重新生成另外 10 個全新的嚴苛情境**，再次進行驗證打分。
4. 反覆此重構與驗證過程，直到滿足平均 95 分的要求，才能輸出最終的專案總結與啟動指令交給我。