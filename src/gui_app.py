"""
gui_app.py — Safety Net Inspector (Streamlit GUI)
通用視覺化：YOLO 觸發 → SAM 去背 → DINO 比對 → True/False 裁決，全程可解釋。

效能設計：與閾值無關的重運算（偵測/去背/嵌入）由 pipeline.build_gui_cache 預先算一次、
存成 pickle；GUI 直接載入快取 → 瞬開。模型只在「上傳新圖即時推論」時才延遲載入。

三個分頁：
  🔍 Inspector  原圖(綠/紅框) + 管線瀑布 + top-3 Golden + 分數溫度計 + 👍/👎 人在迴路
  📊 Dashboard  誤報率/攔截率/保留率/Precision + 分數直方圖 + 分數排序縮圖牆 + No_Detection
  ⚙️ Advanced   Golden 庫總覽 + 特徵空間 2D(PCA) 投影

啟動：  python run.py gui      (首次會自動預算快取)
   或   streamlit run src/gui_app.py
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, ROOT, imread, classify_box

WS = ROOT / "Workspace"
CHECK_DIR = WS / "Check"
TRAIN_DIR = WS / "Training"
CHECK_GT = WS / "scenes" / "check" / "gt.json"
GUI_CACHE = WS / "gui_cache" / "check_records.pkl"
BANK = WS / "vector_bank.npz"
HARDNEG = WS / "HardNegatives"
# 顏色以 RGB 定義（draw_boxes 在 RGB 影像上作畫，st.image 直接顯示 RGB）
GREEN, RED = (40, 200, 80), (235, 60, 60)
GT_TGT, GT_OOD = (235, 205, 40), (40, 200, 235)  # 真值: 黃=目標 / 青=非目標物件

st.set_page_config(page_title="Safety Net Inspector", layout="wide",
                   initial_sidebar_state="expanded")


# ----------------------------- 載入（快取，瞬間）-----------------------------
@st.cache_data(show_spinner=False)
def load_meta():
    cfg = load_config()
    classes = cfg["classes"]
    if BANK.exists():
        classes = [str(c) for c in np.load(BANK, allow_pickle=True)["classes"]]
    return cfg, classes


@st.cache_resource(show_spinner="載入預算快取…")
def load_cache():
    if not GUI_CACHE.exists():
        return None
    with open(GUI_CACHE, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_gt():
    if not CHECK_GT.exists():
        return {}
    gt = json.loads(CHECK_GT.read_text())
    return {sc["image"]: dict(
        targets=[(o["box"], o["cls"]) for o in sc["objects"]],
        oods=[o["box"] for o in sc.get("ood", [])]) for sc in gt["scenes"]}


@st.cache_resource(show_spinner="載入模型（YOLO+SAM+DINO）供即時推論…")
def get_net_and_gallery():
    from pipeline import SafetyNet, _golden_gallery
    net = SafetyNet(load_config())
    return net, _golden_gallery(net)


COLOR_KEY = ("🟩 綠框 = 通過(DINO 認證的真目標+類別)　🟥 紅框 = 攔截(DINO 判定的誤報)"
             "　🟨 黃框 = 真值GT-目標　🟦 青框 = 真值GT-非目標物件(評測模式)")


def render_help(default_open=True):
    """📖 報表怎麼看：修正心智模型 + True/False 定義 + 名詞說明 + 顏色圖例。"""
    with st.expander("📖 報表怎麼看（先看這個）", expanded=default_open):
        st.markdown(
            "#### 流程（每個框會經過這 4 關）\n"
            "📷 **原圖** → 🔍 **YOLO**：這裡有東西嗎？*（只框出可疑位置，**不分類、不判對錯**）* "
            "→ ✂️ **SAM**：把框內背景去掉，只留物件 "
            "→ 🧠 **DINO**：這物件**像哪個標準樣本(Golden)**？算相似度。\n\n"
            "#### ✅ True 還是 ❌ False，是誰決定的？\n"
            "**全是 DINO 決定的，不是 YOLO。** DINO 把去背後的物件跟每個類別的 Golden 標準樣本比 "
            "**cosine 相似度 (0~1，越大越像)**，取最像的類別與分數：\n"
            "- 🟩 **True（通過）**：最高相似度 **≥ 閾值** → 認定是**真目標**，標上 DINO 判定的**類別**。\n"
            "- 🟥 **False（攔截）**：跟**所有** Golden 都**不夠像（< 閾值）** → 判定為 YOLO 的**誤報**，擋下。\n\n"
            "> ⚠️ 常見誤解：YOLO **不會**給你類別、也**不會**判 True/False。它只是「高靈敏觸發器」"
            "（寧可多框、容許犯錯）；真正的「這是不是目標、是哪一類」由後面的 DINO 認定。"
        )
        g1, g2 = st.columns(2)
        g1.markdown(
            "#### 名詞說明\n"
            "- **觸發 (YOLO)**：框出疑似有物件的位置。\n"
            "- **去背 (SAM)**：剔除框內背景，只留物件本體。\n"
            "- **相似度 (DINO)**：物件轉成特徵向量，與 Golden 比 cosine（0~1）。\n"
            "- **Golden 標準樣本**：每類少量「標準長相」範例，組成特徵庫。\n"
            "- **閾值 Threshold**：相似度及格線（側邊滑桿可調）。\n"
            "- **攔截 Intercept**：把 YOLO 的誤報判為 False、擋下來。")
        g2.markdown(
            "#### 指標怎麼算\n"
            "- **False Alarm 率**：YOLO 觸發框中，*非真目標*(非目標物件/背景) 的比例。\n"
            "- **DINO 攔截率**：那些誤報中，被正確判成 False 的比例（越高越好）。\n"
            "- **真目標保留率**：真目標中，被正確放行(True) 的比例。\n"
            "- **Precision（安全網後）**：放行的框中真的是目標的比例。\n\n"
            f"#### 顏色圖例\n{COLOR_KEY}")


def bgr2rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if getattr(x, "size", 0) else x


def decide(rec, thr):
    return "True" if rec["score"] >= thr else "False"


def draw_boxes(img, recs, thr, gt=None, sel=None):
    out = bgr2rgb(img).copy()  # 先轉 RGB，之後以 RGB 顏色作畫（避免 BGR/RGB 互換把紅畫成藍）
    if gt:
        for b, _ in gt["targets"]:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GT_TGT, 2)
        for b in gt["oods"]:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GT_OOD, 2)
    for i, r in enumerate(recs):
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        passed = decide(r, thr) == "True"
        col = GREEN if passed else RED
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 5 if i == sel else 3)
        tag = f'{r["pred_class"]} {r["score"]:.2f}' if passed else f'FALSE {r["score"]:.2f}'
        cv2.putText(out, tag, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


def thermometer(score, thr, passed):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[score], y=[""], orientation="h",
                         marker_color=("#28c850" if passed else "#eb3c3c"),
                         width=0.5, hoverinfo="x"))
    fig.add_vline(x=thr, line_dash="dash", line_color="#222",
                  annotation_text=f"閾值 {thr:.2f}", annotation_position="top")
    fig.update_layout(xaxis=dict(range=[0, 1], title="cosine 相似度"),
                      yaxis=dict(showticklabels=False), height=130,
                      margin=dict(l=10, r=10, t=24, b=24), showlegend=False)
    return fig


# ----------------------------- 📤 資料與設定頁 -----------------------------
DINO_OPTS = ["dinov2_vits14（快, 預設）", "dinov2_vitb14（準, CPU 慢）", "dinov2_vitl14（最準, 最慢）"]


def render_setup_page():
    ss = st.session_state
    up = WS / "uploads"; up.mkdir(parents=True, exist_ok=True)
    st.header("📤 資料與設定 — 上傳你的資料 → 一鍵建置（零訓練）")
    st.caption("不訓練任何模型：依你的標註自動裁出 Golden、用 DINO 嵌入成特徵庫、(有權重)自動校準閾值。")
    with st.expander("每個上傳的用途（先看）", expanded=False):
        st.markdown(
            "- **① GT ZIP（必填）**：影像 + YOLO **標註** txt（+`data.yaml`）→ 自動裁每類 Golden、當校準正樣本。\n"
            "- **② Check ZIP（必填）**：影像 + YOLO **預測** txt（含誤報）→ 要被驗證的框（直接用，不再跑 YOLO）。\n"
            "- **③ 權重 .pt（可選）**：自動校準閾值 + Inspector 即時單張。**只上傳你信任的權重。**\n"
            "- GT 影像會**切兩半**：一半裁 Golden、一半當 hold-out 校準集（避免閾值樂觀）。\n"
            "- Check 沒有真值 → 只給框數 + 分數直方圖 + 逐框 Golden 證據（無攔截率/保留率）。")

    # ① GT ZIP
    st.subheader("① Ground Truth ZIP")
    gtz = st.file_uploader("影像 + YOLO 標註 txt（+ data.yaml）", type="zip", key="gtz")
    names = None
    if gtz is not None:
        from setup_build import safe_extract, resolve_class_names, list_pairs, class_ids_present
        if ss.get("gtz_name") != gtz.name:
            (up / "gt.zip").write_bytes(gtz.getbuffer())
            safe_extract(up / "gt.zip", up / "gt")
            pairs = list_pairs(up / "gt")
            ss.gtz_name = gtz.name
            ss.gt_labeled = sum(1 for _, l in pairs if l)
            ss.gt_names = resolve_class_names(up / "gt")
            ss.gt_ids = class_ids_present(pairs)
        st.success(f"解壓完成：{ss.gt_labeled} 份標註；類別編號 {ss.gt_ids}")
        if ss.gt_names:
            st.info(f"自動偵測類別名：{ss.gt_names}")
            names = ss.gt_names
        else:
            st.warning("找不到 data.yaml / classes.txt — 請手動填每個編號的類別名：")
            cols = st.columns(min(4, max(1, len(ss.gt_ids))))
            names = [cols[k % len(cols)].text_input(f"類別 {cid}", value=f"class{cid}", key=f"nm{cid}")
                     for k, cid in enumerate(ss.gt_ids)]
        ss.names = names

    # ② Check ZIP
    st.subheader("② Check ZIP（含誤報的 YOLO 預測）")
    ckz = st.file_uploader("影像 + YOLO 預測 txt", type="zip", key="ckz")
    if ckz is not None:
        from setup_build import safe_extract, list_pairs
        if ss.get("ckz_name") != ckz.name:
            (up / "check.zip").write_bytes(ckz.getbuffer())
            safe_extract(up / "check.zip", up / "check")
            ss.ckz_name = ckz.name
            ss.ck_imgs = len(list_pairs(up / "check"))
        st.success(f"解壓完成：{ss.ck_imgs} 張影像")

    # ③ 權重
    st.subheader("③ YOLO 權重 .pt（可選）")
    wz = st.file_uploader("自動校準 + 即時單張用；Check 不需要它", type=["pt"], key="wz")
    if wz is not None and ss.get("wz_name") != wz.name:
        (up / "weights.pt").write_bytes(wz.getbuffer()); ss.wz_name = wz.name
    if ss.get("wz_name"):
        st.success(f"已上傳權重：{ss.wz_name}")

    # ④ 模型 + 建置
    st.subheader("④ 模型與一鍵建置")
    variant = st.selectbox("DINO 特徵模型", DINO_OPTS, index=0).split("（")[0]
    per_class = st.slider("每類 Golden 取幾張", 5, 40, 20)
    ready = bool(gtz and ckz and ss.get("names") and all(ss.get("names", [])))
    if not ready:
        st.info("完成 ①②（並確認類別名都填好）即可建置。")
    if st.button("🛠️ 建立特徵庫 + 校準（非訓練）", type="primary", disabled=not ready):
        from setup_build import run_build
        bar = st.progress(0.0, "開始…")
        frac = {"split": .05, "golden": .3, "bank": .5, "calib": .7, "cache": .95, "done": 1.}
        lab = {"split": "切分GT", "golden": "裁Golden", "bank": "建特徵庫", "calib": "校準",
               "cache": "預算Check", "done": "完成"}
        def pcb(stage, i, n, msg):
            bar.progress(frac.get(stage, .5), f"{lab.get(stage, stage)} {i}/{n}  {msg}")
        weights = str(up / "weights.pt") if ss.get("wz_name") else None
        try:
            with st.spinner("建置中（CPU 視資料量數分鐘）…"):
                summ = run_build(up / "gt", up / "check", ss.names, weights=weights,
                                 dino_variant=variant, per_class=per_class, progress=pcb)
            load_cache.clear(); load_meta.clear()
            ss.build_summary = summ
            st.success(f"✅ 建置完成！Golden {sum(summ['golden'].values())} 張、"
                       f"Check {summ['n_check_imgs']} 圖/{summ['n_check_boxes']} 框、"
                       f"閾值 {summ['threshold']}（{'自動校準' if summ['calibrated'] else '預設'}）")
            st.info("👉 切到側邊「🔍 檢視結果」即可看 Inspector / Dashboard。")
        except Exception as e:
            st.error(f"建置失敗：{e}")
    if ss.get("build_summary"):
        with st.expander("最近建置摘要"):
            st.json(ss.build_summary)


# ----------------------------- 啟動 -----------------------------
cfg, CLASSES = load_meta()
cache = load_cache()
# 上傳的 Check 沒有真值 → 不顯示需要 GT 的指標(攔截率/保留率/Precision)，避免誤導
GT = {} if (cache and cache.get("source") == "upload") else load_gt()
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

st.title("🛡️ Safety Net Inspector")
mode = st.sidebar.radio("模式", ["📤 資料與設定", "🔍 檢視結果"],
                        index=(1 if cache is not None else 0))
st.sidebar.divider()

if mode.startswith("📤"):
    render_setup_page()
    st.stop()

# ---- 🔍 檢視結果 ----
st.caption("YOLO（高靈敏觸發器，只偵測「有物件」）→ SAM 去背 → DINOv2 特徵 → "
           "與 Golden 樣本比對 → 近=通過(綠) / 遠=攔截誤報(紅)。**分類由 DINO 決定，非 YOLO。**")
render_help(default_open=False)
if cache is None:
    st.info("尚未建立資料。請切到側邊「📤 資料與設定」上傳你的 GT/Check ZIP 並一鍵建置。")
    st.stop()

with st.sidebar:
    st.header("⚙️ 控制")
    thr = st.slider("DINO 判決閾值 (cosine)", 0.0, 1.0, float(cfg["matching"]["threshold"]), 0.01,
                    help="拉高=攔更兇(precision↑/可能誤殺)；拉低=放更多過(recall↑)")
    st.caption(f"類別：{', '.join(CLASSES)}")
    st.caption(f"快取：{cache['device']} · DINO {cache['dino']} · {len(cache['images'])} 張"
               + ("（上傳）" if cache.get("source") == "upload" else ""))
    show_gt = st.checkbox("疊上 Ground Truth（評測模式）", value=False, disabled=not GT,
                          help="有 GT 時顯示真目標(黃)/非目標物件(青)")
    st.divider()
    st.caption("資料：config.yaml + vector_bank.npz + gui_cache")

tab_insp, tab_dash, tab_adv = st.tabs(["🔍 Inspector", "📊 Dashboard", "⚙️ Advanced"])
names = sorted(cache["images"].keys())


# =========================== 🔍 INSPECTOR ===========================
with tab_insp:
    src = st.radio("影像來源", ["載入 Check/（瞬間）", "上傳新圖（即時推論，需載模型）"],
                   horizontal=True, key="insp_src")
    recs = full = gtinfo = None
    if src.startswith("載入"):
        name = st.selectbox("選擇影像", names, key="insp_img")
        recs = cache["images"][name]
        full = imread(CHECK_DIR / name)
        gtinfo = GT.get(name) if show_gt else None
    else:
        up = st.file_uploader("上傳影像", type=["jpg", "jpeg", "png"])
        if up is not None:
            from pipeline import process_for_gui
            net, gal = get_net_and_gallery()
            with st.spinner("跑完整管線中（CPU 約 10 餘秒）…"):
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                recs = process_for_gui(net, img, gal); full = img

    if recs is not None and full is not None:
        n_true = sum(decide(r, thr) == "True" for r in recs)
        c1, c2 = st.columns([3, 4])
        with c1:
            st.markdown(f"**原圖**　YOLO 觸發 {len(recs)} 框 → "
                        f"<span style='color:#28a050'>通過 {n_true}</span> / "
                        f"<span style='color:#e03c3c'>攔截 {len(recs)-n_true}</span>",
                        unsafe_allow_html=True)
            sel = None
            if recs:
                labels = [f"#{i} [{'✅' if decide(r,thr)=='True' else '❌'}] "
                          f"{r['pred_class']} {r['score']:.2f}" for i, r in enumerate(recs)]
                sel = labels.index(st.selectbox("選一個偵測框（看判決理由）", labels, key="insp_det"))
            st.image(draw_boxes(full, recs, thr, gtinfo, sel), width="stretch")
            st.caption(COLOR_KEY)
        with c2:
            if recs and sel is not None:
                r = recs[sel]; passed = decide(r, thr) == "True"
                st.markdown(f"### 管線瀑布　—　偵測框 #{sel}")
                w1, w2, w3 = st.columns(3)
                w1.caption("① YOLO 觸發"); w1.image(bgr2rgb(r["raw"]), width="stretch")
                w1.caption(f"「有物件」conf={r['conf']:.2f}")
                w2.caption("② SAM 去背"); w2.image(bgr2rgb(r["sam"]), width="stretch")
                w2.caption("✓ 去背成功" if r["sam_ok"] else "⚠ fallback：用原始框")
                w3.caption("③ DINO 最相似 Golden")
                for gp, gc, gs in r["golden"]:
                    a, b = w3.columns([1, 2])
                    a.image(bgr2rgb(imread(gp)), width=46); b.caption(f"{gc}\n`{gs:.2f}`")
                st.markdown("#### ④ 裁決")
                if passed:
                    st.success(f"✅ TRUE — 判定為 **{r['pred_class']}**　"
                               f"(score {r['score']:.2f} ≥ 閾值 {thr:.2f})")
                else:
                    st.error(f"❌ FALSE ALARM 攔截 — 不屬任何 Golden 類別　"
                             f"(最近 {r['pred_class']} {r['score']:.2f} < 閾值 {thr:.2f})")
                st.plotly_chart(thermometer(r["score"], thr, passed), width="stretch")
                f1, f2, _ = st.columns([1, 1, 4])
                key = f"{st.session_state.get('insp_img','up')}#{sel}"
                if f1.button("👍 同意", key=f"ok{key}"):
                    st.session_state.feedback[key] = "agree"; st.toast("已記錄：同意")
                if f2.button("👎 不同意 → 存 Hard Negative", key=f"no{key}"):
                    d = HARDNEG / r["pred_class"]; d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{key.replace('#','_')}_{r['score']:.2f}.png"), r["sam"])
                    st.session_state.feedback[key] = "disagree"; st.toast(f"已匯出 → {d}")


# =========================== 📊 DASHBOARD ===========================
with tab_dash:
    all_recs = [(n, r) for n in names for r in cache["images"][n]]
    no_det = sum(1 for n in names if not cache["images"][n])
    scores = np.array([r["score"] for _, r in all_recs])
    passed = scores >= thr
    st.subheader("指標")
    m = st.columns(5)
    m[0].metric("YOLO 觸發框", len(all_recs), help="YOLO 框出的所有可疑物件數（含誤報）")
    m[1].metric("通過 (True)", int(passed.sum()), help="DINO 認證為真目標、放行的框")
    m[2].metric("攔截 (False)", int((~passed).sum()), help="DINO 判為誤報、擋下的框")
    m[3].metric("No_Detection 圖", no_det, help="YOLO 完全沒觸發的影像數")
    if GT:
        tp = fa = inter = rok = rtot = 0
        for name, r in all_recs:
            g = GT.get(name)
            if not g:
                continue
            kind, _ = classify_box(r["box"], g["targets"], g["oods"])
            p = r["score"] >= thr
            if kind == "target":
                rtot += 1; rok += int(p); tp += 1
            elif kind in ("ood", "bg"):
                fa += 1; inter += int(not p)
        m[4].metric("False Alarm 率", f"{(fa/(tp+fa)*100 if tp+fa else 0):.0f}%",
                    help="YOLO 觸發框中，非真目標(非目標物件＋背景)的比例＝YOLO 誤報多嚴重")
        k = st.columns(3)
        k[0].metric("DINO 攔截率", f"{(inter/fa*100 if fa else 100):.1f}%",
                    help="誤報中被 DINO 正確判為 False 的比例（安全網價值，越高越好）")
        k[1].metric("真目標保留率", f"{(rok/rtot*100 if rtot else 100):.1f}%",
                    help="真目標中被正確放行(True)的比例（沒誤殺，越高越好）")
        k[2].metric("Precision（安全網後）", f"{(rok/max(1,rok+(fa-inter)))*100:.1f}%",
                    help="放行的框中真的是目標的比例（對比 YOLO 單獨的精準度）")

    st.subheader("分數分佈（綠=通過 / 紅=攔截，虛線=閾值）")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores[passed], name="通過", marker_color="#28c850",
                               xbins=dict(start=0, end=1, size=0.03)))
    fig.add_trace(go.Histogram(x=scores[~passed], name="攔截", marker_color="#eb3c3c",
                               xbins=dict(start=0, end=1, size=0.03)))
    fig.add_vline(x=thr, line_dash="dash", line_color="#222")
    fig.update_layout(barmode="stack", height=260, xaxis_title="cosine 相似度",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("偵測縮圖牆")
    f1, f2, f3 = st.columns(3)
    fdec = f1.selectbox("判決", ["全部", "只看通過", "只看攔截"])
    fcls = f2.selectbox("類別", ["全部"] + CLASSES)
    order = f3.selectbox("排序", ["分數高→低", "分數低→高"])
    items = [(n, r) for n, r in all_recs
             if (fdec == "全部" or (fdec == "只看通過") == (r["score"] >= thr))
             and (fcls == "全部" or r["pred_class"] == fcls)]
    items.sort(key=lambda x: x[1]["score"], reverse=(order == "分數高→低"))
    PAGE = 24
    npages = max(1, (len(items) + PAGE - 1) // PAGE)
    pg = st.number_input(f"頁（共 {npages}，{len(items)} 筆）", 1, npages, 1) - 1
    cols = st.columns(6)
    for j, (name, r) in enumerate(items[pg * PAGE:(pg + 1) * PAGE]):
        p = r["score"] >= thr
        with cols[j % 6]:
            st.image(bgr2rgb(r["sam"]), width="stretch")
            st.markdown(f"<div style='border-top:4px solid {'#28c850' if p else '#eb3c3c'};"
                        f"text-align:center;font-size:12px'>{r['pred_class']} {r['score']:.2f}</div>",
                        unsafe_allow_html=True)
    st.caption(f"已標記回饋：{len(st.session_state.feedback)} 筆（👎 已存於 {HARDNEG}）")


# =========================== ⚙️ ADVANCED ===========================
with tab_adv:
    st.subheader("Golden 標準樣本庫")
    gpaths = []
    for c in CLASSES:
        gpaths += [(str(p), c) for p in sorted((TRAIN_DIR / c).glob("*.png"))[:8]]
    cols = st.columns(min(8, max(1, len(gpaths))))
    for i, (p, c) in enumerate(gpaths):
        with cols[i % len(cols)]:
            st.image(bgr2rgb(imread(p)), caption=c, width="stretch")

    st.subheader("特徵空間 2D 投影（PCA）— 近=同類、遠=誤報")
    st.caption("各類 Golden 質心(大圓) + 選定影像的偵測框(星, 綠=通過/紅=攔截)。")
    try:
        from sklearn.decomposition import PCA
        b = np.load(BANK, allow_pickle=True)
        protos = b["protos"]
        dname = st.selectbox("疊加哪張 Check 影像", names, key="adv_img")
        drecs = cache["images"][dname]
        dv = np.stack([r["vec"] for r in drecs]) if drecs else np.zeros((0, protos.shape[1]))
        allv = np.vstack([protos, dv]) if len(dv) else protos
        p2 = PCA(n_components=2).fit_transform(allv)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p2[:len(protos), 0], y=p2[:len(protos), 1], mode="markers+text",
                                 text=CLASSES, textposition="top center", name="Golden 質心",
                                 marker=dict(size=20, opacity=0.85)))
        if len(dv):
            pts = p2[len(protos):]
            cols = ["#28c850" if r["score"] >= thr else "#eb3c3c" for r in drecs]
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name="偵測框",
                                     marker=dict(size=15, symbol="star", color=cols,
                                                 line=dict(width=1, color="#000"))))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.info(f"PCA 投影不可用：{e}")
