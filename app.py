import os
import re
from collections import Counter
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# =========================
# Streamlit 基本設定
# =========================
st.set_page_config(
    page_title="SMS Spam Classifier — Dashboard",
    page_icon="📨",
    layout="wide",
)


# =========================
# 文字清理：和 01_preprocess.ipynb 一致的簡化版本
# =========================
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# 載入資料 / 模型 / 向量器（加 cache）
# =========================
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_vectorizer_and_model():
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    model_path = os.path.join("models", "best_model.pkl")

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model


def token_topn(series: pd.Series, topn: int) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for s in series.astype(str):
        counter.update(s.split())
    return counter.most_common(topn)


# =========================
# 初始化：載入 artifacts
# =========================
vectorizer, model = load_vectorizer_and_model()


# =========================
# Sidebar：基本控制項
# =========================
with st.sidebar:
    st.header("設定 / Inputs")

    data_path = st.text_input(
        "Dataset CSV 路徑",
        value="data/sms_spam_no_header.csv",
        help="相對於專案根目錄的路徑",
    )

    df = load_dataset(data_path)

    # 猜 label / text 欄位（簡化版）
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in ("label", "target", "col_0")]
    text_candidates = [c for c in cols if c.lower() in ("text", "message", "text_clean", "col_1")]

    default_label = label_candidates[0] if label_candidates else cols[0]
    default_text = text_candidates[0] if text_candidates else cols[-1]

    label_col = st.selectbox("Label 欄位", options=cols, index=cols.index(default_label))
    text_col = st.selectbox("文字欄位", options=cols, index=cols.index(default_text))

    # 選正類（多半是 spam）
    unique_labels = sorted(df[label_col].astype(str).unique().tolist())
    default_pos = "spam" if "spam" in [u.lower() for u in unique_labels] else unique_labels[0]
    # 找出 default_pos 對應的原始大小寫
    for u in unique_labels:
        if u.lower() == default_pos.lower():
            default_pos = u
            break

    pos_label = st.selectbox("Positive class（正類，通常是 spam）", options=unique_labels, index=unique_labels.index(default_pos))

    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    threshold = st.slider("決策閾值（threshold）", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    topn_tokens = st.slider("Top-N tokens (per class)", min_value=10, max_value=40, value=20, step=5)


# =========================
# 主標題
# =========================
st.title("📨 SMS Spam Classifier — Dashboard")
st.caption("Data distribution · Token patterns · Model performance · Live inference")

st.markdown("---")


# =========================
# 區塊一：Data Overview
# =========================
st.subheader("1. Data Overview")

c1, c2 = st.columns(2)

with c1:
    st.write("Class distribution")
    label_counts = df[label_col].value_counts().sort_index()
    st.bar_chart(label_counts)

with c2:
    st.write("Dataset head")
    st.dataframe(df.head())


st.markdown("---")


# =========================
# 區塊二：Top Tokens by Class（大致仿老師範例）
# =========================
st.subheader("2. Top Tokens by Class (簡單 token 統計)")

# 先建立清理過文字欄位
df["_clean_text_for_tokens"] = df[text_col].astype(str).apply(clean_text)

col_a, col_b = st.columns(2)
classes_for_top = list(label_counts.index[:2])  # 取前兩個類別展示（多半就是 ham + spam）

for label, col in zip(classes_for_top, [col_a, col_b]):
    with col:
        st.write(f"Class: **{label}**")
        subset = df.loc[df[label_col] == label, "_clean_text_for_tokens"]
        top_tokens = token_topn(subset, topn_tokens)
        if top_tokens:
            toks, freqs = zip(*top_tokens)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=list(freqs), y=list(toks), ax=ax)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Token")
            st.pyplot(fig)
        else:
            st.info("No tokens found for this class.")

st.markdown("---")


# =========================
# 區塊三：Model Performance（Confusion matrix, ROC, PR, Threshold sweep）
# =========================
st.subheader("3. Model Performance on Test Split")

# 建立乾淨文字 + label
X_all = df[text_col].astype(str).apply(clean_text)
y_all = df[label_col].astype(str)

# train/test split（注意要 stratify）
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all,
    y_all,
    test_size=test_size,
    random_state=seed,
    stratify=y_all,
)

# vectorize 測試集
X_te_vec = vectorizer.transform(X_te)

# 二元 0/1 label for curves
y_true_binary = np.array([1 if y == pos_label else 0 for y in y_te])

# 取得 scores / probabilities
y_scores = None
use_proba = False

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_te_vec)
    classes = list(model.classes_)
    if pos_label in classes:
        pos_idx = classes.index(pos_label)
    else:
        pos_idx = 1 if len(classes) > 1 else 0
    y_scores = proba[:, pos_idx]
    use_proba = True
elif hasattr(model, "decision_function"):
    y_scores = model.decision_function(X_te_vec)
    use_proba = False

# 根據 threshold 做預測（如果有 score）
if y_scores is not None:
    y_pred_binary = (y_scores >= threshold).astype(int)
    # 對應回 label
    # 找一個「反類」名稱
    neg_label = [c for c in unique_labels if c != pos_label]
    neg_label = neg_label[0] if neg_label else f"not_{pos_label}"
    y_pred_labels = np.where(y_pred_binary == 1, pos_label, neg_label)
else:
    # fallback：直接用 model.predict 的 label 當作預測
    y_pred_labels = model.predict(X_te_vec)
    # 同時轉 0/1，方便後面至少畫 confusion matrix
    y_pred_binary = np.array([1 if y == pos_label else 0 for y in y_pred_labels])

# ---- Confusion matrix（label 版本） ----
cm = confusion_matrix(y_te, y_pred_labels, labels=unique_labels)
cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in unique_labels], columns=[f"pred_{l}" for l in unique_labels])

c3, c4 = st.columns(2)
with c3:
    st.write(" Confusion Matrix (table)")
    st.dataframe(cm_df)

with c4:
    st.write(" Confusion Matrix (heatmap)")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

# ---- ROC & PR curves（如果有 score）----
if y_scores is not None:
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true_binary, y_scores)
    ap = average_precision_score(y_true_binary, y_scores)

    fig_curves, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    # PR
    ax_pr.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax_pr.set_title("Precision–Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")

    st.pyplot(fig_curves)

    # ---- Threshold sweep ----
    st.write(" Threshold sweep (precision / recall / f1)")
    ths = np.round(np.linspace(0.3, 0.8, 11), 3)
    rows = []
    for t in ths:
        p_bin = (y_scores >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true_binary, p_bin, zero_division=0)),
            "recall": float(recall_score(y_true_binary, p_bin, zero_division=0)),
            "f1": float(f1_score(y_true_binary, p_bin, zero_division=0)),
        })
    st.dataframe(pd.DataFrame(rows))
else:
    st.info("當前 best_model 不支援 predict_proba / decision_function，無法繪製 ROC / PR / threshold sweep。")


st.markdown("---")


# =========================
# 區塊四：Live Inference（互動預測）
# =========================
st.subheader("4. Live Inference（即時預測）")

ex_spam = "Free entry in a weekly contest to win cash now! Click the link to claim your prize."
ex_ham = "Hi, I will arrive around 7 pm, see you then."

c_ex1, c_ex2 = st.columns(2)
with c_ex1:
    if st.button("填入 spam 範例"):
        st.session_state["input_text"] = ex_spam
with c_ex2:
    if st.button("填入 ham 範例"):
        st.session_state["input_text"] = ex_ham

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

user_text = st.text_area("請輸入要分類的訊息（SMS）：", key="input_text", height=100)

if st.button("預測"):
    if user_text.strip():
        clean_user = clean_text(user_text)
        with st.expander("顯示清理後的文字", expanded=False):
            st.code(clean_user)

        X_single = vectorizer.transform([clean_user])

        pred_label = model.predict(X_single)[0]
        score_display = None

        if hasattr(model, "predict_proba"):
            proba_single = model.predict_proba(X_single)[0]
            classes = list(model.classes_)
            if pos_label in classes:
                idx_pos = classes.index(pos_label)
            else:
                idx_pos = 1 if len(classes) > 1 else 0
            score_display = float(proba_single[idx_pos])
        elif hasattr(model, "decision_function"):
            score_display = float(model.decision_function(X_single)[0])

        if str(pred_label).lower() == str(pos_label).lower():
            st.error(f"預測結果：**{pred_label}**  （positive class: {pos_label}）")
        else:
            st.success(f"預測結果：**{pred_label}** ")

        if score_display is not None:
            if use_proba:
                st.write(f"模型對 **{pos_label}** 的信心（機率）：**{score_display:.4f}**")
            else:
                st.write(f"模型 decision score：**{score_display:.4f}**")
    else:
        st.info("請先輸入非空白訊息。")

st.markdown("---")
st.caption("Homework 3 — SMS Spam Classification · Streamlit Visual Dashboard")
