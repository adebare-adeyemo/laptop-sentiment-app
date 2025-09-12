# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from utils_text import detect_columns, clean_text_series

st.set_page_config(page_title="Laptop Sentiment Analysis", layout="wide")
st.title("💻 Laptop Sentiment Analysis — Classical ML (+ optional BERT)")

# Sidebar
st.sidebar.header("Options")
use_uploaded = st.sidebar.checkbox("Use uploaded CSV (else demo)", value=True)
show_bert = st.sidebar.checkbox("Enable Hugging Face BERT (inference only)", value=False)
threshold = st.sidebar.slider("Decision Threshold (LogReg/MLP only, affects proba-based models)", 0.0, 1.0, 0.5, 0.01)

uploaded = st.file_uploader("Upload CSV with text + label columns", type=["csv"])

def load_and_prepare(df):
    text_col, label_col = detect_columns(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = clean_text_series(df[text_col])
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    X_text = df[text_col].values
    return X_text, y, le, text_col, label_col

def train_quick(X_text, y, max_features=3000, ngram_max=1):
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1, ngram_max))
    X = tfidf.fit_transform(X_text)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)
            y_pred = (y_proba.max(axis=1) >= threshold).astype(int)  # rough for 2 classes; for 3 classes use argmax
            y_pred = clf.predict(X_test)  # For multi-class: stick to predicted class (threshold is less direct)
        else:
            y_pred = clf.predict(X_test)

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"report": rep, "cm": cm, "model": clf}

    return tfidf, results, (X_train, X_test, y_train, y_test)

# Load data
if use_uploaded and uploaded is not None:
    df = pd.read_csv(uploaded)
elif not use_uploaded:
    df = pd.read_csv("data/Labeled_Laptop_Reviews.csv")
else:
    st.info("Upload a CSV to begin or uncheck 'Use uploaded CSV' to use the demo dataset in data/.")
    st.stop()

# Process & train
try:
    X_text, y, le, text_col, label_col = load_and_prepare(df)
except Exception as e:
    st.error(f"Column detection failed: {e}")
    st.stop()

tfidf, results, split = train_quick(X_text, y, max_features=3000, ngram_max=1)
X_train, X_test, y_train, y_test = split
class_names = le.classes_.tolist()

# Show metrics table
st.subheader("📊 Model Performance (Test Set)")
table_rows = []
for name, obj in results.items():
    rep = obj["report"]
    table_rows.append({
        "Model": name,
        "Accuracy": round(rep["accuracy"] * 100, 2),
        "Precision (macro)": round(rep["macro avg"]["precision"], 3),
        "Recall (macro)": round(rep["macro avg"]["recall"], 3),
        "F1 (macro)": round(rep["macro avg"]["f1-score"], 3),
    })
perf_df = pd.DataFrame(table_rows)
st.dataframe(perf_df, use_container_width=True)

# Confusion matrices
st.subheader("🧩 Confusion Matrices")
cols = st.columns(2)
i = 0
for name, obj in results.items():
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    sns.heatmap(obj["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    cols[i % 2].pyplot(fig)
    i += 1

# Sample predictions viewer
st.subheader("🔍 Try Sample Predictions")
sample_text = st.text_area("Enter a laptop review", "Battery life is amazing but the keyboard feels cheap.")
if st.button("Predict with best classical model"):
    # pick best by F1 macro
    best = max(results.items(), key=lambda kv: kv[1]["report"]["macro avg"]["f1-score"])
    best_name, best_obj = best
    clf = best_obj["model"]
    X_one = tfidf.transform([sample_text])
    pred_id = clf.predict(X_one)[0]
    st.write(f"**Prediction (best model = {best_name}):** {class_names[pred_id]}")

# Optional: BERT pipeline (quick inference)
if show_bert:
    st.subheader("🤖 BERT (quick inference)")
    try:
        from transformers import pipeline
        hf = pipeline("sentiment-analysis")
        out = hf(sample_text)[0]
        st.write("HF Output:", out)
        st.caption("Tip: For multi-class laptop sentiment, fine-tune a HF model with train_bert.py.")
    except Exception as e:
        st.warning(f"Hugging Face not available ({e}). Install transformers/torch to enable.")
