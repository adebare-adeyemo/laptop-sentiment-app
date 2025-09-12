import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# -----------------------------
# Helpers
# -----------------------------
TEXT_CANDIDATES = ["Sentence", "Review", "Text", "comment", "review_text", "content", "body"]
LABEL_CANDIDATES = ["label", "Sentiment", "VADER_Sentiment", "target", "sentiment"]

def detect_columns(df: pd.DataFrame):
    text_col, label_col = None, None
    for c in df.columns:
        if c in TEXT_CANDIDATES: text_col = c; break
    if text_col is None:
        obj = df.select_dtypes(include="object").columns.tolist()
        if obj: text_col = obj[0]
    for c in df.columns:
        if c in LABEL_CANDIDATES: label_col = c; break
    if label_col is None:
        for c in df.columns:
            if c.lower() in ["label","sentiment","target"]:
                label_col = c; break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Columns: {list(df.columns)}")
    return text_col, label_col

def clean_text(s):
    if not isinstance(s, str): s = str(s)
    s = s.strip().lower()
    return " ".join(s.split())

def plot_cm(cm, classes, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    return fig

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Laptop Sentiment Analysis", layout="wide")
st.title("💻 Laptop Sentiment Analysis — Classical ML")

st.sidebar.header("Options")
use_uploaded = st.sidebar.checkbox("Use uploaded CSV (else demo)", value=True)
uploaded = st.file_uploader("Upload CSV with text + label columns", type=["csv"])

st.caption("Tip: Expected columns look like `Review`/`Text` and `Sentiment`/`label`. The app will auto-detect.")

# Load data
if use_uploaded and uploaded is not None:
    df = pd.read_csv(uploaded)
elif not use_uploaded:
    df = pd.read_csv("data/Labeled_Laptop_Reviews.csv")
else:
    st.info("Upload a CSV to begin, or uncheck 'Use uploaded CSV' to use the demo dataset in `data/`.")
    st.stop()

# Detect columns
try:
    text_col, label_col = detect_columns(df)
except Exception as e:
    st.error(str(e)); st.stop()

# Clean + encode
df = df.dropna(subset=[text_col, label_col]).copy()
df[text_col] = df[text_col].astype(str).apply(clean_text)

le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str))
class_names = le.classes_.tolist()

# Vectorize
tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 1))
X = tfidf.fit_transform(df[text_col].values)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVC": LinearSVC(),
    "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {"model": clf, "report": rep, "cm": cm}

# Summary table
st.subheader("📊 Model Performance (Test Set)")
rows = []
for name, obj in results.items():
    rep = obj["report"]
    rows.append({
        "Model": name,
        "Accuracy (%)": round(rep["accuracy"]*100, 2),
        "Precision (macro)": round(rep["macro avg"]["precision"], 3),
        "Recall (macro)": round(rep["macro avg"]["recall"], 3),
        "F1 (macro)": round(rep["macro avg"]["f1-score"], 3)
    })
perf_df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False, ignore_index=True)
st.dataframe(perf_df, use_container_width=True)

# Confusion matrices
st.subheader("🧩 Confusion Matrices")
cm_cols = st.columns(2)
i = 0
for name, obj in results.items():
    fig = plot_cm(obj["cm"], class_names, f"{name}")
    cm_cols[i % 2].pyplot(fig)
    i += 1

# -----------------------------
# Try Sample Predictions (fixed mapping + optional probs)
# -----------------------------
st.subheader("🔍 Try Sample Predictions")

# Model selector + default to best F1
best_name = perf_df.iloc[0]["Model"]
model_choice = st.selectbox(
    "Choose model for prediction",
    options=list(results.keys()),
    index=list(results.keys()).index(best_name)
)

sample_text = st.text_area(
    "Enter a laptop review",
    "Battery life is amazing but the keyboard feels cheap."
)

if st.button("Predict"):
    clf = results[model_choice]["model"]
    X_one = tfidf.transform([clean_text(sample_text)])
    pred_id = int(clf.predict(X_one)[0])
    pred_label = class_names[pred_id]

    # Optional probabilities (if supported)
    proba_txt = ""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_one)[0]
        proba_txt = "  |  " + ", ".join([f"{lbl}: {p:.2f}" for lbl, p in zip(class_names, probs)])

    st.success(f"Prediction ({model_choice}): **{pred_label}** (id={pred_id}){proba_txt}")
    st.caption(f"Class order used by the model: {class_names}")
