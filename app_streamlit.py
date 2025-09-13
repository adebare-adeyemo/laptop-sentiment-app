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

from utils_text import detect_columns, clean_text_series, basic_clean   # <-- new import

st.set_page_config(page_title="Laptop Sentiment Analysis", layout="wide")
st.title("💻 Laptop Sentiment Analysis — Classical ML")

st.sidebar.header("Options")
mode = st.sidebar.radio("Data source", ["Upload CSV", "Use demo file"], index=0)
uploaded = st.file_uploader("Upload CSV (text + label columns)", type=["csv"])

# Load data
if mode == "Upload CSV":
    if uploaded is None:
        st.info("Upload a CSV to begin."); st.stop()
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/Labeled_Laptop_Reviews.csv")

# Detect & clean
try:
    text_col, label_col = detect_columns(df)
except Exception as e:
    st.error(str(e)); st.stop()

df = df.dropna(subset=[text_col, label_col]).copy()
df[label_col] = df[label_col].replace({0: "Negative", 1: "Neutral", 2: "Positive"})
df[text_col] = clean_text_series(df[text_col])

le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str))
class_names = le.classes_.tolist()

tfidf = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1,2))
X = tfidf.fit_transform(df[text_col].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Models (NB tuned)
models = {
    "NaiveBayes": MultinomialNB(alpha=0.5, fit_prior=False),
    "LogisticRegression": LogisticRegression(max_iter=4000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, random_state=42)
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {"model": clf, "report": rep, "cm": cm}

# Performance table
st.subheader("📊 Model Performance (Test Set)")
perf = pd.DataFrame([{
    "Model": k,
    "Accuracy (%)": round(v["report"]["accuracy"] * 100, 2),
    "Precision (macro)": round(v["report"]["macro avg"]["precision"], 3),
    "Recall (macro)": round(v["report"]["macro avg"]["recall"], 3),
    "F1 (macro)": round(v["report"]["macro avg"]["f1-score"], 3)
} for k,v in results.items()]).sort_values("F1 (macro)", ascending=False, ignore_index=True)
st.dataframe(perf, use_container_width=True)

# Confusion matrices
st.subheader("🧩 Confusion Matrices")
cols = st.columns(2); i = 0
for name, obj in results.items():
    fig, ax = plt.subplots(figsize=(4.2,3.5))
    sns.heatmap(obj["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    cols[i % 2].pyplot(fig); i += 1

# ------------------------------ SANITY CHECKER ------------------------------
st.subheader("🔍 Try Sample Predictions")

# Default to BEST model (by F1 macro)
best_name = perf.iloc[0]["Model"]
all_names = list(results.keys())
default_index = all_names.index(best_name)

model_choice = st.selectbox(
    "Choose model for prediction",
    options=all_names,
    index=default_index,
    help="Defaults to the best-performing model on this split."
)

sample_text = st.text_area("Enter a laptop review", "Battery life is bad; I don’t like this laptop.")

# Simple cue lists (you can extend these easily)
NEG_CUES = {"bad","terrible","awful","hate","regret","slow","poor","worse","worst","disappointing","buggy","crash",
            "don_t_like","not_good","not_recommend","battery_bad","overheat","lag"}
POS_CUES = {"good","great","excellent","love","amazing","fast","smooth","awesome","perfect","recommend","satisfied","happy"}

if st.button("Predict"):
    # Clean like training & search for cues
    cleaned = basic_clean(sample_text)
    toks = set(cleaned.split())
    matched_neg = sorted(list(NEG_CUES.intersection(toks)))
    matched_pos = sorted(list(POS_CUES.intersection(toks)))

    # Predict
    clf = results[model_choice]["model"]
    X_one = tfidf.transform([sample_text])  # vectorizer already handles lower/stopwords
    pred_id = int(clf.predict(X_one)[0])
    pred_label = class_names[pred_id]
    proba_txt = ""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_one)[0]
        proba_txt = "  |  " + ", ".join([f"{lbl}: {p:.2f}" for lbl, p in zip(class_names, probs)])

    st.success(f"Prediction ({model_choice}): **{pred_label}** (id={pred_id}){proba_txt}")
    st.caption(f"Class order: {class_names}")

    # Show sanity checker cues
    st.markdown("**Sanity Checker (detected cues in your text):**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔻 Negative cues:")
        if matched_neg:
            for w in matched_neg: st.markdown(f"- `{w}`")
        else:
            st.write("_None_")
    with col2:
        st.write("🔺 Positive cues:")
        if matched_pos:
            for w in matched_pos: st.markdown(f"- `{w}`")
        else:
            st.write("_None_")
