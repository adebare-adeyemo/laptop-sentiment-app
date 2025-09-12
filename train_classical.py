import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# ---------- Utils ----------
TEXT_CANDIDATES = ["Sentence","Review","Text","comment","review_text","content","body"]
LABEL_CANDIDATES = ["label","Sentiment","VADER_Sentiment","target","sentiment"]

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

def clean_text(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.lower().strip()
    s = " ".join(s.split())
    # simple negation handling: join "not good" -> "not_good"
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] in {"not","no","never","cannot","can't","dont","don't","isn't","wasn't","won't"} and i+1 < len(tokens):
            out.append(tokens[i] + "_" + tokens[i+1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def plot_confusion(cm, classes, title, outpath):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

# ---------- Training ----------
def main(args):
    data_path = Path(args.data)
    out_dir   = Path(args.out_dir);   (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    rep_dir   = out_dir / "reports";  rep_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    text_col, label_col = detect_columns(df)
    df = df.dropna(subset=[text_col, label_col]).copy()

    # 1) Normalize labels to HUMAN-READABLE strings BEFORE encoding
    #    If your CSV already has words, this does nothing.
    numeric_to_name = {0: "Negative", 1: "Neutral", 2: "Positive"}  # adjust if your dataset uses other codes
    df[label_col] = df[label_col].replace(numeric_to_name)

    # 2) Clean text (with simple negation handling)
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    # 3) Encode labels to integers (ordered by string sort of class names)
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str))
    class_names = le.classes_.tolist()

    # 4) TF-IDF with bigrams + larger vocab
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,   # default 20000
        ngram_range=(1, args.ngram_max)   # default (1,2) -> unigrams + bigrams
    )
    X = tfidf.fit_transform(df[text_col].values)

    # 5) Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # 6) Train models (add class_weight to LR & LinearSVC to combat imbalance)
    models = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=4000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, random_state=42)
    }

    summary = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        report_text = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        # Save confusion matrix
        plot_confusion(cm, class_names, f"{name} Confusion Matrix", out_dir / "figures" / f"confusion_{name}.png")

        # Save per-model artifacts
        joblib.dump(clf, model_dir / f"{name}.joblib")

        # Save classification report
        with open(rep_dir / f"{name}_classification_report.txt", "w") as f:
            f.write(report_text)

        summary.append({
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(p), 4),
            "recall_macro": round(float(r), 4),
            "f1_macro": round(float(f1), 4)
        })

    # Persist vectorizer + encoder for the app
    joblib.dump(tfidf, model_dir / "tfidf.joblib")
    joblib.dump(le,    model_dir / "label_encoder.joblib")

    # Save summary JSON & quick accuracy bar
    with open(rep_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Accuracy comparison chart
    import matplotlib.pyplot as plt
    models_names = [m["model"] for m in summary]
    accs = [m["accuracy"]*100 for m in summary]
    plt.figure(figsize=(7,4)); plt.bar(models_names, accs)
    plt.ylabel("Accuracy (%)"); plt.title("Model Accuracy Comparison")
    plt.tight_layout(); plt.savefig(out_dir / "figures" / "accuracy_comparison.png", dpi=200); plt.close()

    print("✅ Training complete.")
    print("Classes (id order):", class_names)
    print("Artifacts saved in:", out_dir, "and", model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Labeled_Laptop_Reviews.csv", type=str)
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument("--model_dir", default="models/classical", type=str)
    parser.add_argument("--test_size", default=0.2, type=float)        # 80/20 split
    parser.add_argument("--max_features", default=20000, type=int)     # larger vocab
    parser.add_argument("--ngram_max", default=2, type=int)            # bigrams ON
    args = parser.parse_args()
    main(args)
