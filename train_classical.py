# train_classical.py
import argparse, json, os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from utils_text import detect_columns, clean_text_series

def plot_confusion(cm, classes, title, outpath):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main(args):
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    text_col, label_col = detect_columns(df)

    # Clean text
    df[text_col] = clean_text_series(df[text_col])

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    class_names = le.classes_.tolist()

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english",
                            max_features=args.max_features,
                            ngram_range=(1, args.ngram_max))
    X = tfidf.fit_transform(df[text_col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Models
    models = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=None),
        "LinearSVC": LinearSVC(),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    }

    summary = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        # report & confusion
        report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Save figures
        fig_path = out_dir / "figures" / f"confusion_{name}.png"
        plot_confusion(cm, class_names, f"{name} Confusion Matrix", fig_path)

        # Append summary
        summary.append({
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(p), 4),
            "recall_macro": round(float(r), 4),
            "f1_macro": round(float(f1), 4)
        })

        # Save per-model artifacts
        joblib.dump(model, model_dir / f"{name}.joblib")

        # Save reports
        with open(out_dir / "reports" / f"{name}_classification_report.txt", "w") as f:
            f.write(report_text)

    # Save vectorizer & label encoder for later inference
    joblib.dump(tfidf, model_dir / "tfidf.joblib")
    joblib.dump(le,    model_dir / "label_encoder.joblib")

    # Save metrics summary
    with open(out_dir / "reports" / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save comparison bar chart
    fig = plt.figure(figsize=(7,4))
    models_names = [m["model"] for m in summary]
    accs = [m["accuracy"]*100 for m in summary]
    plt.bar(models_names, accs)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "accuracy_comparison.png", dpi=200)
    plt.close()

    print("✅ Done. Artifacts saved to:", out_dir, "and", model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Labeled_Laptop_Reviews.csv", type=str)
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument("--model_dir", default="models/classical", type=str)
    parser.add_argument("--test_size", default=0.2, type=float)     # train/test split ratio: 80/20
    parser.add_argument("--max_features", default=3000, type=int)
    parser.add_argument("--ngram_max", default=1, type=int)         # set 2 for bigrams if desired
    args = parser.parse_args()
    main(args)
