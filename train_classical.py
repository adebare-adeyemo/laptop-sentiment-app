"""
Balanced classical sentiment training that outputs the three artifacts your
predict-only app needs:
  models/classical/tfidf.joblib
  models/classical/label_encoder.joblib
  models/classical/best.joblib

What’s improved:
- Negation-aware cleaning (don’t like -> don_t_like, not good -> not_good)
- Stratified split (no leakage)
- Class-balanced training (oversample minority classes only on TRAIN)
- Stronger TF-IDF features (1–3 grams, up to 30k features)
- Compares LinearSVC and LogisticRegression; picks best by macro-F1
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import vstack as sp_vstack

# -------------------- config --------------------
DATA = "data/Labeled_Laptop_Reviews.csv"     # <-- adjust if needed
OUT = Path("models/classical")               # artifacts go here
OUT_FIG = Path("outputs/figures")            # optional plots
OUT_REP = Path("outputs/reports")            # txt/json reports
MAX_FEAT = 30000
NGRAM_MAX = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_REP.mkdir(parents=True, exist_ok=True)

# -------------------- helpers --------------------
def clean_text(s: str) -> str:
    s = str(s).lower().replace("’","'").strip()
    toks = s.split()
    neg = {"not","no","never","cant","can't","dont","don't","isnt","isn't","doesnt",
           "doesn't","didnt","didn't","won't","wont","wasn't","wasnt"}
    out=[]; i=0
    while i < len(toks):
        if toks[i] in neg and i+1 < len(toks):
            out.append(toks[i].replace("'","_")+"_"+toks[i+1]); i += 2
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)

def detect_cols(df: pd.DataFrame):
    text_cands = ["Review","Text","Sentence","comment","review_text","content","body"]
    label_cands = ["Sentiment","label","target","VADER_Sentiment","sentiment"]
    text = next((c for c in text_cands if c in df.columns), None)
    if text is None:
        objs = df.select_dtypes(include="object").columns.tolist()
        if not objs: raise ValueError("No text column found")
        text = objs[0]
    label = next((c for c in label_cands if c in df.columns), None)
    if label is None: raise ValueError("No label column found")
    return text, label

def balance_train(X_tr, y_tr):
    counts = Counter(y_tr); maxc = max(counts.values())
    idx_by = {c: np.where(y_tr==c)[0] for c in counts}
    parts_X, parts_y = [], []
    for c, idxs in idx_by.items():
        if len(idxs) < maxc:
            up = np.random.choice(idxs, size=maxc, replace=True)
            parts_X.append(X_tr[up]); parts_y.append(np.full(maxc, c))
        else:
            parts_X.append(X_tr[idxs]); parts_y.append(np.full(len(idxs), c))
    Xb = sp_vstack(parts_X); yb = np.concatenate(parts_y)
    perm = np.random.permutation(Xb.shape[0])
    return Xb[perm], yb[perm]

# -------------------- load & prep --------------------
df = pd.read_csv(DATA)
text_col, label_col = detect_cols(df)
df[label_col] = df[label_col].replace({0:"Negative",1:"Neutral",2:"Positive"})
df[text_col] = df[text_col].astype(str).apply(clean_text)
df = df.dropna(subset=[text_col, label_col]).copy()

le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str))
classes = le.classes_.tolist()

tfidf = TfidfVectorizer(stop_words="english", max_features=MAX_FEAT, ngram_range=(1, NGRAM_MAX))
X = tfidf.fit_transform(df[text_col].values)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE, stratify=y)

# balance only the training split
X_tr_bal, y_tr_bal = balance_train(X_tr, y_tr)

# -------------------- train models --------------------
models = {
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "LogReg":    LogisticRegression(max_iter=4000, class_weight="balanced"),
}

metrics = []
trained = {}
for name, clf in models.items():
    clf.fit(X_tr_bal, y_tr_bal)
    pred = clf.predict(X_te)
    f1 = f1_score(y_te, pred, average="macro")
    acc = accuracy_score(y_te, pred)
    rep = classification_report(y_te, pred, target_names=classes, zero_division=0)
    cm  = confusion_matrix(y_te, pred)

    trained[name] = clf
    metrics.append({"model": name, "f1_macro": float(f1), "accuracy": float(acc)})

    # save per-model report
    (OUT_REP / f"{name}_report.txt").write_text(rep)

# pick best by macro-F1
metrics = sorted(metrics, key=lambda m: m["f1_macro"], reverse=True)
best_name = metrics[0]["model"]
best_clf  = trained[best_name]

# -------------------- save artifacts --------------------
joblib.dump(tfidf, OUT / "tfidf.joblib")
joblib.dump(le,    OUT / "label_encoder.joblib")
joblib.dump(best_clf, OUT / "best.joblib")

# also record a JSON summary (nice to keep)
(Path(OUT_REP) / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))

print("✅ Done.")
print("Saved artifacts in:", OUT.resolve())
print("Best model:", best_name, " | F1_macro:", round(metrics[0]["f1_macro"], 4))
