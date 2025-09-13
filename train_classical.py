"""
Fast classical fix + keyword boosting:
- LogisticRegression (probabilities + thresholding)
- Negation-aware cleaning ("don't like" -> "don_t_like")
- Class-balanced oversampling on TRAIN split
- TF-IDF (1–3 grams, up to 20k features)
- Keyword boosting: repeats severe negative tokens so TF-IDF learns them

Outputs (for app_predict_only.py):
  models/classical/tfidf.joblib
  models/classical/label_encoder.joblib
  models/classical/best.joblib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.sparse import vstack as sp_vstack

# ---------- Config ----------
DATA = "data/Labeled_Laptop_Reviews.csv"  # change if needed
OUT_DIR = Path("models/classical")
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 20000
NGRAM_MAX = 3

# Keyword boosting (repeat tokens to increase TF-IDF weight)
NEG_KEYWORDS = {
    "bad","terrible","awful","hate","regret","waste","slow","buggy","crash",
    "overheat","lag","poor","worse","worst","don_t_like","not_good","not_recommend",
}
BOOST_FACTOR = 3  # each matched token is repeated (original + 3x)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def clean_text(s: str) -> str:
    s = str(s).lower().replace("’","'").strip()
    toks = s.split()
    neg = {"not","no","never","cant","can't","dont","don't","isnt","isn't","doesnt","doesn't",
           "didnt","didn't","won't","wont","wasn't","wasnt"}
    out=[]; i=0
    while i < len(toks):
        if toks[i] in neg and i+1 < len(toks):
            out.append(toks[i].replace("'","_")+"_"+toks[i+1]); i += 2
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)

def detect_cols(df: pd.DataFrame):
    text_candidates = ["Review","Text","Sentence","comment","review_text","content","body"]
    label_candidates = ["Sentiment","label","target","VADER_Sentiment","sentiment"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        obj = df.select_dtypes(include="object").columns.tolist()
        if not obj: raise ValueError("No text column found")
        text_col = obj[0]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError("No label column found")
    return text_col, label_col

def oversample_train(X_tr, y_tr):
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

def boost_keywords(doc: str) -> str:
    """Repeat NEG_KEYWORDS in the doc to boost their TF-IDF weights."""
    toks = doc.split()
    extra = []
    for t in toks:
        if t in NEG_KEYWORDS:
            extra.extend([t] * BOOST_FACTOR)
    return " ".join(toks + extra)  # original + boosted copies

# ---------- Load & prep ----------
df = pd.read_csv(DATA)
text_col, label_col = detect_cols(df)

# Map numeric labels if present
df[label_col] = df[label_col].replace({0:"Negative",1:"Neutral",2:"Positive"})

# Clean + Boost
df[text_col] = df[text_col].astype(str).apply(clean_text).apply(boost_keywords)
df = df.dropna(subset=[text_col, label_col]).copy()

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str))

# Features
tfidf = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, ngram_range=(1, NGRAM_MAX))
X = tfidf.fit_transform(df[text_col].values)

# Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Balance training split only
X_tr_bal, y_tr_bal = oversample_train(X_tr, y_tr)

# ---------- Train Logistic Regression ----------
clf = LogisticRegression(max_iter=4000, class_weight="balanced")
clf.fit(X_tr_bal, y_tr_bal)

# Evaluate
pred = clf.predict(X_te)
f1 = f1_score(y_te, pred, average="macro")
acc = accuracy_score(y_te, pred)
print(f"Macro-F1: {f1:.4f} | Accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_te, pred, target_names=le.classes_.tolist(), zero_division=0))

# ---------- Save artifacts ----------
joblib.dump(tfidf, OUT_DIR / "tfidf.joblib")
joblib.dump(le,    OUT_DIR / "label_encoder.joblib")
joblib.dump(clf,   OUT_DIR / "best.joblib")

print("\nSaved: models/classical/{tfidf.joblib,label_encoder.joblib,best.joblib}")
