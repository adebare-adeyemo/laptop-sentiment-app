import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# Setup & artifact loading
# ===============================
MODEL_DIR = Path("models/classical")

st.set_page_config(page_title="Laptop Sentiment — Quick Predict", layout="wide")
st.title("💻 Laptop Sentiment — Quick Predict")

try:
    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    le    = joblib.load(MODEL_DIR / "label_encoder.joblib")
    clf   = joblib.load(MODEL_DIR / "best.joblib")
    CLASS_NAMES = list(le.classes_)
    st.sidebar.success("Artifacts loaded ✓")
except Exception as e:
    st.sidebar.error(
        "Could not load artifacts from models/classical/.\n\n"
        "Expected: tfidf.joblib, label_encoder.joblib, best.joblib\n\n"
        f"Details: {e}"
    )
    st.stop()

# ===============================
# Cue sets & helpers
# ===============================
def basic_clean(s: str) -> str:
    return str(s).lower().replace("’", "'").strip()

NEG_CUES_BASE = {
    "bad","terrible","awful","hate","regret","waste","slow","buggy","crash",
    "overheat","lag","poor","worse","worst","don_t_like","not_good","not_recommend"
}
POS_CUES_BASE = {
    "good","great","love","amazing","excellent","fantastic","awesome",
    "fast","smooth","recommend","best","like","thanks","thank","thank_you"
}
# Hard override cues
STRONG_NEG_CUES = {"hate","terrible","awful","regret","worst"}
STRONG_POS_CUES = {"love","amazing","excellent","fantastic","awesome","best","thanks","thank_you"}

# ===============================
# Prediction logic (no probabilities shown)
# ===============================
def predict_label(text: str,
                  close_gap: float,
                  neg_floor: float,
                  pos_floor: float,
                  neg_cues: set,
                  pos_cues: set,
                  strong_neg_cues: set,
                  strong_pos_cues: set):
    X = tfidf.transform([text])
    pred_id = int(clf.predict(X)[0])
    pred_label = CLASS_NAMES[pred_id]
    toks = set(basic_clean(text).split())

    try:
        neg_idx = CLASS_NAMES.index("Negative")
        pos_idx = CLASS_NAMES.index("Positive")
    except ValueError:
        neg_idx = pos_idx = None

    # --- HARD OVERRIDES ---
    if neg_idx is not None and any(tok in toks for tok in strong_neg_cues):
        return "Negative", neg_idx
    if pos_idx is not None and any(tok in toks for tok in strong_pos_cues):
        return "Positive", pos_idx

    # --- SOFT RULES (if model supports predict_proba) ---
    if hasattr(clf, "predict_proba") and neg_idx is not None and pos_idx is not None:
        probs = clf.predict_proba(X)[0]
        neg_prob, pos_prob = float(probs[neg_idx]), float(probs[pos_idx])
        gap = abs(pos_prob - neg_prob)
        has_neg = any(tok in toks for tok in neg_cues)
        has_pos = any(tok in toks for tok in pos_cues)

        if has_neg and (gap < close_gap) and (neg_prob >= neg_floor):
            return "Negative", neg_idx
        elif has_pos and (gap < close_gap) and (pos_prob >= pos_floor):
            return "Positive", pos_idx

    return pred_label, pred_id

# ===============================
# Sidebar — fixed but tweakable thresholds
# (remove this block if you want them hard-coded)
# ===============================
st.sidebar.header("⚙️ Soft-rule thresholds")
if "close_gap" not in st.session_state: st.session_state.close_gap = 0.15
if "neg_floor" not in st.session_state: st.session_state.neg_floor = 0.30
if "pos_floor" not in st.session_state: st.session_state.pos_floor = 0.30

st.session_state.close_gap = st.sidebar.slider("CLOSE_GAP", 0.01, 0.50, float(st.session_state.close_gap), 0.01)
st.session_state.neg_floor = st.sidebar.slider("NEG_FLOOR", 0.10, 0.80, float(st.session_state.neg_floor), 0.01)
st.session_state.pos_floor = st.sidebar.slider("POS_FLOOR", 0.10, 0.80, float(st.session_state.pos_floor), 0.01)

# ===============================
# Single prediction (no probabilities shown)
# ===============================
st.subheader("🔎 Single Review")
sample = "Thank you for the laptop — it is amazing!"
text = st.text_area("Enter a laptop review", sample, height=120)

if st.button("Predict sentiment"):
    label, _ = predict_label(
        text=text,
        close_gap=st.session_state.close_gap,
        neg_floor=st.session_state.neg_floor,
        pos_floor=st.session_state.pos_floor,
        neg_cues=NEG_CUES_BASE,
        pos_cues=POS_CUES_BASE,
        strong_neg_cues=STRONG_NEG_CUES,
        strong_pos_cues=STRONG_POS_CUES,
    )
    st.success(f"Prediction: **{label}**")

# ===============================
# Batch CSV prediction (no probabilities shown)
# ===============================
st.subheader("📦 Batch Predict (CSV)")
st.caption("Upload a CSV. I’ll detect a text column and add a **pred_label** column.")

csv = st.file_uploader("Upload CSV", type=["csv"])
if csv is not None:
    try:
        df = pd.read_csv(csv)

        # Heuristically pick a text column
        preferred = ["Review","Text","Sentence","comment","review_text","content","body","text"]
        text_col = next((c for c in preferred if c in df.columns), None)
        if text_col is None:
            obj_cols = df.select_dtypes(include="object").columns.tolist()
            text_col = obj_cols[0] if obj_cols else None

        if text_col is None:
            st.error("Could not find a text column in this CSV.")
        else:
            # Predict each row
            labels = []
            for t in df[text_col].astype(str).fillna(""):
                lbl, _ = predict_label(
                    text=t,
                    close_gap=st.session_state.close_gap,
                    neg_floor=st.session_state.neg_floor,
                    pos_floor=st.session_state.pos_floor,
                    neg_cues=NEG_CUES_BASE,
                    pos_cues=POS_CUES_BASE,
                    strong_neg_cues=STRONG_NEG_CUES,
                    strong_pos_cues=STRONG_POS_CUES,
                )
                labels.append(lbl)

            out = df.copy()
            out["pred_label"] = labels
            st.dataframe(out.head(30), use_container_width=True)
            st.download_button(
                "⬇️ Download predictions",
                data=out.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")
