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
st.title("💻 Laptop Sentiment — Quick Predict (No Training / No Dataset Upload)")

try:
    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    le    = joblib.load(MODEL_DIR / "label_encoder.joblib")
    clf   = joblib.load(MODEL_DIR / "best.joblib")
    CLASS_NAMES = list(le.classes_)
    st.sidebar.success(f"Artifacts loaded ✓  | classes: {CLASS_NAMES}")
except Exception as e:
    st.sidebar.error(
        "Could not load artifacts from models/classical/.\n\n"
        "Expected files: tfidf.joblib, label_encoder.joblib, best.joblib\n\n"
        f"Details: {e}"
    )
    st.stop()

# ===============================
# Helpers
# ===============================
NEG_CUES_BASE = {
    "bad","terrible","awful","hate","regret","waste","slow","buggy","crash",
    "overheat","lag","poor","worse","worst","don_t_like","not_good","not_recommend"
}
# NEW: hard-override cues — if any appear, force Negative
STRONG_NEG_CUES = {"hate","terrible","awful","regret","worst"}

def basic_clean(s: str) -> str:
    return str(s).lower().replace("’", "'").strip()

def predict_with_safety(text: str, close_gap: float, neg_floor: float,
                        neg_cues: set, strong_neg_cues: set):
    """
    Base prediction + probability-based safety override + HARD override for strong negative cues.
    """
    X = tfidf.transform([text])
    pred_id = int(clf.predict(X)[0])
    pred_label = CLASS_NAMES[pred_id]
    proba_txt = ""

    toks = set(basic_clean(text).split())

    # --- HARD OVERRIDE ---
    # If any strong negative cue is present, force Negative immediately.
    if any(tok in toks for tok in strong_neg_cues):
        try:
            neg_idx = CLASS_NAMES.index("Negative")
            pred_label = "Negative"
            pred_id = neg_idx
            # still show probabilities if available
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                proba_txt = "  |  " + ", ".join(
                    [f"{lbl}: {p:.2f}" for lbl, p in zip(CLASS_NAMES, probs)]
                )
            return pred_label, pred_id, proba_txt
        except ValueError:
            pass  # fall back if "Negative" not in classes

    # --- Soft (probability-based) rule ---
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        proba_txt = "  |  " + ", ".join(
            [f"{lbl}: {p:.2f}" for lbl, p in zip(CLASS_NAMES, probs)]
        )
        try:
            neg_idx = CLASS_NAMES.index("Negative")
            pos_idx = CLASS_NAMES.index("Positive")
            neg_prob = float(probs[neg_idx])
            pos_prob = float(probs[pos_idx])
            has_neg = any(tok in toks for tok in neg_cues)

            # If clear negative cues & close call, lean Negative
            if has_neg and (abs(pos_prob - neg_prob) < close_gap) and (neg_prob >= neg_floor):
                pred_label = "Negative"
                pred_id = neg_idx
        except Exception:
            pass

    return pred_label, pred_id, proba_txt

def evaluate_thresholds(df: pd.DataFrame, close_gap_vals, neg_floor_vals, neg_cues: set):
    """Grid-search thresholds to maximize F1 for the Negative class."""
    if not hasattr(clf, "predict_proba"):
        st.warning("This model has no probabilities (predict_proba). Auto-tune disabled.")
        return None, None, None

    X_all = tfidf.transform(df["text"].astype(str).tolist())
    probs_all = clf.predict_proba(X_all)
    label_to_idx = {lbl: i for i, lbl in enumerate(CLASS_NAMES)}
    y_true = np.array([label_to_idx.get(y, -1) for y in df["label"].astype(str)])
    neg_idx = label_to_idx.get("Negative", None)
    pos_idx = label_to_idx.get("Positive", None)
    if neg_idx is None or pos_idx is None:
        st.error("Labels must include 'Negative' and 'Positive' for auto-tune.")
        return None, None, None

    tokens = [set(basic_clean(t).split()) for t in df["text"]]
    has_neg_cue = np.array([any(tok in neg_cues for tok in toks) for toks in tokens])

    best_f1_neg = -1.0
    best_gap, best_floor = None, None

    for gap in close_gap_vals:
        for floor in neg_floor_vals:
            preds = []
            for i in range(len(df)):
                p = probs_all[i]
                base = int(np.argmax(p))
                pred = base
                if has_neg_cue[i] and abs(p[pos_idx] - p[neg_idx]) < gap and p[neg_idx] >= floor:
                    pred = neg_idx
                preds.append(pred)

            preds = np.array(preds)
            tp = np.sum((preds == neg_idx) & (y_true == neg_idx))
            fp = np.sum((preds == neg_idx) & (y_true != neg_idx))
            fn = np.sum((preds != neg_idx) & (y_true == neg_idx))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1_neg:
                best_f1_neg = f1
                best_gap, best_floor = gap, floor

    return best_gap, best_floor, best_f1_neg

# ===============================
# Sidebar — Auto-Tune (optional)
# ===============================
st.sidebar.header("⚙️ Auto-Tune thresholds (optional)")
st.sidebar.caption("Upload a small labeled CSV to learn best thresholds for the soft safety rule.")
cal_csv = st.sidebar.file_uploader("CSV with columns: text, label", type=["csv"], key="cal_csv")

DEFAULT_CLOSE_GAP = 0.15
DEFAULT_NEG_FLOOR = 0.30

if "close_gap" not in st.session_state:
    st.session_state.close_gap = DEFAULT_CLOSE_GAP
if "neg_floor" not in st.session_state:
    st.session_state.neg_floor = DEFAULT_NEG_FLOOR

if cal_csv is not None:
    try:
        df_cal = pd.read_csv(cal_csv)
        if not {"text", "label"}.issubset(set(df_cal.columns)):
            st.sidebar.error("CSV must have 'text' and 'label' columns.")
        else:
            gaps = np.arange(0.05, 0.31, 0.05)
            floors = np.arange(0.20, 0.61, 0.05)
            gap, floor, f1neg = evaluate_thresholds(df_cal, gaps, floors, NEG_CUES_BASE)
            if gap is not None:
                st.session_state.close_gap = float(gap)
                st.session_state.neg_floor = float(floor)
                st.sidebar.success(
                    f"Tuned ✅  CLOSE_GAP={gap:.2f}, NEG_FLOOR={floor:.2f} | F1(Neg)≈{f1neg:.3f}"
                )
            else:
                st.sidebar.warning("Auto-tune could not determine thresholds.")
    except Exception as e:
        st.sidebar.error(f"Auto-tune error: {e}")

st.sidebar.markdown("### Thresholds (soft rule)")
st.session_state.close_gap = st.sidebar.slider(
    "CLOSE_GAP (closeness of Pos vs Neg)", 0.01, 0.50, float(st.session_state.close_gap), 0.01
)
st.session_state.neg_floor = st.sidebar.slider(
    "NEG_FLOOR (min Negative probability)", 0.10, 0.80, float(st.session_state.neg_floor), 0.01
)

# ===============================
# Main — Single prediction
# ===============================
st.subheader("🔎 Single Review")
sample = "Battery life is bad; I hate this laptop."
text = st.text_area("Enter a laptop review", sample, height=120)

if st.button("Predict sentiment"):
    label, pid, proba_txt = predict_with_safety(
        text=text,
        close_gap=st.session_state.close_gap,
        neg_floor=st.session_state.neg_floor,
        neg_cues=NEG_CUES_BASE,
        strong_neg_cues=STRONG_NEG_CUES,
    )
    st.success(f"Prediction: **{label}** (id={pid}){proba_txt}")
    st.caption(f"Class order: {CLASS_NAMES}")
    st.markdown("**🔎 Sanity Checker (tokens):**")
    st.code(", ".join(sorted(set(basic_clean(text).split()))) or "(none)")

# ===============================
# Optional — Batch predict CSV
# ===============================
st.subheader("📦 Batch Predict (optional CSV)")
csv = st.file_uploader("Upload a CSV for predictions. I will auto-pick a text column.", type=["csv"])
if csv is not None:
    df = pd.read_csv(csv)
    text_col = next((c for c in ["Review","Text","Sentence","comment","review_text","content","body","text"]
                     if c in df.columns), None)
    if text_col is None:
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        text_col = obj_cols[0] if obj_cols else None

    if text_col is None:
        st.error("Could not find a text column in this CSV.")
    else:
        out = df.copy()
        labels, ids = [], []
        for t in out[text_col].astype(str).fillna(""):
            lbl, pid, _ = predict_with_safety(
                text=t,
                close_gap=st.session_state.close_gap,
                neg_floor=st.session_state.neg_floor,
                neg_cues=NEG_CUES_BASE,
                strong_neg_cues=STRONG_NEG_CUES,
            )
            labels.append(lbl); ids.append(pid)
        out["pred_id"] = ids
        out["pred_label"] = labels
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("⬇️ Download predictions", data=out.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
