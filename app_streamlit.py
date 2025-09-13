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
# Helpers / cue sets
# ===============================
def basic_clean(s: str) -> str:
    return str(s).lower().replace("’", "'").strip()

# Soft cues (for close-call nudges)
NEG_CUES_BASE = {
    "bad","terrible","awful","hate","regret","waste","slow","buggy","crash",
    "overheat","lag","poor","worse","worst","don_t_like","not_good","not_recommend"
}
POS_CUES_BASE = {
    "good","great","love","amazing","excellent","fantastic","awesome",
    "fast","smooth","recommend","best","like","thanks","thank","thank_you"
}

# HARD overrides — if any appear, force class regardless of probabilities
STRONG_NEG_CUES = {"hate","terrible","awful","regret","worst"}
STRONG_POS_CUES = {"love","amazing","excellent","fantastic","awesome","best","thanks","thank_you"}

def predict_with_safety(text: str,
                        close_gap: float,
                        neg_floor: float,
                        pos_floor: float,
                        neg_cues: set,
                        pos_cues: set,
                        strong_neg_cues: set,
                        strong_pos_cues: set):
    """
    Base prediction + (1) HARD overrides for strong cues, then (2) soft, probability-based nudges.
    Soft rules are symmetric: lean Negative (or Positive) on close calls with cues present.
    """
    X = tfidf.transform([text])
    pred_id = int(clf.predict(X)[0])
    pred_label = CLASS_NAMES[pred_id]
    proba_txt = ""
    toks = set(basic_clean(text).split())

    # ---------- HARD OVERRIDES ----------
    try:
        neg_idx = CLASS_NAMES.index("Negative")
        pos_idx = CLASS_NAMES.index("Positive")
    except ValueError:
        neg_idx = pos_idx = None

    # 1) If strong neg cue present, force Negative
    if neg_idx is not None and any(tok in toks for tok in strong_neg_cues):
        pred_label, pred_id = "Negative", neg_idx
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[0]
            proba_txt = "  |  " + ", ".join([f"{lbl}: {v:.2f}" for lbl, v in zip(CLASS_NAMES, p)])
        return pred_label, pred_id, proba_txt

    # 2) If strong pos cue present (AND no strong neg), force Positive
    if pos_idx is not None and any(tok in toks for tok in strong_pos_cues):
        pred_label, pred_id = "Positive", pos_idx
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[0]
            proba_txt = "  |  " + ", ".join([f"{lbl}: {v:.2f}" for lbl, v in zip(CLASS_NAMES, p)])
        return pred_label, pred_id, proba_txt

    # ---------- SOFT RULES (probability-based) ----------
    if hasattr(clf, "predict_proba") and (neg_idx is not None) and (pos_idx is not None):
        probs = clf.predict_proba(X)[0]
        proba_txt = "  |  " + ", ".join([f"{lbl}: {p:.2f}" for lbl, p in zip(CLASS_NAMES, probs)])

        neg_prob = float(probs[neg_idx])
        pos_prob = float(probs[pos_idx])
        gap = abs(pos_prob - neg_prob)

        has_neg = any(tok in toks for tok in neg_cues)
        has_pos = any(tok in toks for tok in pos_cues)

        # Lean Negative on close-call negatives
        if has_neg and (gap < close_gap) and (neg_prob >= neg_floor):
            pred_label, pred_id = "Negative", neg_idx

        # Lean Positive on close-call positives (only if we didn't already flip to Negative)
        elif has_pos and (gap < close_gap) and (pos_prob >= pos_floor):
            pred_label, pred_id = "Positive", pos_idx

    return pred_label, pred_id, proba_txt

# ===============================
# Sidebar — Auto-Tune (optional)
# ===============================
st.sidebar.header("⚙️ Auto-Tune thresholds (optional)")
st.sidebar.caption("Upload a small labeled CSV (columns: text, label) to tune the soft rule.\n"
                   "Hard overrides always apply for strong cues.")

DEFAULT_CLOSE_GAP = 0.15
DEFAULT_NEG_FLOOR = 0.30
DEFAULT_POS_FLOOR = 0.30

if "close_gap" not in st.session_state: st.session_state.close_gap = DEFAULT_CLOSE_GAP
if "neg_floor" not in st.session_state: st.session_state.neg_floor = DEFAULT_NEG_FLOOR
if "pos_floor" not in st.session_state: st.session_state.pos_floor = DEFAULT_POS_FLOOR

cal_csv = st.sidebar.file_uploader("CSV with columns: text, label", type=["csv"], key="cal_csv")
if cal_csv is not None and hasattr(clf, "predict_proba"):
    try:
        df_cal = pd.read_csv(cal_csv)
        if not {"text","label"}.issubset(df_cal.columns):
            st.sidebar.error("CSV must have 'text' and 'label' columns.")
        else:
            Xc = tfidf.transform(df_cal["text"].astype(str))
            Pc = clf.predict_proba(Xc)
            label_to_idx = {lbl:i for i,lbl in enumerate(CLASS_NAMES)}
            if "Negative" in label_to_idx and "Positive" in label_to_idx:
                neg_i, pos_i = label_to_idx["Negative"], label_to_idx["Positive"]
                toks = [set(basic_clean(t).split()) for t in df_cal["text"]]
                has_neg = np.array([any(x in NEG_CUES_BASE for x in tk) for tk in toks])
                has_pos = np.array([any(x in POS_CUES_BASE for x in tk) for tk in toks])
                y_true = np.array([label_to_idx.get(y, -1) for y in df_cal["label"].astype(str)])

                def f1_for(gap, nfloor, pfloor):
                    preds = []
                    for i in range(len(df_cal)):
                        p = Pc[i]; base = int(np.argmax(p)); pred = base
                        gap_i = abs(p[pos_i]-p[neg_i])
                        if has_neg[i] and (gap_i < gap) and (p[neg_i] >= nfloor):
                            pred = neg_i
                        elif has_pos[i] and (gap_i < gap) and (p[pos_i] >= pfloor):
                            pred = pos_i
                        preds.append(pred)
                    preds = np.array(preds)

                    # F1 for both classes (macro of Pos/Neg only)
                    def f1_for_class(c):
                        tp = np.sum((preds==c)&(y_true==c))
                        fp = np.sum((preds==c)&(y_true!=c))
                        fn = np.sum((preds!=c)&(y_true==c))
                        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
                        return (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
                    return (f1_for_class(neg_i)+f1_for_class(pos_i))/2.0

                gaps   = np.arange(0.05, 0.31, 0.05)
                nfloors= np.arange(0.20, 0.61, 0.05)
                pfloors= np.arange(0.20, 0.61, 0.05)

                best=(None, None, None); best_score=-1.0
                for g in gaps:
                    for nf in nfloors:
                        for pf in pfloors:
                            score = f1_for(g, nf, pf)
                            if score>best_score:
                                best=(g,nf,pf); best_score=score

                if best[0] is not None:
                    st.session_state.close_gap = float(best[0])
                    st.session_state.neg_floor = float(best[1])
                    st.session_state.pos_floor = float(best[2])
                    st.sidebar.success(
                        f"Tuned ✅ CLOSE_GAP={best[0]:.2f} | NEG_FLOOR={best[1]:.2f} | "
                        f"POS_FLOOR={best[2]:.2f} | Macro F1(Neg/Pos)≈{best_score:.3f}"
                    )
    except Exception as e:
        st.sidebar.error(f"Auto-tune error: {e}")

# Manual sliders
st.sidebar.markdown("### Thresholds (soft rule)")
st.session_state.close_gap = st.sidebar.slider("CLOSE_GAP (|Pos−Neg|)", 0.01, 0.50, float(st.session_state.close_gap), 0.01)
st.session_state.neg_floor = st.sidebar.slider("NEG_FLOOR (min P(Neg))", 0.10, 0.80, float(st.session_state.neg_floor), 0.01)
st.session_state.pos_floor = st.sidebar.slider("POS_FLOOR (min P(Pos))", 0.10, 0.80, float(st.session_state.pos_floor), 0.01)

# ===============================
# Main — Single prediction
# ===============================
st.subheader("🔎 Single Review")
sample = "Thank you for the laptop — it is amazing!"
text = st.text_area("Enter a laptop review", sample, height=120)

if st.button("Predict sentiment"):
    label, pid, proba_txt = predict_with_safety(
        text=text,
        close_gap=st.session_state.close_gap,
        neg_floor=st.session_state.neg_floor,
        pos_floor=st.session_state.pos_floor,
        neg_cues=NEG_CUES_BASE,
        pos_cues=POS_CUES_BASE,
        strong_neg_cues=STRONG_NEG_CUES,
        strong_pos_cues=STRONG_POS_CUES,
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
                pos_floor=st.session_state.pos_floor,
                neg_cues=NEG_CUES_BASE,
                pos_cues=POS_CUES_BASE,
                strong_neg_cues=STRONG_NEG_CUES,
                strong_pos_cues=STRONG_POS_CUES,
            )
            labels.append(lbl); ids.append(pid)
        out["pred_id"] = ids
        out["pred_label"] = labels
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("⬇️ Download predictions", data=out.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
