import streamlit as st
import joblib
from pathlib import Path

# -------------------------------
# Load artifacts
# -------------------------------
MODEL_DIR = Path("models/classical")

try:
    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    clf = joblib.load(MODEL_DIR / "best.joblib")
    class_names = list(label_encoder.classes_)
    st.sidebar.success(
        f"Loaded model: best.joblib | classes: {class_names}"
    )
except Exception as e:
    st.sidebar.error(f"Could not load artifacts from {MODEL_DIR}.\n\nDetails: {e}")
    st.stop()

# -------------------------------
# Helper: basic clean
# -------------------------------
def basic_clean(s: str) -> str:
    return str(s).lower().replace("’", "'").strip()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💻 Laptop Sentiment — Quick Predict (No Training/No Dataset Upload)")
st.write("Enter a laptop review below and get a sentiment prediction instantly.")

text = st.text_area("Enter a laptop review", "The battery life is great and I love this laptop.")

if st.button("Predict sentiment"):
    if not text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        # Transform
        X = tfidf.transform([text])

        # Base prediction
        pred_id = int(clf.predict(X)[0])
        pred_label = class_names[pred_id]
        proba_txt = ""

        # -------------------------------
        # Probabilities + safety rule
        # -------------------------------
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            proba_txt = "  |  " + ", ".join(
                [f"{lbl}: {p:.2f}" for lbl, p in zip(class_names, probs)]
            )

            try:
                neg_idx = class_names.index("Negative")
                pos_idx = class_names.index("Positive")
                neg_prob = float(probs[neg_idx])
                pos_prob = float(probs[pos_idx])

                NEG_CUES = {
                    "bad","terrible","awful","hate","regret","waste","slow","buggy","crash",
                    "overheat","lag","poor","worse","worst","don_t_like","not_good","not_recommend"
                }
                toks = set(basic_clean(text).split())
                has_neg_cue = any(tok in toks for tok in NEG_CUES)

                # thresholds – tweak if you like
                CLOSE_GAP = 0.15   # if |pos - neg| < 0.15, it's a close call
                NEG_FLOOR = 0.30   # require at least 0.30 prob on Negative

                if has_neg_cue and (abs(pos_prob - neg_prob) < CLOSE_GAP) and (neg_prob >= NEG_FLOOR):
                    pred_label = "Negative"
                    pred_id = neg_idx
            except Exception:
                pass

        # -------------------------------
        # Show result
        # -------------------------------
        st.success(f"Prediction: **{pred_label}** (id={pred_id}){proba_txt}")

        # Sanity checker: show tokens
        st.write("### 🔍 Sanity Checker")
        st.write(set(basic_clean(text).split()))
