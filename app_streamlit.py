import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils_text import basic_clean  # uses the same cleaning/negation rules

st.set_page_config(page_title="Laptop Sentiment — Quick Predict", layout="wide")
st.title("💬 Laptop Sentiment — Quick Predict (No Training / No Dataset Upload)")

# ---------- Load pre-trained artifacts ----------
MODEL_DIR = "models/classical"

try:
    tfidf = joblib.load(f"{MODEL_DIR}/tfidf.joblib")
    le    = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")

    # Prefer a single canonical "best" model if you saved one
    try:
        clf = joblib.load(f"{MODEL_DIR}/best.joblib")
        model_name = "best.joblib"
    except Exception:
        # fallback order if best.joblib isn't present
        clf = None; model_name = None
        for cand in ["MLP.joblib", "LogReg.joblib", "LinearSVC.joblib", "NaiveBayes.joblib"]:
            try:
                clf = joblib.load(f"{MODEL_DIR}/{cand}")
                model_name = cand
                break
            except Exception:
                pass
        if clf is None:
            raise FileNotFoundError("No model file found in models/classical/")

    class_names = le.classes_.tolist()
    st.success(f"Loaded artifacts ✓  Model: **{model_name}** | Classes: {class_names}")
except Exception as e:
    st.error(f"Could not load artifacts from `{MODEL_DIR}`. Make sure these exist:\n"
             f"- tfidf.joblib\n- label_encoder.joblib\n- best.joblib (or MLP/LogReg/LinearSVC/NaiveBayes .joblib)\n\nDetails: {e}")
    st.stop()

# ---------- Single text prediction ----------
st.subheader("🔎 Single Review")
text = st.text_area("Enter a laptop review", "Battery life is bad; I don’t like this laptop.")

if st.button("Predict sentiment"):
    X = tfidf.transform([text])
    pred_id = int(clf.predict(X)[0])
    pred_label = class_names[pred_id]

    proba_txt = ""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        proba_txt = "  |  " + ", ".join([f"{lbl}: {p:.2f}" for lbl, p in zip(class_names, probs)])

    st.success(f"Prediction: **{pred_label}** (id={pred_id}){proba_txt}")

    # Simple sanity checker using the same cleaner
    NEG_CUES = {"bad","terrible","awful","hate","regret","slow","poor","worse","worst",
                "disappointing","buggy","crash","don_t_like","not_good","not_recommend","overheat","lag"}
    POS_CUES = {"good","great","excellent","love","amazing","fast","smooth","awesome","perfect","recommend","satisfied","happy"}
    toks = set(basic_clean(text).split())
    neg = sorted(NEG_CUES.intersection(toks))
    pos = sorted(POS_CUES.intersection(toks))

    st.markdown("**Sanity Checker (detected cues):**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("🔻 Negative:")
        st.write(", ".join(neg) or "_None_")
    with c2:
        st.write("🔺 Positive:")
        st.write(", ".join(pos) or "_None_")

# ---------- Optional: batch predictions from a small CSV ----------
st.subheader("📦 Batch Predict (optional CSV)")
csv = st.file_uploader("Upload a CSV (only used to predict). The app will auto-pick a text column.", type=["csv"])
if csv is not None:
    df = pd.read_csv(csv)
    text_col = next((c for c in ["Review","Text","Sentence","comment","review_text","content","body"] if c in df.columns), None)
    if text_col is None:
        # fallback to first object/string column
        objs = df.select_dtypes(include="object").columns.tolist()
        if not objs:
            st.error("Could not find a text column. Please include a string/text column.")
        else:
            text_col = objs[0]

    if text_col:
        Xall = tfidf.transform(df[text_col].astype(str))
        preds = clf.predict(Xall)
        labels = [class_names[int(i)] for i in preds]
        out = df.copy()
        out["pred_label"] = labels
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("⬇️ Download predictions", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
