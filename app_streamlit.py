import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Try to import the same cleaner; fall back if it's missing
try:
    from utils_text import basic_clean
except Exception:
    def basic_clean(s: str) -> str:
        s = (s or "").lower().replace("’", "'").strip()
        return " ".join(s.split())

st.set_page_config(page_title="Laptop Sentiment — Quick Predict", layout="wide")
st.title("💬 Laptop Sentiment — Quick Predict (No Training / No Dataset Upload)")

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "classical"

def list_models_dir():
    if MODEL_DIR.exists():
        files = "\n".join([f"- {p.name}" for p in sorted(MODEL_DIR.glob('*'))])
        return files or "(empty)"
    return "(directory not found)"

# ---------- Load artifacts with diagnostics ----------
st.subheader("🔧 Loading pre-trained artifacts")
missing, model_error = [], None

try:
    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
except Exception as e:
    missing.append(f"tfidf.joblib (error: {e})")

try:
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    class_names = le.classes_.tolist()
except Exception as e:
    missing.append(f"label_encoder.joblib (error: {e})")

clf, model_name = None, None
if not missing:
    for cand in ["best.joblib", "MLP.joblib", "LogReg.joblib", "LinearSVC.joblib", "NaiveBayes.joblib"]:
        try:
            clf = joblib.load(MODEL_DIR / cand)
            model_name = cand
            break
        except Exception as e:
            model_error = str(e)

if missing or clf is None:
    st.error(
        "Could not load artifacts from `models/classical/`.\n\n"
        "Make sure these exist in your repo (exact names):\n"
        "  • tfidf.joblib\n  • label_encoder.joblib\n"
        "  • best.joblib  (or MLP/LogReg/LinearSVC/NaiveBayes .joblib)\n\n"
        f"Details:\n{chr(10).join(missing)}\nmodel load error: {model_error}\n\n"
        f"Working dir: {Path.cwd()}\n"
        f"App dir: {APP_DIR}\n"
        f"models/classical listing:\n{list_models_dir()}"
    )
    st.stop()

st.success(f"Loaded model: **{model_name}** | classes: {class_names}")

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

    # quick sanity cues
    NEG = {"bad","terrible","awful","hate","regret","slow","poor","worse","worst",
           "disappointing","buggy","crash","don_t_like","not_good","not_recommend","overheat","lag"}
    POS = {"good","great","excellent","love","amazing","fast","smooth","awesome","perfect","recommend","satisfied","happy"}
    toks = set(basic_clean(text).split())
    st.caption("Sanity Checker")
    c1, c2 = st.columns(2)
    with c1: st.write("🔻", ", ".join(sorted(NEG.intersection(toks))) or "_None_")
    with c2: st.write("🔺", ", ".join(sorted(POS.intersection(toks))) or "_None_")

# ---------- Optional CSV batch ----------
st.subheader("📦 Batch Predict (optional CSV)")
csv = st.file_uploader("Upload a CSV just to get predictions (no training).", type=["csv"])
if csv is not None:
    df = pd.read_csv(csv)
    text_col = next((c for c in ["Review","Text","Sentence","comment","review_text","content","body"] if c in df.columns), None)
    if text_col is None:
        objs = df.select_dtypes(include="object").columns.tolist()
        text_col = objs[0] if objs else None
    if text_col is None:
        st.error("Could not find a text column.")
    else:
        Xall = tfidf.transform(df[text_col].astype(str))
        preds = clf.predict(Xall)
        labels = [class_names[int(i)] for i in preds]
        out = df.copy(); out["pred_label"] = labels
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("⬇️ Download predictions", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
