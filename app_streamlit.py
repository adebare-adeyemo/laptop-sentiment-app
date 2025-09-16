import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ===============================
# Setup & artifact loading
# ===============================
MODEL_DIR = Path("models/classical")
st.set_page_config(page_title="Laptop Sentiment — Quick Predict", layout="wide")
st.title("💻 Laptop Sentiment — Quick Predict")

# Load artifacts
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
# Cue loading (JSON + safe fallbacks)
# ===============================
def load_cues(json_path: str, key: str, fallback_list: list) -> set:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cues = set(map(str.lower, data.get(key, [])))
        return cues if cues else set(map(str.lower, fallback_list))
    except Exception:
        return set(map(str.lower, fallback_list))

NEG_FALLBACK = [
    "don't like","didn't like","doesn't work","did not work","not good","not great","not happy","not satisfied",
    "not working","not worth","not recommended","not usable","not helpful","not useful","not reliable",
    "not impressed","not efficient","not effective","not durable",
    "bad","terrible","horrible","awful","poor","pathetic","ridiculous","useless","worthless","stupid",
    "garbage","junk","fake","broken","faulty","cheap","fragile","slow","laggy","buggy","worst","lousy",
    "hate","hated","angry","frustrated","disappointed","disappointing","upset","annoyed","mad",
    "regret","regretted","unhappy","miserable",
    "return","returned","returning","refund","refunded","refunds","exchange","replacing","replacement",
    "stop working","stopped working","quit working","doesn't turn on","won't turn on","dead on arrival",
    "too slow","too small","too big","too heavy","overheats","overheating","lagging","crashes",
    "freezes","freeze","glitchy","unresponsive","hangs","sluggish",
    "waste of money","waste of time","not worth the price","overpriced","scam","cheated",
    "expensive for nothing","robbery","rip off",
    "wrong item","damaged","scratched","missing parts","missing accessories","broken screen",
    "bad packaging","late delivery","never arrived","fake product","counterfeit",
    "better options","worse than expected","inferior","declined","downgrade","step back","doesn't compare",
    "battery died","battery drains","battery problem","charging issue","charger not working",
    "screen issue","keyboard issue","trackpad issue","speaker problem","fan noise","heating problem"
]

POS_FALLBACK = [
    "good","great","excellent","fantastic","amazing","awesome","brilliant","outstanding","wonderful",
    "perfect","superb","fabulous","top-notch","best","love","loved","like it","liked it","enjoyed",
    "satisfied","happy","pleased","impressed","delighted","exceptional","super","cool","nice","awesome deal",
    "worth it","value for money","works well","works perfectly","does the job","recommend","recommended",
    "highly recommend","smooth","fast","durable","reliable","long-lasting","helpful","useful",
    "better than expected","met expectations","exceeded expectations"
]

NEGATIVE_CUES = load_cues("negative_cues.json", "negative_cues", NEG_FALLBACK)
POSITIVE_CUES = load_cues("positive_cues.json", "positive_cues", POS_FALLBACK)

# Strong cue subsets (force override immediately)
STRONG_NEG = {"hate","terrible","awful","worst","refund","return","broken","damaged","defective","faulty"}
STRONG_POS = {"love","amazing","excellent","fantastic","awesome","best","thanks","thank you","perfect"}

# Common complaint phrases (explicit patterns)
NEG_PHRASES = [
    "not what i ordered", "this was not what i ordered",
    "i am returning it", "i'm returning it", "will return it",
    "asking for a refund", "want a refund", "request a refund",
    "item arrived broken", "product arrived damaged", "came damaged", "came broken",
    "did not work", "does not work", "doesn't work", "didn't work",
    "not as described"
]

# ===============================
# Helpers
# ===============================
def basic_clean(s: str) -> str:
    return str(s).lower().replace("’", "'").strip()

def detect_cue_label(text: str):
    """Return 'Negative'/'Positive'/None based on cue & phrase detection."""
    txt = basic_clean(text)

    # Phrase-level negatives
    for ph in NEG_PHRASES:
        if ph in txt:
            return "Negative"

    # Token-level strong overrides
    for tok in STRONG_NEG:
        if tok in txt:
            return "Negative"
    for tok in STRONG_POS:
        if tok in txt:
            return "Positive"

    # General cue lists
    for cue in NEGATIVE_CUES:
        if cue in txt:
            return "Negative"
    for cue in POSITIVE_CUES:
        if cue in txt:
            return "Positive"

    return None

def predict_label_model_only(text: str):
    X = tfidf.transform([text])
    pred_id = int(clf.predict(X)[0])
    return CLASS_NAMES[pred_id]

def predict_label(text: str):
    # 1) Cue/phrase override
    cue = detect_cue_label(text)
    if cue is not None:
        return cue
    # 2) Model fallback
    return predict_label_model_only(text)

# ===============================
# UI — Single prediction
# ===============================
st.subheader("🔎 Single Review")
sample = "This was not what I ordered — I am returning it."
text = st.text_area("Enter a laptop review", sample, height=120)

if st.button("Predict sentiment"):
    st.success(f"Prediction: **{predict_label(text)}**")

# ===============================
# UI — Batch CSV prediction
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
            out = df.copy()
            out["pred_label"] = [
                predict_label(t) for t in out[text_col].astype(str).fillna("")
            ]
            st.dataframe(out.head(30), use_container_width=True)
            st.download_button(
                "⬇️ Download predictions",
                data=out.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")
