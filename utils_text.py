# utils_text.py
import re
import pandas as pd

TEXT_CANDIDATES = ["Sentence", "Review", "Text", "comment", "review_text", "content", "body"]
LABEL_CANDIDATES = ["label", "Sentiment", "VADER_Sentiment", "target", "sentiment"]

def detect_columns(df: pd.DataFrame):
    text_col = None
    label_col = None
    # find text
    for c in df.columns:
        if c in TEXT_CANDIDATES:
            text_col = c
            break
    if text_col is None:
        # fall back to first object column
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            text_col = obj_cols[0]
    # find label
    for c in df.columns:
        if c in LABEL_CANDIDATES:
            label_col = c
            break
    if label_col is None:
        # common fallbacks
        for c in df.columns:
            if c.lower() in ["label", "sentiment", "target"]:
                label_col = c
                break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Found columns: {list(df.columns)}.\n"
                         f"Please rename your text column to one of: {TEXT_CANDIDATES}\n"
                         f"and label column to one of: {LABEL_CANDIDATES}")
    return text_col, label_col

def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(basic_clean)
