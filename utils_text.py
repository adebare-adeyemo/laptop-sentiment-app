# utils_text.py
import re
import unicodedata
import pandas as pd

TEXT_CANDIDATES = ["Sentence","Review","Text","comment","review_text","content","body"]
LABEL_CANDIDATES = ["label","Sentiment","VADER_Sentiment","target","sentiment"]

def detect_columns(df: pd.DataFrame):
    text_col, label_col = None, None
    for c in df.columns:
        if c in TEXT_CANDIDATES: text_col = c; break
    if text_col is None:
        obj = df.select_dtypes(include="object").columns.tolist()
        if obj: text_col = obj[0]
    for c in df.columns:
        if c in LABEL_CANDIDATES: label_col = c; break
    if label_col is None:
        for c in df.columns:
            if c.lower() in ["label","sentiment","target"]:
                label_col = c; break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Columns: {list(df.columns)}")
    return text_col, label_col

def _normalize_punct(s: str) -> str:
    # Normalize unicode punctuation (curly quotes etc.) to ASCII
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("’", "'").replace("‘", "'")
           .replace("“", '"').replace("”", '"'))
    return s

def basic_clean(s: str) -> str:
    """Lowercase, trim, normalize punctuation, join simple negations:
       don't like -> don_t_like, not good -> not_good"""
    if not isinstance(s, str):
        s = str(s)
    s = _normalize_punct(s)
    s = s.lower().strip()
    s = " ".join(s.split())

    tokens = s.split()
    out = []
    i = 0
    NEG = {
        "not","no","never","cannot","cant","can't","dont","don't","don’t",
        "isnt","isn't","wasnt","wasn't","wont","won't","doesnt","doesn't","didnt","didn't"
    }
    while i < len(tokens):
        if tokens[i] in NEG and i + 1 < len(tokens):
            first = tokens[i].replace("'", "_")
            out.append(first + "_" + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(basic_clean)
