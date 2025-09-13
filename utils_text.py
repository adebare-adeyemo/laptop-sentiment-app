import re
import pandas as pd

TEXT_CANDIDATES = ["Sentence","Review","Text","comment","review_text","content","body"]
LABEL_CANDIDATES = ["label","Sentiment","VADER_Sentiment","target","sentiment"]

def detect_columns(df: pd.DataFrame):
    text_col, label_col = None, None
    for c in df.columns:
        if c in TEXT_CANDIDATES:
            text_col = c; break
    if text_col is None:
        obj = df.select_dtypes(include="object").columns.tolist()
        if obj: text_col = obj[0]
    for c in df.columns:
        if c in LABEL_CANDIDATES:
            label_col = c; break
    if label_col is None:
        for c in df.columns:
            if c.lower() in ["label","sentiment","target"]:
                label_col = c; break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Columns: {list(df.columns)}")
    return text_col, label_col

def basic_clean(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.lower().strip()
    s = " ".join(s.split())
    toks = s.split()
    out = []
    i = 0
    neg = {"not","no","never","cannot","can't","dont","don't","isn't","wasn't","won't","didn't","doesn't"}
    while i < len(toks):
        if toks[i] in neg and i+1 < len(toks):
            out.append(toks[i] + "_" + toks[i+1]); i += 2
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)

def clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(basic_clean)
