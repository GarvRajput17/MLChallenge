"""Shared IO, paths and text normalisation for the entity-resolution pipeline."""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # work/
DATA = os.path.join(os.path.dirname(ROOT), "student_resource", "dataset")   # student_resource/dataset
CACHE = os.path.join(ROOT, "cache")
OUTPUT = os.path.join(ROOT, "output")
REPORTS = os.path.join(ROOT, "reports")
for _d in (CACHE, OUTPUT, REPORTS):
    os.makedirs(_d, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- loading ---
def load_source(split: str, src: int) -> pd.DataFrame:
    """split in {train,test}; src in {1,2,3}. Cached as parquet."""
    pq = os.path.join(CACHE, f"{split}_source{src}.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    tsv = os.path.join(DATA, split, f"{split}_source{src}.tsv")
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.to_parquet(pq, index=False)
    return df


def load_ground_truth() -> pd.DataFrame:
    pq = os.path.join(CACHE, "train_ground_truth.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    df = pd.read_csv(os.path.join(DATA, "train", "train_ground_truth.tsv"), sep="\t",
                     dtype=str, keep_default_na=False, na_filter=False)
    df.to_parquet(pq, index=False)
    return df


# ---------------------------------------------------- unicode / normalise ---
_INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]


def is_indic_char(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in _INDIC_RANGES)


def has_indic(s: str) -> bool:
    return any(is_indic_char(c) for c in s)


_PUNCT_RE = re.compile(r"[^\w\sऀ-෿]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NULLS = {"null", "nan", "none", "n a", "na", "nil", ""}


def strip_accents(s: str) -> str:
    """Fold Latin diacritics (Énterprises -> enterprises) but leave Indic intact."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch) and not is_indic_char(ch):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def basic_norm(s: str) -> str:
    """Lowercase, fold accents, replace punctuation with space, squeeze whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s)
    s = s.lower()
    s = s.replace("&", " and ").replace("+", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def drop_null_tokens(tokens):
    return [t for t in tokens if t not in _NULLS]


def tokens(s: str):
    return drop_null_tokens(s.split())
