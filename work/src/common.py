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
# Overridable so the pipeline can be pointed at a synthetic fixture for smoke tests.
DATA = os.environ.get(
    "ER_DATA", os.path.join(os.path.dirname(ROOT), "student_resource", "dataset"))
CACHE = os.environ.get("ER_CACHE", os.path.join(ROOT, "cache"))
OUTPUT = os.environ.get("ER_OUTPUT", os.path.join(ROOT, "output"))
REPORTS = os.path.join(ROOT, "reports")

# Shared machines: never grab every core. Override with ER_THREADS.
THREADS = int(os.environ.get("ER_THREADS", min(16, os.cpu_count() or 4)))
for _d in (CACHE, OUTPUT, REPORTS):
    os.makedirs(_d, exist_ok=True)


def mem_gb() -> float:
    """Peak RSS of this process, in GB. Used to keep the memory budget visible in
    the logs -- the blocking stage is the one that can exhaust a box."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "linux" else peak / (1024 ** 3)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} {mem_gb():5.1f}G]", *a, flush=True)


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
def _is_folded_base(o: int) -> bool:
    """Scripts whose diacritics are folded away (Latin, Greek, Cyrillic): é -> e.
    Marks on any other script (Indic, Arabic, Thai, ...) are part of the letter."""
    return o < 0x0530 or 0x1E00 <= o <= 0x1FFF


def has_non_latin(s: str) -> bool:
    """Any letter outside Latin -- a transliteration candidate. Script-agnostic."""
    return any(c.isalpha() and ord(c) > 0x024F and not 0x1E00 <= ord(c) <= 0x1EFF
               for c in s)


# \w excludes combining marks, so without this Indic/Arabic/Thai words split apart.
_MARKS, _run = "", None
for _c in range(sys.maxunicode + 2):
    _is_m = _c <= sys.maxunicode and unicodedata.category(chr(_c)).startswith("M")
    if _is_m and _run is None:
        _run = _c
    elif not _is_m and _run is not None:
        _MARKS += f"{re.escape(chr(_run))}-{re.escape(chr(_c - 1))}"
        _run = None
_PUNCT_RE = re.compile(rf"[^\w\s{_MARKS}]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NULLS = {"null", "nan", "none", "n a", "na", "nil", ""}
# Latin letters NFKC does not decompose.
_LIGATURES = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss",
                            "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                            "þ": "th", "Þ": "TH"})


def strip_accents(s: str) -> str:
    """Fold diacritics on Latin/Greek/Cyrillic letters; keep marks on other scripts."""
    out, base = [], 0
    for ch in unicodedata.normalize("NFD", s.translate(_LIGATURES)):
        if unicodedata.combining(ch):
            if _is_folded_base(base):
                continue
        else:
            base = ord(ch)
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


# Names only -- each measured on train ground-truth pairs; on addresses they hurt India.
_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z][./] ?)+[A-Za-z]\b\.?")      # S.A.S. L.L.C. a/k/a
_APOS_RE = re.compile(r"(?<=\w)[’'`´](?=\w)")                         # L'Atelier -> latelier
# Digit-for-letter typos (a1lied, harb0r): S1 names never mix letters and digits in a
# token, and aligned train pairs give exactly these substitutions (1->i seen twice).
_LEET_RE = re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]+\b")
_LEET = str.maketrans("01568", "olsgb")
_ALIAS_RE = re.compile(r"^.*\b(?:dba|fka|aka|formerly|doing business as|trading as|ta)\b\s*")  # S1 name follows


def name_norm(s: str) -> str:
    """basic_norm plus: join dotted acronyms, delete (not space) apostrophes, and drop
    everything up to an alias marker ('Korbrixx D.B.A. Obsidian' -> 'obsidian')."""
    s = _ACRONYM_RE.sub(lambda m: re.sub(r"[./ ]", "", m.group()), s)
    s = _APOS_RE.sub("", s)
    s = basic_norm(s)
    s = _LEET_RE.sub(lambda m: m.group().translate(_LEET), s)
    return _ALIAS_RE.sub("", s) or s


def drop_null_tokens(tokens):
    return [t for t in tokens if t not in _NULLS]


def tokens(s: str):
    return drop_null_tokens(s.split())
