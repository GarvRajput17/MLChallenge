"""Stage 5: pairwise feature extraction.

Design constraint: ~30M candidate pairs, so every feature must be vectorised.
Two engines do all the work, both multithreaded C:

  * sparse cosines  -- reuse of the blocking machinery. A TF-IDF cosine over a token
    view IS the IDF-weighted overlap feature, so one primitive covers the whole
    set-theoretic family across several views of the record.
  * rapidfuzz cpdist -- elementwise edit-distance scorers over paired string lists,
    each encoding a different invariance (transposition, junk insertion, prefix).

Nothing here loops over pairs in Python.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Prefix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from blocking import pair_cosine
from common import log

N_THREADS = max(1, os.cpu_count() or 4)

# (feature name, record field, vectoriser kwargs) -- each is one "view" of the record
COSINE_VIEWS = [
    ("name_tok",  "name_c",    dict(analyzer="word", ngram_range=(1, 1))),
    ("name_c3",   "name_sq",   dict(analyzer="char", ngram_range=(3, 3))),
    ("name_c4",   "name_sq",   dict(analyzer="char", ngram_range=(4, 4))),
    ("ncore_tok", "name_core", dict(analyzer="word", ngram_range=(1, 1))),
    ("addr_tok",  "addr_c",    dict(analyzer="word", ngram_range=(1, 1))),
    ("addr_c3",   "addr_c",    dict(analyzer="char", ngram_range=(3, 3))),
    ("acore_tok", "addr_core", dict(analyzer="word", ngram_range=(1, 1))),
    ("num_tok",   "nums",      dict(analyzer="word", ngram_range=(1, 1))),
]

# views where we also want raw set overlap / containment, not just IDF-weighted cosine
OVERLAP_VIEWS = ["name_tok", "ncore_tok", "addr_tok", "acore_tok", "num_tok"]

RF_SCORERS = [
    ("ratio",  fuzz.ratio),                 # raw typo tolerance
    ("tsort",  fuzz.token_sort_ratio),      # word-transposition invariant
    ("tset",   fuzz.token_set_ratio),       # tolerant of *added* junk tokens
    ("part",   fuzz.partial_ratio),         # DBA prefixes, long-vs-short
]


def _vectorise(field_index, field_query, kwargs, binary=False):
    cls = CountVectorizer if binary else TfidfVectorizer
    extra = dict(binary=True, dtype=np.float32) if binary else dict(
        sublinear_tf=True, dtype=np.float32, norm=None)
    vec = cls(min_df=1, lowercase=False, **kwargs, **extra)
    D = vec.fit_transform(field_index)
    Q = vec.transform(field_query)
    if not binary:
        D, Q = normalize(D, copy=False), normalize(Q, copy=False)
    return D, Q


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b > 0)


class FeatureBuilder:
    """Builds the per-country feature matrix for a set of candidate pairs."""

    def __init__(self, s1, others):
        self.s1, self.others = s1, others
        self.cos_D, self.cos_Q = {}, {}
        self.bin_D, self.bin_Q = {}, {}
        for name, field, kw in COSINE_VIEWS:
            self.cos_D[name], self.cos_Q[name] = _vectorise(
                s1[field].values, others[field].values, kw)
            if name in OVERLAP_VIEWS:
                self.bin_D[name], self.bin_Q[name] = _vectorise(
                    s1[field].values, others[field].values, kw, binary=True)
            log(f"   view {name}: |V|={self.cos_D[name].shape[1]:,}")
        self.bin_sizes_D = {k: np.asarray(v.sum(axis=1)).ravel().astype(np.float32)
                            for k, v in self.bin_D.items()}
        self.bin_sizes_Q = {k: np.asarray(v.sum(axis=1)).ravel().astype(np.float32)
                            for k, v in self.bin_Q.items()}

    # -------------------------------------------------------------- features --
    def build(self, s1_idx: np.ndarray, oth_idx: np.ndarray) -> dict:
        f = {}
        for name, _, _ in COSINE_VIEWS:
            f[f"cos_{name}"] = pair_cosine(self.cos_Q[name], self.cos_D[name],
                                           oth_idx, s1_idx)
        for name in OVERLAP_VIEWS:
            inter = pair_cosine(self.bin_Q[name], self.bin_D[name], oth_idx, s1_idx)
            a = self.bin_sizes_D[name][s1_idx]
            b = self.bin_sizes_Q[name][oth_idx]
            f[f"jac_{name}"] = _safe_div(inter, a + b - inter)
            f[f"cont_{name}"] = _safe_div(inter, np.minimum(a, b))
            f[f"n_{name}_a"] = a
            f[f"n_{name}_b"] = b
        log("   cosine/overlap views done")

        for field in ("name_c", "addr_c"):
            left = list(self.s1[field].values[s1_idx])
            right = list(self.others[field].values[oth_idx])
            tag = field.split("_")[0]
            for sname, scorer in RF_SCORERS:
                f[f"{tag}_{sname}"] = process.cpdist(
                    left, right, scorer=scorer, workers=-1,
                    dtype=np.float32, score_multiplier=0.01)
            f[f"{tag}_jw"] = process.cpdist(
                left, right, scorer=JaroWinkler.normalized_similarity,
                workers=-1, dtype=np.float32)
            f[f"{tag}_pref"] = process.cpdist(
                left, right, scorer=Prefix.normalized_similarity,
                workers=-1, dtype=np.float32)
            la = np.fromiter((len(x) for x in left), np.float32, len(left))
            lb = np.fromiter((len(x) for x in right), np.float32, len(right))
            f[f"{tag}_len_a"], f[f"{tag}_len_b"] = la, lb
            f[f"{tag}_len_ratio"] = _safe_div(np.minimum(la, lb), np.maximum(la, lb))
            f[f"{tag}_empty_b"] = (lb == 0).astype(np.float32)
            log(f"   rapidfuzz {tag} done")

        # squashed-name similarity: immune to tokenisation and to the domain form
        left = list(self.s1["name_sq"].values[s1_idx])
        right = list(self.others["name_sq"].values[oth_idx])
        f["sq_ratio"] = process.cpdist(left, right, scorer=fuzz.ratio, workers=-1,
                                       dtype=np.float32, score_multiplier=0.01)
        f["sq_jw"] = process.cpdist(left, right,
                                    scorer=JaroWinkler.normalized_similarity,
                                    workers=-1, dtype=np.float32)
        f["sq_exact"] = np.fromiter((a == b for a, b in zip(left, right)),
                                    np.float32, len(left))

        # postal-code agreement: high-entropy, rarely coincidental
        pa = self.s1["postal"].values[s1_idx]
        pb = self.others["postal"].values[oth_idx]
        f["postal_match"] = np.fromiter(
            (1.0 if (x and y and bool(set(x.split()) & set(y.split()))) else 0.0
             for x, y in zip(pa, pb)), np.float32, len(pa))
        f["postal_known"] = np.fromiter(
            (1.0 if (x and y) else 0.0 for x, y in zip(pa, pb)), np.float32, len(pa))

        # provenance: S2 and S3 have measurably different noise profiles
        f["src_is_s3"] = np.fromiter(
            (x[1] == "3" for x in self.others["entity_id"].values[oth_idx]),
            np.float32, len(oth_idx))
        return f
