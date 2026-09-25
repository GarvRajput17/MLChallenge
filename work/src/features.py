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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from blocking import pair_cosine
from common import THREADS as N_THREADS, log

# (feature name, record field, vectoriser kwargs) -- each is one "view" of the record.
# char 3- and 4-grams share a single vectoriser: two fits over 5M documents to produce
# near-duplicate signals was the single largest avoidable cost in this stage.
COSINE_VIEWS = [
    ("name_tok",  "name_c",    dict(analyzer="word", ngram_range=(1, 1))),
    ("name_ch",   "name_sq",   dict(analyzer="char", ngram_range=(3, 4))),
    ("ncore_tok", "name_core", dict(analyzer="word", ngram_range=(1, 1))),
    ("addr_tok",  "addr_c",    dict(analyzer="word", ngram_range=(1, 1))),
    ("addr_ch",   "addr_c",    dict(analyzer="char", ngram_range=(3, 4))),
    ("acore_tok", "addr_core", dict(analyzer="word", ngram_range=(1, 1))),
    ("num_tok",   "nums",      dict(analyzer="word", ngram_range=(1, 1))),
]

# Views where raw set overlap / containment is wanted alongside the IDF-weighted
# cosine. The binary matrix is derived from the TF-IDF one by flattening its data
# array -- same vocabulary, same tokenisation, so a second fit_transform over
# millions of documents would compute exactly the same sparsity pattern twice.
OVERLAP_VIEWS = ["name_tok", "ncore_tok", "addr_tok", "acore_tok", "num_tok"]

if os.environ.get("ER_FAST", "0") == "1":
    # Drop the two most expensive, most redundant views (char n-grams over the full
    # address, and the name-core token view) when wall-clock is the constraint.
    _DROP = {"addr_ch", "ncore_tok"}
    COSINE_VIEWS = [v for v in COSINE_VIEWS if v[0] not in _DROP]
    OVERLAP_VIEWS = [v for v in OVERLAP_VIEWS if v not in _DROP]

RF_SCORERS = [
    ("ratio",  fuzz.ratio),                 # raw typo tolerance
    ("tsort",  fuzz.token_sort_ratio),      # word-transposition invariant
    ("tset",   fuzz.token_set_ratio),       # tolerant of *added* junk tokens
    ("part",   fuzz.partial_ratio),         # DBA prefixes, long-vs-short
]


def _vectorise(field_index, field_query, kwargs):
    vec = TfidfVectorizer(min_df=1, lowercase=False, sublinear_tf=True,
                          dtype=np.float32, norm=None, **kwargs)
    try:
        D = vec.fit_transform(field_index)
    except ValueError:
        return None, None            # whole view empty for this slice
    if D.shape[1] == 0:
        return None, None
    Q = vec.transform(field_query)
    return normalize(D, copy=False), normalize(Q, copy=False)


def _binarise(X):
    """Set-membership matrix sharing X's sparsity pattern -- no second fit needed."""
    B = X.copy()
    B.data[:] = 1.0
    return B


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b > 0)


class FeatureBuilder:
    """Builds the per-country feature matrix for a set of candidate pairs."""

    def __init__(self, s1, others):
        self.s1, self.others = s1, others
        self.cos_D, self.cos_Q = {}, {}
        self.bin_D, self.bin_Q = {}, {}
        for name, field, kw in COSINE_VIEWS:
            D, Q = _vectorise(s1[field].values, others[field].values, kw)
            if D is None:
                log(f"   view {name}: empty for this slice -- skipped")
                continue
            self.cos_D[name], self.cos_Q[name] = D, Q
            if name in OVERLAP_VIEWS:
                self.bin_D[name], self.bin_Q[name] = _binarise(D), _binarise(Q)
            log(f"   view {name}: |V|={D.shape[1]:,}")
        self.bin_sizes_D = {k: np.asarray(v.sum(axis=1)).ravel().astype(np.float32)
                            for k, v in self.bin_D.items()}
        self.bin_sizes_Q = {k: np.asarray(v.sum(axis=1)).ravel().astype(np.float32)
                            for k, v in self.bin_Q.items()}

    # -------------------------------------------------------------- features --
    def build(self, s1_idx: np.ndarray, oth_idx: np.ndarray) -> dict:
        f = {}
        n_pairs = len(s1_idx)
        zeros = np.zeros(n_pairs, dtype=np.float32)
        for name, _, _ in COSINE_VIEWS:
            f[f"cos_{name}"] = (pair_cosine(self.cos_Q[name], self.cos_D[name],
                                            oth_idx, s1_idx)
                                if name in self.cos_D else zeros)
        for name in OVERLAP_VIEWS:
            if name not in self.bin_D:
                for k in ("jac", "cont", "n_%s_a" % name, "n_%s_b" % name):
                    f[f"{k}_{name}" if k in ("jac", "cont") else k] = zeros
                continue
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
                    left, right, scorer=scorer, workers=N_THREADS,
                    dtype=np.float32, score_multiplier=0.01)
            f[f"{tag}_jw"] = process.cpdist(
                left, right, scorer=JaroWinkler.normalized_similarity,
                workers=N_THREADS, dtype=np.float32)
            f[f"{tag}_pref"] = process.cpdist(
                left, right, scorer=Prefix.normalized_similarity,
                workers=N_THREADS, dtype=np.float32)
            la = np.fromiter((len(x) for x in left), np.float32, len(left))
            lb = np.fromiter((len(x) for x in right), np.float32, len(right))
            f[f"{tag}_len_a"], f[f"{tag}_len_b"] = la, lb
            f[f"{tag}_len_ratio"] = _safe_div(np.minimum(la, lb), np.maximum(la, lb))
            f[f"{tag}_empty_b"] = (lb == 0).astype(np.float32)
            log(f"   rapidfuzz {tag} done")

        # squashed-name similarity: immune to tokenisation and to the domain form
        left = list(self.s1["name_sq"].values[s1_idx])
        right = list(self.others["name_sq"].values[oth_idx])
        f["sq_ratio"] = process.cpdist(left, right, scorer=fuzz.ratio, workers=N_THREADS,
                                       dtype=np.float32, score_multiplier=0.01)
        f["sq_jw"] = process.cpdist(left, right,
                                    scorer=JaroWinkler.normalized_similarity,
                                    workers=N_THREADS, dtype=np.float32)
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
