"""Stage 4: candidate generation.

Optimises two competing objectives at once:
  * pair completeness (recall ceiling)  -- nothing downstream can recover a dropped pair
  * candidates per S1 entity            -- judged directly in the final ranking

Strategy: retrieve wide and cheaply, then prune hard.

  retrieval   several complementary channels, each returning a small top-k
              (a) name  : char 4-gram TF-IDF cosine on the space-squashed name
              (b) addr  : word TF-IDF cosine on the canonical address
              (c) exact : hash joins on exact canonical name / squashed name / address
  pruning     union the channels, combine the per-channel cosines into one score,
              keep the best `max_per_entity` per S1 entity above `min_score`

Retrieval runs S2/S3 -> S1 because S1 is the smaller side and, by the one-to-one
property of the data, every S2/S3 record has at most one correct parent -- so a small
top-k already captures nearly all of the achievable recall. A reverse S1 -> S2/S3 pass
is unioned in as a recall safety net.

All matrix work is chunked; nothing materialises a dense N x M anything.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from common import log

N_THREADS = max(1, (os.cpu_count() or 4))
QUERY_CHUNK = 400_000


# ----------------------------------------------------------------- vectors ---
def prune_rows(X: sp.csr_matrix, max_nnz: int) -> sp.csr_matrix:
    """Keep only the `max_nnz` highest-weight entries of each row, then renormalise.

    Bounds the cost of the sparse product: retrieval only ever touches a row's most
    discriminative (highest-IDF) features, which is where the cosine mass lives anyway.
    Fully vectorised -- no Python loop over rows.
    """
    X = X.tocsr()
    nnz = X.nnz
    if nnz == 0:
        return X
    counts = np.diff(X.indptr)
    if counts.max() <= max_nnz:
        return normalize(X, copy=False)
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), counts)
    order = np.lexsort((-X.data, rows))          # by row, then descending weight
    rank = np.arange(nnz, dtype=np.int64) - X.indptr[rows[order]]
    sel = order[rank < max_nnz]
    out = sp.coo_matrix((X.data[sel], (rows[sel], X.indices[sel])),
                        shape=X.shape, dtype=np.float32).tocsr()
    return normalize(out, copy=False)


def build_vectors(index_texts, query_texts, *, analyzer, ngram_range,
                  min_df=2, max_df=0.3, max_nnz=24):
    """Fit the vocabulary/IDF on the index side (S1) and project both sides into it."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                          min_df=min_df, max_df=max_df, sublinear_tf=True,
                          dtype=np.float32, norm=None, lowercase=False)
    D = vec.fit_transform(index_texts)
    Q = vec.transform(query_texts)
    return prune_rows(D, max_nnz), prune_rows(Q, max_nnz)


def topk_pairs(Q: sp.csr_matrix, D: sp.csr_matrix, k: int, threshold: float):
    """For each query row, the `k` best index rows above `threshold`.

    Returns (query_idx, index_idx, score) int32/float32 arrays.
    """
    Dt = D.T.tocsr()
    qi_all, di_all, sc_all = [], [], []
    for start in range(0, Q.shape[0], QUERY_CHUNK):
        chunk = Q[start:start + QUERY_CHUNK]
        if chunk.nnz == 0:
            continue
        C = sp_matmul_topn(chunk, Dt, top_n=k, threshold=threshold,
                           sort=False, n_threads=N_THREADS)
        counts = np.diff(C.indptr)
        qi = np.repeat(np.arange(chunk.shape[0], dtype=np.int64), counts) + start
        qi_all.append(qi.astype(np.int64))
        di_all.append(C.indices.astype(np.int64))
        sc_all.append(C.data.astype(np.float32))
        log(f"      chunk {start:,}-{min(start + QUERY_CHUNK, Q.shape[0]):,}: "
            f"{C.nnz:,} pairs")
    if not qi_all:
        empty_i = np.empty(0, np.int64)
        return empty_i, empty_i, np.empty(0, np.float32)
    return (np.concatenate(qi_all), np.concatenate(di_all), np.concatenate(sc_all))


# -------------------------------------------------------------- exact keys ---
def exact_join(index_keys, query_keys, *, skip_empty=True, max_bucket=64):
    """Hash join on an exact key. Buckets larger than `max_bucket` are dropped as
    non-discriminative (e.g. the 253 entities literally named `primary care group`)."""
    keys, inv = np.unique(np.concatenate([index_keys, query_keys]), return_inverse=True)
    n_idx = len(index_keys)
    idx_code, qry_code = inv[:n_idx], inv[n_idx:]
    if skip_empty:
        bad = {i for i, k in enumerate(keys) if not k}
    else:
        bad = set()

    order = np.argsort(idx_code, kind="stable")
    sorted_codes = idx_code[order]
    starts = np.searchsorted(sorted_codes, qry_code, side="left")
    ends = np.searchsorted(sorted_codes, qry_code, side="right")
    sizes = ends - starts
    ok = (sizes > 0) & (sizes <= max_bucket)
    if bad:
        ok &= ~np.isin(qry_code, list(bad))
    if not ok.any():
        e = np.empty(0, np.int64)
        return e, e
    qsel = np.flatnonzero(ok)
    reps = sizes[qsel]
    qi = np.repeat(qsel, reps)
    offsets = np.arange(reps.sum(), dtype=np.int64) - np.repeat(
        np.cumsum(reps) - reps, reps)
    di = order[np.repeat(starts[qsel], reps) + offsets]
    return qi, di


# ------------------------------------------------------------------ merging ---
def merge_channels(n_index, channels):
    """Union per-channel (query, index, score) triples into one scored pair table.

    channels: list of (name, qi, di, scores). Returns (qi, di, score_matrix) where
    score_matrix has one column per channel (0 where that channel did not retrieve).
    """
    keys, cols = [], []
    for _, qi, di, sc in channels:
        keys.append(qi * np.int64(n_index) + di)
        cols.append(sc)
    all_keys = np.concatenate(keys)
    uniq, inverse = np.unique(all_keys, return_inverse=True)
    scores = np.zeros((len(uniq), len(channels)), dtype=np.float32)
    pos = 0
    for c, sc in enumerate(cols):
        seg = inverse[pos:pos + len(sc)]
        np.maximum.at(scores[:, c], seg, sc)
        pos += len(sc)
    return uniq // n_index, uniq % n_index, scores


def prune_per_entity(s1_idx, score, max_per_entity, min_score):
    """Indices of the best `max_per_entity` candidates per S1 entity above `min_score`.

    Returns positions into the input arrays so the caller can slice every parallel
    feature column consistently. This is the step that produces the *reported*
    candidate set, so it is deliberately aggressive -- everything upstream exists
    only to feed this filter.
    """
    kept = np.flatnonzero(score >= min_score)
    if kept.size == 0:
        return kept
    s1_k, sc_k = s1_idx[kept], score[kept]
    order = np.lexsort((-sc_k, s1_k))
    s1_sorted = s1_k[order]
    starts = np.searchsorted(s1_sorted, s1_sorted, side="left")
    rank = np.arange(order.size, dtype=np.int64) - starts
    return kept[order[rank < max_per_entity]]   # grouped by s1_idx via the lexsort


def pair_cosine(Q: sp.csr_matrix, D: sp.csr_matrix, qi, di, chunk=2_000_000):
    """Exact cosine for an arbitrary list of (query row, index row) pairs.

    After the channel union, most pairs were retrieved by only one channel, so the
    other channel's score is unknown rather than zero. Recomputing both exactly is
    what lets the pruning step cut to a small candidate set without losing recall.
    Chunked so the fancy-indexed intermediate never blows up.
    """
    out = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        e = min(s + chunk, len(qi))
        a = Q[qi[s:e]]
        b = D[di[s:e]]
        out[s:e] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out
