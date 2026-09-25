"""Stage 7: decision layer -- choose, per S1 entity, the match set that maximises
expected F_0.5.

Why not a threshold. F_beta has a closed form that removes precision and recall:

    F_beta = (1 + b) * TP / (b * K + |S|),    b = beta^2 = 0.25

with K the true match count and |S| our predicted set size. Given calibrated
per-candidate probabilities we can therefore compute E[F_0.5] for a candidate set
*exactly*, and pick the maximiser. A fixed threshold cannot express this: for
probabilities [0.60, 0.58, 0.55, 0.52] a p>0.70 rule returns the empty set
(E[F]=0.036) while the optimum takes all four (E[F]=0.603).

Two structural facts make the search cheap:
  * The optimal set is a prefix of the probability-sorted candidate list -- swapping a
    lower-probability member for a higher-probability non-member never lowers E[TP]
    while leaving |S| unchanged.
  * Candidate lists are short (<= max_per_entity), so we evaluate every prefix.

TP and the number of true-but-unselected candidates are Poisson-binomial. We build
their distributions with forward and backward convolutions, vectorised over entities
(one pass per distinct candidate-count group), so nothing loops per entity.
"""
from __future__ import annotations

import numpy as np

B = 0.25          # beta^2
GAIN = 1.0 + B


def _forward_pb(probs: np.ndarray) -> list[np.ndarray]:
    """fwd[s][e, t] = P(exactly t of the first s candidates of entity e are true)."""
    n_e, n = probs.shape
    out = [np.ones((n_e, 1), dtype=np.float64)]
    cur = out[0]
    for s in range(n):
        p = probs[:, s:s + 1]
        nxt = np.zeros((n_e, cur.shape[1] + 1), dtype=np.float64)
        nxt[:, :-1] += cur * (1.0 - p)
        nxt[:, 1:] += cur * p
        out.append(nxt)
        cur = nxt
    return out


def _backward_pb(probs: np.ndarray) -> list[np.ndarray]:
    """bwd[s][e, m] = P(exactly m of candidates s..n-1 of entity e are true)."""
    n_e, n = probs.shape
    out = [None] * (n + 1)
    cur = np.ones((n_e, 1), dtype=np.float64)
    out[n] = cur
    for s in range(n - 1, -1, -1):
        p = probs[:, s:s + 1]
        nxt = np.zeros((n_e, cur.shape[1] + 1), dtype=np.float64)
        nxt[:, :-1] += cur * (1.0 - p)
        nxt[:, 1:] += cur * p
        out[s] = nxt
        cur = nxt
    return out


def expected_f05(probs: np.ndarray, miss_prior: float = 0.0) -> np.ndarray:
    """E[F_0.5] for every prefix size, for a block of entities with the same
    candidate count.

    probs      (n_entities, n_candidates), sorted descending along axis 1
    miss_prior expected number of true matches that blocking never surfaced; it
               inflates K and so makes larger predicted sets relatively safer
    returns    (n_entities, n_candidates + 1)
    """
    n_e, n = probs.shape
    probs = np.clip(probs.astype(np.float64), 1e-9, 1 - 1e-9)
    fwd, bwd = _forward_pb(probs), _backward_pb(probs)

    scores = np.empty((n_e, n + 1), dtype=np.float64)
    for s in range(n + 1):
        f, b = fwd[s], bwd[s]
        if s == 0:
            # empty prediction scores 1.0 exactly when the entity truly has no match
            scores[:, 0] = b[:, 0] if miss_prior == 0.0 else 0.0
            continue
        acc = np.zeros(n_e, dtype=np.float64)
        for tp in range(1, f.shape[1]):          # tp = 0 contributes nothing
            ftp = f[:, tp]
            if not ftp.any():
                continue
            for m in range(b.shape[1]):
                k = tp + m + miss_prior
                acc += ftp * b[:, m] * (GAIN * tp / (B * k + s))
        scores[:, s] = acc
    return scores


def choose_counts(probs: np.ndarray, miss_prior: float = 0.0,
                  size_penalty: float = 0.0) -> np.ndarray:
    """Optimal number of candidates to keep for each entity in the block."""
    scores = expected_f05(probs, miss_prior)
    if size_penalty:
        scores -= size_penalty * np.arange(scores.shape[1])
    return scores.argmax(axis=1)


def select(group_starts: np.ndarray, group_sizes: np.ndarray, probs: np.ndarray,
           miss_prior: float = 0.0, size_penalty: float = 0.0) -> np.ndarray:
    """Boolean mask over a flat, entity-grouped, probability-sorted candidate array.

    group_starts / group_sizes describe the contiguous block of each S1 entity.
    Entities are processed in batches sharing the same candidate count so the
    Poisson-binomial recursions stay vectorised.
    """
    keep = np.zeros(len(probs), dtype=bool)
    for size in np.unique(group_sizes):
        if size == 0:
            continue
        sel = np.flatnonzero(group_sizes == size)
        starts = group_starts[sel]
        offs = starts[:, None] + np.arange(size)[None, :]
        block = probs[offs]
        counts = choose_counts(block, miss_prior, size_penalty)
        take = np.arange(size)[None, :] < counts[:, None]
        keep[offs[take]] = True
    return keep
