"""Shared post-scoring logic: one-to-one arbitration, grouping, and set selection.

Used identically for validation and for the final test inference so the two can
never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from decide import select


def one_to_one(cand_ids: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Mask keeping, for each S2/S3 record, only its single best S1 entity.

    In the ground truth no S2/S3 record is ever shared between two S1 entities, so
    the true answer is a partial one-to-one assignment. Enforcing it removes a whole
    class of false merges for free -- and a false merge is the expensive error here.
    """
    order = np.argsort(cand_ids, kind="stable")
    sorted_ids = cand_ids[order]
    sorted_p = probs[order]
    # boundaries of each candidate's block
    new_block = np.empty(len(sorted_ids), dtype=bool)
    new_block[0] = True
    np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=new_block[1:])
    block_id = np.cumsum(new_block) - 1
    n_blocks = block_id[-1] + 1
    best = np.full(n_blocks, -np.inf, dtype=np.float64)
    np.maximum.at(best, block_id, sorted_p)
    keep_sorted = sorted_p >= best[block_id]
    # ties: keep only the first occurrence
    first = np.ones(len(sorted_ids), dtype=bool)
    idx = np.flatnonzero(keep_sorted)
    dup = np.zeros(len(sorted_ids), dtype=bool)
    if len(idx):
        b = block_id[idx]
        dup[idx[1:]] = b[1:] == b[:-1]
    keep_sorted &= ~dup
    keep = np.zeros(len(cand_ids), dtype=bool)
    keep[order[keep_sorted]] = True
    return keep


def rival_margin(cand_ids: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """best - second-best probability across the S1 entities a candidate competes for.

    A candidate that fits one entity and nothing else is far safer to merge than one
    that is mildly similar to nine. Pairwise models cannot see this.
    """
    order = np.lexsort((-probs, cand_ids))
    sid = cand_ids[order]
    sp = probs[order]
    out = np.zeros(len(probs), dtype=np.float32)
    same_next = np.zeros(len(sid), dtype=bool)
    same_next[:-1] = sid[1:] == sid[:-1]
    second = np.where(same_next, np.roll(sp, -1), 0.0)
    is_first = np.ones(len(sid), dtype=bool)
    is_first[1:] = sid[1:] != sid[:-1]
    out[order] = np.where(is_first, sp - second, 0.0).astype(np.float32)
    return out


def choose(df: pd.DataFrame, probs: np.ndarray, *, miss_prior=0.0,
           size_penalty=0.0, enforce_one_to_one=True) -> pd.DataFrame:
    """Return the accepted (s1_entity_id, cand_entity_id) rows."""
    p = probs.astype(np.float64)
    if enforce_one_to_one:
        p = np.where(one_to_one(df["cand_entity_id"].values, p), p, 0.0)

    order = np.lexsort((-p, df["s1_entity_id"].values))
    s1_sorted = df["s1_entity_id"].values[order]
    p_sorted = p[order]

    new = np.ones(len(s1_sorted), dtype=bool)
    new[1:] = s1_sorted[1:] != s1_sorted[:-1]
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, len(s1_sorted)))

    keep_sorted = select(starts, sizes, p_sorted, miss_prior, size_penalty)
    return df.iloc[order[keep_sorted]]


def to_submission(accepted: pd.DataFrame, all_s1_ids: np.ndarray) -> pd.DataFrame:
    """One row per S1 entity, comma-joined IDs, empty for predicted singletons."""
    grouped = (accepted.groupby("s1_entity_id")["cand_entity_id"]
               .agg(lambda v: ",".join(dict.fromkeys(v))))
    return pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": pd.Series(all_s1_ids).map(grouped).fillna("").values,
    })
