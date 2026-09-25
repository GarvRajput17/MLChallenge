"""Driver for candidate generation: runs the channels per country, merges, prunes,
and reports pair completeness + candidates-per-entity.

Usage:
    python3 src/run_blocking.py --split train [--countries India US] [--limit-s1 N]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from blocking import (build_vectors, exact_join, merge_channels, pair_cosine,
                      prune_per_entity, topk_pairs)
from common import CACHE, log

CANON_COLS = ["entity_id", "name_c", "name_core", "name_sq", "addr_c", "addr_core"]


# --------------------------------------------------------------------- data ---
def load_country(split: str, src: int, country: str, columns=CANON_COLS) -> pd.DataFrame:
    path = os.path.join(CACHE, f"{split}_s{src}_canon.parquet")
    tbl = pq.read_table(path, columns=columns,
                        filters=[("country", "==", country)])
    return tbl.to_pandas()


def load_others(split: str, country: str, columns=CANON_COLS) -> pd.DataFrame:
    parts = [load_country(split, s, country, columns) for s in (2, 3)]
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return df


# ---------------------------------------------------------------- channels ---
class Config:
    """All retrieval runs with S1 as the *index* and S2/S3 as the *query*.

    That orientation matters: S1 is the small side, so the transposed index stays
    small, and -- because every S2/S3 record has at most one true parent -- a modest
    top-k already captures nearly all achievable recall. Retrieving in the other
    direction would mean transposing a 100M-nnz matrix per channel.

    Complementary *views* replace a reverse pass: each view ranks candidates
    differently, so their union recovers pairs that any single view would crowd out
    (a name shared by 253 entities cannot be resolved by the name view alone).
    """
    views = [
        # (label, field, analyzer, ngram_range, top_k, threshold)
        ("name_sq",   "name_sq",   "char_wb", (3, 4), 14, 0.32),
        ("addr",      "addr_c",    "word",    (1, 1), 12, 0.30),
        ("name_tok",  "name_c",    "word",    (1, 1),  8, 0.42),
        ("addr_core", "addr_core", "word",    (1, 1),  8, 0.40),
    ]
    max_nnz = 28
    exact_keys = ("name_c", "name_sq", "addr_c")
    exact_bucket = 40
    max_per_entity = 14      # size of the reported candidate set
    min_score = 0.26
    w_name = 0.62
    w_addr = 0.38


def sweep(s1_idx, other_idx, score, n_oth, truth_keys, n_s1, cfg):
    """Pair completeness vs. candidates-per-entity, so the prune can be tuned against
    both objectives at once (recall ceiling AND reported candidate-set size)."""
    keys = s1_idx * np.int64(n_oth) + other_idx
    pos = np.searchsorted(truth_keys, keys)
    pos[pos >= len(truth_keys)] = 0
    is_true = truth_keys[pos] == keys
    n_truth = len(truth_keys)
    log(f"   union recall ceiling: {is_true.sum()/max(n_truth,1):.4f} "
        f"({is_true.sum():,}/{n_truth:,})")
    log("   m    min_score   pairs/entity   pair completeness")
    for m in (3, 4, 5, 6, 8, 10, 14):
        for ms in (0.20, 0.26, 0.32):
            sel = prune_per_entity(s1_idx, score, m, ms)
            log(f"   {m:<4} {ms:<11.2f} {len(sel)/max(n_s1,1):<14.2f} "
                f"{is_true[sel].sum()/max(n_truth,1):.4f}")


def block_country(s1: pd.DataFrame, others: pd.DataFrame, cfg=Config,
                  truth_keys=None):
    n_s1, n_oth = len(s1), len(others)
    log(f"   S1={n_s1:,}  S2+S3={n_oth:,}")

    channels, keep_vecs = [], {}
    for label, field, analyzer, ngram, k, thr in cfg.views:
        D, Q = build_vectors(s1[field].values, others[field].values,
                             analyzer=analyzer, ngram_range=ngram,
                             max_nnz=cfg.max_nnz)
        log(f"   [{label}] vectors: S1 nnz={D.nnz:,} others nnz={Q.nnz:,}")
        qi, di, sc = topk_pairs(Q, D, k, thr)
        log(f"   [{label}] {len(qi):,} pairs ({len(qi)/max(n_oth,1):.1f} per record)")
        channels.append((label, qi, di, sc))
        if label in ("name_sq", "addr"):
            keep_vecs[label] = (Q, D)     # kept for exact rescoring of the union
        else:
            del D, Q
        gc.collect()

    ex_q, ex_d = [], []
    for col in cfg.exact_keys:
        qi, di = exact_join(s1[col].values, others[col].values,
                            max_bucket=cfg.exact_bucket)
        ex_q.append(qi); ex_d.append(di)
        log(f"   exact[{col}]: {len(qi):,} pairs")
    channels.append(("exact", np.concatenate(ex_q), np.concatenate(ex_d),
                     np.ones(sum(len(x) for x in ex_q), np.float32)))
    del ex_q, ex_d
    gc.collect()

    other_idx, s1_idx, ch_scores = merge_channels(n_s1, channels)
    exact_hit = ch_scores[:, -1].copy()
    n_union = len(s1_idx)
    log(f"   union: {n_union:,} pairs ({n_union/max(n_s1,1):.1f} per S1 entity)")
    del channels, ch_scores
    gc.collect()

    # Recompute both principal cosines exactly on the union. Most pairs were found by
    # only one view, so that view's score alone is a biased ranking signal; exact
    # rescoring is what lets the prune cut hard without shedding recall.
    Qn, Dn = keep_vecs["name_sq"]
    name_cos = pair_cosine(Qn, Dn, other_idx, s1_idx)
    del Qn, Dn, keep_vecs["name_sq"]
    gc.collect()
    Qa, Da = keep_vecs["addr"]
    addr_cos = pair_cosine(Qa, Da, other_idx, s1_idx)
    del Qa, Da, keep_vecs
    gc.collect()

    score = cfg.w_name * name_cos + cfg.w_addr * addr_cos + 0.20 * exact_hit

    if truth_keys is not None:
        sweep(s1_idx, other_idx, score, n_oth, truth_keys, n_s1, cfg)

    sel = prune_per_entity(s1_idx, score, cfg.max_per_entity, cfg.min_score)
    log(f"   pruned: {len(sel):,} pairs ({len(sel)/max(n_s1,1):.2f} per S1 entity, "
        f"{len(sel)/max(n_union,1):.1%} of union)")

    return pd.DataFrame({
        "s1_entity_id": s1["entity_id"].values[s1_idx[sel]],
        "cand_entity_id": others["entity_id"].values[other_idx[sel]],
        "name_cos": name_cos[sel],
        "addr_cos": addr_cos[sel],
        "exact_hit": exact_hit[sel],
        "block_score": score[sel],
    })


# -------------------------------------------------------------- evaluation ---
def evaluate(cands: pd.DataFrame, split: str, s1_ids_all: np.ndarray):
    from common import load_ground_truth
    gt = load_ground_truth()
    truth = {}
    for sid, ids in zip(gt["source1_entity_id"].values,
                        gt["matched_entity_ids"].str.split(",")):
        s = {i for i in ids if i}
        if s:
            truth[sid] = s
    have = cands.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
    tot = found = 0
    covered = 0
    relevant = 0
    for sid in s1_ids_all:
        t = truth.get(sid)
        if not t:
            continue
        relevant += 1
        got = have.get(sid, ())
        hit = len(t & got)
        tot += len(t)
        found += hit
        covered += (hit == len(t))
    log(f"   pair completeness : {found/max(tot,1):.4f}  ({found:,}/{tot:,})")
    log(f"   entities fully covered: {covered/max(relevant,1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--limit-s1", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="report the recall / candidate-size curve (train only)")
    args = ap.parse_args()

    s1_all = pq.read_table(os.path.join(CACHE, f"{args.split}_s1_canon.parquet"),
                           columns=CANON_COLS + ["country", "postal"]).to_pandas()
    countries = args.countries or sorted(s1_all["country"].unique())

    truth = None
    if args.sweep and args.split == "train":
        from common import load_ground_truth
        gt = load_ground_truth()
        truth = {sid: [i for i in ids if i] for sid, ids in
                 zip(gt["source1_entity_id"].values,
                     gt["matched_entity_ids"].str.split(","))}

    out = []
    for c in countries:
        log(f"=== {args.split} / {c} ===")
        s1 = s1_all[s1_all["country"] == c].reset_index(drop=True)
        if args.limit_s1:
            s1 = s1.head(args.limit_s1).reset_index(drop=True)
        others = load_others(args.split, c)

        truth_keys = None
        if truth is not None:
            oth_pos = pd.Index(others["entity_id"])
            rows, cols = [], []
            for i, sid in enumerate(s1["entity_id"].values):
                for m in truth.get(sid, ()):
                    rows.append(i); cols.append(m)
            j = oth_pos.get_indexer(np.asarray(cols)) if cols else np.empty(0, np.int64)
            ok = j >= 0
            truth_keys = np.sort(np.asarray(rows)[ok] * np.int64(len(others)) + j[ok])
            log(f"   {len(truth_keys):,} true pairs reachable in this country slice")

        out.append(block_country(s1, others, truth_keys=truth_keys))
        del others
        gc.collect()

    cands = pd.concat(out, ignore_index=True)
    path = os.path.join(CACHE, f"{args.split}_candidates.parquet")
    cands.to_parquet(path, index=False)
    log(f"wrote {path}  ({len(cands):,} pairs, "
        f"{len(cands)/len(s1_all):.2f} per S1 entity)")


if __name__ == "__main__":
    main()
