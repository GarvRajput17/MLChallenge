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

from blocking import (build_vectors, combine, exact_join, merge_channels,
                      pair_cosine, prune_per_entity, topk_pairs)
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
    """Retrieval config. S1 is always the index (small side) and S2/S3 the query.

    ER_FAST=1 drops to two views with a tighter feature budget: roughly 3x faster
    at a few points of recall. Used when wall-clock matters more than the last
    percent of pair completeness.
    """
    _FAST = os.environ.get("ER_FAST", "0") == "1"

    # (label, field, analyzer, ngram_range, top_k, threshold, max_df)
    if _FAST:
        views = [
            ("name_sq", "name_sq", "char_wb", (3, 4), 10, 0.36, 0.20),
            ("addr",    "addr_c",  "word",    (1, 1),  8, 0.34, 0.02),
        ]
        max_nnz = 16
        max_per_entity = 8
    else:
        views = [
            ("name_sq",   "name_sq",   "char_wb", (3, 4), 14, 0.32, 0.20),
            ("addr",      "addr_c",    "word",    (1, 1), 12, 0.30, 0.02),
            ("name_tok",  "name_c",    "word",    (1, 1),  8, 0.42, 0.05),
            ("addr_core", "addr_core", "word",    (1, 1),  8, 0.40, 0.02),
        ]
        max_nnz = 28
        max_per_entity = int(os.environ.get("ER_M", 12))

    score_mode = os.environ.get("ER_SCORE", "sum")   # sum | geometric | robust | weighted
    exact_keys = ("name_c", "name_sq", "addr_c")
    exact_bucket = 40
    min_score = float(os.environ.get('ER_MIN_SCORE', 0.20))


def sweep_report(s1_idx, score, is_true, n_s1, cfg, n_truth_total):
    """Pair completeness vs. candidates-per-entity -- the two graded objectives.

    Completeness is measured against ALL reachable true pairs (n_truth_total), not
    just those the union happened to retrieve, so the union's own recall ceiling is
    visible rather than definitionally 1.0.
    """
    found = int(is_true.sum())
    log(f"   union recall ceiling: {found / max(n_truth_total, 1):.4f} "
        f"({found:,}/{n_truth_total:,})")
    log("   m    pairs/entity   pair completeness")
    for m in (3, 4, 5, 6, 8, 10, 14):
        sel = prune_per_entity(s1_idx, score, m, cfg.min_score)
        log(f"   {m:<4} {len(sel)/max(n_s1,1):<14.2f} "
            f"{is_true[sel].sum()/max(n_truth_total,1):.4f}")


def block_country(s1: pd.DataFrame, others: pd.DataFrame, cfg=Config,
                  truth_keys=None, save_union=None):
    n_s1, n_oth = len(s1), len(others)
    log(f"   S1={n_s1:,}  S2+S3={n_oth:,}")

    channels, keep_vecs = [], {}
    for label, field, analyzer, ngram, k, thr, mxdf in cfg.views:
        D, Q = build_vectors(s1[field].values, others[field].values,
                             analyzer=analyzer, ngram_range=ngram,
                             max_nnz=cfg.max_nnz, max_df=mxdf)
        if D is None:
            log(f"   [{label}] empty vocabulary for this slice -- view skipped")
            continue
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
    def rescore(key):
        if key not in keep_vecs:
            return np.zeros(n_union, dtype=np.float32)
        Q, D = keep_vecs.pop(key)
        out = pair_cosine(Q, D, other_idx, s1_idx)
        del Q, D
        gc.collect()
        return out

    name_cos = rescore("name_sq")
    addr_cos = rescore("addr")
    del keep_vecs
    gc.collect()

    score = combine(name_cos, addr_cos, exact_hit, cfg.score_mode)

    if truth_keys is not None:
        keys = s1_idx * np.int64(n_oth) + other_idx
        pos = np.searchsorted(truth_keys, keys)
        pos[pos >= len(truth_keys)] = 0
        is_true = truth_keys[pos] == keys
        del keys, pos
        if save_union:
            # Cache the scored union: ranking experiments then cost seconds instead
            # of re-running retrieval. This is what makes the prune tunable at all.
            pd.DataFrame({
                "s1_idx": s1_idx.astype(np.int32),
                "oth_idx": other_idx.astype(np.int32),
                "name_cos": name_cos, "addr_cos": addr_cos,
                "exact_hit": exact_hit, "is_true": is_true,
            }).to_parquet(save_union, index=False)
            log(f"   cached union -> {save_union}")
        sweep_report(s1_idx, score, is_true, n_s1, cfg, len(truth_keys))

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
    ap.add_argument("--save-union", default=None,
                    help="parquet path to cache the scored union for ranker tuning")
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

        su = (args.save_union.replace('.parquet', f'_{c}.parquet')
              if args.save_union else None)
        out.append(block_country(s1, others, truth_keys=truth_keys,
                                 save_union=su))
        del others
        gc.collect()

    cands = pd.concat(out, ignore_index=True)
    path = os.path.join(CACHE, f"{args.split}_candidates.parquet")
    cands.to_parquet(path, index=False)
    log(f"wrote {path}  ({len(cands):,} pairs, "
        f"{len(cands)/len(s1_all):.2f} per S1 entity)")


if __name__ == "__main__":
    main()
