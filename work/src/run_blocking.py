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

import scipy.sparse as sp
from sklearn.preprocessing import normalize

# blocking (sparse_dot_topn) must load before lightgbm: the reverse order loads two
# OpenMP runtimes and segfaults in the first sp_matmul_topn call (seen on macOS).
from blocking import (PRUNE_CHANNELS, PRUNER_FEATURES, build_vectors, combine,
                      exact_join, group_stats, merge_channels, pair_cosine,
                      pruner_features, rich_features, topk_pairs)
import lightgbm as lgb
from common import CACHE, THREADS, log
from model import entity_fold

CANON_COLS = ["entity_id", "name_c", "name_core", "name_sq", "addr_c", "addr_core"]
RICH_COLS = ["entity_id", "name_c", "name_sq", "addr_c", "addr_core", "nums"]


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


# ------------------------------------------------------------ dense channel ---
class Dense:
    """Output of dense_encoder.py (GPU): learned multilingual embeddings for every
    record, each S2/S3 record's nearest S1 records, and S2<->S3 nearest neighbours.
    Optional -- when absent (no GPU run for this split) blocking runs without it."""

    def __init__(self, split: str):
        self.dir = os.environ.get("ER_DENSE_DIR", os.path.join(CACHE, "dense"))
        self.split = split
        self.ok = os.path.exists(os.path.join(self.dir, "emb", f"{split}_s1.npy"))
        if self.ok:
            self.emb = {s: np.load(os.path.join(self.dir, "emb", f"{split}_s{s}.npy"))
                        for s in (1, 2, 3)}
            self.ids = {s: pd.Index(pd.read_parquet(os.path.join(
                self.dir, "emb", f"{split}_s{s}_ids.parquet"))["entity_id"]) for s in (1, 2, 3)}
        log(f"   dense channel: {'on' if self.ok else 'OFF (no embeddings for ' + split + ')'}")

    def vectors(self, ids: np.ndarray) -> np.ndarray:
        """float16 embeddings aligned to `ids` (any mix of S1/S2/S3 entity ids).

        `ids` may come straight off a parquet-read column, which pandas 3.x backs
        with pyarrow strings -- np.char ufuncs don't have a dispatch loop for that
        dtype and raise UFuncNoLoopError. pd.Series.str is pandas' own vectorised
        string layer and handles Arrow-backed and plain-object columns alike.
        """
        out = np.zeros((len(ids), self.emb[1].shape[1]), np.float16)
        src = pd.Series(ids)
        for s in (1, 2, 3):
            m = src.str.startswith(f"S{s}-").to_numpy()
            if m.any():
                r = self.ids[s].get_indexer(ids[m])
                assert (r >= 0).all(), f"{(r < 0).sum()} S{s} ids missing from embeddings"
                out[m] = self.emb[s][r]
        return out

    def pairs(self, country, s1_pos: pd.Index, oth_pos: pd.Index):
        path = os.path.join(self.dir, f"dense_{self.split}_{country}.parquet")
        if not os.path.exists(path):
            return None
        d = pd.read_parquet(path)
        qi = oth_pos.get_indexer(d["cand_entity_id"].values)
        di = s1_pos.get_indexer(d["s1_entity_id"].values)
        ok = (qi >= 0) & (di >= 0)
        return qi[ok].astype(np.int64), di[ok].astype(np.int64), d["dense_cos"].values[ok].astype(np.float32)

    def links(self, country, oth_pos: pd.Index):
        parts = [os.path.join(self.dir, f"links_{self.split}_{country}_{t}.parquet") for t in ("s2s3", "s3s2")]
        parts = [pd.read_parquet(p) for p in parts if os.path.exists(p)]
        if not parts:
            return None
        d = pd.concat(parts, ignore_index=True)
        a, b = oth_pos.get_indexer(d["a_id"].values), oth_pos.get_indexer(d["b_id"].values)
        ok = (a >= 0) & (b >= 0)
        return a[ok].astype(np.int64), b[ok].astype(np.int64), d["link_cos"].values[ok].astype(np.float32)


def pair_dot(E1, Eo, s1_idx, oth_idx, chunk=2_000_000):
    out = np.empty(len(s1_idx), np.float32)
    for s in range(0, len(s1_idx), chunk):
        e = min(s + chunk, len(s1_idx))
        out[s:e] = np.einsum("ij,ij->i", E1[s1_idx[s:e]].astype(np.float32),
                             Eo[oth_idx[s:e]].astype(np.float32))
    return out


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
    else:
        views = [
            ("name_sq",   "name_sq",   "char_wb", (3, 4), 14, 0.32, 0.20),
            ("addr",      "addr_c",    "word",    (1, 1), 12, 0.30, 0.02),
            ("name_tok",  "name_c",    "word",    (1, 1),  8, 0.42, 0.05),
            ("addr_core", "addr_core", "word",    (1, 1),  8, 0.40, 0.02),
        ]
        max_nnz = 28
    # Joint name+address channel: 39% of S1 names are shared, so a name-only top-k
    # fills up with same-name entities and the address never gets to break the tie.
    # Retrieving on both fields at once is what the per-field channels cannot do.
    joint_k = int(os.environ.get("ER_JOINT_K", 20))
    joint_thr = 0.25

    score_mode = os.environ.get("ER_SCORE", "sum")   # sum | geometric | robust | weighted
    exact_keys = ("name_c", "name_sq", "addr_c")
    exact_bucket = 40
    min_score = float(os.environ.get('ER_MIN_SCORE', 0.20))
    # Learned prune: keep a candidate if the pruner's p >= prune_t, at most
    # max_per_entity per S1. The per-entity cut adapts: one obvious match keeps one
    # or two candidates, an ambiguous entity keeps more.
    prune_t = float(os.environ.get("ER_PRUNE_T", 0.01))
    # Stage 1 (cheap features, whole union) only has to be loose; stage 2 (rich
    # features, survivors only) makes the reported cut.
    stage1_t = float(os.environ.get("ER_STAGE1_T", 0.002))
    stage1_max = int(os.environ.get("ER_STAGE1_M", 30))
    link_top = 3                 # each S1's best candidates whose S2<->S3 twins are added
    max_per_entity = int(os.environ.get("ER_M", 12))
    pruner_rows = int(os.environ.get("ER_PRUNER_ROWS", 40_000_000))


def block_country(s1: pd.DataFrame, others: pd.DataFrame, cfg=Config, truth_keys=None,
                  dense=None, country=None):
    """Retrieve, union and score one country. Returns the union (above min_score, or
    found by the dense/link channels) as (s1_idx, other_idx, pruner features, is_true)."""
    n_s1, n_oth = len(s1), len(others)
    log(f"   S1={n_s1:,}  S2+S3={n_oth:,}")
    s1_pos, oth_pos = pd.Index(s1["entity_id"]), pd.Index(others["entity_id"])
    use_dense = dense is not None and dense.ok

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

    if "name_sq" in keep_vecs and "addr" in keep_vecs:
        (Qn, Dn), (Qa, Da) = keep_vecs["name_sq"], keep_vecs["addr"]
        Qj = normalize(sp.hstack([Qn, Qa], format="csr"), copy=False)
        Dj = normalize(sp.hstack([Dn, Da], format="csr"), copy=False)
        qi, di, sc = topk_pairs(Qj, Dj, cfg.joint_k, cfg.joint_thr)
        log(f"   [joint] {len(qi):,} pairs ({len(qi)/max(n_oth,1):.1f} per record)")
        channels.append(("joint", qi, di, sc))
        del Qj, Dj
        gc.collect()

    if use_dense:
        dp = dense.pairs(country, s1_pos, oth_pos)
        if dp is not None:
            channels.append(("dense",) + dp)
            log(f"   [dense] {len(dp[0]):,} pairs ({len(dp[0])/max(n_oth,1):.1f} per record)")

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

    labels = [c[0] for c in channels]
    other_idx, s1_idx, ch_scores = merge_channels(n_s1, channels)
    n_union = len(s1_idx)
    log(f"   union: {n_union:,} pairs ({n_union/max(n_s1,1):.1f} per S1 entity)")
    del channels
    gc.collect()

    # Recompute both principal cosines exactly on the union. Most pairs were found by
    # only one view, so that view's score alone is a biased ranking signal.
    def cos(key, oi, si):
        if key not in keep_vecs:
            return np.zeros(len(si), dtype=np.float32)
        Q, D = keep_vecs[key]
        return pair_cosine(Q, D, oi, si)

    name_cos = cos("name_sq", other_idx, s1_idx)
    addr_cos = cos("addr", other_idx, s1_idx)
    exact_hit = ch_scores[:, labels.index("exact")].copy()
    score = combine(name_cos, addr_cos, exact_hit, cfg.score_mode)
    channel = {c: ch_scores[:, labels.index(c)] for c in labels}
    del ch_scores

    # Cross-links: most businesses have matches in BOTH S2 and S3, and a noisy S3
    # record is often closer to its noisy S2 twin than to the clean S1 original. So
    # each S1's best candidates pull in their nearest other-source neighbours.
    link = dense.links(country, oth_pos) if use_dense else None
    if link is not None:
        a, b, lc = link
        top = group_stats(s1_idx, score)[:, 0] < cfg.link_top
        order = np.argsort(a, kind="stable")
        a, b, lc = a[order], b[order], lc[order]
        lo = np.searchsorted(a, other_idx[top], "left")
        hi = np.searchsorted(a, other_idx[top], "right")
        cnt = hi - lo
        rep = np.repeat(np.arange(len(lo)), cnt)
        pos = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        new_s1, new_oth, new_lc = s1_idx[top][rep], b[pos], lc[pos]
        existing = s1_idx * np.int64(n_oth) + other_idx
        new_key = new_s1 * np.int64(n_oth) + new_oth
        new_key, first = np.unique(new_key, return_index=True)
        fresh = ~np.isin(new_key, existing)
        new_s1, new_oth, new_lc = new_s1[first][fresh], new_oth[first][fresh], new_lc[first][fresh]
        n_new = len(new_s1)
        log(f"   [link] {n_new:,} new pairs via S2<->S3 twins of each S1's top-{cfg.link_top}")
        nn, na = cos("name_sq", new_oth, new_s1), cos("addr", new_oth, new_s1)
        s1_idx, other_idx = np.concatenate([s1_idx, new_s1]), np.concatenate([other_idx, new_oth])
        name_cos, addr_cos = np.concatenate([name_cos, nn]), np.concatenate([addr_cos, na])
        exact_hit = np.concatenate([exact_hit, np.zeros(n_new, np.float32)])
        score = combine(name_cos, addr_cos, exact_hit, cfg.score_mode)
        channel = {c: np.concatenate([v, np.zeros(n_new, np.float32)]) for c, v in channel.items()}
        channel["link"] = np.concatenate([np.zeros(len(s1_idx) - n_new, np.float32), new_lc])
    del keep_vecs
    gc.collect()

    zeros_all = np.zeros(len(s1_idx), np.float32)
    channel = {c: channel.get(c, zeros_all) for c in PRUNE_CHANNELS}
    dense_cos = zeros_all
    if use_dense:
        E1 = dense.vectors(s1["entity_id"].values)
        Eo = dense.vectors(others["entity_id"].values)
        dense_cos = pair_dot(E1, Eo, s1_idx, other_idx)
        del E1, Eo

    is_true = None
    if truth_keys is not None:
        keys = s1_idx * np.int64(n_oth) + other_idx
        pos = np.searchsorted(truth_keys, keys)
        pos[pos >= len(truth_keys)] = 0
        is_true = truth_keys[pos] == keys
        # which channel is solely responsible for how many true pairs (methodology evidence)
        nz = np.column_stack([channel[c] > 0 for c in PRUNE_CHANNELS])
        only = nz.sum(1) == 1
        log("   true pairs found ONLY by: " + ", ".join(
            f"{c}={int((is_true & only & nz[:, i]).sum()):,}" for i, c in enumerate(PRUNE_CHANNELS)))

    keep = (score >= cfg.min_score) | (channel["dense"] > 0) | (channel["link"] > 0)
    s1_idx, other_idx = s1_idx[keep], other_idx[keep]
    X = pruner_features(s1_idx, other_idx, name_cos[keep], addr_cos[keep], exact_hit[keep],
                        score[keep], dense_cos[keep], {c: v[keep] for c, v in channel.items()})
    del channel, name_cos, addr_cos, exact_hit, score, dense_cos
    gc.collect()
    if is_true is not None:
        is_true = is_true[keep]
        log(f"   union recall ceiling: {is_true.sum() / max(len(truth_keys), 1):.4f} "
            f"({int(is_true.sum()):,}/{len(truth_keys):,}), "
            f"{len(s1_idx)/max(n_s1,1):.1f} pairs per S1 kept for pruning")
    return s1_idx, other_idx, X, is_true


# ------------------------------------------------------------------ pruner ---
PRUNER_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31,
                     min_data_in_leaf=500, verbosity=-1)


def select(s1_idx, p, cfg=Config):
    """Rows kept by the learned prune: p >= prune_t, best max_per_entity per S1."""
    rank = group_stats(s1_idx, p)[:, 0]
    return (p >= cfg.prune_t) & (rank < cfg.max_per_entity)


def train_pruner(unions, holdout_country, cfg=Config, names=PRUNER_FEATURES, tag="stage 1"):
    """Fit on S1 entities outside validation fold 0 (and outside a simulated-unseen
    country), so the matcher's validation score never sees a pruner that trained
    on its entities. Reports the held-out size/recall curve for the threshold."""
    rng = np.random.default_rng(0)
    parts_X, parts_y, n_rows = [], [], sum(len(u["s1"]) for u in unions.values())
    frac = min(1.0, cfg.pruner_rows / max(n_rows, 1))
    for c, u in unions.items():
        if c == holdout_country:
            continue
        m = (u["fold"][u["s1"]] != 0) & (rng.random(len(u["s1"])) < frac)
        parts_X.append(u["X"][m]); parts_y.append(u["is_true"][m])
    X, y = np.concatenate(parts_X), np.concatenate(parts_y)
    log(f"   pruner {tag}: training on {len(y):,} pairs ({y.mean():.3%} true)")
    booster = lgb.train(dict(PRUNER_PARAMS, num_threads=THREADS),
                        lgb.Dataset(X, y, feature_name=list(names)),
                        num_boost_round=150)
    del X, y, parts_X, parts_y
    gc.collect()

    log(f"   {tag} held-out fold 0:  prune_t   cand/S1   pair completeness")
    for c, u in unions.items():
        v = u["fold"][u["s1"]] == 0
        p = booster.predict(u["X"][v], num_threads=THREADS)
        n_s1 = int((u["fold"] == 0).sum())
        n_true = int((u["fold"][u["truth_keys"] // u["n_oth"]] == 0).sum())
        for t in (0.003, 0.01, 0.03, 0.1):
            keep = (p >= t) & (group_stats(u["s1"][v], p)[:, 0] < cfg.max_per_entity)
            log(f"   {c:<10} {t:<9} {keep.sum()/max(n_s1,1):<9.2f} "
                f"{u['is_true'][v][keep].sum()/max(n_true,1):.4f}")
    return booster


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
    args = ap.parse_args()
    holdout = os.environ.get("ER_HOLDOUT_COUNTRY")   # simulate an unseen country

    s1_all = pq.read_table(os.path.join(CACHE, f"{args.split}_s1_canon.parquet"),
                           columns=CANON_COLS + ["country"]).to_pandas()
    countries = args.countries or sorted(s1_all["country"].unique())

    truth = None
    if args.split == "train":
        from common import load_ground_truth
        gt = load_ground_truth()
        truth = {sid: [i for i in ids if i] for sid, ids in
                 zip(gt["source1_entity_id"].values,
                     gt["matched_entity_ids"].str.split(","))}

    dense = Dense(args.split)
    unions = {}
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
            truth_keys = np.sort(np.asarray(rows, dtype=np.int64)[ok] * np.int64(len(others)) + j[ok])
            log(f"   {len(truth_keys):,} true pairs reachable in this country slice")

        s1_idx, other_idx, X, is_true = block_country(s1, others, truth_keys=truth_keys,
                                                       dense=dense, country=c)
        unions[c] = dict(s1=s1_idx, oth=other_idx, X=X, is_true=is_true,
                         s1_ids=s1["entity_id"].values, oth_ids=others["entity_id"].values,
                         fold=entity_fold(s1["entity_id"].values),
                         truth_keys=truth_keys, n_oth=len(others))
        del others
        gc.collect()
    del dense

    # ---- stage 1: cheap pruner over the whole union, loose cut ----------------
    p1_path, p2_path = os.path.join(CACHE, "pruner.txt"), os.path.join(CACHE, "pruner2.txt")
    if args.split == "train":
        b1 = train_pruner(unions, holdout, tag="stage 1")
        b1.save_model(p1_path)
    else:
        b1 = lgb.Booster(model_file=p1_path)
    for c, u in unions.items():
        p1 = b1.predict(u["X"], num_threads=THREADS).astype(np.float32)
        keep = (p1 >= Config.stage1_t) & (group_stats(u["s1"], p1)[:, 0] < Config.stage1_max)
        s1r = load_country(args.split, 1, c, RICH_COLS).set_index("entity_id").loc[u["s1_ids"]].reset_index()
        oth = load_others(args.split, c, RICH_COLS)
        R, rich_names = rich_features(s1r, oth, u["s1"][keep], u["oth"][keep], threads=THREADS)
        u.update(s1=u["s1"][keep], oth=u["oth"][keep],
                 X=np.hstack([u["X"][keep], p1[keep, None], R]),
                 is_true=None if u["is_true"] is None else u["is_true"][keep])
        msg = f"   {c}: stage 1 keeps {keep.sum():,} pairs ({keep.sum()/max(len(u['s1_ids']),1):.1f} per S1)"
        if u["is_true"] is not None:
            msg += f", pair completeness {u['is_true'].sum()/max(len(u['truth_keys']),1):.4f}"
        log(msg)
        del oth, s1r, R
        gc.collect()
    names2 = list(PRUNER_FEATURES) + ["stage1_p"] + rich_names

    # ---- stage 2: rich-feature pruner on survivors, the reported cut ---------
    if args.split == "train":
        b2 = train_pruner(unions, holdout, names=names2, tag="stage 2")
        b2.save_model(p2_path)
    else:
        b2 = lgb.Booster(model_file=p2_path)

    out = []
    for c, u in unions.items():
        p = b2.predict(u["X"], num_threads=THREADS).astype(np.float32)
        sel = select(u["s1"], p)
        X = u["X"]
        cands = pd.DataFrame({
            "s1_entity_id": u["s1_ids"][u["s1"][sel]],
            "cand_entity_id": u["oth_ids"][u["oth"][sel]],
            "name_cos": X[sel, 0], "addr_cos": X[sel, 1],
            "exact_hit": X[sel, 2], "block_score": X[sel, 3], "dense_cos": X[sel, 4],
            "prune_p": p[sel],
        })
        msg = f"   {c}: {len(cands):,} candidates ({len(cands)/max(len(u['s1_ids']),1):.2f} per S1)"
        if u["is_true"] is not None:
            msg += f", pair completeness {u['is_true'][sel].sum()/max(len(u['truth_keys']),1):.4f}"
        log(msg)
        out.append(cands)
        unions[c] = None
        gc.collect()

    cands = pd.concat(out, ignore_index=True)
    path = os.path.join(CACHE, f"{args.split}_candidates.parquet")
    if args.countries and os.path.exists(path):
        # partial re-run (e.g. self_train.py re-blocking one country): keep the others' rows
        old = pd.read_parquet(path)
        country_of = dict(zip(s1_all["entity_id"].values, s1_all["country"].values))
        redone = old["s1_entity_id"].map(country_of).isin(set(args.countries))
        cands = pd.concat([old[~redone], cands], ignore_index=True)
    cands.to_parquet(path, index=False)
    log(f"wrote {path}  ({len(cands):,} pairs, "
        f"{len(cands)/len(s1_all):.2f} per S1 entity)")


if __name__ == "__main__":
    main()
