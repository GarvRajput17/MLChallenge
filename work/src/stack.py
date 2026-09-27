"""Stage 9 (optional): combine the LightGBM matcher with the cross-encoder.

The cross-encoder (cross_encoder.py, GPU) was fine-tuned on S1 entities outside
validation fold 0, so its scores on fold 0 are out-of-sample. A small stacker learns,
on fold 0 only, how much to trust each model; it is checked by 2-way cross-validation
over fold-0 entities before being refit on all of them and applied to test.

    python3 src/stack.py --ce-dir /tmp/ce                       # train + report
    python3 src/stack.py --ce-dir /tmp/ce --predict --outdir output_ce
        [--ce-countries US India]    # cross-encoder only there; matcher elsewhere
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from assemble import add_context, choose, to_submission
from common import CACHE, OUTPUT, load_ground_truth, log
from evaluate import macro_f05
from model import entity_fold, stage2_predict
from train import load_features

STACK_COLS = ["l_lgb", "l_ce", "ctx_margin", "ctx_best", "ctx_rank", "ctx_n_cand"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=200, verbosity=-1)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def matcher_probs(df, M):
    raw = M["booster"].predict(df[M["feat_cols"]].to_numpy(np.float32),
                               num_iteration=M["booster"].best_iteration)
    add_context(df, raw)
    df["p_lgb"] = M["iso"].predict(stage2_predict(M, df[M["feat_cols2"]].to_numpy(np.float32)))
    df["l_lgb"] = logit(df["p_lgb"].to_numpy())
    return df


def with_ce(df, path):
    ce = pd.read_parquet(path, columns=["s1_entity_id", "cand_entity_id", "ce_p"])
    df = df.merge(ce, on=["s1_entity_id", "cand_entity_id"], how="left")
    miss = df["ce_p"].isna().mean()
    assert miss < 0.001, f"{miss:.2%} of pairs have no cross-encoder score"
    df["l_ce"] = logit(df["ce_p"].fillna(0.5).to_numpy())
    return df


def fit(df):
    y = df["label"].to_numpy()
    b = lgb.train(PARAMS, lgb.Dataset(df[STACK_COLS], y), 200)
    iso = IsotonicRegression(out_of_bounds="clip").fit(b.predict(df[STACK_COLS]), y)
    return b, iso


def apply(model, df):
    b, iso = model
    return iso.predict(b.predict(df[STACK_COLS]))


def train(args, M):
    df = matcher_probs(load_features("train"), M)
    df = df[entity_fold(df["s1_entity_id"].values) == 0].reset_index(drop=True)
    df = with_ce(df, os.path.join(args.ce_dir, "eval_ce.parquet"))
    gt = load_ground_truth()
    truth = {s: {i for i in ids if i} for s, ids in
             zip(gt["source1_entity_id"], gt["matched_entity_ids"].str.split(","))}
    ents = df["s1_entity_id"].unique()
    side = dict(zip(ents, np.random.default_rng(0).integers(0, 2, len(ents))))
    half = df["s1_entity_id"].map(side).to_numpy()

    def f05(p, mask):
        acc = choose(df.loc[mask, ["s1_entity_id", "cand_entity_id"]], p,
                     miss_prior=M["miss_prior"], size_penalty=M["size_penalty"])
        pred = acc.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
        return macro_f05(pred, truth, df.loc[mask, "s1_entity_id"].unique())

    base, stacked = [], []
    for h in (0, 1):
        te = half == h
        base.append(f05(df.loc[te, "p_lgb"].to_numpy(), te))
        stacked.append(f05(apply(fit(df[~te]), df[te]), te))
    log(f"fold-0 2-way CV over {len(ents):,} entities: matcher {np.mean(base):.5f}  "
        f"matcher+cross-encoder {np.mean(stacked):.5f}  "
        f"gain {np.mean(stacked) - np.mean(base):+.5f}")
    model = fit(df)
    with open(os.path.join(CACHE, "stacker.pkl"), "wb") as fh:
        pickle.dump(dict(model=model, cv_gain=float(np.mean(stacked) - np.mean(base))), fh)
    log("wrote cache/stacker.pkl")


def predict(args, M):
    with open(os.path.join(CACHE, "stacker.pkl"), "rb") as fh:
        S = pickle.load(fh)
    df = with_ce(matcher_probs(load_features("test"), M),
                 os.path.join(args.ce_dir, "score_ce.parquet"))
    p = apply(S["model"], df)
    if args.ce_countries:
        use = df["country"].isin(args.ce_countries).to_numpy()
        p = np.where(use, p, df["p_lgb"].to_numpy())
        log(f"cross-encoder applied to {use.mean():.1%} of pairs ({', '.join(args.ce_countries)})")
    pairs = df[["s1_entity_id", "cand_entity_id"]]
    accepted = choose(pairs, p, miss_prior=M["miss_prior"], size_penalty=M["size_penalty"])
    s1_ids = pq.read_table(os.path.join(CACHE, "test_s1_canon.parquet"),
                           columns=["entity_id"]).to_pandas()["entity_id"].values
    os.makedirs(args.outdir, exist_ok=True)
    matching = to_submission(accepted, s1_ids)
    matching.to_csv(os.path.join(args.outdir, "matching_results.tsv"), sep="\t", index=False)
    cand = to_submission(pairs, s1_ids)
    cand.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand.to_csv(os.path.join(args.outdir, "candidate_pairs.tsv"), sep="\t", index=False)
    n_empty = (matching["matched_entity_ids"] == "").sum()
    log(f"{args.outdir}: {len(accepted):,} matches, {n_empty:,} predicted singletons "
        f"({n_empty / len(matching):.2%}), {len(pairs) / len(s1_ids):.2f} candidates per S1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-dir", required=True, help="dir with eval_ce.parquet / score_ce.parquet")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--outdir", default=OUTPUT)
    ap.add_argument("--ce-countries", nargs="*", default=None)
    args = ap.parse_args()
    with open(os.path.join(CACHE, "matcher.pkl"), "rb") as fh:
        M = pickle.load(fh)
    predict(args, M) if args.predict else train(args, M)


if __name__ == "__main__":
    main()
