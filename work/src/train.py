"""Stage 6 driver: train the pairwise matcher, calibrate it, and tune the decision
layer against macro F_0.5 on a held-out split of S1 *entities*.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from assemble import CTX_COLS, add_context, choose
from common import CACHE, REPORTS, load_ground_truth, log
from evaluate import breakdown, macro_f05
from model import (calibration_report, entity_fold, fit_calibrator, importance,
                   stage2_predict, train, train_xgb)

ID_COLS = ("s1_entity_id", "cand_entity_id", "label", "country")
VAL_FOLDS = (0,)


def load_features(split: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(CACHE, f"{split}_feat_*.parquet")))
    if not paths:
        raise SystemExit(f"no feature shards for split={split}; run build_features.py")
    frames = []
    for p in paths:
        country = os.path.basename(p).split("_feat_")[1][:-len(".parquet")]
        df = pd.read_parquet(p)
        df["country"] = country
        frames.append(df)
        log(f"   loaded {os.path.basename(p)}: {len(df):,}")
    return pd.concat(frames, ignore_index=True)


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in ID_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--miss-prior", type=float, default=0.0)
    args = ap.parse_args()

    df = load_features("train")
    # Simulate an unseen country (France-like): train on the others only, then
    # score the held-out country end to end at the bottom.
    holdout = os.environ.get("ER_HOLDOUT_COUNTRY")
    df_h = None
    if holdout:
        df_h = df[df["country"] == holdout].reset_index(drop=True)
        df = df[df["country"] != holdout].reset_index(drop=True)
        log(f"holding out {holdout}: {len(df_h):,} pairs set aside as an unseen country")
    folds = entity_fold(df["s1_entity_id"].values)
    feat_cols = feature_columns(df)
    log(f"{len(df):,} pairs x {len(feat_cols)} features")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)
    booster, val_mask = train(X, y, folds, feat_cols, val_folds=VAL_FOLDS,
                              num_boost_round=args.rounds)
    log("feature importance (gain):\n" + importance(booster, feat_cols))

    raw = booster.predict(X, num_iteration=booster.best_iteration)

    # --- second pass: context features the pairwise model cannot see -------------
    add_context(df, raw)
    feat_cols2 = feat_cols + CTX_COLS
    X2 = df[feat_cols2].to_numpy(dtype=np.float32)
    booster2, _ = train(X2, y, folds, feat_cols2, val_folds=VAL_FOLDS,
                        num_boost_round=args.rounds)
    log("stage-2 importance (gain):\n" + importance(booster2, feat_cols2))
    art = dict(booster2=booster2, xgb2=None, blend=False)
    if os.environ.get("ER_XGB", "1") == "1":
        from sklearn.metrics import log_loss
        art["xgb2"] = train_xgb(X2, y, val_mask)
        va = val_mask
        ll = {b: log_loss(y[va], np.clip(stage2_predict(art, X2[va], use_blend=b), 1e-7, 1 - 1e-7))
              for b in (False, True)}
        art["blend"] = ll[True] < ll[False]
        log(f"stage-2 held-out logloss: LightGBM {ll[False]:.5f}  LightGBM+XGBoost {ll[True]:.5f}"
            f"  -> using {'blend' if art['blend'] else 'LightGBM only'}")
    raw2 = stage2_predict(art, X2)

    # --- calibration, fitted on the held-out entities only -----------------------
    iso = fit_calibrator(raw2[val_mask], y[val_mask])
    p_val = iso.predict(raw2[val_mask])
    log("calibration on validation fold:\n" + calibration_report(p_val, y[val_mask]))

    # --- decision layer: tune miss_prior / size_penalty against macro F0.5 --------
    gt = load_ground_truth()
    truth_map = {s: {i for i in ids if i} for s, ids in
                 zip(gt["source1_entity_id"].values,
                     gt["matched_entity_ids"].str.split(","))}
    s1_country = dict(zip(df["s1_entity_id"].values, df["country"].values))
    val_df = df.loc[val_mask, ["s1_entity_id", "cand_entity_id"]].reset_index(drop=True)
    val_ids = np.unique(val_df["s1_entity_id"].values)
    log(f"validating on {len(val_ids):,} held-out S1 entities")

    best = (None, -1.0)
    for miss in (0.0, 0.1, 0.2, 0.35, 0.5):
        for pen in (0.0, 0.002, 0.005):
            acc = choose(val_df, p_val, miss_prior=miss, size_penalty=pen)
            pred = acc.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
            score = macro_f05(pred, truth_map, val_ids)
            log(f"   miss_prior={miss:<5} size_penalty={pen:<6} "
                f"macro F0.5 = {score:.5f}  (avg |S| {len(acc)/len(val_ids):.2f})")
            if score > best[1]:
                best = ((miss, pen), score)
    (miss, pen), score = best
    log(f"BEST macro F0.5 = {score:.5f}  at miss_prior={miss} size_penalty={pen}")

    acc = choose(val_df, p_val, miss_prior=miss, size_penalty=pen)
    pred = acc.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
    log("by country: " + json.dumps(
        {k: (round(v[0], 4), v[1]) for k, v in
         breakdown(pred, truth_map, val_ids, s1_country).items()}))
    sizes = {e: len(truth_map.get(e, ())) for e in val_ids}
    log("by true-list size: " + json.dumps(
        {str(k): (round(v[0], 4), v[1]) for k, v in
         breakdown(pred, truth_map, val_ids,
                   {e: min(sizes[e], 6) for e in val_ids}).items()}))

    if df_h is not None:
        raw_h = booster.predict(df_h[feat_cols].to_numpy(dtype=np.float32),
                                num_iteration=booster.best_iteration)
        add_context(df_h, raw_h)
        p_h = iso.predict(stage2_predict(art, df_h[feat_cols2].to_numpy(dtype=np.float32)))
        acc = choose(df_h[["s1_entity_id", "cand_entity_id"]], p_h,
                     miss_prior=miss, size_penalty=pen)
        pred_h = acc.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
        s1c = pd.read_parquet(os.path.join(CACHE, "train_s1_basic.parquet"),
                              columns=["entity_id", "country"])
        ids_h = s1c.loc[s1c["country"] == holdout, "entity_id"].values   # all, incl. no-candidate
        log(f"UNSEEN-COUNTRY SIMULATION: {holdout} macro F0.5 = "
            f"{macro_f05(pred_h, truth_map, ids_h):.5f} over {len(ids_h):,} entities")

    with open(os.path.join(CACHE, "matcher.pkl"), "wb") as fh:
        pickle.dump(dict(booster=booster, booster2=booster2, xgb2=art["xgb2"], blend=art["blend"], iso=iso,
                         feat_cols=feat_cols, feat_cols2=feat_cols2,
                         miss_prior=miss, size_penalty=pen, val_score=score), fh)
    log("wrote cache/matcher.pkl")


if __name__ == "__main__":
    main()
