"""Export candidate pairs with raw text for the cross-encoder (GPU) stage.

Entities are split by the same hash folds as the LightGBM matcher, so cross-encoder
scores on fold 0 are directly comparable with the matcher's validation scores.

    python3 src/build_ce_pairs.py --split train --train-entities 250000 --eval-entities 40000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import CACHE, load_ground_truth, load_source, log
from model import entity_fold


def pairs_with_text(split: str) -> pd.DataFrame:
    c = pd.read_parquet(os.path.join(CACHE, f"{split}_candidates.parquet"),
                        columns=["s1_entity_id", "cand_entity_id"])
    s1 = load_source(split, 1).rename(columns={"entity_id": "s1_entity_id",
                                               "business_name": "a_name",
                                               "business_address": "a_addr"})
    oth = pd.concat([load_source(split, 2), load_source(split, 3)]).rename(
        columns={"entity_id": "cand_entity_id", "business_name": "b_name",
                 "business_address": "b_addr"}).drop(columns="country")
    return c.merge(s1, on="s1_entity_id").merge(oth, on="cand_entity_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--train-entities", type=int, default=0, help="0 = all non-fold-0")
    ap.add_argument("--eval-entities", type=int, default=0, help="0 = all fold-0")
    ap.add_argument("--out", default=os.path.join(CACHE, "ce"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pairs_with_text(args.split)
    if args.split != "train":
        df.to_parquet(os.path.join(args.out, f"{args.split}_pairs.parquet"), index=False)
        log(f"{args.split}: {len(df):,} pairs")
        return

    gt = load_ground_truth()
    true = {(s, m) for s, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"].str.split(","))
            for m in ids if m}
    df["label"] = np.fromiter(((s, m) in true for s, m in
                               zip(df["s1_entity_id"], df["cand_entity_id"])), np.int8, len(df))
    ents = df["s1_entity_id"].unique()
    fold = entity_fold(ents)
    rng = np.random.default_rng(0)
    for name, pool, n in (("train", ents[fold != 0], args.train_entities),
                          ("eval", ents[fold == 0], args.eval_entities)):
        pick = rng.choice(pool, n, replace=False) if n and n < len(pool) else pool
        sub = df[df["s1_entity_id"].isin(set(pick))]
        sub.to_parquet(os.path.join(args.out, f"{name}.parquet"), index=False)
        log(f"{name}: {len(pick):,} entities, {len(sub):,} pairs, {sub['label'].mean():.1%} true")


if __name__ == "__main__":
    main()
