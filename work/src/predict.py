"""Stage 8: score the test candidates and write the two submission files."""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from assemble import add_context, choose, to_submission
from common import CACHE, OUTPUT, log
from model import stage2_predict
from train import load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--outdir", default=OUTPUT)
    args = ap.parse_args()

    with open(os.path.join(CACHE, "matcher.pkl"), "rb") as fh:
        art = pickle.load(fh)
    log(f"loaded matcher (validation macro F0.5 = {art['val_score']:.5f})")

    df = load_features(args.split)
    X = df[art["feat_cols"]].to_numpy(dtype=np.float32)
    raw = art["booster"].predict(X, num_iteration=art["booster"].best_iteration)
    del X

    add_context(df, raw)

    X2 = df[art["feat_cols2"]].to_numpy(dtype=np.float32)
    raw2 = stage2_predict(art, X2)
    del X2
    probs = art["iso"].predict(raw2)
    log(f"scored {len(probs):,} candidate pairs; mean p = {probs.mean():.4f}")

    pairs = df[["s1_entity_id", "cand_entity_id"]]
    accepted = choose(pairs, probs, miss_prior=art["miss_prior"],
                      size_penalty=art["size_penalty"])
    log(f"accepted {len(accepted):,} matches")

    s1_ids = pq.read_table(os.path.join(CACHE, f"{args.split}_s1_canon.parquet"),
                           columns=["entity_id"]).to_pandas()["entity_id"].values

    os.makedirs(args.outdir, exist_ok=True)
    matching = to_submission(accepted, s1_ids)
    matching.to_csv(os.path.join(args.outdir, "matching_results.tsv"),
                    sep="\t", index=False)

    cand_out = to_submission(pairs, s1_ids)
    cand_out.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand_out.to_csv(os.path.join(args.outdir, "candidate_pairs.tsv"),
                    sep="\t", index=False)

    n_match = (matching["matched_entity_ids"] != "").sum()
    log(f"matching_results.tsv : {len(matching):,} rows, "
        f"{len(matching) - n_match:,} predicted singletons "
        f"({(len(matching)-n_match)/len(matching):.2%})")
    log(f"candidate_pairs.tsv  : {len(pairs)/len(s1_ids):.2f} candidates per S1 entity")


if __name__ == "__main__":
    main()
