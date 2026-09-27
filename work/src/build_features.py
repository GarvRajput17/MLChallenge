"""Stage 5 driver: turn the candidate table into a labelled feature matrix.

Written out per country as parquet shards so nothing has to hold ~30M x 50 floats
in memory at once.
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

from common import CACHE, load_ground_truth, log
from features import FeatureBuilder
from run_blocking import CANON_COLS, load_country, load_others

# FeatureBuilder reads every canonical view, including the numeric ones that
# blocking does not need.
FEAT_COLS = CANON_COLS + ["nums", "postal"]
CHUNK = 4_000_000


def truth_pairs() -> set:
    gt = load_ground_truth()
    out = set()
    for sid, ids in zip(gt["source1_entity_id"].values,
                        gt["matched_entity_ids"].str.split(",")):
        for i in ids:
            if i:
                out.add((sid, i))
    return out


def build(split: str, countries=None):
    cands = pd.read_parquet(os.path.join(CACHE, f"{split}_candidates.parquet"))
    log(f"{len(cands):,} candidate pairs")
    s1_all = pq.read_table(os.path.join(CACHE, f"{split}_s1_canon.parquet"),
                           columns=FEAT_COLS + ["country"]).to_pandas()
    ctry_of = dict(zip(s1_all["entity_id"].values, s1_all["country"].values))
    cands["country"] = [ctry_of[e] for e in cands["s1_entity_id"].values]

    truth = truth_pairs() if split == "train" else None
    countries = countries or sorted(cands["country"].unique())

    for country in countries:
        out_path = os.path.join(CACHE, f"{split}_feat_{country}.parquet")
        if os.path.exists(out_path):
            log(f"{country}: cached")
            continue
        log(f"=== features {split} / {country} ===")
        sub = cands[cands["country"] == country].reset_index(drop=True)
        s1 = s1_all[s1_all["country"] == country].reset_index(drop=True)
        others = load_others(split, country, columns=FEAT_COLS)

        s1_pos = pd.Index(s1["entity_id"]).get_indexer(sub["s1_entity_id"].values)
        oth_pos = pd.Index(others["entity_id"]).get_indexer(sub["cand_entity_id"].values)
        assert (s1_pos >= 0).all() and (oth_pos >= 0).all()

        fb = FeatureBuilder(s1, others)
        frames = []
        for start in range(0, len(sub), CHUNK):
            end = min(start + CHUNK, len(sub))
            log(f"   pairs {start:,}-{end:,}")
            feats = fb.build(s1_pos[start:end], oth_pos[start:end])
            block = pd.DataFrame(feats)
            for col in ("name_cos", "addr_cos", "exact_hit", "block_score", "dense_cos", "prune_p"):
                block[f"blk_{col}"] = sub[col].values[start:end]
            block.insert(0, "s1_entity_id", sub["s1_entity_id"].values[start:end])
            block.insert(1, "cand_entity_id", sub["cand_entity_id"].values[start:end])
            frames.append(block)
        del fb
        gc.collect()

        out = pd.concat(frames, ignore_index=True)
        del frames
        if truth is not None:
            out["label"] = np.fromiter(
                ((a, b) in truth for a, b in zip(out["s1_entity_id"].values,
                                                 out["cand_entity_id"].values)),
                dtype=np.int8, count=len(out))
            log(f"   label rate {out['label'].mean():.3%}")
        out.to_parquet(out_path, index=False)
        log(f"   wrote {out_path} ({len(out):,} x {out.shape[1]})")
        del out, others, s1, sub
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--countries", nargs="*", default=None)
    a = ap.parse_args()
    build(a.split, a.countries)
