"""Stage 8: self-train the cleaning lexicon for countries that have no labels, then redo them.

Countries the model never saw labels for (France, or any the organisers add) start with the
weak label-free lexicon. Once the first pass has scored their candidate pairs, the model's own
confident matches are used as stand-in labels: on a simulated unseen country they were 99.56%
precise and covered 80% of all true pairs, and a lexicon learned from them scored as well as
one learned from real labels (0.9808 vs 0.9804 F0.5, against 0.9747 label-free).

    python3 src/self_train.py --split test        # then re-run predict.py

For each unlabelled country: score its pairs -> keep p >= P_MIN one-to-one matches -> learn its
lexicon with the same code that learns labelled countries -> re-normalise, re-block (that
country only), rebuild its features. Countries with too few confident pairs keep their
label-free lexicon. Nothing here names a country: "unlabelled" means "in the split but never
in train".
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from assemble import one_to_one
from common import CACHE, log
from normalize import build_canon
from prep_basic import build
from pseudo_lexicon import learn_from_pairs
from stack import matcher_probs

P_MIN = float(os.environ.get("ER_SELF_P", 0.99))
MIN_PAIRS = int(os.environ.get("ER_SELF_MIN_PAIRS", 20000))
HERE = os.path.dirname(os.path.abspath(__file__))


def unlabelled_countries(split: str) -> list[str]:
    return sorted(set(build(split, 1)["country"]) - set(build("train", 1)["country"]))


def confident_pairs(split: str, country: str, matcher) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(CACHE, f"{split}_feat_{country}.parquet"))
    p = matcher_probs(df, matcher)["p_lgb"].to_numpy()
    p = np.where(one_to_one(df["cand_entity_id"].values, p.astype(np.float64)), p, 0.0)
    return df.loc[p >= P_MIN, ["s1_entity_id", "cand_entity_id"]].reset_index(drop=True)


def _canon(args):
    build_canon(*args)


def run(script: str, *args: str):
    subprocess.run([sys.executable, "-u", os.path.join(HERE, script), *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    countries = unlabelled_countries(args.split)
    if not countries:
        log("no unlabelled countries -- nothing to self-train")
        return
    with open(os.path.join(CACHE, "matcher.pkl"), "rb") as fh:
        matcher = pickle.load(fh)
    lex_path = os.path.join(CACHE, "lexicons.json")
    backup = os.path.join(CACHE, "lexicons.pass1.json")
    if not os.path.exists(backup):                  # keep the label-free version, once
        shutil.copy(lex_path, backup)
    with open(backup, encoding="utf-8") as fh:
        lex = json.load(fh)

    done = []
    for country in countries:
        pairs = confident_pairs(args.split, country, matcher)
        log(f"{country}: {len(pairs):,} confident pairs (p >= {P_MIN})")
        if len(pairs) < MIN_PAIRS:
            log(f"{country}: fewer than {MIN_PAIRS:,} -- keeping the label-free lexicon")
            continue
        lex[country] = learn_from_pairs(args.split, pairs, country)
        done.append(country)
    if not done:
        return
    with open(lex_path, "w", encoding="utf-8") as fh:
        json.dump(lex, fh, ensure_ascii=False)

    # canonical views are cached per file for all countries at once: rebuild them
    for src in (1, 2, 3):
        pq = os.path.join(CACHE, f"{args.split}_s{src}_canon.parquet")
        if os.path.exists(pq):
            os.remove(pq)
    with Pool(3) as pool:
        pool.map(_canon, [(args.split, s) for s in (1, 2, 3)])
    for country in done:
        shard = os.path.join(CACHE, f"{args.split}_feat_{country}.parquet")
        if os.path.exists(shard):
            os.remove(shard)
    run("run_blocking.py", "--split", args.split, "--countries", *done)
    run("build_features.py", "--split", args.split, "--countries", *done)
    log(f"self-trained lexicons for {', '.join(done)}; now re-run predict.py")


if __name__ == "__main__":
    main()
