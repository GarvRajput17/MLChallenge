"""Learn a country's cleaning lexicon from *predicted* matches instead of ground truth.

For a country with no labels (France, or any unseen one) the label-free induction in
learn_lexicons.py is weak: measured on a simulated unseen US it scored 0.954 against
~0.98 with labels. The matcher's own high-confidence matches are a far better signal --
on that simulation, matches at p >= 0.99 were 99.56% precise and covered 80% of all true
pairs. This script feeds them through the SAME alignment + distillation code that
learns the labeled countries' lexicons (learn_lexicons.collect / distil), so the result
has the same shape and the same guards (subsequence/JW checks, min counts, non-transitive
phrase links).

    python3 src/pseudo_lexicon.py --pairs cache/pseudo_pairs_US_p99.parquet \
        --split train --country US --out cache/lex_US_pseudo.json

`--pairs` is any parquet with s1_entity_id, cand_entity_id. The result is a dict shaped like
one country's entry of lexicons.json ({name_b: {...}, addr_b: {...}}); `--merge` writes it
into an existing lexicons.json in place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import CACHE, log
from learn_lexicons import collect, distil
from prep_basic import build


def learn_from_pairs(split: str, pairs: pd.DataFrame, country: str) -> dict:
    s1, s2, s3 = build(split, 1), build(split, 2), build(split, 3)
    i1 = pd.Index(s1["entity_id"]).get_indexer(pairs["s1_entity_id"].values)
    cand = pairs["cand_entity_id"]
    is2 = cand.str.startswith("S2-").to_numpy()
    other = np.where(is2,
                     pd.Index(s2["entity_id"]).get_indexer(cand.values),
                     pd.Index(s3["entity_id"]).get_indexer(cand.values))
    ok = (i1 >= 0) & (other >= 0)
    if not ok.all():
        log(f"dropping {(~ok).sum():,} pairs whose ids are not in the {split} records")
    i1, other, is2 = i1[ok], other[ok], is2[ok]
    in_country = (s1["country"].values[i1] == country)
    i1, other, is2 = i1[in_country], other[in_country], is2[in_country]
    log(f"{len(i1):,} pairs for {country}")

    stats = collect(s1, s2, s3, i1, other, is2)
    out = {}
    for field in ("name_b", "addr_b"):
        translit, variant, expand, generic, prefix = distil(stats[(country, field)])
        out[field] = dict(translit=translit, variant=variant, expand=expand,
                          generic=generic, prefix=prefix if field == "name_b" else [])
        log(f"{country}/{field}: translit={len(translit):,} variant={len(variant):,} "
            f"expand={len(expand):,} generic={len(generic):,} prefix={out[field]['prefix']}")
        log(f"   variant sample: {list(variant.items())[:12]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--country", required=True)
    ap.add_argument("--out", required=True, help="write this country's lexicon here (json)")
    ap.add_argument("--merge", default=None,
                    help="also write it into this lexicons.json under the country key")
    args = ap.parse_args()

    lex = learn_from_pairs(args.split, pd.read_parquet(args.pairs), args.country)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(lex, fh, ensure_ascii=False)
    log("wrote", args.out)
    if args.merge:
        with open(args.merge, encoding="utf-8") as fh:
            full = json.load(fh)
        full[args.country] = lex
        with open(args.merge, "w", encoding="utf-8") as fh:
            json.dump(full, fh, ensure_ascii=False)
        log(f"merged into {args.merge}")


if __name__ == "__main__":
    main()
