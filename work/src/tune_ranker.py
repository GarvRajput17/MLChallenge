"""Tune the blocking prune score against the cached union.

The union holds ~99% of true pairs; the question is purely how to RANK them so the
top-m per entity keeps as many as possible. Since the union is cached with both
cosines and the truth label, every candidate scoring function can be evaluated in
seconds instead of re-running retrieval.

Both graded objectives are reported: pair completeness AND candidates per entity.
"""
from __future__ import annotations
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from blocking import prune_per_entity
from common import CACHE, log


def scorers():
    """Candidate ranking functions. `n`/`a` are the exact name/address cosines."""
    return {
        "0.62n+0.38a (current)": lambda n, a, e: 0.62 * n + 0.38 * a + 0.20 * e,
        "0.50n+0.50a":           lambda n, a, e: 0.50 * n + 0.50 * a + 0.20 * e,
        "max(n,a)":              lambda n, a, e: np.maximum(n, a) + 0.20 * e,
        "max + 0.5*min":         lambda n, a, e: np.maximum(n, a) + 0.5 * np.minimum(n, a) + 0.2 * e,
        "max + 0.3*min":         lambda n, a, e: np.maximum(n, a) + 0.3 * np.minimum(n, a) + 0.2 * e,
        "sqrt(n*a) harmonic-ish": lambda n, a, e: np.sqrt(np.clip(n, 0, None) * np.clip(a, 0, None)) + 0.2 * e,
        "max + min + exact":     lambda n, a, e: np.maximum(n, a) + np.minimum(n, a) + 0.3 * e,
        "n + a (sum)":           lambda n, a, e: n + a + 0.2 * e,
        "max, n^2 tiebreak":     lambda n, a, e: np.maximum(n, a) + 0.25 * n * n + 0.25 * a * a + 0.2 * e,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", default=os.path.join(CACHE, "union_*.parquet"))
    ap.add_argument("--ms", nargs="*", type=int, default=[4, 5, 6, 8, 10])
    args = ap.parse_args()

    paths = sorted(glob.glob(args.union))
    if not paths:
        raise SystemExit(f"no cached union at {args.union}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    log(f"union: {len(df):,} pairs, {int(df['is_true'].sum()):,} true")

    s1 = df["s1_idx"].to_numpy(np.int64)
    n = df["name_cos"].to_numpy(np.float32)
    a = df["addr_cos"].to_numpy(np.float32)
    e = df["exact_hit"].to_numpy(np.float32)
    truth = df["is_true"].to_numpy(bool)
    n_truth = int(truth.sum())
    n_s1 = len(np.unique(s1))
    del df

    header = "  " + "".join(f"m={m:<14}" for m in args.ms)
    log(f"{'scorer':<26}{header}")
    log(f"{'':<26}" + "  " + "".join(f"{'recall / per-ent':<16}" for _ in args.ms))
    best = (None, -1.0)
    for name, fn in scorers().items():
        sc = fn(n, a, e).astype(np.float32)
        cells = []
        for m in args.ms:
            sel = prune_per_entity(s1, sc, m, 0.0)
            rec = truth[sel].sum() / max(n_truth, 1)
            cells.append(f"{rec:.4f}/{len(sel)/n_s1:4.1f}   ")
            if m == 6 and rec > best[1]:
                best = (name, rec)
        log(f"{name:<26}  " + "".join(cells))
    log(f"\nBEST at m=6: {best[0]}  (pair completeness {best[1]:.4f})")


if __name__ == "__main__":
    main()
