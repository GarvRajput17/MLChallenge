"""Stage 1b: adaptive stop-word configuration, fitted on the records at hand.

Runs BEFORE normalisation, needs no labels, and hard-codes no language, so it works
unchanged for any country (France, or one the organisers add later).

Per (country, field) it decides
  1. which tokens are stop words -- the head of the frequency distribution, cut at the knee
     of the rank-frequency curve (so the cut adapts to each dataset's own shape): they
     appear in so many records that agreeing on them says almost nothing about identity;
  2. how they group -- stop words are clustered by *distributional role*: how often they
     start, end, or sit inside a multi-word string, and which tokens they sit next to. Legal forms
     (inc llc ltd / sarl sas eurl), street types (rue avenue allee / road street drive),
     and place words fall into separate groups without anyone naming them.

normalize.py consumes the result (cache/norm_config.json): stop words are dropped from the
`*_core` views. The groups are also reported (--show) and available to downstream code.
(Replacing each end-of-string group by a placeholder was tried and measured harmful,
-0.0142 F0.5: those groups mix legal forms with descriptive words like center/group.)

    python3 src/adaptive_config.py [--show]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from common import CACHE, log
from prep_basic import build

KNEE_POOL = 1000     # the knee is searched among this many most frequent tokens
MAX_STOP = 300       # hard cap on stop words per field
MIN_DF = 5           # a stop word must appear in at least this many sampled records
CTX_K = 100          # neighbour vocabulary: the K most frequent tokens
POS_WEIGHT = 2.0     # relative weight of "where in the string" vs "next to what"
MIN_SPLIT = 0.3      # never split groups closer than this cosine distance (noise, not roles)
ROLE_RATE = 0.5      # a group is 'trail'/'lead' when its members end/start multi-word strings this often
SAMPLE = 300_000     # records sampled per (country, field, side)


def fit_field(docs: list[list[str]]) -> dict:
    """docs: token lists (pooled S1 + S2 + S3 records of one country and field)."""
    n_docs = len(docs)
    tf, df = collections.Counter(), collections.Counter()
    for d in docs:
        tf.update(d)
        df.update(set(d))
    ranked = [(t, c) for t, c in tf.most_common(KNEE_POOL) if not t.isdigit()]
    stop = [t for t, _ in ranked[:_knee([c for _, c in ranked])] if df[t] >= MIN_DF][:MAX_STOP]
    mass = sum(tf[t] for t in stop) / (sum(v for t, v in tf.items() if not t.isdigit()) or 1)
    if not stop:
        return dict(stop={}, groups={}, n_docs=n_docs)

    ctx = ["<s>", "</s>"] + [t for t, _ in tf.most_common(CTX_K)]
    cidx = {t: i for i, t in enumerate(ctx)}
    sidx = {t: i for i, t in enumerate(stop)}
    width = len(ctx)
    left = np.zeros((len(stop), width), np.float32)
    right = np.zeros((len(stop), width), np.float32)
    pos = np.zeros((len(stop), 3), np.float32)        # start / middle / end of a multi-word string
    seen = np.zeros(len(stop), np.float32)
    seen_multi = np.zeros(len(stop), np.float32)
    for d in docs:
        n = len(d)
        for i, tok in enumerate(d):
            k = sidx.get(tok)
            if k is None:
                continue
            seen[k] += 1
            lt = d[i - 1] if i else "<s>"
            rt = d[i + 1] if i + 1 < n else "</s>"
            if lt in cidx:
                left[k, cidx[lt]] += 1
            if rt in cidx:
                right[k, cidx[rt]] += 1
            if n >= 2:                                   # a lone word is not at any position
                pos[k, 0 if i == 0 else 2 if i == n - 1 else 1] += 1
                seen_multi[k] += 1
    seen = np.maximum(seen, 1)
    seen_multi = np.maximum(seen_multi, 1)

    def unit(m):
        m = np.sqrt(m / seen[:, None])
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    X = np.hstack([unit(left), unit(right), POS_WEIGHT * pos / seen_multi[:, None]]) + 1e-6
    ids = _cluster(X)

    groups = {}
    for g in sorted(set(ids)):
        idx = [k for k in range(len(stop)) if ids[k] == g]
        idx.sort(key=lambda k: -df[stop[k]])
        w = [df[stop[k]] for k in idx]
        sr = float(np.average(pos[idx, 0] / seen_multi[idx], weights=w))
        er = float(np.average(pos[idx, 2] / seen_multi[idx], weights=w))
        role = "trail" if er >= ROLE_RATE else "lead" if sr >= ROLE_RATE else "mid"
        groups[str(g)] = dict(role=role, start_rate=round(sr, 3), end_rate=round(er, 3),
                              size=len(idx), members=[stop[k] for k in idx[:14]])

    return dict(stop={stop[k]: str(ids[k]) for k in range(len(stop))}, groups=groups,
                n_docs=n_docs, mass=round(mass, 3))


def _knee(freqs: list[int]) -> int:
    """Rank where the log-log rank-frequency curve bends away from its chord (Kneedle):
    everything above it is the 'head' -- the stop words."""
    n = len(freqs)
    if n < 3:
        return n
    x = np.log(np.arange(1, n + 1))
    y = np.log(np.asarray(freqs, dtype=np.float64))
    x, y = (x - x[0]) / (x[-1] - x[0] or 1), (y - y[-1]) / (y[0] - y[-1] or 1)
    return int(np.argmax(y - (1 - x))) + 1            # distance above the chord (1 - x)


def _cluster(X: np.ndarray) -> list[int]:
    """Average-linkage cosine clustering; the cut is the largest jump in merge heights
    (adaptive number of groups -- no k to set)."""
    n = len(X)
    if n < 4:
        return [1] * n
    Z = linkage(X, "average", metric="cosine")
    h = Z[:, 2]
    tail = h[len(h) // 2:]                         # ignore the noisy small merges
    if len(tail) < 2:
        return list(fcluster(Z, t=h.max() + 1, criterion="distance"))
    gap = int(np.argmax(np.diff(tail)))
    if tail[gap + 1] < MIN_SPLIT:                  # every merge is trivially small: one role
        return list(fcluster(Z, t=h.max() + 1, criterion="distance"))
    cut = (tail[gap] + tail[gap + 1]) / 2
    return list(fcluster(Z, t=cut, criterion="distance"))


def fit_all(splits=("train", "test"), sample=SAMPLE) -> dict:
    rng = np.random.default_rng(0)
    frames = {}
    for split in splits:
        for src in (1, 2, 3):
            try:
                frames[(split, src)] = build(split, src)
            except (FileNotFoundError, OSError):
                pass
    countries = sorted({c for f in frames.values() for c in f["country"].unique()})
    out = {}
    for country in countries:
        out[country] = {}
        for field in ("name_b", "addr_b"):
            docs = []
            for f in frames.values():
                col = f.loc[f["country"] == country, field].values
                if len(col) > sample:
                    col = col[rng.choice(len(col), sample, replace=False)]
                docs += [x.split() for x in col if x]
            out[country][field] = fit_field(docs)
            e = out[country][field]
            log(f"{country}/{field}: {len(docs):,} records -> {len(e['stop'])} stop words "
                f"({e.get('mass', 0):.0%} of tokens) in {len(e['groups'])} groups")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print the groups")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    args = ap.parse_args()
    cfg = fit_all(tuple(args.splits))
    path = os.path.join(CACHE, "norm_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False)
    log("wrote", path)
    if args.show:
        for country, fields in cfg.items():
            for field, e in fields.items():
                print(f"\n== {country} / {field} ==")
                for g, v in sorted(e["groups"].items(), key=lambda kv: -kv[1]["size"]):
                    print(f"  [{g}] {v['role']:<5} start={v['start_rate']:<5} end={v['end_rate']:<5} n={v['size']:<3} {' '.join(v['members'])}")


if __name__ == "__main__":
    main()
