"""Stage 6: pairwise matcher -- LightGBM over the engineered features, plus the
probability calibration the decision layer depends on.

Calibration is not optional here. The expected-F_0.5 optimiser in decide.py consumes
p(match) as a genuine probability, not just a ranking score, and raw GBDT outputs are
not calibrated. An isotonic fit on a held-out entity split fixes that; it is the step
that turns a good ranker into a good decision maker.

Splitting is by S1 *entity*, never by pair: pairs sharing an S1 entity are not
independent and a pair-level split would leak.
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from common import THREADS, log

PARAMS = dict(
    objective="binary",
    metric=["binary_logloss", "auc"],
    learning_rate=0.06,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=THREADS,
    verbosity=-1,
)


def entity_fold(entity_ids: np.ndarray, n_folds: int = 10) -> np.ndarray:
    """Deterministic hash of the S1 entity id -> fold. Stable across runs and machines."""
    return np.fromiter(
        (int(hashlib.blake2b(e.encode(), digest_size=4).hexdigest(), 16) % n_folds
         for e in entity_ids), dtype=np.int16, count=len(entity_ids))


def train(X: np.ndarray, y: np.ndarray, folds: np.ndarray, feature_names,
          val_folds=(0,), num_boost_round=1200, early_stopping=60):
    val_mask = np.isin(folds, val_folds)
    tr, va = ~val_mask, val_mask
    log(f"train pairs={tr.sum():,} (pos {y[tr].mean():.3%})  "
        f"val pairs={va.sum():,} (pos {y[va].mean():.3%})")
    dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=list(feature_names))
    dva = lgb.Dataset(X[va], label=y[va], feature_name=list(feature_names),
                      reference=dtr)
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                   lgb.log_evaluation(100)])
    log(f"best iteration {booster.best_iteration}, "
        f"val auc {booster.best_score['val']['auc']:.5f}")
    return booster, va


def fit_calibrator(raw_scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw_scores, y)
    return iso


def calibration_report(p: np.ndarray, y: np.ndarray, bins=10) -> str:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = ["   bin   mean_p   actual    n"]
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(f"   {b:>3}  {p[m].mean():7.4f}  {y[m].mean():7.4f}  {m.sum():>8,}")
    return "\n".join(rows)


def importance(booster, feature_names, top=25) -> str:
    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)[:top]
    total = gain.sum() or 1.0
    return "\n".join(f"   {feature_names[i]:<18} {gain[i]/total:6.2%}" for i in order)
