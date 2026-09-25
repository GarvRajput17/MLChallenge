"""Macro F_0.5 exactly as the brief specifies: computed per Source 1 entity, then
averaged over ALL entities including singletons."""
from __future__ import annotations

import numpy as np

B = 0.25
GAIN = 1.0 + B


def f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    return GAIN * tp / (B * len(truth) + len(pred))      # closed form


def macro_f05(pred_map: dict, truth_map: dict, entity_ids) -> float:
    total = 0.0
    for eid in entity_ids:
        total += f05(pred_map.get(eid, frozenset()), truth_map.get(eid, frozenset()))
    return total / max(len(entity_ids), 1)


def breakdown(pred_map: dict, truth_map: dict, entity_ids, groups: dict) -> dict:
    """Macro F_0.5 sliced by an arbitrary per-entity key (country, true-list size, ...)."""
    acc, cnt = {}, {}
    for eid in entity_ids:
        g = groups.get(eid, "?")
        acc[g] = acc.get(g, 0.0) + f05(pred_map.get(eid, frozenset()),
                                       truth_map.get(eid, frozenset()))
        cnt[g] = cnt.get(g, 0) + 1
    return {g: (acc[g] / cnt[g], cnt[g]) for g in sorted(acc, key=str)}


if __name__ == "__main__":
    # the worked example from the problem statement
    got = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(got - 0.714) < 1e-3, got
    assert f05(set(), set()) == 1.0 and f05({"x"}, set()) == 0.0
    assert f05(set(), {"x"}) == 0.0
    print("evaluate.py: brief's worked example reproduces ->", round(got, 3))
