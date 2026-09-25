"""Stage 2: induce token lexicons from the training data + ground truth only.

Three artefacts, all estimated by aligning S1 records against their ground-truth
matches.  No external dictionary, gazetteer or API is involved.

  translit  : Indic token      -> Latin token      (लिमिटेड -> limited)
  variant   : misspelt/abbrev  -> canonical form   (akon -> akron, ave -> avenue)
  expand    : abbreviation     -> multi-word form  (tn -> tamil nadu)
  generic   : tokens the noise process freely adds/drops (inc, llc, com, null, ...)

Everything is learned *per country*, so `tn` resolves to Tamil Nadu in India and to
Tennessee in the US.

Design notes (a naive version of this over-merges badly):
  * Same-script links go through union-find, but only when the two tokens are
    genuinely string-related (typo / prefix / subsequence abbreviation).  Raw
    co-occurrence frequency is NOT sufficient evidence -- it collapses the whole
    legal-suffix family (inc/llc/group/enterprises) into one token.
  * Cross-script links are directed: Indic -> Latin, never the reverse.
  * Unigram -> phrase links (tn -> tamil nadu) are kept in a separate, NON-transitive
    map and require the key to be an acronym or prefix of the value.  Feeding phrases
    into union-find is what produces nonsense like `chennai -> nadu`.
  * Legal suffixes are deliberately NOT canonicalised into each other; they are
    exported as `generic` so downstream features can down-weight them instead.
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from common import CACHE, has_indic, load_ground_truth, log
from prep_basic import build

MIN_COUNT = 25          # a correspondence must be observed this often
MIN_SHARE = 0.10        # ... and explain this share of the noisy token's alignments
JW_MIN = 0.87           # string-similarity floor for a same-script link
GENERIC_TOP = 600       # size of the exported generic-token inventory


# ------------------------------------------------------------------ helpers ---
def is_subsequence(short: str, long: str) -> bool:
    """`ltd` inside `limited`, `blvd` inside `boulevard` -- the abbreviation test."""
    if len(short) >= len(long):
        return False
    it = iter(long)
    return all(c in it for c in short)


def acronym(phrase: str) -> str:
    return "".join(w[0] for w in phrase.split() if w)


def same_script_link(a: str, b: str) -> bool:
    """Is (a, b) a plausible spelling variant rather than two different words?"""
    if is_subsequence(a, b) or is_subsequence(b, a):
        return True
    return JaroWinkler.similarity(a, b) >= JW_MIN


def phrase_link(key: str, phrase: str) -> bool:
    """Is `key` a plausible abbreviation of the multi-word `phrase`?"""
    if has_indic(key):
        return True
    return key == acronym(phrase) or phrase.startswith(key)


class UnionFind:
    __slots__ = ("parent",)

    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:            # path compression
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ------------------------------------------------------------- pair mining ---
def gt_pairs():
    """Row-index arrays linking each S1 record to each of its ground-truth matches."""
    gt = load_ground_truth()
    s1, s2, s3 = build("train", 1), build("train", 2), build("train", 3)
    left, right = [], []
    for sid, ids in zip(gt["source1_entity_id"].values,
                        gt["matched_entity_ids"].str.split(",")):
        for i in ids:
            if i:
                left.append(sid)
                right.append(i)
    left, right = np.asarray(left), np.asarray(right)
    is2 = np.char.startswith(right.astype(str), "S2-")
    i1 = pd.Index(s1["entity_id"]).get_indexer(left)
    other = np.where(is2,
                     pd.Index(s2["entity_id"]).get_indexer(right),
                     pd.Index(s3["entity_id"]).get_indexer(right))
    assert (i1 >= 0).all() and (other >= 0).all(), "unresolved entity_id in ground truth"
    return s1, s2, s3, i1, other, is2


def _take(counter, tok) -> bool:
    if counter[tok]:
        counter[tok] -= 1
        return True
    return False


def align(clean_toks, noisy_toks):
    """Candidate (noisy, clean) correspondences after cancelling exact matches."""
    common = collections.Counter(clean_toks) & collections.Counter(noisy_toks)
    rem = collections.Counter(common)
    ra = [t for t in clean_toks if not _take(rem, t)]
    rem = collections.Counter(common)
    rb = [t for t in noisy_toks if not _take(rem, t)]
    if not ra or not rb:
        return (), ra, rb
    if len(ra) == len(rb):
        return tuple(zip(rb, ra)), (), ()
    if len(rb) == 1 and len(ra) == 2:                 # tn -> tamil nadu
        return ((rb[0], " ".join(ra)),), (), ()
    return (), ra, rb


def collect(s1, s2, s3, i1, other, is2):
    """Accumulate alignment statistics keyed by (country, field)."""
    country = s1["country"].values
    fields = ("name_b", "addr_b")
    col1 = {f: s1[f].values for f in fields}
    col2 = {f: s2[f].values for f in fields}
    col3 = {f: s3[f].values for f in fields}

    def fresh():
        return dict(pair=collections.Counter(), noisy=collections.Counter(),
                    clean=collections.Counter(), generic=collections.Counter())

    stats = collections.defaultdict(fresh)
    n = len(i1)
    for k in range(n):
        ia, ib = i1[k], other[k]
        c = country[ia]
        src = col2 if is2[k] else col3
        for f in fields:
            a = col1[f][ia].split()
            b = src[f][ib].split()
            if not a or not b:
                continue
            st = stats[(c, f)]
            clean_c = st["clean"]
            for t in a:
                clean_c[t] += 1
            corr, ra, rb = align(a, b)
            pair_c, noisy_c = st["pair"], st["noisy"]
            for bt, at in corr:
                pair_c[(bt, at)] += 1
                noisy_c[bt] += 1
            gen = st["generic"]
            for t in rb:                      # present only in the noisy record
                gen[t] += 1
            for t in ra:                      # present only in the clean record
                gen[t] += 1
        if k and k % 2_000_000 == 0:
            log(f"  aligned {k:,}/{n:,}")
    return stats


# -------------------------------------------------------------- distilling ---
def distil(st):
    """Turn raw alignment counts into the four lexicons."""
    pair, noisy, clean = st["pair"], st["noisy"], st["clean"]

    uf = UnionFind()
    translit_src, expand, phrase_best = set(), {}, {}

    for (bt, at), c in pair.items():
        if c < MIN_COUNT or bt == at or c / noisy[bt] < MIN_SHARE:
            continue
        multiword = " " in at
        if multiword:
            if not phrase_link(bt, at):
                continue
            if c > phrase_best.get(bt, (None, 0))[1]:
                phrase_best[bt] = (at, c)
            continue
        if has_indic(bt) != has_indic(at):
            if has_indic(at):                 # only Indic -> Latin
                continue
            uf.union(bt, at)
            translit_src.add(bt)
        elif same_script_link(bt, at):
            uf.union(bt, at)

    # representative of each component: prefer Latin, then the form S1 uses most
    groups = collections.defaultdict(list)
    for tok in list(uf.parent):
        groups[uf.find(tok)].append(tok)

    translit, variant = {}, {}
    for members in groups.values():
        latin = [m for m in members if not has_indic(m)]
        pool = latin or members
        rep = max(pool, key=lambda t: (clean[t], -len(t), t))
        for m in members:
            if m == rep:
                continue
            (translit if has_indic(m) else variant)[m] = rep

    # unigram -> phrase, non-transitive; skip keys already canonicalised elsewhere
    for k, (v, _) in phrase_best.items():
        if k not in translit and k not in variant:
            expand[k] = v

    generic = dict(st["generic"].most_common(GENERIC_TOP))
    return translit, variant, expand, generic


def main():
    s1, s2, s3, i1, other, is2 = gt_pairs()
    log(f"{len(i1):,} positive pairs")
    stats = collect(s1, s2, s3, i1, other, is2)

    out = {}
    for (country, field), st in sorted(stats.items()):
        translit, variant, expand, generic = distil(st)
        out.setdefault(country, {})[field] = dict(
            translit=translit, variant=variant, expand=expand, generic=generic)
        log(f"{country}/{field}: translit={len(translit):,} variant={len(variant):,} "
            f"expand={len(expand):,} generic={len(generic):,}")

    path = os.path.join(CACHE, "lexicons.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    log("wrote", path)

    for country in out:
        for field in out[country]:
            d = out[country][field]
            log(f"--- {country}/{field} ---")
            log("   translit:", list(d["translit"].items())[:8])
            log("   variant :", list(d["variant"].items())[:10])
            log("   expand  :", list(d["expand"].items())[:10])
            log("   generic :", list(d["generic"])[:18])


if __name__ == "__main__":
    main()
