"""Stage 2: induce token lexicons from the training data + ground truth only.

Three artefacts, all estimated by aligning S1 records against their ground-truth
matches.  No external dictionary, gazetteer or API is involved.

  translit  : non-Latin token  -> Latin token      (लिमिटेड -> limited)
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
  * Cross-script links are directed: non-Latin -> Latin, never the reverse.
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

from common import CACHE, has_non_latin, load_ground_truth, log
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
    if has_non_latin(key):
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
                    clean=collections.Counter(), generic=collections.Counter(),
                    lead=collections.Counter())

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
            st["lead"][b[0]] += 1
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
PREFIX_MIN = 25         # leading noisy token must be seen this often
PREFIX_RATIO = 0.1      # ... and occur in clean names at most this often relative to it


def junk_prefix(lead, clean):
    """Tokens the noise prepends that clean names essentially never contain (Shri, Smt,
    Dr, The). Measured on train pairs: strips them without raising similar-name false
    matches -- unlike stripping the whole generic list, which holds real words (safe,
    heritage, dental) and doubled them. Non-Latin tokens are transliteration gaps, not
    junk, so they are never stripped."""
    return sorted(t for t, v in lead.items()
                  if v >= PREFIX_MIN and clean.get(t, 0) < PREFIX_RATIO * v
                  and not has_non_latin(t))


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
        if has_non_latin(bt) != has_non_latin(at):
            if has_non_latin(at):                 # only non-Latin -> Latin
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
        latin = [m for m in members if not has_non_latin(m)]
        pool = latin or members
        rep = max(pool, key=lambda t: (clean[t], -len(t), t))
        for m in members:
            if m == rep:
                continue
            (translit if has_non_latin(m) else variant)[m] = rep

    # unigram -> phrase, non-transitive; skip keys already canonicalised elsewhere
    for k, (v, _) in phrase_best.items():
        if k not in translit and k not in variant:
            expand[k] = v

    generic = dict(st["generic"].most_common(GENERIC_TOP))
    return translit, variant, expand, generic, junk_prefix(st["lead"], clean)


# ------------------------------------------------ unlabelled countries ---
UNL_MIN_P = 2e-4        # token must be at least this frequent to be considered
UNL_OVER = 3.0          # noisy-side surplus for an abbreviation candidate
UNL_UNDER = 1.15        # S1-side surplus for an expansion candidate
UNL_CTX_COS = 0.9       # neighbour-context cosine: rejects 'no'->'nouvelle' (0.07)
UNL_GENERIC = 2.0       # frequency ratio (either direction) marking a token generic


def _token_stats(texts):
    tok, ctx = collections.Counter(), collections.defaultdict(collections.Counter)
    for x in texts:
        t = x.split()
        tok.update(t)
        for i, w in enumerate(t):
            ctx[w]["L" + (t[i - 1] if i else "^")] += 1
            ctx[w]["R" + (t[i + 1] if i + 1 < len(t) else "$")] += 1
    return tok, ctx


def _cos(a, b):
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = (sum(v * v for v in a.values()) * sum(v * v for v in b.values())) ** 0.5
    return num / den if den else 0.0


def induce_unlabelled(clean_texts, noisy_texts):
    """Lexicon for a country with no ground truth (France), from S1 vs S2/S3 token
    statistics alone.  An abbreviation is over-represented on the noisy side, is a
    subsequence of an S1-over-represented word with the same first letter, and occurs
    in the same neighbour context (r->rue, rte->route, crs->cours).  Tokens whose
    frequency differs strongly between the sides are what the noise adds or drops
    (departments swapped for regions, SARL/SAS, filler words) -> generic."""
    t1, c1 = _token_stats(clean_texts)
    t2, c2 = _token_stats(noisy_texts)
    n1, n2 = sum(t1.values()) or 1, sum(t2.values()) or 1
    p1 = lambda t: t1.get(t, 0) / n1
    p2 = lambda t: t2.get(t, 0) / n2
    over = [t for t in t2 if p2(t) > UNL_MIN_P and p2(t) > UNL_OVER * p1(t) and not t.isdigit()]
    under = [t for t in t1 if p1(t) > UNL_MIN_P and p1(t) > UNL_UNDER * p2(t) and not t.isdigit()]
    variant = {}
    for s in over:
        scored = [(_cos(c2[s], c1[l]), l) for l in under
                  if l[0] == s[0] and len(l) > len(s) and is_subsequence(s, l)]
        if scored:
            cos, best = max(scored)
            if cos >= UNL_CTX_COS:
                variant[s] = best
    mapped = set(variant) | set(variant.values())
    gen = {t: abs(p1(t) - p2(t)) for t in set(t1) | set(t2)
           if t not in mapped and not t.isdigit() and max(p1(t), p2(t)) > UNL_MIN_P
           and max(p1(t), p2(t)) > UNL_GENERIC * min(p1(t), p2(t))}
    generic = dict(sorted(gen.items(), key=lambda kv: -kv[1])[:GENERIC_TOP])
    lead = collections.Counter(x.split()[0] for x in noisy_texts if x.strip())
    # rescale clean counts to the noisy side's record count: S2+S3 hold ~2.7x as many
    # records as S1, so raw counts made the cut ~2.7x too loose (it kept 'le', 'cc')
    scale = len(noisy_texts) / max(len(clean_texts), 1)
    s1_count = collections.Counter(t for x in clean_texts for t in set(x.split()))
    prefix = junk_prefix(lead, {t: c * scale for t, c in s1_count.items()})
    return dict(translit={}, variant=variant, expand={}, generic=generic, prefix=prefix)


def add_unlabelled(out, split="test"):
    t1, t2, t3 = build(split, 1), build(split, 2), build(split, 3)
    for country in sorted(set(t1["country"]) - set(out)):
        side = lambda d, f: d.loc[d["country"] == country, f].values
        out[country] = {f: induce_unlabelled(side(t1, f),
                                             list(side(t2, f)) + list(side(t3, f)))
                        for f in ("name_b", "addr_b")}
        out[country]["addr_b"]["prefix"] = []           # validated on names only
        for f, d in out[country].items():
            log(f"{country}/{f} (unlabelled): variant={d['variant']} "
                f"generic[:25]={list(d['generic'])[:25]} prefix={d['prefix']}")


def main():
    s1, s2, s3, i1, other, is2 = gt_pairs()
    log(f"{len(i1):,} positive pairs")
    stats = collect(s1, s2, s3, i1, other, is2)

    out = {}
    for (country, field), st in sorted(stats.items()):
        translit, variant, expand, generic, prefix = distil(st)
        out.setdefault(country, {})[field] = dict(
            translit=translit, variant=variant, expand=expand, generic=generic,
            prefix=prefix if field == "name_b" else [])
        log(f"{country}/{field}: translit={len(translit):,} variant={len(variant):,} "
            f"expand={len(expand):,} generic={len(generic):,}")

    # ER_HOLDOUT_COUNTRY simulates an unseen country: discard its label-derived
    # lexicon and induce it label-free from its train records, exactly as France is.
    holdout = os.environ.get("ER_HOLDOUT_COUNTRY")
    if holdout:
        out.pop(holdout, None)
        add_unlabelled(out, "train")
    add_unlabelled(out)

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
