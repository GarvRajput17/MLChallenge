"""Stage 3: apply the induced lexicons to produce canonical, comparable fields.

For every record we cache:
    name_c     canonical name tokens        (transliterated, de-abbreviated)
    name_core  name_c minus generic/legal tokens
    name_sq    name_c with all spaces removed  -- kills tokenisation differences and
               rescues the concatenated domain form (maurewilliamscolombier.com)
    addr_c     canonical address tokens
    addr_core  addr_c minus generic tokens and bare numbers
    nums       distinct digit runs in the address (house numbers, PIN/ZIP)
    postal     digit runs of length 5 or 6 (US ZIP / India PIN)

Countries absent from the lexicons (France) simply fall through with basic
normalisation only -- nothing hard-codes the training country set.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import CACHE, log
from prep_basic import build

NULLS = frozenset(("null", "nan", "none", "na", "nil", "n", "a", "unknown", "nu11"))
GENERIC_NAME_TOP = 120
GENERIC_ADDR_TOP = 120
_DIGITS = re.compile(r"\d+")


class FieldMaps:
    """Canonicalisation tables for one (country, field)."""
    __slots__ = ("sub", "expand", "generic")

    def __init__(self, lex: dict | None, generic_top: int):
        lex = lex or {}
        self.sub = {**lex.get("variant", {}), **lex.get("translit", {})}
        self.expand = {k: v.split() for k, v in lex.get("expand", {}).items()}
        self.generic = frozenset(list(lex.get("generic", {}))[:generic_top])

    def apply(self, text: str) -> list[str]:
        out = []
        sub, expand = self.sub, self.expand
        for tok in text.split():
            if tok in NULLS:
                continue
            tok = sub.get(tok, tok)
            exp = expand.get(tok)
            if exp:
                out.extend(exp)
            else:
                out.append(tok)
        return out


def load_maps() -> dict:
    with open(os.path.join(CACHE, "lexicons.json"), encoding="utf-8") as fh:
        lex = json.load(fh)
    maps = {}
    for country, fields in lex.items():
        maps[country] = (FieldMaps(fields.get("name_b"), GENERIC_NAME_TOP),
                         FieldMaps(fields.get("addr_b"), GENERIC_ADDR_TOP))
    maps[None] = (FieldMaps(None, 0), FieldMaps(None, 0))     # unseen countries
    return maps


def build_canon(split: str, src: int) -> pd.DataFrame:
    pq = os.path.join(CACHE, f"{split}_s{src}_canon.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)

    base = build(split, src)
    maps = load_maps()
    fallback = maps[None]
    log(f"canonicalising {split} s{src} ({len(base):,})")

    countries = base["country"].values
    names = base["name_b"].values
    addrs = base["addr_b"].values

    name_c = np.empty(len(base), dtype=object)
    name_core = np.empty(len(base), dtype=object)
    name_sq = np.empty(len(base), dtype=object)
    addr_c = np.empty(len(base), dtype=object)
    addr_core = np.empty(len(base), dtype=object)
    nums = np.empty(len(base), dtype=object)
    postal = np.empty(len(base), dtype=object)

    # group row indices by country so the map lookup happens once per block
    order = pd.Series(countries).groupby(countries).indices
    for country, idx in order.items():
        nm, am = maps.get(country, fallback)
        n_generic, a_generic = nm.generic, am.generic
        n_apply, a_apply = nm.apply, am.apply
        for i in idx:
            toks = n_apply(names[i])
            name_c[i] = " ".join(toks)
            name_core[i] = " ".join([t for t in toks if t not in n_generic]) or name_c[i]
            name_sq[i] = "".join(toks)

            raw_addr = addrs[i]
            toks = a_apply(raw_addr)
            addr_c[i] = " ".join(toks)
            addr_core[i] = " ".join([t for t in toks
                                     if t not in a_generic and not t.isdigit()])
            digits = _DIGITS.findall(raw_addr)
            seen, uniq = set(), []
            for d in digits:
                d = d.lstrip("0") or "0"
                if d not in seen:
                    seen.add(d)
                    uniq.append(d)
            nums[i] = " ".join(uniq)
            postal[i] = " ".join(d for d in uniq if len(d) in (5, 6))
        log(f"   {split} s{src} {country}: {len(idx):,} done")

    out = pd.DataFrame({
        "entity_id": base["entity_id"].values,
        "country": countries,
        "name_c": name_c, "name_core": name_core, "name_sq": name_sq,
        "addr_c": addr_c, "addr_core": addr_core,
        "nums": nums, "postal": postal,
    })
    out.to_parquet(pq, index=False)
    return out


if __name__ == "__main__":
    for split in ("train", "test"):
        for src in (1, 2, 3):
            d = build_canon(split, src)
            log(split, src, len(d))
            print(d.head(3).to_string(max_colwidth=46))
