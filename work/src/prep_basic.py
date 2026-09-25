"""Stage 1: cache basic-normalised name/address for every record."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import CACHE, load_source, basic_norm, log


def build(split: str, src: int) -> pd.DataFrame:
    pq = os.path.join(CACHE, f"{split}_s{src}_basic.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    df = load_source(split, src)
    log(f"normalising {split} s{src} ({len(df):,})")
    out = pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "country": df["country"].values,
        "name_b": [basic_norm(x) for x in df["business_name"].values],
        "addr_b": [basic_norm(x) for x in df["business_address"].values],
    })
    out.to_parquet(pq, index=False)
    return out


if __name__ == "__main__":
    for split in ("train", "test"):
        for src in (1, 2, 3):
            d = build(split, src)
            log(split, src, len(d), d["name_b"].iloc[0])
