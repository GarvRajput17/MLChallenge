"""Self-check for the name cleaning and the unlabelled-country lexicon induction."""
from common import basic_norm, has_non_latin, name_norm
from learn_lexicons import induce_unlabelled, junk_prefix
from normalize import FieldMaps

assert name_norm("Pie College S.A.S.") == "pie college sas"
assert name_norm("Espinoza & Davis L.L.C.") == "espinoza and davis llc"
assert name_norm("L'Architecture") == "larchitecture"
assert name_norm("Korbrixx D.B.A. Obsidian, LLC") == "obsidian llc"
assert name_norm("Pyravantagebelo a/k/a Lycée de Anciennes") == "lycee de anciennes"
assert name_norm("Solbrix Doing Business As Shivam Traders") == "shivam traders"
assert name_norm("A1lied Harb0r L0gistics 5ervices") == "allied harbor logistics services"
assert name_norm("24x7 3M") == "24x7 3m"                   # only the measured digits map
assert name_norm("DBA") == "dba"                        # never empties a name

clean = ["12 rue de la paix nantes", "4 route du port calais"] * 200 + ["9 place nord lille"] * 50
noisy = ["12 r de la paix nantes", "4 rte du port calais", "no 12 r de la paix nantes"] * 200
lx = induce_unlabelled(clean, noisy)
assert lx["variant"] == {"r": "rue", "rte": "route"}, lx["variant"]
assert "no" in lx["generic"] and "rue" not in lx["generic"], lx["generic"]
# junk prefixes: learned, stripped from the front only, never empties a name
import collections
lead = collections.Counter({"shri": 400, "sunrise": 300, "राम": 90})
clean = collections.Counter({"sunrise": 900, "shri": 3})
assert junk_prefix(lead, clean) == ["shri"]                # real words + non-Latin kept
fm = FieldMaps({"prefix": ["shri", "dr"]}, 0)
assert fm.apply("shri dr ganesh traders") == ["ganesh", "traders"]
assert fm.apply("ganesh shri traders") == ["ganesh", "shri", "traders"]
assert fm.apply("shri") == ["shri"]

# script-agnostic: fold Latin/Greek/Cyrillic accents, keep other scripts' marks intact
assert basic_norm("Œuvre Café, Straße") == "oeuvre cafe strasse"
assert basic_norm("Phở Hà Nội") == "pho ha noi"
assert basic_norm("Ελληνική") == "ελληνικη"
assert basic_norm("राम ट्रेडर्स") == "राम ट्रेडर्स"            # marks kept, not split
assert basic_norm("ร้านอาหาร") == "ร้านอาหาร"
assert has_non_latin("東京") and has_non_latin("петр") and not has_non_latin("pho ha noi")
# pandas 3.x backs parquet-read string columns with pyarrow, and np.char ufuncs
# (e.g. np.char.startswith) raise UFuncNoLoopError on that dtype -- caught in the
# real v5 blocking run; regression-guard it here without needing real embeddings.
import pandas as pd
_arrow_ids = pd.DataFrame({"entity_id": ["S1-1", "S2-2", "S3-3"]})["entity_id"].values
_mask = pd.Series(_arrow_ids).str.startswith("S2-").to_numpy()
assert list(_mask) == [False, True, False]
try:
    import numpy as np
    np.char.startswith(_arrow_ids.astype(str), "S2-")
    print("note: np.char.startswith no longer breaks on Arrow strings in this pandas version")
except Exception:
    pass  # expected on pandas 3.x -- the pd.Series.str path above is what run_blocking.py uses

# adaptive stop words: fitted from the records alone, groups by role, no language known
import random
from adaptive_config import fit_field
_rng = random.Random(0)
_words = [f"w{i}x" for i in range(400)]                # long tail of identity words
_legal = ["inc", "llc", "ltd", "corp"]
_docs = [[_rng.choice(_words), _rng.choice(_words)] + ([_rng.choice(_legal)] if _rng.random() < 0.8 else [])
         for _ in range(4000)]
_streets = ["street", "road", "avenue", "drive"]
_addr = [[str(_rng.randrange(1, 999)), _rng.choice(_words), _rng.choice(_streets), _rng.choice(["ohio", "texas"])]
         for _ in range(4000)]
_n = fit_field(_docs)
assert {t for t in _legal} <= set(_n["stop"]), _n["stop"]
assert len({_n["stop"][t] for t in _legal}) == 1, "legal forms should share one group"
assert all(_n["groups"][_n["stop"][t]]["role"] == "trail" for t in _legal)
_a = fit_field(_addr)
assert len({_a["stop"][t] for t in _streets}) == 1, "street types should share one group"
assert _a["stop"]["street"] != _a["stop"]["ohio"], "street types and place words are different roles"
assert not any(t.isdigit() for t in _a["stop"]), "house numbers are never stop words"

print("ok")
