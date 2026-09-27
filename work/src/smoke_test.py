"""End-to-end smoke test on a small synthetic fixture.

Generates a miniature dataset with the same noise characteristics as the real one
(transliteration, typos, suffix swaps, address reordering, singletons, distractors),
then runs every stage. Catches library-version breakage and wiring bugs in ~2 minutes
instead of after an hour of real compute.
"""
from __future__ import annotations
import os, random, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEVA = "राम श्री कुमार शक्ति विजय आदित्य".split()
LAT = "ram shri kumar shakti vijay aditya".split()
XLIT = dict(zip(DEVA, LAT))
STREETS = ["oak street", "main road", "park avenue", "hill drive"]
CITIES = [("chennai", "tamil nadu", "tn"), ("jaipur", "rajasthan", "rj"),
          ("peoria", "illinois", "il"), ("akron", "ohio", "oh")]
SUFFIX = ["private limited", "llc", "inc", "ltd", "llp"]


def typo(s, rng):
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def gen(path, n_entities=1500, seed=0, countries=("India", "US")):
    rng = random.Random(seed)
    os.makedirs(path, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    for e in range(n_entities):
        india = rng.random() < 0.5
        core = f"{rng.choice(LAT)} {rng.choice(['traders','motors','foods','systems'])}"
        suf = rng.choice(SUFFIX)
        name = f"{core} {suf}"
        city, state, abbr = rng.choice(CITIES[:2] if india else CITIES[2:])
        num = rng.randrange(10, 9999)
        addr = f"{num} {rng.choice(STREETS)}, {city}, {state}"
        # test adds labels never seen in training: the pipeline must treat country as open
        country = ("India" if india else "US") if rng.random() < 0.6 else rng.choice(countries)
        sid = f"S1-{e:06d}"
        s1.append((sid, name.title(), addr.title(), country))

        matches = []
        if rng.random() > 0.08:                      # ~8% singletons, as in the real data
            for k in range(rng.randrange(1, 4)):
                nm = name
                if india and rng.random() < 0.5:     # transliterate the head token
                    head = core.split()[0]
                    back = {v: k2 for k2, v in XLIT.items()}.get(head)
                    if back:
                        nm = nm.replace(head, back, 1)
                if rng.random() < 0.4:
                    nm = typo(nm, rng)
                if rng.random() < 0.3:               # drop the legal suffix
                    nm = nm.replace(" " + suf, "")
                ad = f"{num} {rng.choice(STREETS)}, {city}, {abbr}"
                if rng.random() < 0.25:
                    ad = ""                          # address destroyed
                elif rng.random() < 0.3:
                    ad = f"{city}, {abbr}, {num}"     # component reordering
                mid = f"S2-{e:06d}{k}" if k % 2 == 0 else f"S3-{e:06d}{k}"
                (s2 if mid.startswith("S2") else s3).append(
                    (mid, nm.title(), ad.title(), country))
                matches.append(mid)
        gt.append((sid, ",".join(matches)))

    for i in range(n_entities // 3):                 # distractors matching nothing
        country = rng.choice(countries)
        city, state, _ = rng.choice(CITIES)
        row = (f"S2-D{i:06d}", f"Zeta {i} Holdings", f"{i} Far Lane, {city}, {state}", country)
        s2.append(row)
        s3.append((f"S3-D{i:06d}",) + row[1:])

    def dump(fn, rows, header):
        with open(os.path.join(path, fn), "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")

    s1.append(("S1-LONE", "Solo Traders", "1 Nowhere Road", "Atlantis"))   # no S2/S3 at all
    gt.append(("S1-LONE", ""))
    hdr = ["entity_id", "business_name", "business_address", "country"]
    dump("train_source1.tsv", s1, hdr); dump("train_source2.tsv", s2, hdr)
    dump("train_source3.tsv", s3, hdr)
    dump("train_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])
    return len(s1), len(s2), len(s3)


def main():
    tmp = tempfile.mkdtemp(prefix="er_smoke_")
    data = os.path.join(tmp, "dataset")
    for split in ("train", "test"):
        n = gen(os.path.join(data, split), seed=0 if split == "train" else 1,
                countries=("India", "US") if split == "train" else ("France", "Japan"))
        if split == "test":   # test dir needs test_* names and no ground truth
            d = os.path.join(data, "test")
            for f in os.listdir(d):
                if f.startswith("train_ground_truth"):
                    os.remove(os.path.join(d, f))
                elif f.startswith("train_"):
                    os.rename(os.path.join(d, f),
                              os.path.join(d, f.replace("train_", "test_", 1)))
        print(f"  {split}: {n}")
        n_test_s1 = n[0]
    env = dict(os.environ, ER_DATA=data, ER_CACHE=os.path.join(tmp, "cache"),
               ER_OUTPUT=os.path.join(tmp, "output"), ER_SELF_MIN_PAIRS="50")
    steps = [
        ["prep_basic.py"], ["adaptive_config.py"], ["learn_lexicons.py"], ["normalize.py"],
        ["run_blocking.py", "--split", "train"],
        ["run_blocking.py", "--split", "test"],
        ["build_features.py", "--split", "train"],
        ["build_features.py", "--split", "test"],
        ["train.py", "--rounds", "80"],
        ["predict.py", "--split", "test"],
        ["self_train.py", "--split", "test"],
        ["predict.py", "--split", "test"],
    ]
    for step in steps:
        print(f"\n===== {' '.join(step)} =====", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, step[0])] + step[1:],
                           env=env, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()[-14:]
        print("\n".join(tail))
        if r.returncode != 0:
            print("--- STDERR ---")
            print((r.stderr or "").strip()[-3000:])
            print(f"\nSMOKE TEST FAILED at {step[0]}")
            sys.exit(1)
    out = env["ER_OUTPUT"]
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = os.path.join(out, f)
        n_rows = sum(1 for _ in open(p)) - 1
        print(f"\n{f}: {n_rows} rows")
        assert n_rows == n_test_s1, f"{f}: {n_rows} rows for {n_test_s1} S1 entities"
        print("".join(open(p).readlines()[:4]))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
