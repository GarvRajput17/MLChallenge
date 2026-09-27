# Business Entity Resolution — pipeline

Resolves each Source 1 business record to its matching Source 2 / Source 3 records.

## Reproduce

```bash
pip install -r requirements.txt
./src/run_all.sh
```

Expects the challenge data at `../student_resource/dataset/{train,test}/`.
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Every stage caches to `cache/`, so a rerun resumes rather than recomputing.

## Stages

| # | Script | What it does |
|---|---|---|
| 1 | `prep_basic.py` | TSV → parquet; unicode NFKC, Latin accent folding (Indic marks preserved), case folding, punctuation → space, `&`/`+` → `and` |
| 2 | `learn_lexicons.py` | **Induces** transliteration, abbreviation and generic-token lexicons by aligning S1 records against their ground-truth matches. Per country, so `tn` → Tamil Nadu in India and Tennessee in the US. No external dictionary or API. |
| 3 | `normalize.py` | Applies the lexicons; emits canonical views (`name_c`, `name_core`, `name_sq`, `addr_c`, `addr_core`, `nums`, `postal`) |
| 4 | `run_blocking.py` | Candidate generation — four complementary TF-IDF retrieval views plus exact-key hash joins, unioned, exactly rescored, then hard-pruned per entity |
| 5 | `build_features.py` | ~50 pairwise features via sparse cosines and `rapidfuzz.cpdist`; no Python loop over pairs |
| 6 | `train.py` | LightGBM matcher (two stages: pairwise, then with cross-candidate context), isotonic calibration, decision-layer tuning |
| 7 | `predict.py` | Test inference → the two submission TSVs |

## Design notes

**Lexicons are learned, not written.** The names appear in nine Indic scripts
(Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam). A
Devanagari name and its Latin original share *zero* characters, so every string
similarity between them is exactly 0. `learn_lexicons.py` recovers the mapping by
aligning the 7.6M labelled pairs — e.g. all nine scripts' spellings of *Raj* converge
onto `raj`. Same-script links are guarded by a subsequence/acronym/Jaro–Winkler test,
because raw co-occurrence frequency alone collapses unrelated tokens together.

**Blocking always indexes the small side.** Retrieval runs S2/S3 → S1. S1 is smaller,
so the transposed index stays small; and since no S2/S3 record is ever shared between
two S1 entities, a modest top-k already captures nearly all achievable recall.
Complementary *views* (squashed-name char n-grams, address tokens, name tokens, address
core) replace a reverse pass: each ranks differently, so the union recovers pairs any
single view would crowd out.

**The candidate set is deliberately small.** A joint name+address channel retrieves on
both fields at once, so the address breaks ties between the many S1 entities that share
a name. After the union, both principal cosines are recomputed *exactly*, and a small
learned pruner (LightGBM on blocking-stage signals only: channel scores, and each pair's
rank/gap among its S1's and its S2/S3 record's competing candidates) keeps a candidate
only if `p >= ER_PRUNE_T` (default 0.01), at most `ER_M` per entity. The cut adapts per
entity instead of a fixed top-k. The pruner trains on non-validation entities only and
logs its held-out candidates-per-entity / pair-completeness curve.

**Decisions maximise expected F₀.₅ directly.** `F_β = (1+β²)·TP / (β²K + |S|)`, so given
calibrated probabilities the expected score of a candidate set is exactly computable.
`decide.py` evaluates every prefix of the probability-sorted list via forward/backward
Poisson-binomial convolutions and takes the maximiser. A fixed threshold cannot express
this: for probabilities `[0.60, 0.58, 0.55, 0.52]` a `p > 0.70` rule returns the empty
set (E[F] = 0.036) while the optimum takes all four (E[F] = 0.603). Singleton handling
falls out of the same computation instead of needing a separate rule.

**One-to-one arbitration.** No S2/S3 record is ever shared between two S1 entities in
the ground truth, so each candidate is assigned to its single best entity before set
selection, and the best-vs-second-best margin is fed back as a feature.

## Layout

```
src/        pipeline (run in the order above; run_all.sh chains them)
cache/      parquet intermediates and the fitted matcher
output/     matching_results.tsv, candidate_pairs.tsv
reports/    stage logs, including the blocking sweep
docs/       problem analysis, approach, AWS setup
```
