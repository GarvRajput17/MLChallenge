# Business Entity Resolution — pipeline

Resolves each Source 1 business record to its matching Source 2 / Source 3 records.

## Reproduce

Expects the challenge data at `../student_resource/dataset/{train,test}/`. Every stage caches
to `cache/`, so a rerun resumes rather than recomputing.

**1. Dense retriever (GPU, ~1 h on one L4/L40S).** Fine-tunes `multilingual-e5-small` and writes
embeddings plus nearest-neighbour candidate lists; blocking reads them from `cache/dense/`
(it runs without them, minus that channel).

```bash
pip install -r requirements-gpu.txt
SM_CHANNEL_RECORDS=<dir with {train,test}_source{1,2,3}.parquet + train_ground_truth.parquet> \
  python src/dense_encoder.py --s3-out cache/dense      # a local path works as well as s3://
```

**2. CPU pipeline (~5 h on 64 cores / 512 GB; peak RAM ~125 GB).**

```bash
pip install -r requirements.txt
ER_THREADS=64 ./src/run_all.sh
```

Writes `output/matching_results.tsv` (matcher only) and `output/candidate_pairs.tsv` (final).

**3. Cross-encoder + stacking (GPU, then CPU).**

```bash
python src/build_ce_pairs.py --split train --train-entities 0 --eval-entities 0 --out ce_data
python src/build_ce_pairs.py --split test --out ce_data
mkdir -p ce_data/train ce_data/eval ce_data/score        # one directory per channel
mv ce_data/train.parquet ce_data/train/; mv ce_data/eval.parquet ce_data/eval/; mv ce_data/test_pairs.parquet ce_data/score/
# fine-tune (optional; skip to reuse a model) and score -- cross_encoder.py reads SageMaker-style
# channels from env vars: SM_CHANNEL_TRAIN / _EVAL / _SCORE (dirs of parquet), SM_OUTPUT_DATA_DIR
SM_CHANNEL_TRAIN=ce_data/train SM_CHANNEL_EVAL=ce_data/eval SM_CHANNEL_SCORE=ce_data/score \
  SM_OUTPUT_DATA_DIR=ce_scores SM_MODEL_DIR=ce_model python src/cross_encoder.py --epochs 1 --bs 256
python src/stack.py --ce-dir ce_scores                      # prints the validation gain
python src/stack.py --ce-dir ce_scores --predict --outdir output
```

`--ce-dir` takes several score directories to ensemble cross-encoders. `stack.py --export-kit`
/ `--kit` lets step 3 run on a machine without the CPU caches (see `HANDOFF.md`).
`scripts/sm_cross_encoder.sh` and `scripts/sm_dense.sh` launch steps 1 and 3 as SageMaker jobs.

Check: `python src/test_cleaning.py` (self-checks) and `python src/smoke_test.py` (end-to-end on
a synthetic fixture with unseen countries).

## Stages

| # | Script | What it does |
|---|---|---|
| 1 | `prep_basic.py` | TSV → parquet; NFKC, accent folding (Latin/Greek/Cyrillic only), acronym joining, digit-typo repair, alias-prefix stripping |
| 1b | `adaptive_config.py` | Label-free per-country stop words and their roles (start / middle / end of string), fitted from the records |
| 2 | `learn_lexicons.py` | **Induces** transliteration, abbreviation, generic-token and junk-prefix lexicons from the labelled pairs, per country; label-free induction for countries without labels |
| 3 | `normalize.py` | Applies the lexicons; canonical views (`name_c`, `name_core`, `name_sq`, `addr_c`, `addr_core`, `nums`) |
| 4 | `run_blocking.py` | TF-IDF views + joint name/address view + dense channel + S2↔S3 cross-links + exact keys → union → two-stage learned pruner (~6 candidates per entity) |
| 5 | `build_features.py` | ~70 pair features (sparse cosines, rapidfuzz scorers, house-number distance, name multiplicity) |
| 6 | `train.py` | Two-stage LightGBM (second stage sees competitors) + XGBoost blend, isotonic calibration, decision-layer tuning |
| 7 | `predict.py` | Test inference → the two submission TSVs |
| 8 | `self_train.py` | Countries without labels: pseudo-labels from confident matches → re-learned lexicon → re-cleaned, re-blocked, re-scored |
| 9 | `cross_encoder.py`, `stack.py` | Fine-tuned `xlm-roberta-base` pair scores, stacked with the matcher, then expected-F0.5 set selection |

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
