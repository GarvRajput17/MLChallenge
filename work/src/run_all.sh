#!/usr/bin/env bash
# End-to-end reproduction: raw TSV -> output/matching_results.tsv + candidate_pairs.tsv
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/7 parquet cache + basic normalisation =="
python3 -u src/prep_basic.py

echo "== 2/7 lexicon induction (transliteration, abbreviations, generic tokens) =="
python3 -u src/learn_lexicons.py

echo "== 3/7 canonicalisation =="
python3 -u src/normalize.py

echo "== 4/7 candidate generation =="
python3 -u src/run_blocking.py --split train --sweep
python3 -u src/run_blocking.py --split test

echo "== 5/7 feature extraction =="
python3 -u src/build_features.py --split train
python3 -u src/build_features.py --split test

echo "== 6/7 train matcher + tune decision layer =="
python3 -u src/train.py

echo "== 7/7 inference + submission files =="
python3 -u src/predict.py --split test

echo "== validating =="
python3 ../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test
