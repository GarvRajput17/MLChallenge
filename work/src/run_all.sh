#!/usr/bin/env bash
# End-to-end reproduction: raw TSV -> output/matching_results.tsv + candidate_pairs.tsv
set -euo pipefail
cd "$(dirname "$0")/.."

# Stages 1 and 3 are per-file single-threaded Python: run the 6 (split, source) files
# as parallel processes. `wait $pid` propagates each job's failure under set -e.
per_file() {   # per_file <module> <function>
    local pids=()
    for split in train test; do for src in 1 2 3; do
        python3 -u -c "import sys; sys.path.insert(0, 'src'); from $1 import $2; $2('$split', $src)" &
        pids+=($!)
    done; done
    for p in "${pids[@]}"; do wait "$p"; done
}

echo "== 1/7 parquet cache + basic normalisation =="
per_file prep_basic build

echo "== 1b/7 adaptive stop-word configuration (fitted on the records at hand, no labels) =="
python3 -u src/adaptive_config.py

echo "== 2/7 lexicon induction (transliteration, abbreviations, generic tokens) =="
python3 -u src/learn_lexicons.py

echo "== 3/7 canonicalisation =="
per_file normalize build_canon

echo "== 4/7 candidate generation =="
python3 -u src/run_blocking.py --split train
python3 -u src/run_blocking.py --split test

echo "== 5/7 feature extraction =="
python3 -u src/build_features.py --split train
python3 -u src/build_features.py --split test

echo "== 6/7 train matcher + tune decision layer =="
python3 -u src/train.py

echo "== 7/7 inference + submission files =="
python3 -u src/predict.py --split test

echo "== 8/8 self-train cleaning for countries without labels, then re-score them =="
python3 -u src/self_train.py --split test
python3 -u src/predict.py --split test

echo "== validating =="
python3 ../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test
