#!/usr/bin/env bash
# Fast path to a submittable matching_results.tsv.
#
# Trades training rigour for wall-clock: the matcher trains on a SUBSAMPLE of S1
# entities (a GBDT does not need 31M pairs to fit ~60 features) and fewer boosting
# rounds. The TEST side is always complete -- every test entity must be scored, and
# that is the irreducible cost.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p reports output

TRAIN_S1=${TRAIN_S1:-150000}      # S1 entities per country used for training
ROUNDS=${ROUNDS:-300}
export ER_THREADS=${ER_THREADS:-24}

step() { echo; echo "######## $* ########"; date "+%H:%M:%S"; }

echo "threads: $ER_THREADS   train subsample: $TRAIN_S1/country   rounds: $ROUNDS"

step "1/6 blocking: train (subsample ${TRAIN_S1}/country)"
python3 -u src/run_blocking.py --split train --limit-s1 "$TRAIN_S1" --sweep

step "2/6 features: train"
python3 -u src/build_features.py --split train

step "3/6 train matcher (${ROUNDS} rounds) + tune decision layer"
python3 -u src/train.py --rounds "$ROUNDS"

step "4/6 blocking: test (ALL entities, all countries)"
python3 -u src/run_blocking.py --split test

step "5/6 features: test"
python3 -u src/build_features.py --split test

step "6/6 inference -> submission files"
python3 -u src/predict.py --split test

step "validating submission format"
python3 ../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test

echo; echo "DONE"; date "+%H:%M:%S"
wc -l output/*.tsv
