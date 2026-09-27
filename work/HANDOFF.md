# Handoff: Business Entity Resolution — continuing on the GPU server

Written 27 Sep 2026. Read this top to bottom before running anything. It is written so that a
Claude Code session (or a person) can pick up the work on the GPU server without the AWS boxes.

---

## 1. Where we are

| Version | What changed | Candidates / S1 | Validation F0.5 (US+India) | Leaderboard F0.5 |
|---|---|---|---|---|
| v1 | Original pipeline | 12 | 0.974 | 0.968 |
| v4 | Better blocking + fine-tuned cross-encoder (US/India only) | 7.2 | 0.98906 | **0.978144** |
| v6 (no transformer) | Dense blocking channel, two-stage pruner, adaptive cleaning, France self-training, house-number features, XGBoost blend | 6.06 | 0.99006 | **0.983362** |
| v6 + transformer | v6 matcher blended with the cross-encoder | 6.06 | **0.99203** | not yet submitted |

The leaderboard leader is at **0.991043**.

**Where the remaining error is.** Back-solving the v6 leaderboard score with the test country
mix (India 46.8%, US 38.3%, France 15.0%) and the validated India/US scores gives
**France ≈ 0.945**, against ≈ 0.990 for India and US. France has no training labels at all, so it
is still the largest single gap: roughly 0.007 of the leaderboard score.

Two further facts from an error autopsy on the validation fold:
- False matches are rare (0.24% of predictions). Almost all loss is **missed** matches.
- About 1.6% of all true pairs are **unresolvable by any method**: the noisy record has no
  address and its business name is shared by several Source 1 records. That caps US/India near
  0.995.

---

## 2. The pipeline in one screen

All code is in the GitHub repo `GarvRajput17/MLChallenge`, under `work/src/`. The full run is
`work/src/run_all.sh`.

1. **Cleaning.** `prep_basic.py`, `adaptive_config.py`, `learn_lexicons.py`, `normalize.py`.
   Lowercase, fold Latin/Greek/Cyrillic accents, fix digit typos and alias prefixes. Per-country
   dictionaries (abbreviations, transliteration, filler words) are learned from the labels.
   `adaptive_config.py` fits stop words per country from the records alone.
2. **Blocking.** `run_blocking.py`, `blocking.py`, `dense_encoder.py`. Seven search channels
   (TF-IDF name/address/joint views, exact keys, a fine-tuned `multilingual-e5-small` dense
   channel, S2↔S3 cross-links), then a two-stage learned pruner down to ~6 candidates per S1.
   Recall of true pairs after pruning: 99.47% (US and India).
3. **Matching.** `build_features.py`, `features.py`, `train.py`, `model.py`. About 70 pair
   features, a two-stage LightGBM (the second stage sees each candidate's rivals), an XGBoost
   blend, and isotonic calibration.
4. **Self-training for unlabelled countries.** `self_train.py`. The matcher's own confident
   France matches (p ≥ 0.99) are used as labels to learn France's cleaning dictionary, then
   France is re-cleaned, re-blocked and re-scored.
5. **Transformer.** `build_ce_pairs.py` exports pairs with raw text; `cross_encoder.py` fine-tunes
   or scores `xlm-roberta-base` on them; `stack.py` blends its score with the matcher and checks
   the gain by 2-way cross-validation before trusting it.
6. **Decision.** `assemble.py`, `decide.py`. Each S2/S3 record goes to at most one S1, then each
   S1 gets the set that maximises expected F0.5.

The heavy CPU stages (1–4) need about **125 GB of RAM** at peak and ran on a 64-core AWS box
(`r7i.16xlarge`) in about 5 hours. **They do not fit on the GPU server**, where other users
typically leave 30–50 GB free. The GPU server is for step 5, and the kit below lets you finish
steps 5–6 there without AWS.

---

## 3. The GPU server

| | |
|---|---|
| Host | `172.16.192.168` (institute LAN only), hostname `worker1`, Rocky Linux 8 |
| Account | `garvit` — log in with your own password; **change the default one if you haven't** |
| GPUs | 0, 1: RTX 6000 Ada, 48 GB each. **2: RTX PRO 6000 Blackwell, 96 GB (the fastest)** |
| CPU / RAM | 64 cores, 215 GB — **shared, usually 150+ GB used by others** |
| Disk | `/home` is **97% full (≈60 GB free)** and shrinks as other users write |
| Python | system is 3.6 (too old). Use **`~/er/venv/bin/python`** (3.12, torch 2.14 + CUDA 13, transformers 4.57, pandas 3, LightGBM 4.7) |
| AWS | **no credentials on this machine** (by design). Data came in through temporary S3 links |

### It is a shared machine — rules that avoid trouble

- **Check before you launch:** `nvidia-smi` (who is on which GPU) and `df -h ~` (disk). Other
  users run jobs on all three GPUs at different times. Pick a GPU with free memory and set
  `CUDA_VISIBLE_DEVICES` explicitly on every command.
- **Never touch other users' processes or files.**
- **Keep disk use small.** A full-disk crash already happened once: training finished all
  45,635 steps and then died writing the final weights. Write outputs to `~/er/out/`, delete
  what you no longer need, and use `--save_every` for long training runs.
- **Run everything long in `tmux`**, and set environment variables *inline on the command*.
  A new `tmux` session under an already-running tmux server does not inherit `export`ed
  variables.
- When piping logs through `grep`, add `--line-buffered`, or the log stays empty until the end.

### What is on the server (`~/er/`)

```
~/er/venv/                  Python environment (use this interpreter)
~/er/pipeline/work/src/     the full pipeline source (same as the repo)
~/er/pipeline/student_resource/utils/validate_submission.py
~/er/out/full_model/        fine-tuned xlm-roberta-base cross-encoder (the one behind v4/v5/v6)
~/er/kit_v6/                everything needed to turn transformer scores into a submission:
    train_base.parquet        v6 matcher outputs for the 1.15M validation-fold pairs (with labels)
    test_base.parquet         v6 matcher outputs for all 10.5M test candidate pairs
    test_s1_ids.parquet       every test Source 1 id (a submission needs one row each)
    train_ground_truth.parquet
    meta.json                 the decision-layer settings the matcher was tuned with
    test_source1.tsv          for the format validator
    pairs/train.parquet       10.4M labelled training pairs with raw text (non-validation folds)
    pairs/eval.parquet        1.15M validation-fold pairs with raw text and labels
    pairs/test_pairs.parquet  10.5M test candidate pairs with raw text
    ce_v6base/                the current cross-encoder's scores on eval + test (the baseline)
~/er/pipeline/kit_check.sh  reproduces the v6 + transformer submission from the kit (see §4)
```

All pair files use the same columns: `s1_entity_id, cand_entity_id, a_name, a_addr, b_name,
b_addr, country`, plus `label` on the train and eval files. The validation fold is fixed
(hash of the S1 id), so the training pairs never contain a validation entity.

---

## 4. The loop you can run entirely on the GPU server

Everything below uses only the server. The one input that changes between experiments is a
directory with two files, `eval_ce.parquet` and `score_ce.parquet`, each holding
`s1_entity_id, cand_entity_id, ce_p`.

### 4.1 Sanity check first (≈10 min, CPU only)

```bash
~/er/pipeline/kit_check.sh
```

It must print `matcher 0.99006  matcher+cross-encoder 0.99203`. Those are the numbers the AWS
box produced. If it doesn't, stop and investigate before changing anything.

### 4.2 Score pairs with a cross-encoder (GPU)

`cross_encoder.py` was written as a SageMaker script, so it reads its inputs from environment
variables. Directories are "channels" holding parquet files.

| Variable | Meaning |
|---|---|
| `SM_CHANNEL_TRAIN` | dir of training pairs; if set, the model is fine-tuned first |
| `SM_CHANNEL_EVAL` | dir of labelled pairs to score and report AUC on (→ `eval_ce.parquet`) |
| `SM_CHANNEL_SCORE` | dir of unlabelled pairs to score (→ `score_ce.parquet`) |
| `SM_OUTPUT_DATA_DIR` | where the score files go |
| `SM_MODEL_DIR` | where the fine-tuned model is saved |

Flags: `--model` (a HuggingFace name or a local dir), `--epochs`, `--bs`, `--lr`,
`--max_len` (96), `--eval_bs`, `--workers`, `--save_every`.

Example: score the v6 test pairs with the existing model on the Blackwell GPU. Put the channel
files in their own directories first, since each channel reads every parquet in its directory.

```bash
mkdir -p ~/er/ch/eval ~/er/ch/score
ln -sf ~/er/kit_v6/pairs/eval.parquet ~/er/ch/eval/
ln -sf ~/er/kit_v6/pairs/test_pairs.parquet ~/er/ch/score/
tmux new -d -s score "CUDA_VISIBLE_DEVICES=2 SM_CHANNEL_EVAL=\$HOME/er/ch/eval \
  SM_CHANNEL_SCORE=\$HOME/er/ch/score SM_OUTPUT_DATA_DIR=\$HOME/er/out/myscores \
  SM_MODEL_DIR=/tmp/unused ~/er/venv/bin/python -u ~/er/pipeline/work/src/cross_encoder.py \
  --model \$HOME/er/out/full_model --eval_bs 2048 --workers 12 2>&1 | tee ~/er/out/score.log"
```

Measured speed on the RTX 6000 Ada: **~1,230 pairs/s training, ~5,000 pairs/s scoring** (so the
10.5M test pairs take ~35 min). The Blackwell card should be faster; it hasn't been benchmarked.
A harmless warning about a "Mistral regex" appears when loading the tokenizer. Ignore it.

### 4.3 Blend, check, and build a submission (CPU, ≈10 min)

```bash
cd ~/er/pipeline/work
K=~/er/kit_v6; CE=~/er/out/myscores
ER_THREADS=8 ~/er/venv/bin/python src/stack.py --kit $K --ce-dir $CE
ER_THREADS=8 ~/er/venv/bin/python src/stack.py --kit $K --ce-dir $CE --predict \
    --outdir ~/er/out/sub_new --ce-countries US India
```

The first command prints the **validation gain** of your scores over the matcher alone. That
number is what to compare experiments on (the baseline is +0.00197). The second writes
`matching_results.tsv` and `candidate_pairs.tsv`. Drop `--ce-countries` to apply the
transformer to France too; the current model has never seen French training data. Then run
the validator as in `kit_check.sh`.

---

## 5. What is most worth trying next (all GPU-side)

In order of expected value:

1. **Train the cross-encoder on French data.** It has only ever seen US and India. France is
   the biggest gap and has no labels, but the self-training step showed the matcher's confident
   France matches are ~99.6% precise. Pseudo-labelled France pairs (confident positives plus
   the candidates they beat as negatives) mixed into `pairs/train.parquet` would teach the
   transformer French abbreviations and noise. This needs the France rows of `test_base.parquet`
   (they carry `p_lgb`) joined to `pairs/test_pairs.parquet` for the text. **Guard against
   leakage:** never use France pseudo-labels to *score* the same pairs you trained on without
   thinking it through.
2. **A larger model.** `xlm-roberta-large` (MIT licence, ~560M parameters, far under the 8B limit)
   on the Blackwell GPU. Expect ~3× slower per pair than base.
3. **Train on hard pairs only.** Most pairs are obvious. Training (and scoring) only the pairs the
   matcher is unsure about — say `0.02 < p_lgb < 0.98` in the base files — concentrates the model
   on the cases that change decisions, and makes a large model affordable.
4. **Feed the cross-encoder more than name and address.** It currently reads
   `"name | address"` per side. The house number turned out to be the key signal for decoy
   records (a copy of a real business with the number nudged up by 1–25); making it explicit
   in the input text may help.

Judge every idea by the validation gain `stack.py --kit` prints, **not** by the cross-encoder's
AUC: the AUC is already 0.998, and a higher AUC does not reliably mean a better blend.

---

## 6. Things that went wrong once, so they don't twice

| Problem | Cause | What to do |
|---|---|---|
| Training died at the very last step | Shared `/home` disk full while saving weights | Check `df -h ~`; use `--save_every`; clean up old outputs |
| `UFuncNoLoopError` from `np.char.startswith` | pandas 3 reads string columns as Arrow arrays | Use `pd.Series(x).str.startswith(...)` |
| Segfault on import | `lightgbm` loaded before `sparse_dot_topn` (two OpenMP runtimes) | Import blocking code first (only matters for blocking) |
| Env var ignored in a job | New tmux session didn't inherit `export` | Set variables inline on the command |
| Log file empty for an hour | `grep` in a pipe buffers its output | `grep --line-buffered` |
| Validation looked worse than it was | Context features computed after filtering to the validation fold | Compute them on the whole shard, then filter (the kit already does this) |
| Validator crash on a fresh machine | It needs the test source files | `--test-dir` pointing at a dir with `test_source1.tsv` |

---

## 7. Submission and packaging reminders

- Upload only `matching_results.tsv` to the leaderboard. The final zip also needs
  `candidate_pairs.tsv`, the code under `code/business_entity_resolution/`, a `README.md`,
  pinned `requirements.txt`, and the filled-in `Documentation_template.md`.
- **Candidate-set size is ranked too** (smaller per S1 is better). v6 sits at 6.06 per S1.
- Only models under MIT/Apache 2.0 licences and up to 8B parameters are allowed. `xlm-roberta`
  and `multilingual-e5` are MIT. Llama-based models are not allowed.
- No external data lookup of any kind (APIs, registries, geocoders).
