# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve each Source 1 business to its Source 2/3 records with a scalable **blocking → matching →
decision** pipeline. Blocking combines sparse TF-IDF retrieval, a fine-tuned multilingual dense
retriever and cross-source links, then a two-stage learned pruner keeps **6.06 candidates per
entity** while retaining **99.47%** of true pairs. Matching stacks a two-stage gradient-boosted
matcher with a fine-tuned multilingual cross-encoder, and the final sets maximise **expected
F0.5** per entity. France, which has no training labels, is handled by label-free cleaning plus
self-training on the model's own confident matches.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the 7.6M labelled training pairs, each measured before it shaped the design:

- **The answer is a partial one-to-one assignment.** No Source 2/3 record matches two Source 1
  entities; an entity has at most 5 Source 2 and 6 Source 3 matches; 5.6% of entities are
  singletons; about 26% of Source 2/3 records match nothing (distractors).
- **Names alone are ambiguous.** 30–46% of Source 1 names are shared with another entity (one
  India name occurs 199 times). Among candidate pairs whose names clearly agree, only 67% (US) and
  72% (India) are true matches.
- **Decoys.** Distractors include near-copies of real businesses with the **house number shifted
  up by 1–25**. With name and street agreeing, such pairs are only ~4% true in the US labels,
  whereas a true copy keeps the number or changes it by a large amount.
- **Missing addresses.** 4.4% of true pairs have no address on the noisy side; when the name is
  also shared by several Source 1 entities, no method can tell them apart. We measured this
  irreducible floor at about 1.6% of true pairs.
- **Scripts and languages.** India records appear in nine Indic scripts (Devanagari, Tamil,
  Telugu, …), often mixed with Latin in one name. France (test only) uses French abbreviations
  (`R.`→rue, `Av`→avenue), dotted legal forms (`S.A.S.`), departments in place of regions, and
  accent-stripped domain names.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — blocking + stacked classifier (gradient boosting + transformer
cross-encoder) + expected-F0.5 set selection.

**Core Innovations:**
1. A **learned, two-stage pruner** (supervised meta-blocking) that adapts the candidate count per
   entity instead of a fixed top-k: smaller candidate sets *and* higher recall.
2. **Label-free adaptation to unseen countries**: stop words and their roles fitted from the
   records alone, then **self-training** — the matcher's own high-confidence matches (99.6% precise
   on a simulated unseen country) become labels for that country's cleaning dictionary and for
   fine-tuning the cross-encoder.
3. **Decisions that maximise expected F0.5 exactly** from calibrated probabilities (Poisson-binomial
   expectation over every prefix of the ranked list), which also decides singletons.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys / channels used** (all retrieve the nearest Source 1 records for each Source 2/3
  record; Source 1 is the small side, so its index stays small):
  - TF-IDF character 3–4-grams on the space-squashed name (robust to typos and domain-name forms)
  - TF-IDF word views of the address, the address core and the name
  - A **joint name + address** view, so the address can break ties among same-name entities
  - A **dense multilingual channel**: `multilingual-e5-small` (MIT) fine-tuned with in-batch
    negatives on training pairs, exact nearest-neighbour search on GPU
  - **Cross-source links**: each entity's best candidates pull in their nearest records from the
    other noisy source (an S3 copy is often closer to its S2 twin than to the clean original)
  - Exact-key hash joins on the canonical name, squashed name and address
- **Pruning:** the union (~200 pairs per entity) is scored by a stage-1 LightGBM on cheap
  blocking signals (channel scores and each pair's rank/gap among its competitors). A loose cut
  keeps ~6.7 pairs per entity; a stage-2 LightGBM adds token-overlap, rarest-shared-token and
  fuzzy features and makes the final cut (p ≥ 0.01, at most 12 per entity).
- **Candidate pairs generated (test):** 10,503,807, i.e. **6.06 per Source 1 entity**
  (reduction ratio vs. all pairs within a country ≈ 99.9998%).
- **How we ensured true matches were not lost:** both pruners train only on non-validation
  entities and are judged on held-out entities. Pool recall before pruning is 99.65% (India) and
  99.64% (US); after pruning **99.47%**. Each channel's unique contribution is logged, and the
  whole stage is linear in the number of records.

---

## 4. Matching Model

**Features used (≈70):**
- **Name features:** TF-IDF cosines (tokens, character n-grams, name core without legal forms),
  Jaccard/containment, Levenshtein ratio, token-sort and token-set ratios, Jaro–Winkler, prefix
  similarity, squashed-name equality, length ratios, **name multiplicity** (how many Source 1
  entities share this name)
- **Address features:** TF-IDF cosines on tokens, characters and the address core, token overlap,
  fuzzy ratios, **house-number distance** (signed difference, "shifted up by 1–25" decoy flag,
  ratio, edit distance), empty-address flags
- **Other:** blocking-stage scores and pruner probability, dense-embedding cosine, source (S2 vs
  S3), and second-stage **context features**: the best-vs-second-best margin for each Source 2/3
  record across competing entities, rank among the entity's candidates, and how many of the
  entity's confident candidates carry the identical house number

**Model type:** Two-stage LightGBM (the second stage sees each candidate's competitors), blended
with XGBoost and isotonic-calibrated; stacked with a fine-tuned **`xlm-roberta-base`
cross-encoder** (MIT licence, ~280M parameters) reading both records' raw name and address. The
stacker is validated by 2-way cross-validation on held-out entities before it is used.

**Threshold selection method:** No fixed threshold. For each entity, the one-to-one-arbitrated,
calibrated probabilities are ranked and the prefix with the highest **expected F0.5** is chosen
(computed exactly with Poisson-binomial convolutions). Two decision-layer parameters are tuned on
the validation fold against macro F0.5.

**Text normalisation:** Unicode NFKC, accent folding for Latin/Greek/Cyrillic only (Indic marks
kept), dotted-acronym joining, digit-typo repair (`harb0r`→harbor), alias-prefix stripping
(`X D.B.A. Name`→name). Transliteration, abbreviation and filler-word dictionaries are learned per
country from the labelled pairs; for unlabelled countries they are induced from token statistics
and then re-learned from pseudo-labels (self-training).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation fold (US + India, 220,565 held-out entities):**
  **0.99203** (v6 matcher 0.99006 + cross-encoder).
- **Leaderboard progression:** 0.925 → 0.968 → 0.978 (new blocking + cross-encoder) → 0.983
  (adaptive cleaning, self-training, house-number features) → **0.98709** (cross-encoder on all
  countries).
- **Common false positives (wrong merges):** rare (0.08% of predicted pairs). The remaining ones
  are near-duplicate decoys and same-name records with an identical house number on a
  different street.
- **Common false negatives (missed matches):** 73% of the remaining loss comes from noisy records
  **with no address**. Most of that is the irreducible same-name case (an oracle using
  per-source match counts could resolve only 18.6% of it). The rest are heavily corrupted names
  and blocking misses for address-less records.

**France (no labels).** Comparing leaderboard submissions that differ only in France's rows let us
measure it directly: the France fixes raised it by about +0.028, and applying the cross-encoder
to France added a further +0.007, although that model had never seen French training data.

---

## 6. Conclusion

Most of the gain came from measurement rather than model size. Error autopsies showed where the
loss actually was: decoy house numbers, France's label-free cleaning, blocking recall. Each fix
was validated on held-out entities (and, for France, on paired leaderboard submissions) before
adoption. Several ideas that sounded good were rejected by those tests. The pipeline scales
linearly, generalises to an unseen country without labels, and keeps a small candidate set.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` — entry point `run_all.sh` (CPU stages), then the GPU
cross-encoder and stacking:

| Stage | Script |
|---|---|
| Load + basic normalisation | `prep_basic.py`, `common.py` |
| Adaptive stop-word configuration | `adaptive_config.py` |
| Lexicon induction (labelled + label-free) | `learn_lexicons.py`, `pseudo_lexicon.py` |
| Canonicalisation | `normalize.py` |
| Dense retriever (GPU) | `dense_encoder.py` |
| Blocking + learned pruner | `run_blocking.py`, `blocking.py` |
| Pair features | `build_features.py`, `features.py` |
| Matcher (LightGBM + XGBoost, calibration) | `train.py`, `model.py` |
| Self-training for unlabelled countries | `self_train.py` |
| Cross-encoder (GPU) | `build_ce_pairs.py`, `cross_encoder.py` |
| Stacking + submission files | `stack.py`, `assemble.py`, `decide.py`, `predict.py` |
| Tests | `test_cleaning.py`, `smoke_test.py` |

See `README.md` in that folder for exact run instructions and hardware notes.

### B. Additional Results

| Change (measured on held-out entities unless noted) | Effect |
|---|---|
| Learned pruner vs fixed top-12 | candidates 11.9 → 5.9, pair recall 95% → 98.4% |
| Joint + dense + cross-link channels | pool recall (India) 96.2% → 99.65% |
| House-number distance features | F0.5 0.98821 → 0.98963 |
| Name multiplicity + exact-number context | → 0.98998 |
| Self-trained lexicon, simulated unseen country | +0.0061 |
| Cross-encoder stacking (validation / leaderboard India+US) | +0.0020 / +0.0031 |
| Cross-encoder on France (leaderboard, France rows only) | +0.0073 |
