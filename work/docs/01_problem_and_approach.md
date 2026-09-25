# Business Entity Resolution — Problem Statement, EDA Plan, and Feature Plan

> Working document. Everything numeric in here was measured on the actual files in
> `student_resource/dataset/`, not assumed.

---

## 0. TL;DR

We are given three independent, noisy snapshots of the same universe of businesses.
Source 1 is clean and deduplicated. Sources 2 and 3 are dirty and redundant. For every
Source 1 record we must output the set of Source 2 / Source 3 records that describe the
**same real-world business**.

This is a textbook **entity resolution** problem, and it decomposes into the standard
three stages — **blocking → pairwise scoring → decision** — but with two twists that
change the optimal design:

1. The scoring metric is **macro-averaged F₀.₅ per Source 1 entity**, which makes a false
   merge roughly 2.3× as expensive as a missed link, and makes a false merge on a
   *singleton* catastrophic (it costs the entity's entire 1.0).
2. **No Source 2 or Source 3 record is ever shared between two Source 1 entities.** The
   answer is a *partial one-to-one assignment*, not a set of independent binary decisions.
   Almost nobody exploits this, and it is free precision.

---

## 1. The problem, stated formally

Let

- $A = \{a_1, \dots, a_{N_1}\}$ be the Source 1 records (the reference / "gold" side),
- $B = \{b_1, \dots, b_{N_2}\}$ be the Source 2 records,
- $C = \{c_1, \dots, c_{N_3}\}$ be the Source 3 records,

each record being a triple `(business_name, business_address, country)` with no shared key.

There exists a latent function $\varepsilon(\cdot)$ mapping every record to the real-world
business it describes. We must recover, for each $a \in A$:

$$M(a) = \{\, x \in B \cup C \;:\; \varepsilon(x) = \varepsilon(a) \,\}$$

and output it as one row of `matching_results.tsv`. $M(a)$ may be empty.

### Why this is not just "string similarity"

Because $\varepsilon$ is hidden and the observed strings are corrupted by an unknown noise
channel. Formally we observe $x = \eta(\varepsilon^{-1}(\cdot))$ where $\eta$ is a stochastic
corruption process. Our job is to learn a decision rule that is invariant to $\eta$ —
which is exactly what makes this a machine-learning problem rather than a `==` comparison.
The training ground truth is 7.6M supervised examples of $\eta$ in action, and we should
mine it for everything it is worth.

### Scale

| File | Rows | Countries |
|---|---:|---|
| `train_source1.tsv` | 2,206,821 | US 1,323,633 · India 883,188 |
| `train_source2.tsv` | 5,034,616 | US 3,016,817 · India 2,017,799 |
| `train_source3.tsv` | 5,285,603 | US 3,170,056 · India 2,115,547 |
| `test_source1.tsv` | 1,732,544 | US 663,106 · India 809,986 · **France 259,452** |
| `test_source2.tsv` | 4,887,273 | US 1,871,330 · India 2,312,565 · France 703,378 |
| `test_source3.tsv` | 5,082,316 | US 1,945,701 · India 2,405,000 · France 731,615 |

The naive comparison space on test is $1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$
pairs. At one microsecond per pair that is **200 days**. Blocking is not an optimisation;
it is the only thing that makes the problem exist at all.

---

## 2. What the ground truth tells us

Measured on `train_ground_truth.tsv` (2,206,821 rows):

| Property | Value |
|---|---|
| Mean matches per S1 entity | **3.46** |
| Singletons (empty match list) | **123,247 (5.59%)** |
| Distinct matched S2/S3 IDs | 7,638,365 |
| Matched IDs reused by >1 S1 entity | **0** |
| S2 records that match something | 3,693,619 / 5,034,616 (73.4%) |
| S3 records that match something | 3,944,746 / 5,285,603 (74.6%) |

Match-list size distribution:

| size | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count | 123k | 119k | 375k | **531k** | 484k | 322k | 165k | 64k | 19k | 4.2k | 534 | 37 |

Per-source counts are **bounded**: an S1 entity has at most **5** S2 matches and at most
**6** S3 matches. That is a generator artefact and a hard prior we can use as a cap.

### Three structural facts, and what each one buys us

**(a) Partial one-to-one on the S2/S3 side.**
Zero matched IDs are shared. So the true solution is a bipartite graph where every S2/S3
node has degree ≤ 1. This converts "score each pair independently" into an **assignment
problem**. Operationally: after scoring, each S2/S3 record should go to its *arg-max* S1,
and the **margin between its best and second-best S1** becomes a powerful confidence
feature that no pairwise model can see on its own.

**(b) ~26% of S2/S3 are distractors.**
1.34M S2 and 1.34M S3 records match nothing. Sampling them shows they are *different
businesses altogether* (different names, different addresses, different cities) rather
than adversarial near-duplicates. Good news: blocking will naturally exclude most of them.
Bad news: they still inflate the candidate pool and they are what generates false merges
on singletons.

**(c) Only 5.6% singletons, so recall still matters.**
Predicting "no match" for everything scores 0.056. Despite F₀.₅ being precision-heavy, the
score is dominated by entities with 2–5 true matches. We cannot win by being timid.

---

## 3. The noise model (what $\eta$ actually does)

Read off directly from matched groups in the training data. This taxonomy is the
specification our features must be invariant to.

### Name corruptions

| Type | Example (S1 → S2/S3) |
|---|---|
| Character typos | `Payne Enterprises` → `Payne Enterpires`, `PAYNE-ENRTPRMISES` |
| Latin diacritics injected | `Payne Enterprises` → `Payne Énterprises` |
| Legal suffix added/dropped/swapped | `Lumay Boral` → `Lumay Boral Inc.`; `Ltd` ↔ `Limited`; `Corp` ↔ `Corporation` |
| Word transposition | `Hendricks and Flowers Inc` → `Hendricks and Inc Flowers` |
| Junk token insertion | `Summit` → `Summit Inc Center`, `... Partners`, `... Group`, `... Services` |
| Domain-name form | `Maure Williams Colombier Inc` → `maurewilliamscolombier.com` |
| DBA / alias prefix | `Obsidian, LLC` → `Korbrixx D.B.A. Obsidian, LLC` |
| Partial name | `Crystal Staffing Solutions LLC` → `Crystal` |
| **Full transliteration** | `Raj Investments LLP` → `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` |
| **Name replaced entirely** | `Maure Williams Colombier Inc` → `Dréxkor` (address is the only link) |

### Address corruptions

| Type | Example |
|---|---|
| Component reordering | `630 45th Terrace, Kansas City, MO` → `KANSAS CITY, MO, 630 45ND TERRACE` |
| Street-type abbreviation | `Avenue` ↔ `Ave`, `Street` ↔ `St` ↔ **`Saint`**, `Road` ↔ `Rd` |
| State name ↔ code ↔ native script | `Tamil Nadu` ↔ `TN` ↔ `தமிழ்நாடு`; `Missouri` ↔ `MO` |
| Missing components | drop PIN, drop state, drop house number |
| **Address entirely empty** | 168,967 S2 + 175,916 S3 training rows (~3.4%) |
| Placeholder junk | `null`, `<NULL>`, `NULL` embedded as a component |
| House-number mangling | `45th` → `45ND`, `1056` → `1056c`, `8706` → `870`, `684` → `0684` |
| Prefix noise | `Door No`, `H.NO`, `#`, `Block B-517`, `PO Box 8807` |
| City typos | `Akron` → `AKON`, `Danbury` → ok, `Deer Park` → `DEER PARK CIYT` |

### Scripts present

Latin plus **nine Indic scripts**: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
Telugu, Kannada, Malayalam. This is the single biggest modelling hazard: a Devanagari
name and its Latin original share **zero characters**, so *every* character- or token-level
similarity between them is exactly 0. Without a transliteration step, the whole Indian
half of the dataset is invisible to string matching.

---

## 4. The metric, and what it implies for the decision rule

$$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

computed **per Source 1 entity** and then **averaged over all entities** (macro). Singletons
count: predicting empty for a true singleton scores 1.0, predicting anything scores 0.0.

Macro-averaging is the crucial detail. An entity with 1 true match and an entity with 8
true matches contribute equally. So errors on **small match lists are disproportionately
expensive**, and errors on singletons are maximally expensive.

### Decision-theoretic derivation of the accept threshold

Take an entity with $k$ true matches that we have currently predicted perfectly
($P = R = 1$, score 1.0).

**Cost of adding one false positive** — $P = \frac{k}{k+1}$, $R = 1$:

$$F_{0.5} = \frac{1.25k}{1.25k + 1}$$

**Cost of dropping one true positive** — $P = 1$, $R = \frac{k-1}{k}$:

$$F_{0.5} = \frac{1.25(k-1)}{0.25k + k - 1}$$

| $k$ | loss from one FP | loss from one FN | FP / FN cost ratio |
|---:|---:|---:|---:|
| 0 (singleton) | **1.000** | — | ∞ |
| 1 | 0.444 | 1.000 | 0.44 |
| 2 | 0.286 | 0.167 | 1.71 |
| 3 | 0.211 | 0.091 | 2.32 |
| 4 | 0.167 | 0.063 | 2.67 |
| 5 | 0.138 | 0.048 | 2.88 |

Adding a candidate with predicted probability $p$ has expected gain
$p \cdot \text{gain}_{\text{TP}} - (1-p) \cdot \text{loss}_{\text{FP}}$, so the break-even is

$$p^* = \frac{\text{loss}_{\text{FP}}}{\text{loss}_{\text{FP}} + \text{gain}_{\text{TP}}}$$

giving $p^* \approx 0.70$ at $k=3$, $\approx 0.63$ at $k=2$, $\approx 0.74$ at $k=5$.

**Three consequences that shape the whole decision layer:**

1. The global operating point is roughly $p > 0.7$, **not** $p > 0.5$. Anyone who ships an
   argmax/0.5 classifier leaves several points on the table.
2. The threshold should depend on **how many matches we already believe the entity has** —
   the first match into an empty list is cheap, the sixth is expensive. This argues for an
   **entity-level decision stage**, not a per-pair one.
3. A singleton is a cliff: one FP costs 1.0. It is worth training a dedicated
   **"does this entity match anything at all?"** gate, because the 5.6% of entities that
   are singletons are worth 0.056 of the final score on their own — comparable to the gap
   between a decent and a winning solution.

---

## 5. Pipeline architecture

```
raw TSV
  │
  ├─ [0] normalisation          unicode → case → punctuation → token stream
  │
  ├─ [1] lexicon induction      learn transliteration + abbreviation maps
  │                             FROM TRAIN GROUND TRUTH ONLY  (no external data)
  │
  ├─ [2] blocking               name channel ⊎ address channel → candidate_pairs.tsv
  │                             (recall ceiling is set here — nothing downstream recovers)
  │
  ├─ [3] pairwise scoring       feature vector → LightGBM → p(match)
  │
  ├─ [4] decision layer         one-to-one assignment + margin + per-entity threshold
  │                             + singleton gate → matching_results.tsv
  │
  └─ [5] validation             macro F₀.₅ on held-out S1 entities
```

Stage [1] is the part that is specific to *this* dataset and where most of the
differentiation lives. Stage [4] is where the metric is actually won.

---

## 6. EDA plan

EDA here is not "draw histograms". Every experiment below exists to **settle one design
decision**, and is listed with that decision attached.

### E1 — Noise-channel quantification
*For each corruption type in §3, what fraction of true pairs exhibit it?*
Method: for each ground-truth pair, align token multisets and classify the difference
(exact / typo / suffix-only / transposition / transliteration / disjoint).
→ **Decides:** which invariances the feature set must have, and their priority ordering.

### E2 — Field informativeness
*Conditional on being a true pair, what is the distribution of name similarity? Of address
similarity? How often is exactly one of them destroyed?*
→ **Decides:** whether blocking needs two independent channels (it does if P(name useless) is
non-trivial) and whether the model needs an explicit "trust the other field" interaction.

### E3 — Token IDF landscape
*Document frequency of name and address tokens, per country.*
The head is junk (`private`, `limited`, `inc`, `llc`, `road`, `new`, `delhi`); the tail is
identity-bearing (`Orelee`, `Korbrixx`, `Dréxkor`). Blocking on a head token is useless;
blocking on a rare token is nearly sufficient on its own.
→ **Decides:** blocking key selection, and the weighting scheme for overlap features.

### E4 — Blocking recall ceiling sweep
*For candidate generator $g$ and top-$k$, what fraction of true pairs survive, and what is
the reduction ratio?*
Metrics: **pair completeness** (recall ceiling) and **reduction ratio**
$1 - \frac{|\text{candidates}|}{|A| \times |B \cup C|}$.
Sweep $k$ and channel combinations.
→ **Decides:** the single most important hyper-parameter in the pipeline. Everything
downstream is capped by this number.

### E5 — Transliteration coverage
*What share of Indic tokens (by occurrence) does the induced lexicon cover? What do the
misses look like?*
→ **Decides:** whether token-level transliteration suffices or we need a character-level
model (learned grapheme-cluster alignment) for the tail.

### E6 — Distractor characterisation
*Take unmatched S2/S3 records; find their nearest S1 neighbour. Is the score distribution
separable from true pairs?*
→ **Decides:** how hard precision will be, and whether the singleton gate needs its own
model or falls out of the pairwise scores.

### E7 — Hard-negative structure
*Within a blocking bucket, how often do two different real businesses share a full address
(same building) or a near-identical name (chain / franchise)?*
→ **Decides:** whether we need features that specifically arbitrate "same address,
different business" — the classic ER failure mode.

### E8 — France transfer probe
*Compare the French test distribution to US/India on script, token length, legal-suffix
vocabulary, address format, and name/address token IDF shape.*
→ **Decides:** whether a model trained on US+India transfers, or whether France needs a
country-blind feature subset and a more conservative threshold.

### E9 — Cardinality priors
*Distribution of |M(a)| conditional on country and on observable properties of $a$.*
→ **Decides:** whether the decision layer can use a learned prior over list length.

### E10 — Leakage / shortcut audit
*Are entity IDs, row order, or file order correlated with the labels?*
→ **Decides:** nothing about the model — but it protects us from building on an artefact
that evaporates on the private leaderboard.

---

## 7. Feature engineering plan

Grouped by the NLP/ML concept each family comes from, with the noise type it defends
against. All features are computed on both the **name** field and the **address** field
unless noted.

### 7.1 Normalisation and canonicalisation (pre-feature)

| Technique | Purpose |
|---|---|
| Unicode NFKC + selective NFD accent folding | `Énterprises` → `enterprises`, without touching Indic combining marks |
| Case folding, punctuation → space, `&`/`+` → `and` | punctuation-variation noise |
| Null-token stripping (`null`, `<NULL>`, `n/a`) | placeholder junk |
| **Induced transliteration map** (Indic token → Latin token) | the 9-script problem |
| **Induced abbreviation map**, learned *per country* | `Ave`↔`Avenue`, `TN`↔`Tamil Nadu` (India) vs `TN`↔`Tennessee` (US) |
| **Induced junk-token inventory** | `com`, `inc`, `dba`, `formerly`, `www`, `door`, `po` — down-weighted, not deleted |

Everything in this table is estimated from `train_ground_truth.tsv` by aligning matched
records. **No external dictionary, gazetteer, or API is used** — this is both a rules
requirement and, in fact, the better engineering choice, since the noise is
dataset-specific.

### 7.2 Lexical overlap features — *set-theoretic view of text*

- **Jaccard** $\frac{|T_a \cap T_b|}{|T_a \cup T_b|}$ on token sets — order-invariant, so it
  survives word transposition.
- **Containment / overlap coefficient** $\frac{|T_a \cap T_b|}{\min(|T_a|,|T_b|)}$ — the right
  measure when one side is a *partial name* (`Crystal` ⊂ `Crystal Staffing Solutions LLC`).
- **IDF-weighted overlap** $\frac{\sum_{t \in T_a \cap T_b} \text{idf}(t)}{\sum_{t \in T_a \cup T_b} \text{idf}(t)}$
  — one shared rare token (`Korbrixx`) outweighs five shared stopwords (`private limited`).
- **Max-IDF of the shared tokens** — a single very rare shared token is near-proof of a match.
- The same three, recomputed **after stripping junk/legal tokens**, giving the model both views.

*Concept:* bag-of-words / vector-space model, with IDF as a learned informativeness prior.

### 7.3 Character n-gram features — *sub-word representation*

- **TF-IDF cosine over character 3-grams** of the whole normalised string. This is the
  workhorse: it degrades gracefully under typos, absorbs abbreviations (`ave` ⊂ `avenue`
  shares `av`,`ave`), and — critically — handles the **concatenated domain form**
  (`maurewilliamscolombier.com`) that destroys every token-based feature.
- **Jaccard over character 3-gram sets** (unweighted companion).
- Computed on the **space-stripped** string too, so tokenisation differences vanish.

*Concept:* sub-word / n-gram language modelling — the same insight behind BPE and
FastText, applied to a 10M-document retrieval problem.

### 7.4 Edit-distance family — *sequence alignment*

Via `rapidfuzz` (SIMD, C++; the pure-Python versions are unusable at this scale):

- **Normalised Levenshtein / Indel ratio** — raw typo tolerance.
- **Jaro–Winkler** — prefix-weighted, matches how truncations and abbreviations behave.
- **`token_sort_ratio`** — sorts tokens before comparing → explicit transposition invariance.
- **`token_set_ratio`** — compares intersection vs remainders → explicit tolerance for
  *added* junk tokens, which is exactly the `+ Center / + Partners / + Group` noise.
- **`partial_ratio`** — best-matching substring → handles DBA prefixes and long-vs-short.

*Concept:* string alignment, with each variant encoding a different invariance. We give the
model all of them and let the trees learn which one to trust per regime.

### 7.5 Phonetic features — *equivalence classes over pronunciation*

- **Double Metaphone / Soundex** codes per token (`jellyfish`), then token-set Jaccard over
  codes. Catches `Akron`/`AKON`, `Shaffing`/`Staffing`, and transliteration residue where
  the romanisation differs but the sound does not.
- Deliberately used as a *feature*, never as a blocking key alone — phonetic keys are far
  too lossy (they'd collide half the corpus).

### 7.6 Structural / numeric features — *domain-specific signal*

These are frequently the highest-gain features in address matching and are worth building
carefully:

- **Numeric token agreement.** House numbers, PIN codes, and unit numbers are
  high-entropy and rarely coincidental. Extract all digit runs; compute exact-match count,
  Jaccard, and a *fuzzy* numeric match (`684` vs `0684` after leading-zero strip; `8706`
  vs `870` under prefix relation) — the observed mangling is systematic.
- **PIN / ZIP agreement** as its own feature (6-digit India, 5-digit US patterns).
- **State agreement after canonicalisation** — cheap and near-binary once the per-country
  abbreviation map is applied.
- **City-token agreement**, weighted by city frequency.
- **Acronym features:** does one side's name reduce to the other's initials? (`SS Food` ↔
  `एसएस फूड`). Also first-letter-sequence agreement.
- **Length and cardinality ratios** on both fields — a calibration signal for the trees.

### 7.7 Missingness and provenance features

- Address empty on either side (3.4% of rows) — must be an explicit flag, since all address
  similarities collapse to 0 and the model has to know *why*.
- Source indicator (S2 vs S3) — the two sources have measurably different noise profiles
  (S2 tends to uppercase + reorder, S3 tends to expand state names + typo).
- Country (as a categorical, **not** one-hot restricted to `{US, India}` — France must pass
  through cleanly).

### 7.8 Context features — *the part a pairwise model cannot see*

Computed after a first scoring pass; these encode the one-to-one structure from §2(a):

- **Rank** of this S1 among all S1 entities the candidate was blocked against.
- **Margin**: $s_{\text{best}} - s_{\text{second-best}}$ for the candidate across its S1 options.
  A candidate that loves one S1 and nothing else is a far safer merge than one that is
  mildly similar to nine.
- **Symmetric rank**: rank of this candidate within the S1 entity's own candidate list.
- **Entity-level aggregates**: number of candidates above threshold, score of the best
  candidate, gap to the next — these feed the singleton gate and the per-entity threshold.

*Concept:* this is learning-to-rank / structured prediction rather than pointwise
classification, and it is the standard reason production ER systems beat naive pairwise
classifiers.

---

## 8. Model and decision layer

**Pairwise model:** gradient-boosted decision trees (**LightGBM**). Justification, not habit:
the feature space is ~60 dense, heterogeneous, non-linearly-interacting numeric features
with sharp thresholds (`state matches AND house number matches` → almost certainly a match).
That is precisely the regime where GBDTs dominate, they train on millions of pairs in
minutes on CPU, and they give us monotone-ish, inspectable behaviour for error analysis.

**Training data:** positives from the ground truth; negatives from the blocking output
itself, so the model is trained on exactly the distribution it will score at inference —
avoiding the classic ER mistake of training on random negatives and then meeting only
hard ones in production.

**Decision layer**, in order:
1. Score every candidate pair.
2. **Assignment:** enforce the degree-≤1 constraint on the S2/S3 side (greedy arg-max is
   near-optimal here and is $O(n \log n)$; full Hungarian is unnecessary at this sparsity).
3. **Entity-level threshold** tuned directly against macro F₀.₅ on held-out entities,
   using the break-even analysis of §4 as the starting point and letting the data refine it.
4. **Singleton gate** — an entity-level classifier over the aggregate features from §7.8,
   because the cost of an FP on a singleton is 1.0 and deserves a dedicated decision.
5. Cap per-source counts at the observed maxima (≤5 from S2, ≤6 from S3).

**Neural components (phase 2, on GPU).** With AWS compute available the 8B/MIT allowance
becomes usable rather than theoretical. Two additions, in this priority order:

1. **Fine-tuned bi-encoder for blocking recall.** We have 7.6M labelled positive pairs —
   an enormous contrastive-learning dataset. Train a multilingual sentence encoder with
   InfoNCE / multiple-negatives-ranking loss, using in-batch negatives plus hard negatives
   mined from the lexical blocker. Then embed all 12.5M records and do ANN retrieval
   (FAISS) as a **third blocking channel**. This is the highest-value neural use because it
   attacks the one thing nothing downstream can fix — the recall ceiling — and it is
   precisely where lexical blocking fails: cross-script names whose transliteration the
   lexicon missed, and records where the name was replaced outright.
   Candidate base models (all ≤8 B, MIT/Apache-2.0): **LaBSE** (Apache-2.0, ~471 M,
   translation-pair pretraining so its cross-script alignment is strong out of the box),
   **multilingual-e5-base/large** (MIT), **XLM-R base** (MIT). Verify each licence before
   committing.

2. **Cross-encoder re-ranker on the uncertain band.** A small multilingual encoder
   fine-tuned on (S1, candidate) pairs, applied only where the GBDT is unconfident. Full
   attention over both records catches the compositional cases hand-built features miss
   (word transposition combined with a suffix swap and a typo). Restricting it to the
   ambiguous band keeps inference to tens of millions of pairs — hours, not days.

**The GBDT stays.** It remains the primary scorer and the arbiter of what the cross-encoder
even sees, because it is fast enough to run over the full candidate set, it consumes the
structural and numeric features a text encoder cannot represent (house-number agreement,
IDF-weighted overlap, one-to-one margin), and it is inspectable during error analysis.
The neural models are ensembled *into* it as additional features — bi-encoder cosine and
cross-encoder logit become columns in the GBDT, which is strictly better than averaging
scores at the end.

---

## 9. Validation protocol

- **Split by S1 entity** (hash of ID), not by pair — pairs from the same entity are not
  independent, and splitting by pair would leak.
- **Block against the full S2/S3 pool**, not just the held-out slice. Otherwise the candidate
  set is artificially easy and every threshold we tune is miscalibrated.
- Score with the exact macro-F₀.₅ from the brief, singletons included.
- Report alongside it: blocking pair-completeness, reduction ratio, pairwise PR-AUC, and
  a **per-country and per-list-size breakdown** — a single scalar will hide the France
  problem and the singleton problem, which are the two places this can quietly fail.
- Keep a fixed validation split for the whole project so numbers stay comparable.

---

## 10. Known risks

| Risk | Mitigation |
|---|---|
| **France is unlabelled.** No training signal for French noise. | Country-agnostic features; validate a US-only-trained model on India as a proxy for transfer; conservative threshold for unseen countries. |
| **Blocking recall caps everything.** | Two independent channels (name ⊎ address); measure pair-completeness explicitly before building anything downstream. |
| **Over-fitting the induced lexicon to train.** | Frequency thresholds and string-similarity guards on every induced mapping; hold the lexicon out of the threshold-tuning split. |
| **Compute budget is finite ($600).** | Develop on a cheap CPU box; burst to GPU only for training/inference. Budget alarm at $400. Stop instances when idle. |
| **GPU quota on a fresh AWS account is often zero.** | Request the G-instance vCPU limit increase on day one — approval can take hours to days and is the long pole. |
| **Spot interruption during the final inference run.** | Spot for experiments, on-demand for the run that produces the submission. |
| **Public/private leaderboard gap.** | Never tune on leaderboard feedback; trust the local held-out macro F₀.₅. |

---

## 10b. Compute plan (AWS, $600 credit)

The budget removes the laptop's 16 GB / 8-core ceiling and makes the neural components in
§8 viable. It does not remove the need for discipline — $600 is roughly 100 GPU-hours, so
compute is spent where it changes the score, not where it is convenient.

**Shape of the workload**

| Stage | Bound by | Wants |
|---|---|---|
| Normalisation, lexicons, EDA | single-thread Python | modest CPU, lots of RAM |
| Lexical blocking (sparse TF-IDF matmul) | RAM + memory bandwidth | **high-memory CPU** |
| Feature extraction over ~10⁸ pairs | CPU, embarrassingly parallel | **many vCPU** |
| GBDT training | CPU | many vCPU |
| Bi-encoder fine-tune + 12.5M embeddings | **GPU** | multi-GPU, fast interconnect not required |
| ANN index + search | GPU or high-RAM CPU | either |
| Cross-encoder re-rank | **GPU** | multi-GPU |

**Proposed split** — two instances rather than one, because paying GPU rates to run EDA is
the fastest way to burn the credit:

- **Work box:** memory-optimised CPU instance (r7i / r6i class, ~128–256 GB RAM). Everything
  in Phase A lives here. Cheap enough to leave running.
- **GPU box:** g5 / g6 class (A10G or L40S), started only for training and batch inference
  bursts, stopped immediately after. **Spot for experiments, on-demand for the final run.**
- **Shared state:** dataset and intermediate parquet/feature shards in **S3**; both boxes
  read from it. Avoids rsyncing 10 GB of intermediates between machines.

**Operational notes** (worth doing before any modelling):

1. **Request the G-instance vCPU quota increase immediately.** Fresh AWS accounts commonly
   have a limit of 0 for G and P families; approval can take hours to days. This is the
   single most likely thing to stall us.
2. Use a **Deep Learning AMI** so we are not debugging CUDA/driver versions on the clock.
3. Set an **AWS Budgets alarm at $400** and a second at $550.
4. **Stop, don't terminate**, between sessions — you keep the EBS volume and pay only for
   storage (a few dollars a month for 500 GB gp3).
5. Instance pricing moves and varies by region; verify current on-demand and spot rates in
   the console before committing rather than trusting any figure quoted from memory.

---

## 11. Build order

Strictly sequenced so that we always have a *scoreable* end-to-end pipeline, and every
later stage is justified by measured error analysis rather than by ambition.

**Phase A — baseline (CPU, must complete before anything neural)**

1. ~~Environment + parquet caching + base normalisation~~ ✅
2. **Lexicon induction** — transliteration ✅ works across all 9 scripts; Latin
   canonicalisation ❌ over-merges and needs the redesign in §7.1
3. **EDA E1–E3, E5** → confirm the feature plan against measurements
4. **Blocking + E4 recall sweep** → lock the candidate generator, emit `candidate_pairs.tsv`
5. **Feature extraction** (§7.1–7.7) on train candidates
6. **LightGBM pairwise model** + PR-AUC + error analysis (E6, E7)
7. **Context features (§7.8) + decision layer (§8)** → tune macro F₀.₅
8. **First full test inference → validate → submit.** A scored leaderboard entry is the
   checkpoint that makes everything after it safe.

**Phase B — neural lift (GPU), each step gated on Phase A error analysis**

9. **E8 France probe** → adjust thresholds / feature blindness
10. **Bi-encoder fine-tune** on the 7.6M positive pairs → third blocking channel → re-measure
    pair-completeness. Keep only if the recall ceiling actually moves.
11. **Cross-encoder re-ranker** on the uncertain band → new GBDT feature column → re-tune
    the decision layer.
12. Final full inference → validate → package (code, pinned requirements, README,
    methodology write-up).
