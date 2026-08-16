# Layer 2 corpus design proposal

**Status: proposal. Nothing here is implemented.**

Layer 1 — the corrected explicit-mode W/P task — is **not superseded** by
this. It remains the causal primitive and the blocker-diagnostic environment:
the smallest setting in which "does training order leave a persistent
difference" can be asked at all, and the setting in which the catastrophic
interference now under investigation was found. Layer 2 does not begin until
the Layer-1 diagnostics finish.

## 0. Why a second layer is needed at all

W/P has two families. That is enough to ask whether order matters, and not
enough for almost anything else the program claims.

| Claim | Testable in Layer 1? | Why |
|---|---|---|
| 1. Order produces persistent differences | yes | two families suffice |
| 2. Directional structure not reducible to semantics | **no** | a 2x2 transfer matrix has a single off-diagonal pair; there is no structure to be irreducible |
| 3. Model predicts held-out interventions | **no** | leave-one-pair-out needs pairs to leave out |
| 4. Ontology revision by predictive consequence | **no** | nothing to merge or split |
| 5. Derived curriculum beats reverse/uniform/random | **no** | with two families, curriculum *is* order; there is no allocation to hold fixed |
| Additive `alpha_i + beta_j` vs interaction `gamma_ij` | **no** | zero interaction degrees of freedom at n=2 |

The additive baseline is the sharpest competitor to the whole developmental
claim (`RISKS.md` S2), and it is **untestable** with two families. That alone
justifies Layer 2.

## 1. Candidate skill families

One world, one vocabulary, one record format. Six families, each requested by
an explicit MODE token — the Layer-1 identifiability fix generalized, so that
competence in each family is separately measurable on the same checkpoint.

Every record is a small entity description:

```
[BOS, MODE, k1, v1, k2, v2, k3, v3, k4, v4, SEP, ANSWER]
```

Keys `k` come from a key vocabulary, values `v` from a value vocabulary, and
every family answers into the **same** C-class answer vocabulary, so output
entropy is matched across families by construction.

| Family | Composition | Answer |
|---|---|---|
| F1 | SELECT ∘ MAP | `L(v_a)` — read the field named by MODE's key, map it through lookup `L` |
| F2 | SELECT ∘ AGGREGATE | `(v_a + v_b) mod C` |
| F3 | MAP ∘ AGGREGATE | `(L(v_a) + L(v_b)) mod C` |
| F4 | SELECT ∘ COMPARE | class of `argmax(v_a, v_b)` |
| F5 | CHAIN ∘ SELECT | `i = v_a mod 4`; answer is the bucketed value of field `i` |
| F6 | CHAIN ∘ MAP | `i = v_a mod 4`; answer is `L(v_i)` |

## 2. Shared latent primitives

Five primitives, each used by two or three families:

```
SELECT     read the value at a named key                F1 F2 F4 F5
MAP        apply the learned lookup L                   F1 F3 F6
AGGREGATE  sum modulo C                                 F2 F3
COMPARE    order relation between two values            F4
CHAIN      use one value as the index of another field  F5 F6
```

This is the design's whole point. Families **overlap in mechanism**, so a
developmental relationship between two of them can exist and be discovered.
The W/P pair shares nothing — a trivial associative lookup against a
substantially harder modular rule — which is precisely why its transfer
structure was degenerate.

The primitive-sharing graph is **known latent mechanistic structure**, not
developmental ground truth. It records which families are built from which
primitives, and nothing more.

Sharing a primitive does **not** entail positive transfer, and does not imply
a developmental edge in either direction. Two families over the same
primitive may help each other, interfere with each other, or be
developmentally unrelated; which of those holds is the empirical question the
transfer and interference matrices exist to answer. Treating the composition
graph as the answer would assume exactly what the project sets out to
measure.

It is quarantined under `data/hidden_ground_truth/`, invisible to discovery,
and used only as a **comparison object**: a structure against which inferred
developmental relationships can be contrasted, and a real signal for the
semantic-similarity baseline to be built from.

## 3. Example records

Illustrative, with `C=8`, four fields:

```
F1  SELECT∘MAP        [BOS MODE_F1 kA v3 kB v7 kC v1 kD v5 SEP  L(v3)  ]
F2  SELECT∘AGGREGATE  [BOS MODE_F2 kA v3 kB v7 kC v1 kD v5 SEP (3+7)%8]
F3  MAP∘AGGREGATE     [BOS MODE_F3 kA v3 kB v7 kC v1 kD v5 SEP (L3+L7)%8]
F4  SELECT∘COMPARE    [BOS MODE_F4 kA v3 kB v7 kC v1 kD v5 SEP  cls(B) ]
F5  CHAIN∘SELECT      [BOS MODE_F5 kA v3 kB v7 kC v1 kD v5 SEP  b(v5) ]   3 mod 4 = 3 -> field D
F6  CHAIN∘MAP         [BOS MODE_F6 kA v3 kB v7 kC v1 kD v5 SEP  L(v5) ]
```

In the illustration the four bodies are identical, but **literal byte
identity is not the requirement** and should not be imposed where it would
impoverish the families. Chained and comparison families in particular need
value distributions that make their own computation non-degenerate, and
forcing every family through one record draw would flatten exactly the
richness that makes six families worth having.

The real requirement is threefold:

* **controlled nuisance statistics** — record length, key order, value
  marginals and answer-class marginals are matched across families, and any
  deliberate dependency (as in a chained family, where one value indexes a
  field) is declared and is the *only* uncontrolled statistic;
* **matched surface complexity** — no family is distinguishable from its
  surface form alone by a model that ignores MODE;
* **identifiable capability differences** — MODE remains the only explicit
  task identifier, so competence in each family is separately measurable on
  the same checkpoint.

The Layer-1 identifiability failure was not about bytes. It was that the
answer-generating function varied while nothing in the input said which
function applied. MODE fixes that, and matched nuisance statistics prevent a
model from inferring the family through a back door instead.

### Persistent background

A seventh stream, `BACKGROUND`, carries `MODE=NEUTRAL` and a plain
next-token objective over the same records. It is present at a fixed weight
`alpha_bg` in **every** phase of **every** curriculum. This is what makes a
curriculum change capability emphasis rather than swapping the visible
distribution wholesale, and it is the Layer-2 answer to the Apple finding
below.

## 4. Acquisition-difficulty calibration

The W/P failure mode to avoid: P was learned in ~25 steps and W needed ~600,
so every mixture ratio traded one against the other and the fixed-budget
continuity curve was confounded. Layer 2 equalizes difficulty *before* any
curriculum runs.

Each family exposes complexity knobs — lookup table size, number of
aggregated fields, value range, chain depth. Calibration:

1. train each family **solo** from initialization at the frozen regime;
2. measure `t_90`, steps to reach 90% of observed acquisition;
3. adjust each family's knobs to bring `t_90` inside a prespecified band,
   say a factor of 1.5 between fastest and slowest;
4. re-measure and freeze; record the frozen knobs in `regime.json` alongside
   the model and train configuration;
5. verify each family clears its competence and held-out generalization
   thresholds solo.

Exit criterion: `max_i t_90(i) / min_i t_90(i) <= 1.5`, with every family
clearing competence and generalization. Difficulty matching is a **gate**,
not a nicety: unmatched difficulty makes `T_ij` and `F_ij` partly a
measurement of how hard family `j` is, which is exactly the additive
`beta_j` term the developmental claim has to beat.

Held-out diagnostics per family: a stratified split of the record space, so
each family has its own compositional generalization set, exactly as the
current `digit_table` split works.

## 5. Curricula as time-varying mixture weights

A curriculum is a trajectory `alpha_i(t)` over the six families plus the
background:

```
alpha_bg(t) = alpha_bg          constant, every phase
sum_i alpha_i(t) = 1 - alpha_bg  for all t
```

Two constraints make ordering the only manipulated variable:

```
total tokens          T = integral over t, identical across arms
per-family allocation c_i = integral alpha_i(t) dt, identical across arms
```

So two curricula may differ arbitrarily in **when** each family is
emphasized, while agreeing exactly on total compute and on the aggregate
corpus composition. This is the control that `research.md` 21 demands via the
reversed curriculum, generalized: if a discovered curriculum and its reverse
both beat uniform, the advantage came from allocation, and here allocation is
held identical by construction rather than checked after the fact.

Representative trajectory shapes: staged blocks, linear ramps, overlapping
raised-cosine windows, and the reverse of each.

## 6. Both directions of the pairwise effect

Layer 1 measures one thing badly and the other not at all. Layer 2 measures
both, with matched controls:

**Forward transfer** — does exposure to `i` speed acquisition of `j`?

```
T_ij = integral over the target phase of [ L_j^N(t) - L_j^(i)(t) ] dt
```

against a neutral-prefix control `N`, as already specified.

**Retention / interference** — does acquiring `j` damage `i`?

```
F_ij = A_i(after j) - A_i(control)
```

where the control spends the same tokens on `BACKGROUND` instead of `j`.
`F_ij` is negative under interference. This is the Layer-1 phenomenon
promoted to a first-class estimand rather than a blocker: with six families
there are 30 ordered pairs, so interference itself has structure that can be
predicted, and `F` has its own additive baseline to beat.

`T` and `F` are separate matrices with separate uncertainty and replication
counts, and neither is a proxy for the other.

## 7. Related work bearing on the Layer-1 blocker

Flagged for the record: Apple's 2025 scaling study reports that mixing even a
small fraction of pretraining data into finetuning can strongly protect prior
behaviour, and continual-learning work more broadly treats distribution
continuity as a central stabilizer —
https://machinelearning.apple.com/research/scaling-laws

This is consistent with the direction of the Layer-1 continuity curve, where
mixing the prior family into phase 2 rescued W→P coexistence. Two cautions
before leaning on it. First, this is a pointer supplied for context, not a
claim verified against the paper's own protocol, and the fraction that
suffices is exactly the quantity our diagnostic is measuring rather than
assuming. Second, our rescue appeared between r=0.10 and r=0.25, not at the
1–5% level, though the fixed-budget arm was confounded and the
budget-corrected curve may move that threshold.

The design consequence stands regardless: `alpha_bg > 0` in every phase makes
the persistent common background a **structural property of the corpus**
rather than a continual-learning method bolted onto training. That distinction
matters for the scientific claim — it is a statement about what a
developmental corpus *is*, not about how to optimize on one.

## 8. What has to be decided before implementing

1. Whether the primary operationalization stays pure block-sequential or
   moves to controlled overlapping curricula. Layer 1's continuity curve is
   the evidence; the decision is about the hypothesis and belongs to the
   record, not to a threshold.
2. Whether `F_ij` is co-primary with `T_ij` or secondary.
3. The value of `alpha_bg`, which should itself be calibrated rather than
   guessed.


## 9. Scope: what Layer 2 does and does not establish

Layer 2 is a synthetic corpus with exact ground truth and deterministic
generation. It can establish whether developmental structure exists and is
predictable **in that corpus**.

It does not establish arbitrary-corpus optimization. Applying the frozen
pipeline to an unseen natural corpus is Gate J, a separate later stage, and
neither Layer 1 nor Layer 2 licenses claims about it. Sections of the plan
describing generic corpus abstraction, family proposal from arbitrary text,
and compiled sampling schedules for large corpora are **later portability and
system-identification targets**, not results.


## 10. Frozen Layer-2 family set (today)

`n_values = 64`, `agg_range = 16`. Four families frozen, two dropped.

| family | median t90 | train | held-out | status |
|---|---|---|---|---|
| F1 SELECT∘MAP | 99 | 1.000 | 1.000 | frozen |
| F4 SELECT∘COMPARE | 119 | 0.979 | 0.971 | frozen |
| F6 CHAIN∘MAP | 225 | 1.000 | 1.000 | frozen |
| F5 CHAIN∘SELECT | 232 | 1.000 | 1.000 | frozen |
| F2 SELECT∘AGGREGATE | — | 0.173 | 0.167 | **dropped** |
| F3 MAP∘AGGREGATE | 684 | 0.718 | 0.709 | **dropped** |

Difficulty spread over the frozen four is **2.34x**, inside the adequacy
band. Twelve ordered pairs remain, enough to fit the baseline ladder.

Reducing `agg_range` helped F3 substantially (0.208 to 0.718) by shrinking
its lookup domain, and did not help F2 at all. The reason is that the input
tokens still span 64 values, so F2 must learn a 64-to-16 residue reduction
*before* the modular sum — harder than the original 16-value task even though
the arithmetic range now matches it. AGGREGATE is not reachable from a
shared-vocabulary knob at this model scale, and reopening that is rigor-pass
work rather than today's.

The remaining four still share primitives: SELECT appears in F1, F4 and F5;
MAP in F1 and F6; CHAIN in F5 and F6. Ten of twelve ordered pairs share at
least one primitive, so the structure the design exists to test survives the
drop.

Pre-existing nuisance note, not introduced by this change: F5's answer
marginal deviates more than the others (0.096), because the chain index is
derived from the same value that can be selected. Recorded for the rigor pass.

## 11. Held-out prediction: what may and may not be used

**Pair-ID interaction coefficients cannot predict an unseen pair.** A
coefficient `gamma_ij` fitted per pair has no value for a pair never
observed, so a model built on them is descriptive only and belongs in the
ladder as an **in-sample upper bound**, never as the held-out predictor.

Held-out prediction must use features that exist for unseen pairs:

* frozen source and target primitive indicators;
* primitive-sharing features (count and identity of shared primitives,
  whether sharing is in the first or second composition slot);
* source and target solo difficulty (`t90`), which is the honest form of the
  additive `beta_j` term rather than something to hide.

The ladder therefore reads:

    global mean
      -> source-only / target-only
      -> additive source + target
      -> primitive-sharing / relationship features
      -> [pair-ID interaction: in-sample upper bound, descriptive]

Developmental structure counts as established only if the relationship
features beat the additive baseline **on truly held-out pairs**. Sharing a
primitive is a feature, not a prediction: it may correspond to positive
transfer, interference, or nothing.


## 12. Transfer pilot design and its structural limit

All **12 directed pairs** over the frozen four families, 3 seeds, paired
source/control arms sharing one base checkpoint, target `t=0` evaluation
preserved. The neutral control `N` is a balanced mixture of the two families
that are neither source nor target at the same token budget, so the arms
differ in the identity of the source phase and nothing else.

### Execution units

| quantity | value |
|---|---|
| trajectories | 72 (12 pairs x 2 arms x 3 seeds) |
| bundled units | 36 (paired arms in one process) |
| steps per unit | 2,400 — total 86,400 |
| measured C3 rate | ~190 steps/s aggregate at concurrency 22 |
| 1 VM | 1.6 waves, ~8 min |
| **2 VMs** | **1 wave of 18 units each, ~4 min** |

Two `c3-standard-22` VMs, one wave each. The experiment is not reduced to fit
one machine.

### Predictive ladder, deliberately small

Twelve pair identities is a very small universe, so the ladder stops early:

    global mean
      -> additive source + target
      -> additive + minimal primitive-sharing features

Structural features are limited to **shared-primitive count and identity**.
Composition-slot interactions are *not* fitted: with 12 pairs they would
overfit immediately. Pair-ID interactions remain descriptive and in-sample
only, because a per-pair coefficient cannot predict a pair never observed.

**Held-out split unit is the pair.** Every seed and both arms of a held-out
source-to-target pair stay entirely held out, or the model sees the pair it
is being asked to predict.

### The structural limit, recorded

**10 of the 12 directed pairs share at least one primitive.** Only 2 do not.
That is very little contrast for a feature whose whole purpose is to
distinguish sharing from non-sharing, and it caps what this pilot can
establish: a positive result would rest on two negative-control pairs.

This is a **viability pass**, not evidence of developmental structure. A
broader substrate — more families, deliberately including primitive-disjoint
ones — is required in the rigor pass before any claim about primitive-level
structure is defensible.

### Non-blocking pilot debt

F5's answer marginal deviates 0.096 from uniform, more than the other three,
because its chain index derives from a value that can itself be selected. Not
tuned today. It becomes blocking only if it produces a concrete transfer or
evaluation invalidator, in which case F5's rows and columns are suspect.


## 13. Transfer pilot result (viability pass)

36 bundled units, 72 trajectories, all 12 directed pairs x 3 seeds, 776s.
A second VM was blocked by `CPUS_ALL_REGIONS` quota (limit 32; the running
worker holds 22), so the wave ran in two waves at concurrency 22 on one
machine rather than shrinking the experiment.

### Measurability

| quantity | value |
|---|---|
| between-pair spread of mean `T_aulc` | 0.478 |
| median within-pair seed sd | 0.109 |
| signal-to-noise | **4.37** |

Effects are reproducible across seeds. **Exception: F6->F5 has a within-pair
sd of 0.897**, eight times the median, so that cell is not reproducible and
its row should be treated as suspect.

Most pairs show **negative** transfer — prior exposure to another family
slows target acquisition — strongest at F1->F5 (-1.51). Only F5->F4 (+0.20)
and F6->F1 (+0.12) are positive.

The `t=0` head start and the rate-only component **frequently carry opposite
signs**: F4->F1 has head start -2.31 against rate-only +2.06. A single AULC
number would have hidden two opposing mechanisms, which is precisely why the
`t=0` evaluation is mandatory.

### Held-out prediction, pair as split unit

| model | held-out RMSE | MAE |
|---|---|---|
| global mean | 0.4987 | 0.3745 |
| additive source + target | 0.6743 | 0.5984 |
| **additive + shared-primitive features** | **0.2711** | **0.2354** |
| pair-ID interaction | in-sample only, descriptive |

Structural features beat the additive baseline by **59.8% RMSE**. The
additive model is *worse than the global mean*, which is itself informative:
source and target main effects alone do not explain this matrix, so target
difficulty is not the whole story.

### What this does and does not establish

It establishes **viability**: paired transfer is measurable at this scale,
reproducible across seeds, and predictable on held-out pairs by a model whose
features exist for unseen pairs.

It does **not** establish primitive-level developmental structure. Only 2 of
12 directed pairs are primitive-disjoint, so the structural features rest on
almost no contrast and could reflect those two pairs rather than a general
relationship. A broader substrate with deliberately primitive-disjoint
families is required before any such claim, and that is rigor-pass work.


## 14. Natural-corpus pilot: 20 Newsgroups, proposal frozen

Pipeline run in the required order, with official newsgroup labels excluded
from every step that could influence discovery.

    fetch 18,846 -> dedup 559 (3.0%) -> 18,287 unique
      -> document split BEFORE proposal: train 12,801 / val 2,743 / test 2,743
      -> TF-IDF + LSA + k-means (k=6) fitted on TRAIN ONLY -> frozen
      -> val and test assigned by the frozen proposer
      -> support audit

### Support audit

| family | train docs | train tokens | held-out docs | usable |
|---|---|---|---|---|
| 0 | 319 | 49,841 | 80 | **no** — 49,841 < 200,000 |
| 1 | 6,910 | 788,953 | 1,489 | yes |
| 2 | 2,328 | 755,419 | 462 | yes |
| 3 | 508 | 204,445 | 89 | yes |
| 4 | 2,005 | 387,654 | 454 | yes |
| 5 | 731 | 148,045 | 169 | **no** — 148,045 < 200,000 |

**Four usable families** of six proposed. The audit did its job: families 0
and 5 lack the training tokens to serve as a source, and running transfer on
them would have produced cells measuring corpus scarcity rather than
developmental effect.

Family sizes are also highly unequal — family 1 holds 6,910 documents against
family 3's 508 — which is a real property of k-means on this corpus and
becomes a nuisance variable the exposure matching must absorb.

### Frozen pair split

12 directed pairs over the four usable families: **9 observed, 3 held out**,
frozen to `artifacts/natural_pilot/frozen_pairs.json` **before any transfer
run**, so held-out prediction cannot be evaluated against pairs chosen after
seeing results.

### Descriptive sanity check (labels never used in proposal)

| proposed family | top official groups | purity |
|---|---|---|
| 1 | 8, 7, 12 | 0.09 |
| 2 | 17, 11, 16 | 0.18 |
| 3 | 15, 0, 19 | **0.62** |
| 4 | 2, 5, 1 | 0.23 |

Only family 3 aligns strongly with an official group. The rest are **not**
topic clusters in the newsgroup sense, which is the honest reading: the
proposer found lexical structure that cuts across the official taxonomy.
That is neither good nor bad for the developmental question — the families
only need to be distinguishable and learnable — but it does mean these are
not "topics" and should not be described as such.

## 15. Natural-corpus transfer path (prespecified before results)

Written before any natural transfer result was inspected. One unit — the
`f2 -> f1` smoke — had been run at this point, which fixes no between-pair or
seed quantity.

### Recovered provenance

The 20NG pilot had no committed producer. `scripts/run_natural_pilot.py` is
now that producer, and it **verifies rather than regenerates**: the frozen
document split, family assignment and pair split are frozen scientific
objects, so the script asserts reproduction and refuses to overwrite.

The historical ingestion path was recovered exactly:

    fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
      -> sklearn's own defaults, shuffle=True and random_state=42, which fix
         the document order the split permutation consumes
      -> deduplicate 18,846 -> 18,287 (559 removed)
      -> split_documents(seed=0, fractions=(0.7, 0.15, 0.15))

Thirteen assertions pass: dedup statistics, id-set equality, no remaining
content duplicates, exactly one split per document, exact train/val/test
membership, and all six per-family support-audit token counts.

### Two token units, kept apart

* **Support-audit tokens** — plain `str.split()`, committed as
  `dsi.natural.whitespace_tokens`. The unit the frozen 200,000-token
  source-support gate was computed in. **Not redefined.**
* **LM training tokens** — ids from a train-only vocabulary, one per
  whitespace piece plus one `EOS` per document. The unit the transfer
  intervention matches exposure in, asserted equal per unit.

Because ids are assigned per whitespace piece, the two units differ only by
one token per document, and vocabulary size cannot change how much text an
arm was exposed to.

### Regime

| quantity | value | why |
|---|---|---|
| vocabulary | 8,192 types, train-only, `min_count=2` | UNK rate 0.16-0.26 by family, recorded not optimized |
| `seq_len` | 128 | |
| chunks per phase | 1,536 (196,608 LM tokens) | family 3 supplies 1,601 train chunks; the largest round budget every usable family serves in a **single pass** |
| batch | 16, so 96 steps per phase | |
| model | d64 / l4 / h4, lr 3e-3 | the Layer-1 shape, reused rather than recalibrated |
| eval | 256 val chunks of the target family, fixed across arms and seeds | |
| seeds | 3000-3002 | minimum viable replication |

Single-pass exposure is deliberate. The usable families differ in supply by
nearly 4x (788,953 against 204,445 audit tokens), and cycling a small family
would make the transfer estimate partly a measurement of how many times its
documents were seen.

### The control, and a deliberate strengthening

The neutral control excludes the target **and the source**: it is a balanced
mixture of the two families that are neither. Excluding only the target would
leave source material in the control arm and dilute the very contrast the
unit exists to measure. Its chunks are **interleaved, not blocked**, because
a blocked mixture is itself a two-phase curriculum and Layer 1 established
that abrupt isolated phases produce catastrophic interference.

**What this makes the estimand, recorded before results.** Because `N_ij`
excludes both source and target, the control background is **specific to the
pair `(i, j)`**, not common to the target. The measured quantity is therefore
the effect of source `i` *relative to the complementary background for that
pair*, and not the effect of source `i` relative to a common target-specific
control.

This is internally valid for the pilot and is the choice that avoids source
contamination of the control arm. It carries one cost that must not be
forgotten when the matrix is read: a pair-dependent control can itself
contribute relational structure, because the background differs across cells
in a way that depends on which families the cell leaves out. A predictive
feature could in principle be reading the control's composition rather than
the source's effect.

**Rigor-pass debt: a common-control comparison.** Re-running a subset against
a single target-specific control `N_j` shared across sources would separate
the two readings. Not run today, and the pilot's conclusions are stated
against the pair-specific estimand only.

### Prespecified measurability gate

Between-pair spread of mean `T_aulc` against median within-pair seed sd, the
same quantity the synthetic pilot reported as 4.37.

* **S/N >= 2.0** — proceed to the predictive ladder.
* **S/N < 2.0** — stop before prediction. The ladder is not fitted on noise
  as ceremony.

A failure below the threshold is **not automatically an invalidator**, and
the two cases it can mean are diagnosed rather than assumed:

* **excessive within-pair variance** — cells do not reproduce across seeds.
  That is a measurement or statistical invalidator, and it licenses no
  conclusion about natural developmental structure in either direction.
* **small between-pair spread with reproducible cells** — the apparatus
  works and the effects are simply close together. That is a **valid
  negative result**: weak or absent relational structure *at this scale*,
  reportable as such.

The numeric threshold is fixed at 2.0 and is not revised after results are
seen.

### Predictive ladder, three parameters at most

Nine observed pairs, so:

    global mean (1)
      -> additive source + target (7)
      -> intercept + centroid cosine (2)
      -> intercept + cosine + KL(target || source) (3)

Cosine is symmetric and cannot by itself express direction; the unigram KL is
asymmetric, which is what allows `i->j` to differ from `j->i` at all. Both are
computed from **train documents only** under the frozen family assignment;
nothing is refitted or reclustered. Pair-identity coefficients are excluded by
construction, since they cannot predict a pair never run.

The split unit is the **pair**: all three seeds of a held-out pair are held
out with it. The three held-out pairs were frozen before any natural transfer
run.

### The four-family role conflict, recorded

With four usable families there are exactly **12 directed pairs**, frozen as 9
observed and 3 untouched. Those 3 are the only source-to-target pairs the
apparatus has never seen, and they cannot serve two roles at once:

* as an **independent batch test set**, all 3 are spent at once on validating
  the predictor, leaving no unobserved pair for adaptive selection;
* as an **adaptive candidate pool**, they are consumed one at a time by the
  selector, and no clean 3-pair batch test remains.

**The pilot cannot support both.** This is a structural limit of a four-family
universe, not a choice between analyses, and it is recorded rather than
resolved: the rigor pass needs a larger family substrate so the batch test set
and the adaptive candidate pool can be disjoint. The same enlargement is
already owed for primitive-disjoint contrast on the synthetic side.

**Resolution for this pilot: the 3 untouched pairs are the adaptive pool.**
Model comparison happens entirely inside the 9 observed identities, by
leave-one-pair-out. The frozen 9/3 identities are unchanged and no untouched
pair is run for batch validation.

### Adaptive protocol, frozen before any untouched pair is observed

1. **Model comparison by LOPO** within the 9 observed pairs only. Each
   observed pair is held out in turn, with all its seeds, and the ladder is
   refitted on the remaining 8. This is what decides whether the relational
   predictor beats global and additive baselines.
2. **Fit the selected predictor on all 9** observed pairs.
3. **Freeze predictions and acquisition scores for all 3 untouched pairs**,
   written to disk *before* any of them is run, so a prediction cannot be
   revised after seeing an outcome.
4. **The selector chooses one** by the acquisition rule below.
5. **Run the selected pair.** That is the Stage-5 pilot milestone: a
   previously unobserved natural intervention chosen by the fitted
   developmental model rather than by the experimenter.
6. **Compare observed against the frozen prediction**, update the model, and
   choose among the remaining two for a first sequential closed loop.

**Acquisition rule, frozen now.** Research plan 30 version 1, `e* = argmax
U_e`, with `U_e` the predictor's leave-one-pair-out standard error at the
candidate's feature location — the pair the model is least able to predict.
Importance `I_e` and cost `C_e` are uniform across the three candidates: all
three are one directed pair over the same families at the same budget, so the
budget-aware form reduces to version 1 exactly. Ties broken by the lower
`(source, target)` in sorted order, so the rule is total and cannot be
steered after the fact.

## 16. Natural transfer pilot result

27 units, 9 observed pairs x 3 seeds, single-pass exposure, 196,608 LM tokens
per phase per arm, exposure equality asserted per unit.

### Measurability: the gate passes, and the components pass far more clearly

| metric | between-pair spread | median seed sd | max seed sd | S/N |
|---|---|---|---|---|
| `T_aulc` | 0.0669 | 0.0289 | 0.0698 (`f1->f3`) | **2.31** |
| `T_aulc_rate_only` | 0.1593 | 0.0222 | 0.0446 | **7.18** |
| `head_start` | 0.2080 | 0.0237 | 0.0807 | **8.79** |
| `endpoint` | 0.0393 | 0.0291 | 0.0648 | 1.35 |

`T_aulc` clears the prespecified 2.0 threshold. No cell is anomalously
unstable — the worst is 2.4x the median, against the synthetic pilot's 8x
outlier at `F6->F5` — so this is not a case of one hot cell carrying the
noise.

**The composite is by far the weakest of the four.** Head start and rate-only
carry **opposite signs in 7 of 9 pairs**, and both are three to four times
more reproducible than their sum. `f3->f4` is the clearest case: a head start
of -0.3474 and a rate-only effect of +0.2864 cancelling into an AULC of
-0.0609. A single AULC number would have reported a marginal effect at
S/N 2.31 while concealing two strongly measurable opposing mechanisms. This
is the sharpest vindication yet of making the `t=0` evaluation mandatory back
in Milestone A.

The consistent shape: prior exposure to one family leaves the model **worse
at the target immediately** and **faster to acquire it afterwards**.

**Caveat against over-reading that.** Endpoint effects are small and barely
reproducible (S/N 1.35), so the arms largely converge by the end of the
phase. A rate advantage that mirrors a starting deficit is partly mechanical
— an arm that starts further away has more room to descend toward a shared
asymptote. A second confound: the treatment prefix is a **single** family
while the control prefix is a **two-family mixture**, so the arms differ in
source diversity as well as source identity. Both belong to the rigor pass.

### Prediction: the relational features fail, and the structure is a source main effect

Leave-one-pair-out over the 9 observed identities, pair as split unit:

| model | params | in-sample | LOPO RMSE | MAE |
|---|---|---|---|---|
| global mean | 1 | 0.0630 | 0.0709 | 0.0530 |
| **source-only** | 4 | 0.0316 | **0.0498** | 0.0408 |
| target-only | 4 | 0.0567 | 0.0878 | 0.0620 |
| additive source + target | 7 | 0.0039 | **0.0483** | 0.0292 |
| intercept + centroid cosine | 2 | 0.0625 | 0.0884 | 0.0642 |
| + KL(target \|\| source) | 3 | 0.0565 | 0.1166 | 0.0895 |

**This is the opposite of the synthetic pilot's result.** There, additive was
worse than the global mean and structural features beat it by 59.8%. Here the
relational features are worse than the global mean, and the additive family
wins.

Within the additive family, the signal is **entirely in the source**.
`source_only` reaches 0.0498 with four parameters; `target_only` is worse than
the global mean; the full additive model buys 3% more RMSE for three more
parameters, fitting 7 parameters on 8 points under LOPO with an in-sample
RMSE of 0.0039 that is near interpolation. The parsimonious reading is
`source_only`: **which family you train on first matters, and it matters
about equally whatever you train on next.**

That is a real and reproducible effect. It is a **main effect, not relational
developmental structure**, and it does not distinguish this result from
"some corpora are simply worse to train on first".

### Gate verdict

* **Measurability — PASS.** Not an invalidator and not a weak-structure
  negative: effects are reproducible and separated.
* **Prediction — FAIL.** The criterion is held-out signal meaningfully useful
  over the *simple baselines*, and the winning model is itself a simple
  baseline. The relational predictor is beaten by the global mean.
* **Adaptive step — not licensed.** Selection driven by this predictor would
  exercise the machinery on main effects while testing nothing about
  developmental structure.

Predictions and acquisition scores for the 3 untouched pairs were frozen to
`artifacts/natural_transfer/frozen_predictions.json` before the verdict and
are annotated with it. **No untouched pair has been run**, so the adaptive
candidate pool is intact.

## 17. Prospective validation on the 3 untouched pairs

36 units total. Predictions for all 6 rungs on all 4 metrics were frozen,
hashed (`e60bc265d76e...`) and timestamped `2026-08-16T21:16:32Z` before any
untouched pair ran; the scorer re-derives the hash before comparing. The 3
pairs differ from the first 9 in pair identity alone.

### Primary: does the source-only AULC pattern generalize? **No.**

Observed: `1->4` -0.0453 +- 0.0541, `3->2` +0.0457 +- 0.0051, `4->1` -0.0867 +- 0.0213.

| model | LOPO (frozen) | prospective RMSE | MAE | sign |
|---|---|---|---|---|
| global mean | 0.0709 | 0.0699 | 0.0529 | 2/3 |
| **source-only** | **0.0498** | **0.0711** | 0.0682 | 2/3 |
| target-only | 0.0878 | 0.0984 | 0.0782 | 2/3 |
| **additive** | **0.0483** | **0.1434** | 0.1283 | 3/3 |
| cosine | 0.0884 | 0.0634 | 0.0485 | 2/3 |
| cosine + KL | 0.1166 | 0.0394 | 0.0362 | 2/3 |

**The source-only pattern does not generalize.** At 0.0711 it is worse than
the global mean (0.0699). The provisional explanation carried out of the
9-pair analysis fails its own prospective test.

**Additive was an interpolation artifact, decisively.** Best LOPO (0.0483),
worst prospective (0.1434) — three times worse than the global mean. The
diagnostic cells the divergence was flagged on settle it:

| pair | observed | source-only | additive |
|---|---|---|---|
| `1->4` | -0.0453 | -0.0859 (err -0.0406) | -0.2173 (err **-0.1720**) |
| `4->1` | -0.0867 | -0.1633 (err -0.0766) | -0.2620 (err **-0.1753**) |

7 parameters on 8 LOPO points with an in-sample RMSE of 0.0039 was
interpolation, and it extrapolated to roughly three times the true magnitude.
Its 3/3 sign accuracy against the worst RMSE is the signature: direction
captured, magnitude fabricated.

**The LOPO ranking inverted prospectively.** The two best LOPO models are the
two worst prospectively; the worst LOPO model scores best. At this scale
**model selection itself is unreliable**, which is a stronger and more
uncomfortable finding than any individual model's failure. No conclusion is
drawn from the relational model's 0.0394: it is three points, its advantage
comes mostly from shrinking toward the mean, and it still gets `3->2`'s sign
wrong.

**`3->2` is the informative cell.** Observed +0.0457, and every model except
additive predicts negative. It pairs with `f2->f3` (+0.0493) from the
observed nine: families 2 and 3 help each other in **both** directions, the
only mutually positive pair in the matrix.

### Exploratory: 12-pair component LOPO

Hypothesis generation only. **Cannot rescue the failed primary AULC gate.**

| metric | global | source-only | target-only | additive | cosine | cosine+KL |
|---|---|---|---|---|---|---|
| `T_aulc` | 0.0697 | **0.0598** | 0.0951 | 0.0889 | 0.0745 | 0.0668 |
| `head_start` | 0.2213 | 0.2385 | 0.3040 | 0.3642 | 0.2470 | **0.1766** |
| `rate_only` | 0.1741 | 0.1795 | 0.2381 | 0.2758 | 0.1927 | **0.1415** |
| `endpoint` | 0.0442 | 0.0263 | 0.0604 | 0.0330 | 0.0464 | **0.0263** |

Relational features do best on the two **components** (+20.2%, +18.7% over
global) and worst on the composite. Additive collapses below global on every
metric once 12 pairs constrain it.

### A confound that undermines the one exploratory signal

With four families, `N_ij` is the complement of `{i, j}`, and the centroid
cosine is symmetric, so **both are functions of the unordered pair**. Checked
directly: 6 distinct cosine values, 6 distinct control compositions, a
one-to-one mapping.

**The cosine feature is perfectly confounded with the identity of the two
families in the control arm.** Any predictive value it shows may be a
control-composition effect rather than a source-target relationship. This
converts the common-control comparison from rigor-pass debt into the
load-bearing next test. Directionality is unaffected — `N_ij = N_ji`, so
`i->j` against `j->i` is clean.

### 1. What the experiment demonstrated

* Natural-corpus transfer is **measurable and reproducible** at this scale:
  `T_aulc` S/N 2.31, components 7.18 and 8.79, no anomalous cell.
* The `t=0` decomposition is **load-bearing**: head start and rate-only
  oppose in 7 of 9 observed pairs and are 3-4x more reproducible than their
  sum.
* **No model predicts held-out natural transfer.** Source-only and additive
  both fail prospectively; additive is a demonstrated interpolation artifact.
* **LOPO does not select models reliably at n=9-12.** The ranking inverted.
* The apparatus is sound: frozen provenance, asserted exposure matching,
  shared initialization, sealed test split.

### 2. What remains a hypothesis

* That the **component vector** carries structure the composite destroys.
  Suggested by S/N and by the exploratory table, never tested confirmatorily.
* That **semantic families are not developmental families**. Not concluded —
  the semantic features are confounded with the control, so their failure on
  AULC and their modest success on components are both uninterpretable as
  evidence about ontology.
* That families 2 and 3 are **mutually facilitating**. Two cells.
* That the negative head start / positive rate pattern is developmental
  rather than mechanical regression toward a shared asymptote. Endpoint
  effects are small (S/N 1.35), which is consistent with either.

### 3. Proposed next experiment: **H3, common control**

Re-run the 12 directed pairs against a **common target-specific control
`N_j`**, shared across all sources of a given target, holding tokenizer,
budgets, architecture, seeds and schedule fixed.

It is recommended over H2 and H1 because it is the only one that is currently
**blocking**: the cosine/control confound means the exploratory component
signal — the sole surviving lead, and the entire evidential basis for H2 —
cannot be interpreted until the control no longer varies with the unordered
pair. Running H2's vector-response test first would build a confirmatory
design on a feature whose meaning is unresolved, and H1 would rebuild the
ontology using relational evidence that is currently uninterpretable.

It also costs less than it appears: a common control is computed **once per
target**, so 12 pairs need 12 treatment arms and 4 control arms rather than
24 arms — 16 trajectories per seed against 24, cheaper than the wave already
run.

And it yields fresh data. The component hypothesis (H2) has been examined
exploratorily on these 36 units and cannot be confirmed on them; the
common-control wave produces an independent 12-pair matrix on which the
component vector can be prespecified as the primary response and tested
confirmatorily.

**What it primarily tests:** whether the apparent relational structure is a
property of the source-target relationship or of the background composition —
that is, whether the control design (H3) was generating the structure. It
secondarily supplies the clean dataset H2 requires.

## 18. H3: the common-control estimand, frozen before the run

The direct test of whether the apparent relational structure was generated by
the control design. **Every dimension except the control definition is
unchanged**: the same four frozen families, the same train-only 8,192-type
vocabulary, the same d64/l4/h4 model at lr 3e-3, the same 1,536-chunk
(196,608 LM token) phases at batch 16, the same seeds 3000-3002, the same 11
evaluation offsets including the mandatory `t=0`, and the same primary and
secondary response definitions.

### The estimand

For target `j`, the control `N_j` is a **balanced, interleaved mixture of all
three families other than `j`**, at the same 1,536-chunk budget: 512 chunks
from each. Every source `i` is compared against this same `N_j`.

    T_ij = AULC( L_j^{N_j} ) - AULC( L_j^{(i)} )

**The critical invariant is that `N_j` does not depend on `i`.** It is
enforced structurally rather than by convention: the control stream is
computed by a function whose only arguments are the target and the seed, its
trajectory is cached once per `(j, seed)` and reused by all three sources,
and each unit records a hash of the control stream so cross-source identity
is checkable after the fact.

### What this control necessarily contains, stated plainly

With four families, the only control that is independent of `i` and excludes
the target must draw from all three non-target families — **including the
source**. So `N_j` contains source-family material: one third of its chunks
are drawn from family `i` whenever `i` is the source.

This is a real dilution of the contrast, and it is documented rather than
engineered around. **No post-hoc neutral pool is constructed.** Manufacturing
a source-free common control would require either a fifth family held out for
the purpose, or a control whose composition varies with `i` — which is
precisely the property under test. The dilution is the honest price of the
invariant, it is identical across all sources of a given target, and it
biases `T_ij` toward zero rather than toward apparent structure.

The complementary-control matrix already run is retained unchanged. The two
designs are compared, not substituted.

### Prespecified readings of the outcome

* **Component relational structure survives under common controls** —
  H2 / state-space response becomes the leading hypothesis, and the component
  vector is prespecified as the primary response for the next design.
* **Measurable effects survive but relational structure disappears** —
  H1 / ontology rises: the families are measurable but not relationally
  predictive, and learner-dependent family discovery is motivated.
* **Much of the transfer structure itself disappears** — the original
  complementary-control construction is recorded as **the primary source of
  the apparent structure**, and the natural pilot's relational readings are
  withdrawn rather than merely qualified.

The ontology is **not** rebuilt on the current exploratory component signal.
That signal is confounded with control composition and is exactly what this
run exists to disambiguate.

## 19. Corpus v2 intake: a substrate that can host disjoint pools

The four-family 20NG universe cannot support disjoint observed, held-out and
adaptive pools (section 15). WikiText-103 raw, run through the **same frozen
pipeline order** — dedup, then split, then a proposer fitted on train alone —
and audited against the **unchanged** 200,000 support-audit-token gate:

    900,675 lines -> 14,900 articles -> 14,553 with >= 200 tokens
      -> dedup 62 removed (0.43%) -> 14,491 documents, 50,447,749 tokens
      -> split BEFORE proposal: train 10,144 / val 2,174 / test 2,173

| k | usable | directed pairs |
|---|---|---|
| 8 | **8 / 8** | 56 |
| 10 | 9 / 10 | 72 |
| 12 | 11 / 12 | 110 |

50.4M support-audit tokens against 20NG's 2.3M. At `k=8` every proposed
family clears the gate, and 56 directed pairs is enough for an observed set,
an independent held-out validation set and an untouched adaptive pool to be
**genuinely disjoint** — the role conflict that capped the 20NG pilot.

Recorded nuisance, not tuned: family sizes are very unequal, and at `k=12`
family 0 holds 18.5M training tokens against family 5's 56,366. Article
lengths vary far more than newsgroup posts, so exposure matching carries more
of the load here than it did on 20NG.

**No transfer has been run on this substrate.** Intake and support audit
only, so that the family proposal is frozen before any outcome is visible.
