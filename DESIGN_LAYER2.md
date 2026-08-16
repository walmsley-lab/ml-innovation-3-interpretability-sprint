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
