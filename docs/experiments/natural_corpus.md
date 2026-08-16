# Natural-corpus experiments

Stage-5 work on arbitrary natural corpora: the 20 Newsgroups pilot, its
transfer matrix and prospective validation, the H3 common-control test, and
the frozen WikiText protocol.

Split out of `DESIGN_LAYER2.md` during a documentation restructure. Section
numbering is continuous with `layer2_synthetic.md`, which holds sections 0-13,
because the sections are cross-referenced by number throughout the codebase
and the commit history. **No content was altered in the move.**

---

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

## 20. H3 result (frozen)

36 treatment units over 12 directed pairs x 3 seeds, against 12 common
controls, run on `dsi-cpu-bench`. The complementary design was re-run in the
**same environment** as the comparator, because the original matrix ran
locally under a different jax version and OS; the local matrix is retained
as a cross-environment check, never as the comparator.

**1. The common-control invariant is verified.** All 12 target x seed groups
pass, checked from the control-stream hashes recorded in the artifacts rather
than trusted from the code path: 3 sources each, exactly 1 distinct hash.
`N_j` provably did not vary with source.

**2. Gross transfer structure survives, but AULC measurability is weak.**

| metric | spread | median sd | max sd | S/N |
|---|---|---|---|---|
| `T_aulc` | 0.0416 | 0.0208 | 0.0757 | **2.002** |
| `head_start` | 0.1392 | 0.0383 | 0.0612 | 3.64 |
| `rate_only` | 0.1188 | 0.0259 | 0.0493 | 4.59 |
| `endpoint` | 0.0292 | 0.0178 | 0.0919 | 1.65 |

S/N fell from 2.31 to 2.002, clearing the prespecified 2.0 threshold by
0.002. Reading 3 is excluded — the structure did not disappear — but the
honest description is **effects survive, weakened**, and a margin of 0.002 is
not a comfortable pass.

**3. AULC relational prediction remains effectively absent.** 0.0428 against
the global mean's 0.0434 is a **1.5%** gain, which is nothing. The failed
primary AULC relational gate stays failed.

**4. The component relational signal survives the control repair almost
unchanged.**

| metric | global | relational | gain | under complementary control |
|---|---|---|---|---|
| `head_start` | 0.1454 | 0.1167 | **19.8%** | 20.2% |
| `rate_only` | 0.1241 | 0.1003 | **19.2%** | 18.7% |

This is the informative result. The live alternative explanation was that the
component signal came from the cosine/control confound; breaking that
confound left it essentially where it was.

**5. This promotes the component / state-space response to the leading
hypothesis. It does not establish prospective relational structure.** The
evidence is a LOPO advantage at n=12, and today's prospective pass
demonstrated that this exact metric at this exact scale **inverts**: additive
went from best LOPO to worst prospective. The component signal has survived a
confound test, which is what it was asked to do. It has not survived a
prospective test, which nothing here attempts.

Reading 2 (H1 / ontology) is weakened but not eliminated: the AULC relational
signal is still absent, and only the components carry any.

## 21. WikiText v2 protocol (prespecified, no transfer outcome exists)

### Outcome-blind audit, k=8

| fam | train docs | audit tok | chunks | val chunks | UNK | med len |
|---|---|---|---|---|---|---|
| 0 | 997 | 3,010,425 | 23,526 | 4,611 | 0.112 | 2,400 |
| 1 | 408 | 647,413 | 5,061 | 1,102 | 0.103 | 1,085 |
| 2 | 629 | 2,568,054 | 20,067 | 6,032 | 0.116 | 3,047 |
| 3 | 411 | 1,491,233 | 11,653 | 2,660 | 0.123 | 3,122 |
| 4 | 719 | 1,751,525 | 13,689 | 2,725 | 0.117 | 2,084 |
| 5 | 384 | 833,651 | 6,515 | 1,691 | 0.084 | 1,692 |
| 6 | **6,126** | **23,942,169** | 187,096 | 37,843 | 0.136 | 2,992 |
| 7 | 470 | 1,182,302 | 9,240 | 1,835 | 0.101 | 1,811 |

**Nuisance, recorded not tuned.** Size imbalance is **37x**, far worse than
20NG's 4x, and family 6 alone holds 60% of the training documents. A cluster
that large is likely heterogeneous, and it should be treated as a suspected
catch-all rather than assumed to be a coherent family. Vocabulary coverage is
better than 20NG (UNK 0.084-0.136 against 0.16-0.26), as expected from
encyclopedic prose.

**Runtime.** Family 1 binds at 5,061 train chunks, so the largest single-pass
phase budget every family can serve is **4,992 chunks = 638,976 LM tokens**,
3.25x the 20NG budget. 56 directed pairs; the common-control design needs 56
treatment plus 8 control arms per seed, so 64 trajectories per seed against
the paired design's 112.

### Response: a prospectively frozen multicomponent vector

The primary response is the **2-vector** `(head_start, rate_only)`, frozen
before any outcome. `T_aulc` and `endpoint` are recorded and reported as
secondary.

**Both components are observable proxies, not established latent-state
variables.** They are functionals of the loss curve. Promoting them to
coordinates of a developmental state is the hypothesis under test, not an
assumption of the design, and two alternative readings remain open: a rate
advantage mirroring a starting deficit may be mechanical regression toward a
shared asymptote, and endpoint effects have been weak throughout (S/N 1.65
under the common control). The protocol must not describe them as state
variables in any output.

### Material-improvement rule, prespecified

Today a model "beat the global mean" by **1.5%** and triggered a
relational-success reading. That must not be possible again.

1. **LOPO is a selection device only.** It may choose at most one model to
   freeze. It never constitutes success. Today's prospective pass showed LOPO
   inverting outright at this scale.
2. **Success is prospective**, on a pool frozen before any of it is run.
3. **Material improvement** means prospective RMSE `<= 0.75 x` the global
   mean's prospective RMSE, i.e. a **25% or greater** reduction.
4. **Robust to jackknife**: the 25% must hold with any single held-out pair
   removed, so no one cell can carry the result.
5. **Above the noise floor**: the RMSE reduction must exceed the median
   within-pair seed sd of that response. Predicting below the noise floor is
   not prediction.
6. **Both components**, under the **same** frozen model. Success on one
   component only is reported as a negative with a note, never as success;
   otherwise the vector response becomes two chances at a positive.
7. Sign accuracy is reported and is **not** a criterion. Today additive
   scored 3/3 on sign with the worst RMSE of any model.

**Why 25%.** Today's non-signals sit at 0.5-1.5% and fall far below it.
Today's component LOPO gains of 19-20% also fall below it — deliberately,
because they are LOPO rather than prospective, and the threshold is meant to
demand more than what is currently in hand. The synthetic pilot reached 59.8%
where real structure existed, so the bar is clearable when there is something
to clear.

### Freeze list before any WikiText transfer launches

No transfer runs until all of these are frozen to disk:

* same-environment comparator complete and recorded;
* family audit (above) frozen;
* response definition, primary and secondary;
* model ladder;
* observed / held-out validation / untouched adaptive pools, disjoint;
* success criterion (above).

### A note on the forward rule and H3

The material-improvement rule in section 21 was prespecified **after** H3 ran,
for WikiText. It is **not applied retroactively to H3**, because re-gating a
completed experiment on a criterion chosen after seeing its results is the
move this project's invalidator/falsifier separation exists to prevent. The
H3 verdict stands as recorded in section 20.

It does sharpen what the leading hypothesis has yet to demonstrate. Under the
forward rule the component gains of 19-20% would **not** qualify as material,
so the component signal is: strong enough to survive the confound repair
unchanged, and not yet strong enough to clear the bar the next experiment
must clear. Both facts are part of the record.

### Frozen pair pools

`sha256 548460a776c4aa716cada7d93acf6a29ebc1eb37147231150c6f5eb25e2b3bac`,
seed 20260816, written before any WikiText transfer.

| pool | unordered | directed | role |
|---|---|---|---|
| development | 18 | 36 | fitting and model selection; LOPO lives here only |
| confirmatory | 6 | 12 | batch held-out, never fitted on, scored once |
| adaptive | 4 | 8 | untouched candidates for model-selected execution |

**The partition unit is the unordered pair.** Both directions move together.
Holding out `i->j` while `j->i` sits in development leaks the pair's identity
— the two share families, features and the same unordered-pair nuisance, so a
model fitted on one has effectively seen the other. The 20NG universe could
not provide three disjoint pools at all; this is the structural reason for the
larger substrate.

### Regularized ladder, sized to 36 observed relationships

Ridge throughout, penalty chosen by inner LOPO **inside development only**:

    global mean                                     1
      -> source main effects                        8
      -> target main effects                        8
      -> additive source + target                  15
      -> additive + cosine + KL(target || source)  17

17 parameters against 36 development pairs, regularized, is the ceiling. No
composition-slot interactions, no pair-identity terms, no neural transition
model, no full-trajectory model. Those are responses to specific failure
modes, not prerequisites, and adding them before a clean test would repeat
the additive lesson: 7 parameters on 8 points scored best by LOPO and worst
prospectively.

### Acquisition diagnostics, so the selector cannot pick an extrapolation

Every candidate is scored with, and the frozen record retains:

* **leverage** `x0' (X'X)^-1 x0` at the candidate's feature location;
* **Mahalanobis distance** from the development feature cloud;
* **predicted value against the observed range**, flagged when outside.

**A candidate whose leverage exceeds the 95th percentile of development
leverages is ineligible**, whatever its acquisition score. Today every frozen
additive prediction fell outside the observed range and all three were wrong
by roughly 3x; an uncapped `argmax U_e` selects exactly those points.

### Stopping rules, exact

* **Measurability.** If S/N < 2.0 on **both** components, stop before
  prediction. Diagnose as before: excessive within-pair variance is a
  measurement invalidator; small spread with reproducible cells is a valid
  weak-structure negative.
* **Cell reproducibility.** Any cell with within-pair sd > 4x the median is
  flagged, excluded from fitting, and reported. `F6->F5` in the synthetic
  pilot sat at 8x and should not have been silently carried.
* **Prediction.** If no model meets the material criterion on the
  confirmatory pool, **stop**. No adaptive step, no ladder extension, no
  re-selection on a secondary metric.
* **Adaptive.** If every candidate is leverage-ineligible, stop and report
  that the model cannot select a defensible intervention.

### Curve preservation

The primary response is a pair of summaries, but **complete acquisition
curves for both arms are retained per unit**, as they already are for every
unit run today. The summaries are a projection, the projection is what the
present hypothesis is about, and the projection is exactly what a later
state-space analysis would need to revisit. Discarding curves would make that
revisit impossible.

### The Stage-5 claim, kept narrow

Validated prediction of unseen natural-data developmental interventions,
followed by model-selected execution of **one** untouched intervention.
Nothing about latent state, nothing about causal ontology, nothing about
curriculum control.

## 22. Three outcome-blind design resolutions, before any WikiText transfer

No transfer has been run on this substrate. Nothing below is informed by an
outcome.

### 1. Family imbalance: family 6 is a residual cluster, and is excluded

| fam | docs | tokens | med len | cohesion | margin | top terms |
|---|---|---|---|---|---|---|
| 0 | 997 | 3,010,425 | 2,400 | 0.377 | 0.60 | album, song, music, band |
| 1 | 408 | 647,413 | 1,085 | 0.442 | 0.79 | highway, route, ny, road |
| 2 | 629 | 2,568,054 | 3,047 | 0.291 | 0.52 | army, battalion, aircraft, war |
| 3 | 411 | 1,491,233 | 3,122 | 0.429 | 0.64 | game, player, gameplay, nintendo |
| 4 | 719 | 1,751,525 | 2,084 | 0.375 | 0.63 | episode, series, season, homer |
| 5 | 384 | 833,651 | 1,692 | 0.611 | 0.81 | storm, tropical, hurricane |
| **6** | **6,126** | **23,942,169** | 2,992 | **0.145** | 0.52 | **film, species, new, team, time** |
| 7 | 470 | 1,182,302 | 1,811 | 0.462 | 0.56 | ship, ships, guns, fleet |

Seven families are crisply thematic. Family 6 is not:

* **cohesion 0.145**, less than half the next-worst (0.291) and a third of
  the 0.409 median — its documents are not near each other;
* top terms `film, species, new, team, season, time` span three unrelated
  domains, and its distinctive terms — `kepler, thanhouser, banksia, µm,
  wickets` — span astronomy, silent cinema, Australian flora, microscopy and
  cricket;
* it is the **nearest neighbour of five of the other seven**, the signature
  of a central leftover blob.

It is the residual bin k-means leaves after the coherent structure is carved
off. A source drawn from it measures "generic text" rather than a
developmental family, and including it in a control would make the control
generic.

**Resolution: family 6 is excluded from every role — source, target and
control.** It remains in the corpus and is used for nothing. Seven usable
families, 21 unordered and 42 directed pairs. Re-clustering was rejected: at
`k=12` the method produces the same artifact (family 0 at 4,632 documents and
18.5M tokens), so a residual bin is a property of the proposer, not of `k`.

### 2. Intervention dose: fixed, not corpus-derived

The 638,976-token figure came from taking the phase budget to be *the largest
single full pass the binding family can serve*. That makes treatment strength
a function of corpus size, so a WikiText effect and a 20NG effect would not be
the same intervention, and any difference between substrates would be
partly a dose difference.

**Resolution: the phase budget is fixed at 1,536 chunks = 196,608 LM tokens,
identical to 20NG.** Corpus size now affects **support** — how much headroom
exists, how many families clear the gate, whether exposure stays single-pass
— and not **intervention strength**. Every family still supplies its phase in
a single pass with large margin: the binding family holds 5,061 chunks
against a 1,536 requirement.

There is no principled reason here to define treatment as one full family
pass. "One pass" is a property of the corpus; dose should be a controlled
variable.

### 3. Common-control weighting: equal-family, frozen

`N_j` is an **equal-family** mixture over the six non-target used families,
256 chunks each. Natural-frequency weighting is rejected: at 37x imbalance it
would make the control almost entirely one family, so the control's
composition would be an accident of corpus proportions rather than a
controlled constant, and it would vary in character across targets for
reasons unrelated to the design. Equal weighting keeps the control a fixed,
stated object. (Excluding family 6 removes most of the imbalance anyway; the
weighting is frozen explicitly so it cannot drift.)

### Re-frozen pools

The 8-family partition is **superseded** and retained as
`frozen_pools_k8_superseded.json`. The replacement is built on 7 families:

`sha256 497b9c3fc66e8adbca96ac2eef41e9e2ada14ffcdc0d78bb4cbbf589c42b3c27`

| pool | unordered | directed |
|---|---|---|
| development | 13 | 26 |
| confirmatory | 5 | 10 |
| adaptive | 3 | 6 |

Superseding a frozen partition is legitimate **only** because the defect was
found on outcome-blind grounds and no WikiText transfer has been run. The
supersession, its reason and both hashes are recorded rather than the old
file being deleted.

**Ladder consequence.** With 7 families the additive model is 13 parameters
and the relational model 15, against 26 development relationships. Ridge
throughout, penalty by inner LOPO inside development only. The 25%
material-improvement rule is unchanged and remains **prospective and
WikiText-only**; it is not applied to H3.

## 23. Three clarifications, before transfer starts

### 1. The family-6 exclusion is an amendment, not an algorithm

Excluding family 6 is a **WikiText-specific, outcome-blind protocol
amendment**, justified by a clear and specific residual-cluster pathology:
cohesion 0.145 against a 0.409 median, top terms spanning three unrelated
domains, distinctive terms spanning astronomy, silent cinema, Australian
flora, microscopy and cricket, and nearest-neighbour status to five of the
other seven families.

**This project does not have a general automated residual-family rejection
rule, and nothing here should be read as one.** The decision was a judgement
made by inspecting diagnostics on one corpus, before any outcome existed. It
does not generalize, and it would not survive being applied mechanically to a
corpus whose cluster structure differs.

A future corpus needs a **prospectively defined family-quality gate** —
thresholds on cohesion, distinctiveness and margin, fixed before the
proposal is inspected, in the way the 200,000-token support gate was fixed
before the 20NG audit. Defining that gate is outstanding work, and until it
exists, any family exclusion on a new corpus is a fresh judgement that must
be recorded as one.

### 2. "Single-pass" means no repetition, not a complete pass

The phase budget of 1,536 chunks (196,608 LM tokens) is drawn from families
holding far more:

| family | available chunks | used | fraction |
|---|---|---|---|
| 1 (binding) | 5,061 | 1,536 | 30.3% |
| 5 | 6,515 | 1,536 | 23.6% |
| 7 | 9,240 | 1,536 | 16.6% |
| 3 | 11,653 | 1,536 | 13.2% |
| 4 | 13,689 | 1,536 | 11.2% |
| 2 | 20,067 | 1,536 | 7.7% |
| 0 | 23,526 | 1,536 | 6.5% |

**"Single-pass" means no chunk is seen twice.** It does **not** mean the
source phase traverses the family. A treatment is 1,536 distinct chunks
sampled once, without replacement, from a pool three to fifteen times larger,
taken as a prefix of a seeded document shuffle.

Two consequences, recorded as nuisances rather than corrected:

* **The sampled fraction varies 4.7x across families.** A source drawn from
  family 1 covers 30% of it; one drawn from family 0 covers 6.5%. How
  representative a treatment is of its family therefore differs by family.
* **Which slice is drawn varies with the seed**, so within-pair seed variance
  includes the sampling variance of which documents were seen. That is
  properly part of the intervention's variability rather than measurement
  error, but it means seed replication is doing two jobs at once.

### 3. What the 25% criterion must beat

**A relational-developmental claim must beat the best *simpler* model, not
merely the global mean.**

    simpler models = {global, source_only, target_only, additive}
    required: RMSE(relational) <= 0.75 x min RMSE over the simpler models

A relational model that beats the global mean while a source-only model
predicts better has demonstrated that something about the source matters —
a main effect — and nothing whatever about the relationship between source
and target. The weaker "beats global" form would let precisely that pass, and
on this project it nearly did: on the same matrix where the relational model
reached a 1.5% gain over global, source-only reached 13.8%.

All conditions must hold together, on **both** components, under the **same**
model, prospectively on the frozen confirmatory pool:

1. RMSE <= 0.75 x the **best simpler model**, not the global mean;
2. the margin survives leaving out any single confirmatory pair;
3. the reduction exceeds that component's within-pair seed-noise floor.

Implemented in `scripts/success_criterion.py` so the gate lives in code
rather than only in prose, with a regression check confirming that
"relational beats global but loses to source-only" **fails**.

### Frozen and not to be changed once outcomes begin

The seven-family proposal, equal-family control weighting, the fixed
196,608-token dose, and the hashed 13/5/3 unordered-pair pools
(`497b9c3fc66e8adbca96ac2eef41e9e2ada14ffcdc0d78bb4cbbf589c42b3c27`) are
frozen. None of them may be revised once any developmental outcome exists on
this substrate.
