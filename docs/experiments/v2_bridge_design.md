# V2 bridge: mechanistic identification in a language micro-world

**Status: design proposal. Nothing here is implemented, frozen, or licensed.**
No result in `report.md`, `RESULTS.md` or `docs/experiments/` is revised by this
document. The synthetic Layer-2 viability result stands exactly as recorded,
including its structural limit (2 of 12 directed pairs primitive-disjoint).

---

## 0. Why this experiment, stated honestly

The motivation is **not** that natural-corpus prediction failed. That reading
is available in the record but is wrong on the facts:

| substrate | measurable? | prospectively predictive? |
|---|---|---|
| synthetic Layer 2 | yes, S/N 4.37 | held-out RMSE 0.271 vs additive 0.674 (pilot) |
| 20 Newsgroups | yes, S/N 2.31 | **no** — relational lost to global mean, LOPO inverted |
| WikiText-103 | yes, S/N 5.95–8.32 | **yes on both components** (+30.0%, +37.7%), failed jackknife robustness only |

The honest summary is that transfer is measurable everywhere, weakly and
inconsistently predictable, and — per the `r = 0.20` overlap diagnostic,
Branch 2 — **does not compose into multi-stage orderings** (best − reverse
= −0.0383 against pooled seed sd 0.2556, effect/noise −0.15, with retention
restored and dose decoupled from order).

The actual gap is different and larger than any of those:

> **Every result this project has produced is a correlation between a
> human-supplied family label and a scalar summary of a learning curve. No
> experiment has yet identified an internal mechanism, and no experiment has
> yet run a causal intervention on one.**

One **hypothesis** for why `T_ij` fails to compose is that a scalar edge
between two labels carries no information about *what* was transferred, so
there is nothing for composition to operate on. That is a conjecture, not an
established diagnosis — the observed non-composition is equally consistent
with retention dynamics, dose-position interactions, or there being no
compositional structure to recover at this scale. **V2 tests the conjecture; it
does not presuppose it.** Establishing that a mediator exists and is causally
necessary in one case would make the hypothesis live; it would still not prove
that mediator-blindness is what defeated composition in L0, which would need a
separate test.

The V2 bridge exists to ask whether a specific causal chain can be identified
at all:

```
source corpus  D_A  →  internal mechanism  M  →  increased learnability of  D_B
```

If that chain cannot be established in a setting where we have a strong prior
about what `M` is, it will not be established on an unlabeled corpus where we
do not.

## 1. Layering, restated

| layer | substrate | what it establishes | status |
|---|---|---|---|
| **L0** | synthetic algorithmic families | the instrumentation can distinguish known causal structures | done; treat as validated measuring device, not as an ecological result |
| **L1** | **language micro-world, next-token objective** | a *mechanism* mediates transfer, causally | **this document** |
| **L2** | unconstrained natural corpora | the model's own developmental ontology, inferred rather than supplied | later; frozen designs stay frozen |

L0 is not deprecated and its ontology is **not** to be extended to rescue
anything. The four frozen families, the 12 directed pairs, the pilot result
and its recorded limits are terminal.

## 2. The circularity problem, and the specific fix

The sharpest criticism of L0 is that the world is constructed as

```
primitive ontology → families → training relationships
```

and then asked whether primitive relationships predict transfer. The
hypothesis is partly baked into the substrate. Three properties of the L1
design are chosen specifically to break that loop.

**1. `M` is discovered by the model, not planted by us.** The mediator is an
**induction-style prefix-matching circuit** — a mechanism that arises in
ordinary decoder-only transformers under ordinary next-token prediction, that
nobody labels in the data, and that has an established independent measurement
(per-head prefix-matching score on random repeated-token sequences).

**2. `M` is measured off-distribution.** The prefix-matching probe uses random
token sequences drawn from the vocabulary, **not** micro-world records. The
mediator statistic is therefore not a function of the corpus whose transfer it
is being asked to explain. This is the single most important guard against
measuring our own design back.

**3. There is a pre-registered negative capability.** The design predicts a
*null* as well as a positive. A generic "prior training changes later
training" artifact — optimizer state, capacity competition, distribution shift
— would move both capabilities. Only a mechanism-specific effect moves one.

This does not make L1 ecologically valid. It makes L1 an **instrument
validation for mechanistic mediation**: can the apparatus recover a causal
chain we have reason to expect, and correctly reject the control?

> ### Standing scope statement
>
> Carry this verbatim into any writeup, talk or abstract that reports a V2
> result, including a successful one.
>
> **What a full V2 pass would establish:** that mechanistic developmental
> state can be *identified* — measured during training, shown to mediate a
> transfer effect, and causally verified against a matched control — in a
> controlled language micro-world under an ordinary next-token objective.
> The result is about the **apparatus and the identifiability of
> developmental state**.
>
> **What it would not establish:** that induction-style circuits explain
> dependencies in natural pretraining; that this mediator generalizes to
> other capability pairs; that developmental dependencies are widespread,
> or that they are exploitable for scheduling. `M` is chosen because it is
> the best-characterized mechanism available, which makes it the right
> first target and a poor basis for generalization.
>
> The distance between those two paragraphs is the remaining research
> programme, not a caveat to be softened.

## 3. Substrate

One corpus, one vocabulary, one objective: **full next-token prediction**
(`loss_positions="all"`, already supported in `src/dsi/train.py:114`). No
task heads, no answer-position loss, no MODE tokens.

Vocabulary ≈ 2,048: a small closed English-like function/content lexicon plus
a pool of ~512 nonce nouns (`dax`, `wug`, `blicket`, …) and ~64 attributes.
Sequence length 256.

### Three streams

**`S_ind` — the mechanism-inducing stream (source `D_A`).** Ordinary
micro-world narrative text in which entities recur, and in which the
best available predictor of a recurring span is *prefix matching over the
current context*. Nothing about attribute binding; the entities are just
mentioned and re-mentioned.

**`S_bind` — the target capability `B`.** In-context novel-binding retrieval.
A nonce entity is introduced with an attribute early in the sequence and the
attribute must be retrieved later:

```
... a dax appeared . the dax was green . ... the colour of the dax was ___
```

Nonce identities are **resampled every example**, so the binding cannot be
stored in the weights. `B` is by construction an in-context retrieval
capability, which is why prefix matching should be reusable for it.

**`S_fact` — the negative-control capability `C`.** Parametric recall of a
fixed, globally consistent association held in the weights:

```
... grass was green . ... the colour of grass was ___
```

The association is constant across the whole corpus, so `C` is learnable
purely as a weight-stored bigram-like statistic and should gain nothing from
an in-context retrieval circuit.

`B` and `C` are matched on answer entropy, surface form, position of the
queried token, and query-template distribution. They differ **only** in
whether the answer is recoverable from the context or from the weights.

### The matched-statistics control source `A'` — as implemented

> **Reconciled with `src/dsi/microworld.py` after the shortcut audit. This
> section supersedes the matched-bigram-resampler proposal.**

`S_ind'` draws the **same per-document entity set with the same slot picks**
as `S_ind`, so entity recurrence is identically distributed. The streams
differ in the *binding*: in `A` a recurring entity keeps its value, so the
repeated prefix predicts what follows; in `A'` it draws a fresh value, so
copying from an earlier occurrence predicts nothing.

`A` and `A'` therefore differ in exactly one property: **whether the source
stream rewards prefix matching.**

**Why not the two simpler constructions.** The audit
(`scripts/audit_microworld_shortcuts.py`) tests whether predictors that
*cannot* use long-range context can nonetheless distinguish the streams — if
they can, a model could exploit that instead of the mechanism under test.

- The **bigram resampler** matches marginals only asymptotically and is hard
  to verify. Not implemented.
- **Fresh-entity sampling** (draw every slot i.i.d. so nothing recurs) was
  implemented first and **the audit rejected it**: unigram cross-entropy gap
  0.0060 against a null limit of 0.0030, bigram 0.0117 against 0.0056. The
  cause was 2.88 distinct entities per document in `A` against 7.94 in `A'` —
  removing recurrence necessarily changes entity diversity, and diversity is
  exploitable.
- **Value-rebinding** (implemented) passes: positional gap 0.0003, unigram
  0.0005, bigram 0.0009, all inside their nulls; distinct entities 2.89 vs
  2.89; binding oracle 1.0000 in `A` against 0.0153 in `A'`, where chance is
  0.0156.

**C13, recorded.** The all-position induction oracle scores 0.68 in `A` and
0.55 in `A'` — high in both, because copying an entity token correctly
predicts the structural `REL` token in either stream. That template-copying
component is a large constant common to both arms and is *not* the
manipulation. Any statistic used to measure the mechanism must isolate value
slots, or it will understate the contrast; this is why the gate uses the
binding oracle rather than the all-position one.

**The substrate is frozen at this construction.** The audit is the stopping
rule that licenses freezing it.

## 4. Hypotheses, each stated so it can fail

| id | hypothesis | prediction | fails if |
|---|---|---|---|
| **H2.1** | behavioural transfer, and it is specific | `A→B` advantage > 0; `A→C` advantage ≈ 0 | both move (generic artifact), or neither moves |
| **H2.2** | temporal mediation | `M` rises during `A` **before** `B` accelerates; `M` at end of phase 1 predicts `B` learnability across seeds after partialling out general learning speed | `M` rises after the behavioural gain, or the partial correlation vanishes |
| **H2.3** | necessity, as an **interaction** | targeted `M` ablation eliminates the `A→B` advantage *preferentially* — i.e. it costs the `A` arm more than it costs `A'` and background | `M` ablation hurts `B` roughly equally in all three arms (non-specific damage), or matched-random ablation produces the same interaction, or the `A` advantage survives `M` ablation |
| **H2.4** | architectural dependence, at **matched capacity** | the `A→B` advantage is substantially reduced at depth 1 relative to depth 2 and 4, holding parameter count fixed | depth-1 at matched capacity reproduces the depth-4 advantage |
| **H2.5** | dose / promotion (data-side) | advancing `M` emergence by raising repeat density monotonically advances `B` acquisition | no monotone relation between `M` emergence time and `B` acquisition time |

### H2.3 is a difference-in-differences, not a damage test

Ablating heads hurts a model. "`B` got worse after ablation" is therefore
uninformative on its own, and an earlier draft of this design made that
mistake. The estimand is the **interaction**:

```
Δ  =  [ B(A, no-abl) − B(A', no-abl) ]  −  [ B(A, M-abl) − B(A', M-abl) ]
```

`Δ > 0` says `M` ablation removed specifically the part of `B`-learnability
that came from `A`. The same quantity computed for matched-random ablation is
the specificity control and must be ≈ 0. This requires the ablation to be run
on the **control arms too**, which is why V2-b below is larger than a
single-arm design.

### H2.4, stated at the right strength

A 1-layer transformer cannot implement the canonical two-layer induction
circuit (previous-token head composed with a matching head). It **can** learn
other in-context copying strategies — skip-trigram-like attention, positional
heuristics, weight-stored approximations — so *categorical* impossibility is
the wrong prediction and would be falsified by mechanisms irrelevant to our
claim. The prediction is therefore about **the canonical efficient mechanism
failing at matched capacity**: depth 1 should show a substantially reduced
`A→B` advantage relative to depth 2 and 4 when parameter count is held fixed
(depth-1 models are widened to compensate, so the contrast is depth, not size).

Read the outcome as evidence about *which* mechanism is operating, not as a
binary. A depth-1 model that reproduces the full advantage says the transfer
is carried by something that does not need two-layer composition — which is a
real finding about the mechanism, not merely a null.

### Sufficiency, deferred to the ladder

H2.3 tests necessity and H2.5 tests a **data-side** promotion. Neither is a
direct sufficiency test. The complete causal statement requires

```
do(M+)  →  accelerated B
```

— installing or advancing `M` by intervening on the mechanism itself rather
than on the data, then observing accelerated acquisition of `B`.

**The decisive form of the test is `do(M+)` in a model that never received the
`A` history at all** — graft or induce `M` into a `BG`-only or `A'`-only model
and ask whether `B` then accelerates. Merely holding source exposure constant
is weaker: it leaves open that `A` did several things and `M` is a bystander.
Installing `M` without `A` isolates the mechanism from everything else the
source history changed, and it is the first result that *creates* a
developmental advantage rather than observing or destroying one — the crossing
point from interpretability into control.

Candidate implementations: grafting `M`-scoring heads from a donor checkpoint,
freezing them in place during phase 2, or an auxiliary objective that induces
prefix matching without micro-world content.
**This is not implemented in V2.** Grafting carries its own confound — a
donor's heads arrive with a compatible residual basis, so the transplant
imports more than `M` — and resolving that is a design problem in its own
right. It is recorded here as the next rung so it is not quietly dropped once
necessity passes.

## 5. Observables (the developmental telemetry stack)

Recorded at every telemetry checkpoint (every 50 steps; ~400 per run).

**Behaviour.** Per-capability held-out loss and accuracy for `B`, `C`, and
retention of `A`; `t = 0` evaluation at every phase boundary is **mandatory**
and preserved from the existing protocol. The head-start / rate-only
decomposition is retained per capability — it carried opposite signs in 7 of 9
pairs at 20NG and is 3–4× more reproducible than its sum, and there is no
reason to expect that to stop being true here.

**Optimization.** Per-layer gradient norm and update magnitude; Adam
second-moment norm per layer; **gradient cosine similarity between
capability-specific batches** (`grad(S_ind)` vs `grad(S_bind)` vs
`grad(S_fact)`), which is the observable that distinguishes "A helps B because
their gradients align" from "A helps B despite conflicting gradients, by
building a reusable circuit". Those are different mechanisms and the design
should not be able to confuse them.

**Representation.** Centred kernel alignment between checkpoints and across
seeds; effective rank of the residual stream per layer; linear probe for
"is a binding present in context".

**Circuits — the mediator.** Per-head **prefix-matching score** on held-out
random repeated-token sequences; this is `M`. Also per-head attention entropy,
mean attention distance, and per-head ablation sensitivity on `B`.

> **Freeze before any run:** the definition of the prefix-matching score, the
> probe sequence set and its seed, the value of `k` in "top-`k` heads", the
> aggregation from per-head scores to the scalar `M`, and the emergence-time
> definition (first checkpoint at which `M` crosses a fixed threshold and stays
> above it). Every one of these is a researcher degree of freedom that could
> manufacture H2.2.

## 6. Design and run counts

Base architecture `d_model 256, 4 layers, 8 heads, seq 256`, ≈ 3M
non-embedding parameters. Total token budget identical across every arm, and
per-stream aggregate allocation identical across arms — the L0 constraint
carried forward, since it is the only thing that makes ordering rather than
dose the manipulated variable.

| stage | question | conditions | seeds | runs |
|---|---|---|---|---|
| **V2-0** preflight | is the apparatus sound? **no scientific claims** | see §6.1 | — | **~20** (short, disposable) |
| **V2-a1** scout | H2.1, H2.2 at low cost | 3 arms: `A→B`, `A'→B`, background→`B` | **6 paired** | **18** |
| **V2-a2** expansion | same frozen experiment, powered | identical to V2-a1, seeds 7–12 | +6 | **+18** (36 total) |
| **V2-b** ablation | H2.3 | 3 arms × 2 ablations (`M`-heads / matched-random); no-ablation cells reused from V2-a | 12 | **72** (short) |
| **V2-c** architecture | H2.4 | depth {1, 2, 4} at matched params × arms {`A`, `A'`} | 8 | **48** |
| **V2-d** dose | H2.5 | repeat density {0, 0.05, 0.15, 0.40} | 8 | **32** |
| | | | | **≈ 208 if every gate passes; 38 to the first real decision** |

The number that matters is not 208. It is **38** — preflight plus the scout —
which is what it costs to find out whether the mechanistic bridge exists at
all. Everything after that is conditional.

### 6.1 Preflight (V2-0): burn runs aggressively here

No scientific claim is made from preflight and no preflight run is reported.
Bugs found here are the cheapest bugs in the programme. Seven checks, each
with a pass criterion fixed before it runs:

| # | check | passes if |
|---|---|---|
| P1 | `A` / `A'` match on promised low-order statistics | unigram and bigram marginals, length, and entity-mention counts agree within a frozen tolerance; **repeat structure differs and nothing else does** |
| P2 | `B` / `C` matched except contextual vs parametric retrieval | answer entropy, surface form, query position and template distribution agree; solo `t90` within the L0 difficulty band |
| P3 | the off-distribution `M` metric detects the mechanism | prefix-matching score rises on a model trained on repeat-rich data and stays flat on `A'`-only data; the metric is computed on random token sequences, never on micro-world records |
| P4 | `B` is learnable but does not ceiling | background arm ends with measurable headroom on `B` (the ceiling-scout lesson) |
| P5 | the phase boundary alone does not produce the effect | a `background→background` arm with a boundary shows no `B` advantage; optimizer-state norms recorded across the boundary |
| P6 | determinism | same spec + same seed + same hardware reproduces bit-identically; checkpoint save/load round-trips exactly |
| P7 | ablation machinery | head ablation is applied where intended, matched-random selection is reproducible from a frozen key, and ablating zero heads is a no-op |

**P6 requires checkpoint persistence, which does not exist in this repo
today** (no `tree_serialise`, no Orbax, no weight artifacts in GCS). It is a
blocking prerequisite, shared with the sprint experiment.

### 6.2 The scout gate is four criteria, not one

V2-a1 expands to V2-a2 only if **all four** hold, evaluated on the frozen
statistics:

1. `A` produces substantially more `M` than `A'` — the manipulation worked.
2. `A→B` has a meaningful advantage over `A'→B`.
3. `C` does **not** show the same advantage — the effect is specific.
4. `M` rises **before** the change in `B` acquisition rate, not after.

Criterion 1 is the one most likely to fail quietly and is the cheapest to fix;
criterion 4 is the one most likely to be manufactured by a badly chosen
emergence-time definition, which is why that definition is frozen in §5.

Expansion adds seeds 7–12 to the **identical frozen experiment**. It is not an
opportunity to adjust arms, budget, metric or thresholds. If the scout looks
promising but the design needs changing, that is a new experiment with a new
freeze, and the scout's seeds are spent.

`B` and `C` are evaluated on the same checkpoints, so the negative control
costs **zero additional runs**. The no-ablation cells of V2-b are the V2-a runs
themselves, so the interaction is estimated without paying for that arm twice.
Width is deliberately *not* crossed in the first pass — depth is the
theoretically load-bearing axis and width is a follow-up if depth shows the
predicted pattern.

### Sequential go/no-go staging

The stages are **not** launched concurrently, even though V2-a, V2-c and V2-d
are technically independent. Each later stage is only interpretable if the
earlier one produced a real effect, and launching all of them at once buys
wall-clock at the cost of spending compute on questions that may already be
moot. Every gate is evaluated on the frozen criteria before the next stage is
provisioned.

| gate | after | proceed only if | otherwise |
|---|---|---|---|
| **G0** | V2-0 | all seven preflight checks P1–P7 pass | fix the apparatus. No scout runs until P1–P7 are green; preflight failures are not results |
| **G1a** | V2-a1 (18 runs) | all four scout criteria in §6.2 | **stop.** Report "no mechanism-specific transfer under an LM objective at this scale" — a substantive result that retro-illuminates L0. V2-a2/b/c/d are moot |
| **G1b** | V2-a2 (36 total) | the scout effect survives at 12 seeds against the frozen seed-noise floor (H2.1, H2.2) | the scout was a small-sample artifact. Record it as such; do not re-cut the analysis |
| **G2** | V2-b | the `M`-ablation interaction `Δ > 0` and the matched-random interaction ≈ 0 (H2.3) | stop escalating the mechanism claim. Report transfer as real but mediator unidentified; V2-c/d become optional diagnostics, not claim support |
| **G3** | V2-c | the depth-1 advantage is substantially reduced at matched capacity (H2.4) | the mechanism is not two-layer composition. Retract the induction framing, keep the transfer result, re-open what carries the effect |
| **G4** | V2-d | monotone dose–response of `B` acquisition on `M` emergence time | necessity stands without data-side sufficiency; record the gap and route to the `do(M+)` rung |

**Ordering rationale.** V2-b precedes V2-c because necessity is the pivotal
claim and architecture is elaboration on a claim that may not survive. If V2-b
returns an ambiguous interaction rather than a clean pass or fail, pull V2-c
forward — a depth-1 result can discriminate mechanisms when the ablation
cannot.

### Compute

- ~0.3 TFLOP/optimizer step at batch 64; ~20k steps/run.
- **This is the first experiment in the project that should not run on CPU.**
  The C3 workers are calibrated for `d64/l4/seq32`; this substrate is roughly
  two orders of magnitude more FLOPs per step. Target `pdp-gpu`
  (`g2-standard-4`, L4, currently TERMINATED, disk preserved) at ~$0.85/hr.
- ~20–30 min/full run at concurrency 4; V2-b runs are phase-2 only. Whole
  programme ≈ **20–25 GPU-hours ≈ $20–25** if every gate passes, and
  substantially less if an early one does not — which is a further argument for
  staging. The binding constraint is engineering time, not compute.
- **Within** a stage, runs are fully parallel at concurrency 4. **Across**
  stages, execution is serial behind the gates above.
- **Storage is the real constraint.** Dense full checkpoints would be
  ~2.4 GB/run (400 × 3M × 4B) → ~480 GB across the programme. Mitigation:
  telemetry scalars dense, **full checkpoints only at phase boundaries plus 8
  log-spaced points, and only for V2-a**, which is the only stage V2-b resumes
  from. Orbax becomes "earned" here per `technical.md` §3.

### Engineering delta against the current repo

Smaller than it looks. Already present: decoder-only transformer
(`src/dsi/model.py`), full next-token loss path, dense evaluation hooks
(`train_phase(eval_at=...)`), paired-RNG discipline, content addressing,
frozen-spec machinery. New work: the micro-world generator and its
matched-bigram resampler; `seq_len`/vocab scale-up; the telemetry stack;
head-level ablation; checkpoint persistence; a GPU executor path.

## 7. Confounds — where the design could drive the result

Ordered by how likely each is to produce a false positive.

**C1. Surface / parameter reuse rather than mechanism.** `A→B` could reflect
SGD having already tuned weights useful for overlapping surface statistics.
*Guard:* `A'` holds unigram and bigram statistics fixed and removes only the
repeat structure. This is what makes the contrast interpretable.

**C2. Optimizer state and distribution shift at the phase boundary.** Momentum,
second-moment estimates and abrupt distribution change all have strong causal
effects — the L0 record already shows how severe this can be. *Guards:*
constant learning rate with no warmup restart at the boundary; a
`background→B` arm that experiences a phase boundary without an informative
source; optimizer-state norms recorded as telemetry, not assumed inert.

**C3. Capacity competition at small scale.** At 3M parameters, capabilities may
compete for representational room, producing interference that vanishes at
scale. This is exactly the regime where a 2026 Pythia-scale study found
curriculum effects strongest in small models and diminishing with scale.
*Guard:* the depth sweep, plus an explicit width follow-up if the effect is
found. Any positive result is reported **with its scale caveat attached**, not
as a general claim.

**C4. Ceiling and floor artifacts.** If `B` saturates, transfer collapses into
a timing difference; if `B` never learns, there is nothing to accelerate. *Guard:*
calibrate the phase-2 budget so the `background→B` arm ends **below ceiling**,
the same lesson the ceiling scout paid for (interleaving at 0.987 left no
headroom for anything to beat it).

**C5. Regression toward a shared asymptote.** A rate advantage mirroring a
starting deficit can be mechanical rather than developmental. *Guard:* the
`t = 0` decomposition, reported as two components, never as their sum.

**C6. "Fast seeds are fast."** The same-corpus/different-initialization
analysis (H2.2) is correlational across seeds, and a seed that learns
everything quickly will show early `M` *and* fast `B`. *Guard:* the primary
statistic is a **partial** correlation between `M` emergence time and `B`
acquisition time, controlling for a general learning-speed covariate —
background LM loss trajectory and `C` acquisition time. Without this control
H2.2 is nearly guaranteed to "pass" and would mean nothing.

**C7. Ablation is not surgical.** Removing heads damages the model generally,
so a main effect of ablation on `B` is expected and carries no information.
*Guards:* the estimand is the **arm × ablation interaction** of H2.3, which
differences that general damage out; a matched-random ablation control chosen
to match attention entropy and output norm; `C` as a specificity check
(ablating `M` heads should not hurt `C` much); ablation applied at the *start*
of phase 2 so the question is about learnability, not about scoring a damaged
model.

**C8. Head-selection researcher degrees of freedom.** Choosing `k`, the
threshold, or the score after seeing outcomes would manufacture H2.2 and H2.3.
*Guard:* all frozen and hashed before phase 2 of the first run, on the existing
freeze machinery.

**C9. We planted the dependency.** True, and it is the point at this layer —
but it bounds the claim, per the standing scope statement in §2. Success
licenses "**the apparatus can identify and causally verify a mediating
mechanism, and correctly rejects a matched control**". It does not license
"developmental dependencies exist in pretraining", and it does not license
"induction circuits explain natural pretraining dependencies". Discovering `M`
without being told what to look for is L2.

**C10. One mechanism is not a theory.** Induction-style circuits are the
best-characterized mechanism available and therefore the right first target,
but a single mediator generalizes to nothing on its own. A second mediator on
a different capability pair is the immediate follow-up if the first succeeds.

**C11. Necessity is not sufficiency.** Even a clean H2.3 pass leaves
`do(M+) → accelerated B` untested, so "M is required for the transfer" must not
be reported as "M produces the transfer". The two are routinely conflated in
mediation writeups and the distinction is load-bearing here.

## 8. Reading the outcomes, fixed in advance

| outcome | reading |
|---|---|
| H2.1 + H2.2 + H2.3 + H2.4 all hold | Mechanistic developmental state is identifiable and causally verified in this substrate. Strongest available result; licenses L2 ontology discovery — **and still carries the §2 scope statement**, since sufficiency (`do(M+)`) remains untested. |
| H2.1 + H2.2, H2.3 interaction ≈ 0 | Transfer is real and correlated with `M`, but `M` is not shown necessary. Downgrade to "mediator unidentified"; do **not** report a mechanism. |
| H2.3 interaction `Δ > 0` **but** matched-random shows it too | The ablation is not specific. Treat as non-result on necessity and diagnose C7 before re-running. |
| H2.1 holds, H2.4 fails (depth-1 at matched capacity reproduces the advantage) | The effect is not carried by two-layer composition. Retract the induction framing; the surviving question — what a 1-layer model carries across a phase boundary — is smaller and more tractable. |
| H2.1 fails — `A→B` ≈ `A'→B` | Under a genuine LM objective the effect does not appear at this scale. **This retro-illuminates L0 as substrate-specific** and is a major result for the paper, not a failure of it. |
| `A→C` moves as much as `A→B` | Generic artifact (optimizer/capacity). The design has caught itself; diagnose C2/C3 before anything else. |

## 9. What this deliberately does not do

- Does not extend the L0 synthetic ontology, add families, or re-fit any
  frozen analysis.
- Does not touch the frozen WikiText adaptive pool (3 unordered / 6 directed),
  which remains untouched and unlicensed.
- Does not attempt curriculum compilation, natural-corpus work, UI, or
  training-data attribution. Attribution (MDA-style) is a genuinely attractive
  follow-up **once a mechanism exists to attribute**, and is meaningless before.
- Does not attempt closed-loop scheduling (H3). Control requires a transition
  model; there is no transition model until a mediator is identified.
- Does not implement the sufficiency rung `do(M+) → accelerated B` (§4). It is
  recorded on the ladder, not deferred silently, and it is the first thing to
  design if V2 passes G2.

## 10. Where this sits on the ladder

```
  identified          V2 target:  D_A → M → learnability(D_B), necessity
  ↓
  sufficient          do(M+) → accelerated B, source exposure held constant
  ↓
  general             a second mediator, a second capability pair
  ↓
  discovered          M inferred from training dynamics, not supplied  (L2)
  ↓
  predictive          model state S_t predicts P(ΔS | D, S_t)
  ↓
  controllable        state-dependent allocation beats strong interleaving
```

V2 addresses the first rung only. Each rung below it is a separate design, and
none of them is licensed by V2 passing — the rungs are listed so that the
distance to the north star stays visible rather than being collapsed by a
single encouraging result.
