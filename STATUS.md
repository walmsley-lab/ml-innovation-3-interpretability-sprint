# STATUS

> **Resuming from another machine or session?** Start with
> [HANDOFF.md](HANDOFF.md): what is running, how to restore state, what is
> established, what has been controlled for, and where it can go next.

## Research objective

**Discover measurable structure in how models acquire, transfer, retain, and
interfere with information, then use that structure to organize or represent
training data so models reach greater capability with fewer tokens, less
corpus, and less compute.**

Stage 5 and developmental ontology discovery remain central. But pairwise
transfer matrices, DAGs, semantic families, overlap schedules and static
curricula are **candidate mechanisms and representations, not objectives we
are obligated to vindicate**. Any of them may be discarded without loss if a
cheaper or better-performing representation appears.

### From curriculum ordering to developmental scheduling

The ceiling scout already taught us something the original framing did not
anticipate: **presentation structure alone produced an enormous effect at
identical compute and identical corpus allocation** — interleaved 0.9867 mean
and 0.9510 min accuracy against block-sequential 0.3827 and 0.1062.

That means retention, recurrence, interference and timing are part of the
developmental problem, not merely prerequisites to an ordering. The eventual
intervention may not look like `A -> B -> C -> D` at all. It may be a
time-varying mixture, rehearsal, recurrence, selective exposure, or
state-dependent allocation.

The hypothesis is therefore broadened from **curriculum ordering** to
**developmental scheduling**.

### The Stage-5 target, stated as an outcome

Given a natural corpus and a model state: infer useful developmental units
and relationships from limited measurements, predict which training exposures
or representations carry the highest marginal learning value, organize
training accordingly, and **prospectively demonstrate higher capability at
equal compute, or equal capability with fewer tokens**.

### Ontology, kept alive but grounded

Current families are **provisional measurement units**. A model-native
developmental ontology should be inferred from how examples or families
behave during learning — what they prepare for, interfere with, reinforce,
require, or change representationally. Semantic similarity is one candidate
explanatory signal among several.

If reliable interactions emerge, the question becomes whether **cheap
observables** predict them: gradient alignment, representation similarity,
loss trajectories, learning-speed signatures, probes, activation changes. The
long-run method needs a small calibration set, not exhaustive pairwise
training.

### Rule for experiment selection

> Before proposing a substantial new run: **if this succeeds, does it
> materially improve our ability to predict or control useful learning per
> token?**

If not, it is diagnostic-only or deprioritized.

---

Updated at each frozen barrier. Completed stage, hashes, gate result,
interpretation, risks, next licensed action. Nothing else.

---

## Barrier: WikiText Stage-5 confirmatory gate — **FAIL** (2026-08-16)

**Completed stage.** Natural-corpus Stage 5 on WikiText-103, 7 usable
families, common-control design. 108 units: 78 development (26 directed
pairs x 3 seeds), 30 confirmatory (10 pairs x 3 seeds), 21 shared controls.
All valid; control invariant PASS on every target x seed group.

**Frozen artifacts.**

| object | hash |
|---|---|
| pair pools (13/5/3 unordered) | `497b9c3fc66e8adbca96ac2eef41e9e2ada14ffcdc0d78bb4cbbf589c42b3c27` |
| confirmatory predictions | `2ff3cf08f744dc9a3c8a98984e026e50e0148186fb0bec2368b1f985cc0b71a5` |
| Layer-2 scout manifest | `1deb0ae83655d08dbf930c3ddd9109c6441bedb03941bde349a6a9796ec5ebac` |

**Gate result: FAIL.** Both primary components clear the 25% material
threshold against the best simpler model and the seed-noise floor.
`rate_only` is jackknife-robust across all ten leave-one-pair-out subsets
(29.3%-60.6%). `head_start` is not: dropping `0->4`, `4->0` or `4->3` gives
21.8%, 18.8%, 22.8%. The criterion required every condition on both
components under one model.

**Scientific interpretation.** The relational model was best in development
LOPO **and best prospectively on both components**, beating the best simpler
model by 30.0% (head start) and 37.7% (rate-only). The 20NG inversion — where
the development-best model became prospectively worst — **did not recur**.
The formal gate nonetheless fails because one component's margin leans on
family 4. The honest statement is that a jackknife-robust 25% improvement was
not demonstrated at n=10 confirmatory pairs, not that structure is absent.

This gate is **closed and will not be refit or reinterpreted**. Family-4
sensitivity is a post-hoc diagnostic only, and any ontology-revision work
stays separate from confirmatory claims.

**Current risks.**
- Confirmatory pool underpowered at 10 directed pairs; a real effect could
  fail this criterion by variance alone.
- `head_start` margin concentrated in one family.
- Relational model is p=15 at ridge 1e-4 on 26 development pairs; it
  generalized here, but the parameterization remains fragile.
- Adaptive pool (3 unordered / 6 directed) is **untouched and must stay so**
  unless a gate licenses it.

**Next licensed action.** The Layer-2 ceiling scout, already frozen and now
running. Stage 5 was **not** reached; no adaptive intervention was selected
or launched.

**Against the four questions.**

* *Phenomenon* — **yes, on WikiText.** Development S/N 5.95 and 6.09 on the
  primary components, confirmatory 7.62 and 8.32, under a repaired control,
  with matched composition and asserted exposure equality.
* *Prediction* — **partial.** The relational model beat every simple baseline
  prospectively on both components (30.0%, 37.7%) against frozen predictions,
  and the 20NG inversion did not recur. It failed the frozen robustness
  condition on one component. This is the closest the project has come, and
  it is not a pass.
* *Utility* — **untested.** No curriculum has been compiled or evaluated.
* *Efficiency* — **unmeasured.** 26 development pairs were used to predict 10
  held-out; the minimum sufficient fraction is unknown and is the target of a
  later sample-efficiency program.

---

---

## Barrier: Layer-2 ceiling scout — **FAIL** (2026-08-16)

**Completed stage.** 15/15 units, manifest `1deb0ae83655d08d`, allocation
identical across arms and families.

**Gate result: FAIL.** `predicted_best` 0.3827 vs `exact_reverse` 0.3731 on
final mean accuracy — a +0.0096 difference against pooled between-seed sd of
0.0187, so effect/noise +0.51. Best does **not** beat balanced either: 0.383
against **0.987**.

**Scientific interpretation.** Sequential block curricula retain only the
final family. Every family reaches ceiling at the end of its own phase and
then collapses by 0.69-0.89; only the last survives. The two sequential arms
differ in outcome only in *which* family is last, so their gap is recency,
not order. The balanced arm reaching 0.951-0.998 on all four families shows
the task is fully learnable at this dose — the failure is specific to
block-sequential presentation, and reproduces the Layer-1 catastrophic
interference finding that overlap `r = 0.20` was introduced to fix.

A second failure mode survives even if forgetting is fixed: the pairwise
matrix measured *immediate acquisition of the next family* over two phases,
while the scout composed it into a four-phase ordering scored on *retention
of all families*. Transitivity was never tested.

**Current risks.** The pairwise-graph approach may not compose into
multi-phase curricula at all; that is now an open and testable question
rather than an assumption.

**Next licensed action.** Proposed and **not run**: re-run the same three
arms with Layer-1's validated overlap floor `r = 0.20`, changing that one
thing only. It separates "forgetting swamped the effect" from "pairwise
transfer does not compose". 15 units.

Sharpened by the efficiency framing: the comparator is the **interleaved
arm**, not the reverse arm. The reverse arm remains as the order control, but
the question that matters is whether any ordering beats interleaving at equal
compute. Efficiency metrics (tokens- and steps-to-threshold, accuracy at
fixed budget, min across families, retention) are frozen below and would be
recorded per unit.

**Against the four questions.**

* *Phenomenon* — **no**, for block-sequential Layer-2 curricula at this dose.
  The dominant effect is recency, not order.
* *Prediction* — untested here; the scout never reached a prediction stage.
* *Utility* — **no.** The compiled curriculum lost heavily to plain
  interleaving (0.383 vs 0.987). Any curriculum claim must beat the balanced
  control, and this one does not come close.
* *Efficiency* — the scout cost 15 units to establish that the regime has no
  headroom, which is the cheapest possible way to have learned it.

---

---

## Barrier: r = 0.20 overlap diagnostic — **Branch 2** (2026-08-16)

**Completed stage.** 15/15 units, manifest `cd73c7444818dcfc`, every family at
exactly 600 steps in every arm.

**Gate result.** Retention restored (min acc 0.106 -> 0.53-0.57), order effect
absent (best - reverse = -0.0383 against pooled seed sd 0.2556, effect/noise
**-0.15**), and interleaving wins every efficiency metric (min acc 0.951 vs
0.529; joint steps-to-threshold reached 5/5 seeds at median 950 steps vs 1/5
and 0/5).

**Scientific interpretation.** Overlap at `r = 0.20` substantially fixes the
catastrophic forgetting the ceiling scout exposed, so failure mode 1 was real.
With retention restored and order decoupled from dose, **the ordering effect
is still absent**. Per the branch fixed in advance: local pairwise transfer
does **not** straightforwardly compose into useful multi-stage orderings, and
the static ordering construction is not to be patched further.

Claims kept separate. *Mechanistic*: pairwise transfer is real and measurable
but does not compose additively — a finding about composition, not evidence
that developmental structure is absent. *Practical*: the ordering programme
has no utility at this dose, and the sequential arms are ~15x less stable
across seeds than interleaving (sd 0.27 vs 0.016).

**Current risks.** Interleaving may be near-optimal at this scale; until a
regime exists where it is below ceiling, no method can demonstrate an
efficiency gain over it.

**Next licensed action.** The resource-constrained scheduling benchmark. No
pairwise-derived static ordering carries forward.

**Against the four questions.**

* *Phenomenon* — **yes, but not the hypothesized one.** The reproducible
  effect at equal compute is presentation structure (interleaved vs blocked),
  not ordering.
* *Prediction* — pairwise predictions do not compose into multi-stage
  outcomes. WikiText prospective prediction remains the only partial success.
* *Utility* — **no.** Interleaving wins every frozen efficiency metric.
* *Efficiency* — interleaving reaches joint 0.90 across all families in a
  median 950 steps, 5/5 seeds; the best ordering manages 1/5.

---

## Post-barrier branches (now resolved — Branch 2 fired)

Fixed before the result is visible:

* **Retention not restored** — diagnose retention. Do **not** expand the graph
  programme.
* **Retention restored, predicted-best ~ reverse** — evidence that local
  pairwise transfer does **not** straightforwardly compose into useful
  multi-stage orderings. Do **not** keep patching the static ordering
  construction to make it work.
* **Retention restored, predicted-best > reverse beyond noise** — pairwise
  developmental information has some compositional value. This is **not** yet
  practical utility.

In the latter two cases the next major experiment orients to useful learning
per token, in a **deliberately resource-constrained regime where strong
interleaving is below ceiling**, comparing at equal budget on the already
frozen metrics with primary emphasis on min-across-families and joint
steps/tokens-to-threshold. Candidate methods: uniform interleaving,
randomized interleaving, simple heuristic schedules, any pairwise-derived
schedule that survives this diagnostic, and a **crude adaptive/closed-loop
scheduler** that periodically observes competence, learning rate and
forgetting across families and allocates the next window toward underlearned,
high-value or endangered families.

The scheduler is not to be optimized for sophistication. The first question is
only whether **state-dependent allocation** yields measurable sample- or
compute-efficiency gains over strong interleaving.

## Ontology programme — the north star, not yet licensed

The four-question ladder is the **evidence ladder** for the ontology
programme, not a replacement for it. The end state remains a **model-native
developmental ontology**: units defined by how learning transfers,
interferes, accelerates and changes representations, rather than by human
semantic labels.

Standing constraints, recorded now so they are not rediscovered later:

* The current semantic families and any first recovered graph are
  **provisional measurement units**, not the ontology.
* Ontology optimization does **not** begin from partial or failed results.
  Reliable interaction structure must exist first; as of this barrier it does
  not.
* Once stable non-additive, held-out-predictive structure exists, the next
  question is which observables explain it — representation similarity,
  gradient alignment, loss trajectories, learning speed, probe/activation
  change, semantic structure, or combinations.
* **Every ontology revision must make frozen prospective predictions before
  being considered better.** This is the WikiText standard, applied to
  ontologies.
* The deliverable is a predictive and actionable developmental ontology. A
  graph on its own is not the deliverable, and neither is a curriculum. The
  deliverable is a developmental model of training that yields **more
  capability from less data and compute**.

### Efficiency is the terminal metric

Candidate ontologies and representations are judged by concrete training
outcomes, not by effect size:

* same target performance with **fewer tokens**;
* a given accuracy in **fewer optimization steps**;
* intelligent ordering beating shuffled/interleaved **at equal compute**;
* a better representation needing **less corpus**;
* identifying redundant or low-value data that can be **omitted** without
  harming capability;
* higher final accuracy **at a fixed token/compute budget**.

Once reliable structure exists, subsequent experiments measure sample and
compute efficiency, not merely effect size.

### The scout already answered one efficiency question

Not the one it asked. At **identical token and compute budget**, and
identical aggregate family allocation:

    balanced / interleaved   0.9867 mean acc, 0.9510 min acc
    block-sequential         0.3827 mean acc, 0.1062 min acc

That is a very large capability difference at fixed budget, produced purely
by **presentation structure**. It is the strongest training-efficiency effect
this project has measured, and it points the opposite way from the
hypothesis under test: blocking destroys capability that interleaving
retains.

The consequence for design is concrete. **Interleaved is the baseline to
beat, not a control to beat.** Any ordering technique must show an advantage
*over* interleaving at equal compute, and no result that merely beats
block-sequential ordering is interesting.

### Efficiency metrics to freeze before the next experiment

Recorded now so they are fixed in advance rather than chosen after:

* **tokens-to-threshold** — tokens to reach accuracy tau on each family, and
  on all families jointly;
* **steps-to-threshold** — the same in optimizer steps;
* **accuracy at fixed budget** — final mean and **min** across families, since
  a mean hides a destroyed family;
* **area under the learning curve** per family;
* **retention** — accuracy at the end of the curriculum against accuracy at
  the end of that family's own exposure.

The min-across-families metric is the one that matters most: the scout's
sequential arms looked half-decent on the mean and catastrophic on the min.

## Infrastructure

`dsi-cpu-bench` and `dsi-cpu-w2`, both c3-standard-22, us-west1-a, verified
environment-identical (image `v20260807`, jax 0.11.0, matching vocab, pools
and corpus-cache hashes). Throughput ~3.06 trajectories/min per worker, flat
in concurrency. `pdp-gpu` TERMINATED and preserved.
