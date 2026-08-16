# Blocker and risk register

Maintained alongside the stage gates. A gate that fails on a sound apparatus
is a scientific result, not an entry here. Only apparatus faults, execution
constraints, and open scientific risks are tracked.

---

## 1. Resolved blockers

### Original W/P non-identifiability and impossible retention bound

The two training families presented byte-identical model-visible inputs and
differed only in the answer. For any deterministic predictor this capped
`A_W + A_P <= 1 + 1/K`, so `min(A_W, A_P) <= 0.625` at `K=4`, against a
prespecified `tau_retention = 0.80`. **Gate B was mathematically
unsatisfiable at any capacity, learning rate, or duration**, and every
capacity sweep run against it was measuring an impossible criterion.

Resolved by the explicit MODE redesign (`USE_W` / `USE_P` / `NEUTRAL`), which
makes the request part of the input. An oracle now reaches `A_W = A_P = 1.0`
on the same evaluation populations (`tests/test_identifiability.py`). The
threshold was not moved. Recorded in `research.md` 8b; the failed sweep and
diagnostic artifacts are preserved as provenance.

### B2 return-contract bug

`run_sequence` returned two different shapes for solo and sequential runs, so
every Stage B2 unit died with `KeyError: 'per_phase'` while B1 appeared
healthy. Silent in the sense that mattered: the sweep kept running and would
have reported "no adequate regime" from an empty retention stage.

Fixed, and the completed B1 units resumed from disk rather than re-running,
which is what the per-unit persistence was added for.

---

## 2. Completed negative apparatus result

### Pure block-sequential Gate B: abrupt isolated phases produce catastrophic interference

**This is a finished result and stands unchanged.** Abrupt, isolated training
phases destroy the prior capability at every capacity tested, and no
continual-learning method was introduced to work around it. It is recorded as
a property of the block-sequential apparatus, not as a fault in it and not as
a defect of the hypothesis.

What follows is the evidence and the diagnosis that closed it.

At the fixed d64/l2 regime on the corrected task, worst-order coexistence is
0.238 against 0.80. The first capability collapses to chance within the first
10% of phase 2, before the second begins to rise, in both orders.

This is a valid measurement, not an apparatus fault. It is now **diagnosed
as genuine catastrophic interference**, not a capacity limit.

The eight-regime sweep found coexistence flat at 0.195-0.250 across a 6.4x
parameter range, two depths and two learning rates, while solo competence
varied enormously over the same range. A quantity that does not move when
capacity moves is not limited by capacity.

Diagnostics settled it:

| diagnostic | result |
|---|---|
| A: joint training, same total budget, d64/l2 | **coexistence 0.982** |
| A: joint training, d64/l4 | **coexistence 0.986** |
| sequential, best of eight regimes | 0.250 |
| B: after identical NEUTRAL_ALIGNED integration | 0.246-0.277 |

So of the four candidate explanations:

1. **optimization stability** — *live*. The same architecture reaches 0.98
   when the two skills arrive together and 0.23 when they arrive in
   sequence. The difference is the order of the updates, nothing else.
2. **representational capacity** — **ruled out**. d64/l2 holds both skills
   simultaneously at 0.982.
3. **abrupt phase-transition / task-switch dynamics** — *live*. The first
   skill collapses within the first 10% of phase 2, before the second begins
   to rise, in every regime and both orders.
4. **design mismatch** — *live, and the most consequential*. The lost skill
   was **not behaviourally recovered by the tested identical
   NEUTRAL_ALIGNED integration phase** (0.246-0.277). That is what the
   evidence directly supports; whether the skill is destroyed or merely
   unrecovered by *this* intervention is not settled by it. If a
   developmental history overwrites rather than layers, the phenomenon this
   project set out to study may not survive its own operationalization.

### Continuity-threshold diagnostic, and a confound in it

Mixing a fraction r of the first family into phase 2, branching every ratio
from one shared phase-1 checkpoint, at d64/l4:

| r | W->P coexistence | P->W coexistence |
|---|---|---|
| 0.00 | 0.229 | 0.236 |
| 0.01 | 0.299 | 0.240 |
| 0.05 | 0.289 | 0.492 |
| 0.10 | 0.510 | 0.520 |
| 0.25 | **0.994** | 0.727 |
| 0.50 | **1.000** | 0.479 |

**W->P is rescued by continuity**: 25% mixing takes coexistence from 0.229 to
0.994. **P->W is not**, and is non-monotone, peaking at r=0.25 and falling at
r=0.50.

The asymmetry is a confound in the protocol, not a property of the phenomenon.
The phase-2 *total* budget was held fixed, so higher r leaves the new skill
proportionally fewer of its own tokens: 600 steps at r=0, 300 at r=0.5. W is
the slow skill and P the fast one, so in the P->W direction high r starves
the very skill being acquired. Every P->W arm has A_W **still rising at the
final checkpoint**, which is the signature of an unfinished acquisition
rather than of forgetting.

### Budget-corrected continuity curve — the confound was real

Holding the *new* skill's exposure fixed at 600 pure steps and adding
old-skill samples on top (phase 2 runs 600, 632, 667, 800, 1200 steps):

| r | W->P | P->W | prior capability |
|---|---|---|---|
| 0.00 | 0.229 | 0.236 | remains lost, both |
| 0.05 | 0.293 | 0.467 | remains lost, both |
| 0.10 | 0.627 | 0.531 | lost (W->P) / recovers (P->W) |
| 0.25 | **0.988** | **0.996** | collapses then recovers, both |
| 0.50 | **1.000** | **0.965** | recovers (W->P) / continuously preserved (P->W) |

**Both directions clear tau at r=0.25**, and the directional asymmetry
disappeared: P->W at r=0.5 moved from 0.479 under the confounded protocol to
0.965 under matched exposure. The asymmetry was an artifact of starving the
slower incoming skill, exactly as diagnosed, and is **not** recorded as a
genuine interference asymmetry.

Diagnostic C is **not** run. The A-succeeds/sequential-fails pattern licensed
it, but continuity rescues coexistence without any protection method, so
EWC/OGD would answer a question that is no longer open.

**Status of the r=0.25 result: discovery / proof-of-concept only.** It was
obtained at seed 1000, at one architecture and one task, and the replicated
calibration that followed showed it does not hold on other seeds (0.369 at
seed 1002). It demonstrates that overlap *can* restore coexistence; it is
**not** a calibrated overlap setting and must not be used as one.

r=0.10 to r=0.25 is likewise a sharp transition region **observed in the
discovery run**, not a universal threshold.

The calibrated overlap floor is undetermined. It cannot be selected until the
regime's solo competence is robust across calibration seeds, and when it is
re-run the matched-incoming-exposure arithmetic scales with the frozen
duration: at incoming exposure N, the old-skill sample count is
`N_old = N * r / (1 - r)`.

The open question is now about the hypothesis, not the apparatus: whether the
primary developmental operationalization stays pure block-sequential or moves
to controlled overlapping curricula. Gate B, tau_retention and the
preregistered block-sequential result stand unchanged until that is decided
explicitly.

Constraint carried forward: replay, EWC, OGD, adapters, separate heads and
parameter isolation are **diagnostics only**. None enters the primary design
without first agreeing what it changes about the scientific claim. Any move
from immediate retention to post-integration competence is an explicit change
of hypothesis and gate definition, never a post-hoc threshold workaround.

---

## 2b. New active blocker: the frozen regime is seed-fragile

The overlap calibration over r in {.15,.20,.25,.30} x 3 seeds x 2 directions
found **no robust ratio**:

| r | worst | s1000 | s1001 | s1002 | robust |
|---|---|---|---|---|---|
| 0.15 | 0.250 | 0.680 | 0.869 | 0.250 | no |
| 0.20 | 0.365 | 0.547 | 0.873 | 0.365 | no |
| 0.25 | 0.369 | 0.988 | 0.883 | 0.369 | no |
| 0.30 | 0.391 | 0.551 | 0.980 | 0.391 | no |

In 11 of the 12 limiting cells the failing quantity is **A_W, the skill being
acquired**, not the retained one. At r=0.15/s1002 the retained skill sits at
A_P=0.996 while A_W is 0.250. That is not interference.

The control settles it. Solo acquisition from initialization at d64/l4, 600
steps, across the three calibration seeds:

| seed | solo A_W | solo A_P |
|---|---|---|
| 1000 | 0.998 | 1.000 |
| 1001 | **0.529** | 1.000 |
| 1002 | 0.982 | 1.000 |

**The regime does not reliably learn the rule at all.** Seed 1001 reaches
0.529 solo, far below tau_w = 0.90, and its trace is still climbing at the
final checkpoint, so it is under-trained rather than stuck. P is at ceiling on
every seed; the variance is entirely in W.

Two consequences.

First, **the overlap calibration rests on an uncalibrated regime** and its
numbers cannot be used to select r. A ratio cannot be chosen for its ability
to preserve a skill the regime does not reliably acquire.

Second, **a stage-ordering defect in the sweep**. Seed replication (B3) is
gated behind retention (B2), so a regime that fails retention never has its
*solo* competence replicated across seeds. Retention was failing for an
apparatus reason unrelated to the regime, so B3 never ran and the seed
fragility went unseen. Solo criteria are properties of the regime alone and
should be replicated across seeds inside B1, before any retention measurement
is attempted. The seed-replication requirement was correct; it was placed too
late in the sequence to catch this.

Remedy, in progress: **seed replication is now permanent in B1**. The revised
Gate-B ordering is

* **B1 — replicated solo adequacy.** Solo competence per family, held-out
  generalization and learning-window adequacy, on every calibration seed. A
  regime does not reach retention testing unless it passes on all of them.
* **B2 — sequential coexistence and retention**, on B1-robust regimes only.

A B2 failure can no longer suppress discovery of B1 seed fragility.

### Duration calibration: no duration passes, but the failure migrated

`steps_per_phase` in {600, 900, 1200}, d64/l4, lr=3e-3, seeds 1000-1002, all
else unchanged:

| steps | s1000 | s1001 | s1002 | robust |
|---|---|---|---|---|
| 600 | ok | A_W=0.480, gen=0.430 | ok | no |
| 900 | ok | A_W=0.711, gen=0.527 | ok | no |
| 1200 | ok | ok | R_P=0.115 | no |

**Duration fixes what it was expected to fix.** At 1200 steps the rule is
learned on every seed (A_W = 0.988, 1.000, 1.000) and generalization clears
everywhere (0.871 to 0.996). Seed 1001, which reached 0.480 at 600 steps, was
simply under-trained.

What remains is a different criterion failing for a mechanical reason. The
learning window is defined as a **fraction of the phase**, and the cue's
acquisition is roughly fixed in absolute steps:

| steps | mean absolute cue window | as a fraction |
|---|---|---|
| 600 | 138 | 0.229 |
| 900 | 142 | 0.158 |
| 1200 | 172 | 0.143 |

Lengthening the phase to make the rule learnable therefore shrinks `R_P` as a
fraction, and at 1200 steps `min_window = 0.15` demands an absolute cue
window of 180 steps. The two criteria are in direct tension **because W and P
have very different intrinsic timescales**.

This is the difficulty-mismatch problem that `DESIGN_LAYER2.md` §4 makes a
gate for Layer 2, appearing in Layer 1. The neutral lever is task difficulty:
`n_cues` sets how long the cue takes to acquire, and
`scripts/probe_cue_window.py` already records the relationship (R_P rising
from 0.080 at 16 cues to 0.550 at 1024, measured at 600 steps).

`n_cues` was frozen at Gate B on the cue in isolation, with the stipulation
that it must not be revisited once confirmatory results are visible. No
confirmatory result exists: the conflict condition has never been generated
or evaluated in any run of this cycle. Re-calibrating it on neutral solo
criteria is therefore legitimate, and doing so would be recorded as a
recalibration rather than a silent change.

**Not launched.** Neither `min_window` nor any threshold has been moved.

---

## 3. Execution and infrastructure constraints

### Tiny models are CPU-faster than the L4 at current scale

Measured: ~235 effective runs/hour locally at 3 workers against 71.6 on the
L4 at concurrency 4, with mean GPU utilization never exceeding 10%. At 133k
parameters the work per kernel cannot occupy an L4, and throughput is bound
by dispatch overhead on the machine's 4 vCPUs. The GPU becomes the right
executor when the model grows — at the transfer matrix, or above roughly 10M
parameters — not before. Carries no scientific risk either way.

### L4 multi-process requires JAX preallocation disabled

JAX reserves ~75% of VRAM per process, so three of four workers died at
concurrency 4 under the default. With `XLA_PYTHON_CLIENT_PREALLOCATE=false`
all four succeed at ~350 MiB each. Allocator settings are executor metadata
and must never enter the scientific `RunSpec` identity: two runs differing
only in allocator policy are the same experiment.

---

## 4. Future scientific risks

Each is a way the project produces a valid negative result. None is an
apparatus fault, and each stops the escalation of claims at a defined point.

| # | Risk | Gate | Rough compute |
|---|---|---|---|
| S1 | No persistent order effect after common integration | D | ~40 runs, local, hours |
| S2 | Transfer effects explained by additive source/target difficulty | F | 6-12 families, hundreds of runs, L4, ~$15-40 |
| S3 | Developmental model fails held-out intervention prediction | G | reuses F's runs; analysis only |
| S4 | Ontology revision does not improve predictive compression | H | analysis plus targeted re-runs |
| S5 | Derived curriculum fails against reverse, random and uniform | I | ~5 arms x n seeds, fresh models |
| S6 | Frozen pipeline fails on unseen natural text | J | largest stage; scale-dependent |

S2 deserves particular attention: the additive `alpha_i + beta_j` baseline is
the sharpest competitor to the developmental claim, and the one most likely
to explain a transfer matrix without any pairwise structure at all.

Cost basis for the estimates: local CPU ~$0; L4 `g2-standard-4` at
$0.85/hour, measured at ~72 runs/hour at concurrency 4.
