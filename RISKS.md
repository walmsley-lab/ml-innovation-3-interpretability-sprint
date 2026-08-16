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

## 2. Active blocker under test

### Catastrophic forgetting under corrected explicit-mode sequential training

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

The diagnostic therefore does not license Diagnostic C. The controlled
version holds the *new skill's* token count fixed and lets the phase total
grow with r, separating continuity from acquisition budget.

Diagnostic C remains licensed in principle by the A-succeeds/sequential-fails
pattern, and remains diagnostic only, but should wait for the
budget-corrected continuity curve.

Constraint carried forward: replay, EWC, OGD, adapters, separate heads and
parameter isolation are **diagnostics only**. None enters the primary design
without first agreeing what it changes about the scientific claim. Any move
from immediate retention to post-integration competence is an explicit change
of hypothesis and gate definition, never a post-hoc threshold workaround.

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
