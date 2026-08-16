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

This is a valid measurement, not an apparatus fault, and it is **not yet
diagnosed**. Four candidate explanations, to be discriminated rather than
assumed:

1. **optimization stability** — the update rule destroys the first solution
   even though the parameters could hold both;
2. **representational capacity** — no parameter setting of this architecture
   holds both;
3. **abrupt phase-transition / task-switch dynamics** — the model switches
   solution wholesale rather than accumulating;
4. **design mismatch** — isolated sequential acquisition may not be the right
   operationalization of the developmental phenomenon the project studies.

Discriminating suite: (A) joint-training upper bound, (B) recovery under
common integration, and (C) a minimal protected-learning control, run only if
A succeeds while sequential training fails.

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
