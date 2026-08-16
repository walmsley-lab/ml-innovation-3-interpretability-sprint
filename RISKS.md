# Blocker and risk register

Maintained alongside the stage gates. A gate that fails on a sound apparatus
is a scientific result, not an entry here; only apparatus faults and open
scientific risks are tracked.

Compute estimates are rough and exist so that scientific risk can be told
apart from infrastructure cost. Cost basis: local CPU ~$0, L4
`g2-standard-4` at $0.85/hour, measured at ~72 runs/hour at concurrency 4.

---

## Resolved hard blockers

### R1. Original W/P task was unidentifiable — RESOLVED

The two training families presented byte-identical model-visible inputs and
differed only in the answer. For any deterministic predictor this capped
`A_W + A_P <= 1 + 1/K`, so `min(A_W, A_P) <= 0.625` at `K=4`, against a
prespecified `tau_retention = 0.80`. **Gate B was mathematically
unsatisfiable at any capacity, learning rate or duration.**

Resolved by the explicit MODE redesign: `USE_W` / `USE_P` / `NEUTRAL` makes
the request part of the input. An oracle now reaches `A_W = A_P = 1.0` on the
same evaluation populations (`tests/test_identifiability.py`).

The threshold was not moved. Recorded in `research.md` 8b, with the failed
sweep and diagnostic artifacts preserved as provenance.

---

## Active possible blockers

### B1. Catastrophic forgetting under corrected sequential W/P acquisition

At the fixed d64/l2 regime on the corrected task, worst-order coexistence is
0.238 against 0.80. The first capability collapses to chance within the first
10% of phase 2, before the second begins to rise, in both orders.

This is a valid measurement, not an apparatus fault. What it is *not yet* is
diagnosed. Four candidate explanations, to be distinguished rather than
assumed:

1. **optimization stability** — the update rule destroys the first solution
   even though the parameters could hold both;
2. **representational capacity** — no parameter setting of this architecture
   holds both;
3. **abrupt phase-transition / task-switch dynamics** — the model switches
   solution wholesale rather than accumulating;
4. **design mismatch** — isolated sequential acquisition may not be the right
   operationalization of the developmental phenomenon the project is about.

Discriminating suite: joint-training upper bound (A), recovery under common
integration (B), and, only if A succeeds while sequential fails, a minimal
protected-learning control (C).

Constraint carried forward: replay, EWC, OGD, adapters, separate heads and
parameter isolation are diagnostics only. None enters the primary design
without first agreeing what it changes about the scientific claim. Any move
from immediate retention to post-integration competence is an explicit change
of hypothesis and gate definition, never a post-hoc threshold workaround.

---

## Future scientific risks

Each is a way the project can produce a valid negative result. None is an
apparatus fault, and each stops the escalation of claims at a defined point.

| # | Risk | Gate | Rough compute |
|---|---|---|---|
| S1 | No persistent W/P order effect survives common integration | D | ~40 runs, local, hours |
| S2 | Transfer structure is explained by additive source + target effects, with no pairwise interaction | F | 6-12 families, hundreds of runs, L4, ~$15-40 |
| S3 | The fitted developmental model fails held-out intervention prediction | G | reuses F's runs; analysis only |
| S4 | Ontology revision does not improve predictive compression | H | analysis, plus targeted re-runs |
| S5 | The derived curriculum fails to beat reverse, random and uniform baselines | I | ~5 arms x n seeds fresh models |
| S6 | The frozen method fails on an unseen natural corpus | J | largest stage; scale-dependent |

S2 deserves particular attention: the additive `alpha_i + beta_j` baseline is
the sharpest competitor to the whole developmental claim, and the one most
likely to explain the transfer matrix without any pairwise structure.

---

## Executor note

The L4 is currently *slower* than the local machine for this model size
(~72 runs/hour against ~235), because 133k-parameter models cannot occupy the
GPU and throughput is bound by dispatch on 4 vCPUs. The GPU becomes the right
executor when the model grows, not before. This is an infrastructure fact and
carries no scientific risk either way.
