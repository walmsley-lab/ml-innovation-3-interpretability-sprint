# Developmental System Identification for Pretraining

## What is the question?

Does what a model has already been trained on change **what it is able to learn
next** — and if so, can that be measured well enough to act on?

Not "does curriculum order matter". The sharper question is whether training
history leaves a model in a *developmental state* that determines the value of
subsequent data, and whether that state can be read from the model itself.

## What did we find?

**Training history produces a large, selective difference in future
learnability.** In a controlled language micro-world under ordinary next-token
prediction, a model trained on source `A` reaches **0.1322 ± 0.0149** zero-shot
accuracy on target capability `B`, while two matched controls sit at
**0.0156** — exactly the chance floor. Confirmatory, fresh seeds, frozen
criteria.

**It is selective.** A negative-control capability `C` shows no advantage
(+0.0395 against +0.5265 on `B`), and `C` is genuinely learnable, so the test
is not vacuous.

**It is not memorised content.** With **zero shared entity tokens** between
source and target, the effect is fully retained (111%). The controls differ
from `A` in exactly one property — whether a recurring entity keeps its value —
and are matched on unigram, bigram, positional statistics and entity
recurrence, verified against a same-stream null.

**It survives 32× scale** (786k → 25.2M non-embedding parameters), exploratory.

**The value of data is conditional on model state.** A 48-cell `V(S,D)` matrix
shows a substantial State × Data interaction (share 0.381–0.464) with genuine
**ranking reversals**: the identity of the best next corpus depends on the
incoming state.

## What failed?

Kept visible, because the failures are informative and several are load-bearing.

| | |
|---|---|
| **Static curricula** | Block-sequential training destroys capability at equal compute: 0.383 mean / 0.106 min against interleaved 0.987 / 0.951 |
| **Composition** | With retention repaired, pairwise transfer does **not** compose into useful orderings (effect/noise −0.15) |
| **The pre-registered mechanism** | An off-distribution induction-style probe failed its confirmatory gate (selectivity 0.76 against ≥2.0). A striking single-seed preflight transient did not reproduce anywhere |
| **Causal mediation** | **Inconclusive, not falsified.** The ablation produced no interaction, but also removed none of the capability (zero-shot `B` 0.139 → 0.161) — necessity was never tested at adequate strength |
| **Adaptive selection** | A state-aware selector **loses to a state-blind global-best rule** on held-out states (regret 0.047 vs 0.021). It beats random, so it learned the data main effect, not the conditionality |
| **The preference substrate** | The earlier W/P task could not host a behaviour-matched test: competence pass rate 1/6 |

## Why does it matter?

The failure of adaptive selection is the most useful result, because it
localizes the problem precisely:

> **The conditional training signal exists. State inference is inadequate.**

Data value genuinely depends on model state — that is measured, not assumed.
What is missing is a representation of state good enough to predict *which*
data is best. That is a specific, attackable technical bottleneck, and a better
place to stand than either "curriculum matters" or a premature claim of
adaptive pretraining.

## What remains unresolved?

* **Is the state causal?** Requires an intervention that first demonstrates it
  can move the capability. Top-k head ablation demonstrably cannot.
* **Can behaviourally matched models have different futures?** Prospective test
  with pairs frozen from present observables, pending.
* **When does the relevant state emerge?** A changepoint at ~270 steps is
  established for *acquired competence* (held-out RMSE 0.0171 vs 0.0331
  linear); whether **future learnability** changes at the same moment is being
  measured by dense replay.
* **How far does the effect generalize?** The developmental radius — targets
  ordered by distance from the source — is not yet mapped.
* **Can state be read well enough to predict conditional data value?** The open
  problem. Current evidence argues against a small set of attention heads and
  toward gradient/update geometry.

## How do I reproduce it?

```bash
uv venv && uv pip install -e ".[dev]"

PYTHONPATH=src python scripts/preflight_microworld.py       # substrate checks
PYTHONPATH=src python scripts/audit_microworld_shortcuts.py # A/A' shortcut audit
PYTHONPATH=src python -m pytest tests/ -q                   # unit tests
PYTHONPATH=src python scripts/audit_claims.py               # recompute every headline number
```

Artifacts are gitignored; restore from GCS if available. Every prediction
artifact is hashed before the outcomes it predicts exist:

| frozen object | sha256 (first 16) |
|---|---|
| V2.1 spec | `f92f5831bece0d91` |
| what-next tournament protocol (unrun, trigger failed) | `f9da9fe23b1b2400` |
| downstream protocols | `8fc78c4087e2f87b` |
| latent-state protocol | `afd0bf5174cd8073` |

## Repository layout

| path | contents |
|---|---|
| [RESEARCH.md](RESEARCH.md) | the scientific programme, evidence ladder, standing commitments, deferred work with triggers |
| [RESULTS.md](RESULTS.md) | claim ledger, chronology, barrier log — the canonical record |
| `docs/experiments/` | per-experiment designs and frozen protocols |
| `docs/technical.md`, `docs/RISKS.md` | system specification and risk register |
| `docs/archive/` | superseded plans and the original ordering programme, preserved verbatim |
| `src/dsi/`, `scripts/`, `tests/` | implementation |
