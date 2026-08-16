# Technical Plan: Developmental System Identification for Pretraining

## 1. Engineering Objective

The codebase should be:

* small;
* functional where practical;
* explicit;
* reproducible;
* easy to test;
* cheap to branch;
* easy to parallelize;
* budget aware;
* infrastructure agnostic.

The central principle is:

[
\boxed{
\text{state/configuration as data}
+
\text{behavior as functions}
}
]

The science layer defines experiments.

The executor runs them.

The infrastructure must never decide the scientific design.

---

# 2. Simplicity Rule

No abstraction is implemented merely because the eventual research program might need it.

New modules or frameworks are added only when a current stage requires them.

In particular:

* no graph framework before (T) exists;
* no ontology engine before held-out prediction works;
* no active-selection framework before exhaustive small-scale experiments work;
* no learned control policy before a transparent curriculum compiler works;
* no polished UI before scientific outputs stabilize.

---

# 3. Core Stack

## Initial core

* Python
* JAX
* Equinox
* Optax
* Polars
* PyArrow
* NumPy
* SciPy
* pytest

## Added when earned

### Hydra

At capacity/HPO sweep stage.

### Orbax

When checkpoints must cross processes/machines.

### NumPyro

Only if the transfer pilot justifies hierarchical probabilistic inference.

### NetworkX

For graph views after transfer structure exists.

### Optuna / Ray Tune

Only for calibration/HPO if a simple sweep becomes insufficient.

### Ray

Only if local/GCP process scheduling becomes genuinely difficult.

---

# 4. Milestone A Repository

Initial scaffold:

```text
pyproject.toml

src/dsi/
  rng.py
  specs.py
  stats.py
  data.py
  model.py
  train.py
  eval.py
  artifacts.py

tests/
  test_rng.py
  test_specs.py
  test_stats.py
  test_data.py
  test_train.py

README.md
```

Nothing else is required initially.

---

# 5. Milestone A Ordering

Implementation order:

1. `stats.py`
2. `rng.py`
3. `specs.py`
4. `data.py`
5. `model.py`
6. `train.py`
7. `eval.py`
8. `artifacts.py`

Estimator validation comes first because it requires no model code and satisfies Research Gate A before any accelerator use.

This corrects the earlier plan that placed estimator simulation in Milestone B.

---

# 6. Milestone A Exit Criterion

Milestone A exits only when:

1. `test_stats` recovers known synthetic effects with calibrated CI coverage;
2. `test_rng` proves pair-sharing and null-divergence contracts;
3. canonical `RunSpec` hashing is stable;
4. target-phase (t=0) evaluation exists in the interface;
5. one local (W\rightarrow P) / (P\rightarrow W) pair trains end-to-end;
6. versioned Parquet artifacts are written successfully.

Hydra, Orbax, GCP, GCS, NumPyro, transfer discovery, and UI are deliberately absent.

---

# 7. Immutable Specifications

```python
@dataclass(frozen=True)
class PhaseSpec:
    family: str
    tokens: int
    role: str


@dataclass(frozen=True)
class EvalSpec:
    suite_id: str
    version: str


@dataclass(frozen=True)
class RunSpec:
    schema_version: int
    parent_id: str | None
    phases: tuple[PhaseSpec, ...]
    model_config_id: str
    data_version: str
    init_seed: int
    data_seed: int
    branch_seed: int
    eval_seed: int
    evals: tuple[EvalSpec, ...]
```

Configuration is data.

Execution is functional.

---

# 8. Canonical Content Addressing

Every run receives:

[
run_id
======

H(
RunSpec,
code_version
).
]

Canonical serialization must specify:

* sorted keys;
* stable numeric formatting;
* explicit schema version;
* normalized tuples/lists;
* no hostnames;
* no absolute local paths;
* immutable data/eval IDs;
* pinned code version semantics.

Scientific identity should remain stable across execution infrastructure.

Hardware metadata belongs in the resulting artifact unless hardware is itself experimental.

---

# 9. JAX RNG Discipline

Every run receives one root key.

Derive named streams using `jax.random.fold_in`.

Example:

```python
root = key(seed)

init_key = fold_in(root, INIT)
source_key = fold_in(root, SOURCE_DATA)
target_key = fold_in(root, TARGET_DATA)
train_key = fold_in(root, TRAIN)
eval_key = fold_in(root, EVAL)
```

Phase index and role are also folded in.

Never reuse a JAX key.

Role identifiers are centralized and tested.

---

# 10. Paired RNG Contract

For transfer pair:

[
D_i\rightarrow D_j
]

versus:

[
N\rightarrow D_j,
]

share:

* initialization key;
* target-data ordering;
* evaluation RNG;
* phase-2 stochasticity where possible.

Differ only in source-corpus identity and source realization as prespecified.

The pair-sharing contract is unit tested.

---

# 11. Null RNG Contract

Identity-null pairs must differ in the same logical location as real treatment pairs.

For:

[
N_1\rightarrow D_j
]

versus:

[
N_2\rightarrow D_j,
]

the source draw differs while initialization, target sequence, and evaluation are matched.

This keeps null variance comparable to treatment variance.

---

# 12. Reproducibility Levels

Do not promise universal bitwise reproducibility.

## Same process / same hardware / deterministic mode

Expect exact or extremely tight agreement where JAX/XLA permits.

## Different process / same accelerator class

Expect numerical agreement within strict tolerance.

## Different accelerator class

Expect distributional equivalence rather than pointwise identity.

Tests should distinguish these regimes.

---

# 13. Model

Use a conventional decoder-only transformer.

Initial components:

* token embeddings;
* causal attention;
* RoPE;
* LayerNorm or RMSNorm;
* GELU or SwiGLU;
* residual blocks;
* output projection.

Model architecture remains deliberately boring.

Architecture innovation would confound the curriculum experiment.

---

# 14. Functional Training API

Prefer:

```python
state = init_model(config, key)

state, curve = train_phase(
    state,
    phase_spec,
    data,
    key,
)

metrics = evaluate(
    state,
    eval_spec,
    key,
)
```

Avoid hidden globals.

Training state contains:

* parameters;
* optimizer state;
* step;
* token count;
* RNG lineage.

---

# 15. Statistical Utilities

`stats.py` initially implements:

* AULC;
* paired effect;
* confidence interval;
* equivalence interval check;
* threshold crossing;
* censoring summary.

Synthetic estimator tests operate entirely on arrays.

---

# 16. Target (t=0) Evaluation

The target family must be evaluated immediately before target training.

Store:

[
L_j(0).
]

This is mandatory.

It allows decomposition of:

* immediate transfer;
* acquisition-rate transfer.

Skipping this measurement makes the decomposition unrecoverable without rerunning experiments.

---

# 17. Artifact Schema

Every persisted artifact contains:

```text
schema_version
run_id
parent_id
config_hash
code_version
data_version
eval_version
status
```

Compute artifacts additionally include:

```text
provider
region
accelerator
start_time
end_time
wall_seconds
accelerator_seconds
tokens_seen
estimated_cost_usd
actual_cost_usd
```

---

# 18. Parquet Tables

## runs.parquet

```text
schema_version
run_id
parent_id
seed_family
curriculum
phase
tokens
checkpoint_uri
accelerator
cost_usd
status
```

## evaluations.parquet

```text
schema_version
run_id
checkpoint_step
suite_id
suite_version
example_id
condition
target
prediction
logprob
loss
```

## learning_curves.parquet

```text
schema_version
run_id
family
target_phase_step
tokens_seen
loss
accuracy
```

## transfer.parquet

```text
schema_version
source
target
seed_pair
control
t0_delta
aulc_delta
endpoint_delta
threshold_delta
stderr
cost_usd
```

## nulls.parquet

```text
schema_version
target
seed_pair
null_type
null_delta
hardware
```

---

# 19. Divergence Policy

No run is silently dropped.

Possible statuses:

```text
COMPLETE
INFRA_FAILED
RETRIED
DIVERGED
CANCELLED
```

Define prespecified thresholds for:

* NaNs;
* runaway loss;
* retry limits;
* divergence-rate invalidation.

Infrastructure failures may retry.

Scientific divergence remains in the dataset.

---

# 20. Capacity Calibration

At Milestone B, add Hydra because Cartesian sweeps now justify it.

Generate:

[
\text{model size}\times\text{token budget}.
]

Record:

* params;
* W score;
* P score;
* worst-family score;
* surface generalization;
* compositional generalization;
* (t_{10});
* (t_{90});
* learning window;
* cost.

Automated selector chooses smallest eligible model.

---

# 21. Nuisance HPO

Tune only calibration variables such as:

* learning rate;
* batch tokens;
* weight decay;
* warmup;
* source duration;
* target duration.

Do not tune separately for:

[
W\rightarrow P
]

and:

[
P\rightarrow W.
]

After calibration, freeze nuisance hyperparameters.

---

# 22. Null and Power Planner

Implement:

```python
estimate_noise(...)
plan_power(...)
```

Inputs:

* null paired variance;
* (\delta_{\min});
* target power;
* alpha.

Outputs:

* required paired seeds;
* estimated runs;
* GPU hours;
* expected cost.

No confirmatory batch launches before power planning succeeds.

---

# 23. W/P Experiment Generator

Generate:

```text
W → P
P → W
W → P → MIX
P → W → MIX
W-only
P-only
mixed
```

Experiments are immutable manifests.

All pairing relationships are explicit metadata.

---

# 24. Checkpointing

Orbax is added only when model state must cross:

* process boundaries;
* VM boundaries;
* long-running jobs.

Checkpoint at:

* developmental phase boundaries;
* analysis checkpoints;
* recovery intervals.

Always keep boundary checkpoints.

Prune redundant recovery checkpoints after durable upload.

---

# 25. Generic Corpus Abstraction

Add only when entering the multi-family stage.

```python
@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class Corpus:
    train: tuple[Document, ...]
    validation: tuple[Document, ...]
    test: tuple[Document, ...]
    version: str
```

No large ingestion framework is needed.

---

# 26. Corpus Loaders

Initial loaders may support:

* JSONL;
* Parquet;
* Hugging Face dataset references;
* text directory.

All normalize to `Corpus`.

The experiment code should not care about the source format.

---

# 27. Split Before Proposal

Workflow:

```text
raw corpus
   ↓
document-level split
   ↓
train / validation / test
   ↓
fit proposer on train only
   ↓
freeze proposer
   ↓
assign held-out documents
```

This is a hard invariant.

---

# 28. Family Proposal

Add:

```python
propose_families(corpus, proposer_spec)
```

Version 1 proposers:

* TF-IDF/LSA + clustering;
* frozen embeddings + clustering.

Optional later:

* semantic classifier;
* LLM-generated descriptions.

Do not begin with a plugin system.

---

# 29. Family Assignment Artifact

Persist:

```text
family_assignment.parquet
```

Fields:

```text
schema_version
document_id
split
family_id
proposal_version
representation_version
assignment_score
```

Also persist:

```text
families.json
```

with summaries and representative documents.

---

# 30. Ground-Truth Quarantine

Repository layout:

```text
data/
  public/
  hidden_ground_truth/
```

Discovery modules must not import hidden ground truth.

CI should include a test preventing prohibited import paths where practical.

Generator truth is used only by evaluation code.

---

# 31. Transfer Experiment Generator

For pair (D_i,D_j):

```text
shared checkpoint
      │
      ├── Di → Dj
      └── N  → Dj
```

Store paired observations, not independent arm summaries, as the primary statistical unit.

All cells in one matrix draw from comparable checkpoint pools.

---

# 32. Transfer Model

Start with simple estimators.

First baseline:

[
\mu+\alpha_i+\beta_j.
]

Then add:

[
\gamma_{ij}.
]

If hierarchical inference is warranted, NumPyro may implement:

[
y_{ijs}
\sim
\mathcal N(
\mu+\alpha_i+\beta_j+\gamma_{ij},
\sigma
).
]

Do not introduce NumPyro before the transfer pilot demonstrates need.

---

# 33. Prediction Baselines

Implement baseline predictors before complex models:

```text
global mean
source mean
target mean
source + target additive
symmetric T
semantic cosine
embedding kernel
family size
family frequency
```

Then compare against:

* ridge;
* low-rank factorization;
* developmental latent model.

---

# 34. Directionality Analysis

Implement:

[
S=\frac{T+T^\top}{2}
]

and:

[
A=\frac{T-T^\top}{2}.
]

Compare antisymmetric magnitude to null-calibrated expectations.

Store both components.

---

# 35. Developmental Phenotype

Represent family (i) as:

[
\phi_i=
[T_{i,*},T_{*,i}].
]

If low-rank:

[
\phi_i=[U_i,V_i].
]

This representation feeds merge/split proposal code.

---

# 36. Ontology Revision

Only after held-out prediction works, add:

```python
propose_merge(...)
propose_split(...)
score_revision(...)
```

Version 1 remains narrow.

No general ontology engine.

Acceptance uses a frozen validation criterion and complexity penalty.

Record number of attempted revisions to prevent repeated test-set search.

---

# 37. Held-Out Splits

Implement reusable functions:

```python
leave_pair_out(...)
leave_source_out(...)
leave_target_out(...)
leave_family_out(...)
```

Each split receives a stable ID and seed.

The final test partition is not reused for model-selection iteration.

---

# 38. Higher-Order Check

Add a specific experiment constructor for:

[
A+C\rightarrow B.
]

This tests whether the pairwise curriculum compiler is adequate.

No general higher-order graph machinery is required initially.

---

# 39. Active Experiment Selection

Only after predictive transfer modeling works.

Version 1 acquisition:

```python
score = uncertainty * importance / estimated_cost
```

Use batch diversity.

No Bayesian EIG system until justified.

---

# 40. Curriculum Compiler

Implement:

```python
compile_curriculum(
    transfer_model,
    families,
    uncertainty,
)
```

Output is a plain serializable artifact:

```json
{
  "phases": [
    {"families": ["D1", "D2"], "tokens": 100000},
    {"families": ["D1", "D2", "D3"], "tokens": 150000},
    {"families": ["D3", "D4"], "tokens": 150000},
    {"families": ["ALL"], "tokens": 300000}
  ]
}
```

Also automatically emit its reverse.

---

# 41. Fresh Validation

Generate fresh runs for:

```text
discovered
reversed
uniform
random
semantic/manual
```

with new initialization seeds.

Discovery checkpoints are never reused.

---

# 42. Portability Integration Test

The generic system should eventually support:

```python
result = discover_curriculum(
    corpus=corpus_b,
)
```

This is not intended as magic API design.

It is an architectural test that the pipeline does not contain corpus-specific scientific logic.

Internally:

```text
ingest
split
propose
calibrate
intervene
fit
validate
revise
compile
fresh train
report
```

Each stage produces immutable artifacts.

---

# 43. GCP Executor

Reuse and evolve the existing hardened runner rather than replacing it.

The prior script already includes:

* deadman shutdown;
* shutdown even after failed jobs;
* explicit billing status checks;
* zone fallback for scarce L4 capacity;
* driver bootstrap;
* memory gating;
* launch staggering;
* detached jobs;
* bounded retries.

## Those are valuable operational lessons.

# 44. GCP Interface

Evolve:

```text
up
run
logs
status
fetch
down
```

into:

```text
up
plan
submit
resume
logs
status
cost
fetch
down
```

The executor consumes manifests.

It does not generate experiments.

---

# 45. GCP `plan`

Report:

```text
candidate runs
unique prefix runs
checkpoint reuse
estimated GPU hours
estimated storage
estimated spend
reserved spend
remaining budget
```

No batch launches before planning succeeds.

---

# 46. Budget Reservation

Maintain:

```text
spent
reserved
remaining
```

Launch condition:

[
spent+reserved+estimated_{\mathrm{batch}}
\le
budget.
]

This prevents multiple parallel launches from each independently seeing the same remaining budget.

---

# 47. Budget Config

Example:

```yaml
budget:
  max_total_usd: 75
  max_batch_usd: 15
  max_run_usd: 2
  max_accelerator_hours: 100
  max_wall_hours: 8
  warn_fraction: 0.75
```

Exact values are project-configurable.

---

# 48. Cost Ledger

Persist:

```text
cost.parquet
```

with:

```text
schema_version
run_id
provider
machine
gpu
zone
start_time
end_time
accelerator_seconds
disk_gb_hours
network_bytes
estimated_usd
actual_usd
reserved_usd
```

Derived metrics include:

[
$/\text{paired effect}
]

[
$/\text{transfer cell}
]

[
$/\text{information gain}.
]

---

# 49. GCS Persistence

The VM disk is cache, not source of truth.

Upload incrementally:

* manifest;
* phase-boundary checkpoint;
* evaluation artifact;
* completion record;
* cost ledger.

The earlier GCP runner explicitly warns that artifacts left only on the VM must be fetched before deletion.

The new system removes that fragility by uploading continuously.

---

# 50. No Repo / Runtime Coupling

Hard invariant:

```text
git repository
≠
runtime artifact directory
≠
durable object storage
```

Do not use symlinks that allow Git operations to replace runtime directories.

This directly avoids a failure mode from the predecessor project.

---

# 51. Experiment-Level Parallelism

For small models:

```text
GPU0 → experiment A
GPU1 → experiment B
GPU2 → experiment C
GPU3 → experiment D
```

Do not shard one tiny model across multiple GPUs.

The existing runner's GPU pinning logic is appropriate for this model.

---

# 52. Host Memory Protection

Retain:

* memory gating;
* startup staggering;
* bounded retry.

These already exist because simultaneous corpus loading caused OOM failures in earlier experiments.

Expose thresholds through config.

---

# 53. Colab

Colab remains a lightweight frontend to the same package.

Use for:

* local-style quickstart;
* JAX debugging;
* estimator demonstration;
* W/P pilot;
* reproduction.

Notebooks should import package logic rather than contain unique research code.

---

# 54. Hugging Face

Use as publication/dissemination layer for:

* dataset;
* model checkpoints;
* family assignments;
* developmental graphs;
* Parquet results;
* reproduction notebook;
* interactive artifact.

The public explorer should rely mostly on precomputed artifacts to minimize serving cost.

---

# 55. UI Architecture

The UI sits outside training.

```text
JAX experiments
      ↓
Parquet / JSON / checkpoints
      ↓
query layer
      ↓
single interactive page
```

Start with a minimal explorer.

Build the bespoke HCI version only after the science stabilizes.

---

# 56. UI Data Contract

Expose:

```text
runs.parquet
evaluations.parquet
learning_curves.parquet
transfer.parquet
nulls.parquet
family_assignment.parquet
families.json
developmental_graph.json
checkpoint_metadata.parquet
activation_summaries.parquet
curriculum.json
cost.parquet
```

The UI never imports training internals.

---

# 57. UI Views

## Corpus

Representative documents and current family organization.

## Developmental graph

Clickable measured relationships.

## Intervention evidence

Show actual paired experiments behind each edge.

## Model interrogation

Aligned, conflict, W-only, P-only, counter-evidence, custom.

## Internal state

Probe and activation summaries.

## Developmental timeline

Checkpoint scrubber.

All views share selection state.

---

# 58. Figure Contract

Every empirical paper figure is produced from code.

Example:

```text
fig_01_overview.py
fig_02_capacity.py
fig_03_null_power.py
fig_04_wp_design.py
fig_05_wp_trajectories.py
fig_06_wp_diagnostics.py
fig_07_corpus_intake.py
fig_08_transfer_matrix.py
fig_09_directionality.py
fig_10_prediction_baselines.py
fig_11_ontology_revision.py
fig_12_curriculum_validation.py
fig_13_portability.py
```

No hand-edited experimental values.

---

# 59. Testing Strategy

## Unit tests

* RNG uniqueness;
* pair-sharing contract;
* null-divergence contract;
* canonical hashing;
* AULC;
* CI coverage;
* equivalence tests;
* target (t=0) measurement;
* threshold censoring;
* budget reservation;
* corpus split leakage.

## Integration tests

* identical `RunSpec` produces identical run ID;
* checkpoint resume matches uninterrupted run within tolerance;
* artifacts survive GCS round-trip;
* discovery cannot access hidden ground truth;
* generic Corpus B pipeline runs without corpus-specific code.

---

# 60. Stage-Based Implementation

## Milestone A

Statistics + RNG + specs + local W/P run.

## Milestone B

Capacity calibration + Hydra.

## Milestone C

Null/power planning + confirmatory W/P.

## Milestone D

GCP manifest executor + Orbax + GCS + budget ledger.

## Milestone E

Generic corpus intake + family proposal.

## Milestone F

Transfer matrix + partial pooling + prediction baselines.

## Milestone G

Ontology revision + held-out intervention prediction.

## Milestone H

Curriculum compiler + reverse + fresh validation.

## Milestone I

Blind Corpus B portability run.

## Milestone J

Polished interactive research artifact.

Each milestone is earned by the prior scientific gate.

---

# 61. Minimal End-to-End Loop

```python
validate_estimators()

model_config = calibrate_capacity(corpus)

noise = calibrate_nulls(
    model_config=model_config,
)

design = plan_power(
    sigma_pair=noise.sigma_pair,
    delta_min=config.delta_min,
)

wp_result = run_wp_confirmatory(
    model_config=model_config,
    design=design,
)

if not wp_result.valid:
    report_invalidated()
    stop()

if wp_result.equivalent_after_washout:
    report_negative_path_dependence()
    stop_or_narrow_claim()

corpus = load_corpus(source)

train, val, test = split_corpus(corpus)

families = propose_families(train)

observations = run_transfer_wave(
    families=families,
    design=design,
)

transfer_model = fit_transfer_model(
    observations,
)

prediction = evaluate_held_out(
    transfer_model,
    baselines=BASELINES,
)

if not prediction.beats_baselines:
    report_limited_structure()
    stop_or_narrow_claim()

families = revise_families(
    families,
    transfer_model,
)

curriculum = compile_curriculum(
    families,
    transfer_model,
)

results = fresh_validate(
    discovered=curriculum,
    reversed=reverse(curriculum),
    uniform=uniform(),
    random=random_schedule(),
    semantic=semantic_baseline(),
)

portability = run_frozen_pipeline(
    unseen_corpus=corpus_b,
)

report(
    wp_result,
    prediction,
    results,
    portability,
)
```

---

# 62. Final Architecture

```text
                    SCIENCE

raw corpus
   ↓
provisional family proposal
   ↓
controlled RunSpecs
   ↓
paired training interventions
   ↓
raw transfer measurements
   ↓
partial-pooled developmental model
   ↓
held-out prediction
   ↓
ontology revision
   ↓
curriculum compilation
   ↓
fresh-model validation
   ↓
unseen-corpus portability

────────────────────────────────────

                  EXECUTION

local / Colab / GCP
        ↓
JAX + Equinox + Optax
        ↓
Orbax when needed
        ↓
GCS / local artifact store
        ↓
Parquet / JSON

────────────────────────────────────

                INTERACTION

paper figures
reproduction notebooks
single-page developmental explorer
Hugging Face release
```

The codebase should remain small enough that the scientific argument is visible by reading it.

The system should grow only when experimental evidence earns the next abstraction.
