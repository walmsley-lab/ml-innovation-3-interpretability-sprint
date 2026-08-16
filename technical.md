# Technical Plan: Developmental System Identification for Pretraining

## 1. Engineering Philosophy

The implementation should be:

* functional where practical;
* small;
* explicit;
* reproducible;
* cheap to branch;
* highly parallel at the experiment level;
* deterministic enough to support paired experimentation;
* budget aware;
* infrastructure agnostic.

The core principle is:

[
\boxed{
\text{state/configuration as data}
+
\text{behavior as functions}
}
]

The research code defines experiments.

Execution infrastructure merely runs them.

---

# 2. Primary Technology Stack

## Model and optimization

* **JAX**
* **Equinox**
* **Optax**
* **Orbax**

JAX fits the project particularly well because explicit state, explicit RNG keys, function transformation, vectorization, and immutable-style computation align with controlled developmental branching.

## Configuration

* Hydra

## Analysis

* Polars
* PyArrow / Parquet
* NumPy
* SciPy
* scikit-learn
* NetworkX
* Matplotlib

## Statistical modeling

Initial:

* SciPy / statsmodels where sufficient

If hierarchical inference becomes useful:

* NumPyro is the natural JAX-aligned option.

Do not introduce probabilistic programming until the transfer pilot justifies it.

## HPO

* Optuna or Ray Tune
* used only for nuisance/calibration parameters.

## Cloud execution

* existing GCP runner, evolved rather than replaced.

The prior runner already protects against runaway billing with a deadman shutdown, failed-job shutdown, and delete-not-stop workflow.

It also already handles real operational failures such as GPU quota ambiguity and scarce L4 capacity by trying multiple zones.

---

# 3. Repository Structure

Keep the repository small.

```text
src/
  config.py
  data.py
  model.py
  train.py
  eval.py

  specs.py
  interventions.py
  executor.py

  stats.py
  transfer.py
  discovery.py
  curriculum.py

  artifacts.py
  budget.py
  plots.py

configs/
  model/
  experiment/
  sweep/
  budget/

notebooks/
  00_quickstart.ipynb
  01_capacity.ipynb
  02_wp.ipynb
  03_transfer.ipynb
  04_explorer.ipynb

scripts/
  run_local.sh
  gcp.py

ui/
  ...

tests/
  ...

artifacts/
```

Avoid a large `experiments.py` god-module.

Keep separate:

* spec generation;
* selection;
* execution;
* inference;
* visualization.

---

# 4. Immutable Core Specifications

Use frozen dataclasses or equivalent immutable PyTrees.

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

The scientific layer generates `RunSpec`s.

The executor does not decide what experiment should exist.

---

# 5. Canonical Run Hashing

Every run receives a stable content address:

[
run_id
======

H(
RunSpec,
code_version
).
]

Canonicalization requirements:

* sorted keys;
* explicit schema version;
* stable JSON or msgpack encoding;
* normalized floats;
* no local absolute paths;
* no machine hostname;
* pinned code version;
* immutable dataset/eval version identifiers.

Different infrastructure should produce the same scientific hash.

Hardware metadata belongs in the resulting artifact, not the scientific run identity unless the hardware itself is an experimental variable.

---

# 6. JAX RNG Discipline

Every `RunSpec` receives one root key.

Derive all stochastic streams explicitly using `jax.random.fold_in`.

Conceptually:

```python
root = key(run_seed)

init_key = fold_in(root, INIT_ROLE)
source_key = fold_in(root, SOURCE_DATA_ROLE)
target_key = fold_in(root, TARGET_DATA_ROLE)
branch_key = fold_in(root, BRANCH_ROLE)
eval_key = fold_in(root, EVAL_ROLE)
```

Phase keys also fold in phase index.

Never reuse a JAX key.

Document role constants centrally.

Paired branches share the RNG streams they are intended to share and diverge only on explicitly defined roles.

---

# 7. Model

Use a deliberately conventional decoder-only transformer.

Example structure:

* token embedding;
* 4–8 transformer blocks;
* LayerNorm or RMSNorm;
* causal attention;
* RoPE;
* GELU or SwiGLU;
* output projection.

Do not innovate on architecture while studying curriculum.

Model configuration is frozen after capacity calibration.

---

# 8. Functional Training API

Prefer signatures resembling:

```python
state = init_model(config, key)

state, metrics = train_phase(
    state,
    phase,
    data,
    key,
)

metrics = evaluate(
    state,
    eval_spec,
    key,
)
```

Avoid hidden global state.

Training state contains:

* parameters;
* optimizer state;
* step;
* RNG lineage;
* token count.

---

# 9. Orbax Checkpointing

Checkpoint at:

* phase boundaries;
* analysis checkpoints;
* safe recovery intervals.

Store:

* parameters;
* optimizer state;
* step;
* RNG metadata;
* parent run ID;
* data version;
* config hash;
* code version;
* accelerator metadata.

Retention policy:

* always keep boundary checkpoints;
* keep selected analysis checkpoints;
* prune redundant intermediate recovery checkpoints after verified upload.

---

# 10. Artifact Schema

Every artifact contains:

```text
schema_version
run_id
parent_id
config_hash
code_version
data_version
eval_version
accelerator
provider
region
start_time
end_time
wall_seconds
accelerator_seconds
tokens_seen
estimated_cost_usd
actual_cost_usd
status
```

Use Parquet for tabular results.

---

# 11. Core Tables

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

## transfer.parquet

```text
schema_version
source
target
seed_pair
control
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
null_delta
estimator
hardware
```

---

# 12. Data Lineage

Every dataset has an immutable version ID.

Keep hidden generator metadata physically or logically separate from discovery-visible data.

Suggested structure:

```text
data/
  public/
    examples.parquet
    semantic_features.parquet

  hidden_ground_truth/
    latent_families.parquet
    planted_graph.json
```

Discovery code must not import or access `hidden_ground_truth`.

Evaluation scripts may.

---

# 13. Model Capacity Sweep

Generate a small Cartesian sweep:

```text
model_size × token_budget
```

Use Hydra to produce `RunSpec`s.

Use Optuna only if the search space becomes large.

The capacity calibration output should be a compact table containing:

* params;
* tokens;
* W score;
* P score;
* worst-family score;
* surface generalization;
* compositional generalization;
* (t_{10});
* (t_{90});
* learning window;
* cost.

Automated selection:

```python
eligible = [
    row for row in results
    if row.w >= tau_w
    and row.p >= tau_p
    and row.generalization >= tau_g
    and row.worst_family >= tau_family
    and row.learning_window >= min_window
]

chosen = min(eligible, key=lambda x: x.params)
```

Freeze the resulting configuration.

---

# 14. Nuisance HPO

Tune only:

* learning rate;
* weight decay;
* batch tokens;
* warmup;
* source duration;
* target duration if necessary.

Use calibration data, not the primary W/P comparison.

After tuning, freeze these values.

Do not tune separately by curriculum.

---

# 15. Estimator Unit Tests

Before cloud experiments, simulate known learning curves.

For example:

```python
curve_control = synthetic_curve(...)
curve_effect = synthetic_curve(delta=0.05)
```

Test:

* AULC estimate;
* paired difference;
* CI coverage;
* threshold censoring logic;
* hierarchical shrinkage behavior.

These tests run on CPU.

---

# 16. Null Calibration Runner

Implement identity-null generation as a first-class experimental family.

```python
make_null_pair(
    target="B",
    prefix_family="neutral",
    seed_pair=...
)
```

Use the same checkpoint, token budgets, and target data as scientific transfer experiments.

Null outputs feed the power estimator.

---

# 17. Power Planner

Input:

* pilot paired variance;
* minimum effect size;
* alpha;
* desired power.

Output:

* required seed count;
* expected GPU hours;
* estimated dollar cost.

Example output:

```text
sigma_pair        0.041
delta_min         0.050
power             0.90
required_pairs    9

estimated runs    18
estimated cost    $4.80
```

The planner should run before confirmatory batches can be submitted.

---

# 18. W/P Experiment Generator

Produce matched branches automatically.

```python
make_wp_pair(
    seed_family=s,
    source_tokens=...,
    target_tokens=...,
    washout_tokens=...,
)
```

Generated branches include:

```text
W → P
P → W
W → P → MIX
P → W → MIX
W-only
P-only
mixed baseline
```

All scientific comparisons are declaratively represented in manifest metadata.

---

# 19. Divergence Policy

Runs must never be silently discarded.

Define:

* NaN threshold;
* loss-spike threshold;
* retry count;
* environment-failure classification;
* scientific-divergence classification.

An infrastructure failure may be retried once.

A scientifically diverged training run remains in the dataset with status:

```text
DIVERGED
```

and is analyzed according to a prespecified rule.

---

# 20. Transfer Matrix Execution

For each (D_i,D_j), generate:

```text
Di → Dj
N  → Dj
```

from the same checkpoint pool.

Checkpoint identity should be modeled as a blocking/random effect.

All cells of a matrix use comparable checkpoint pools.

Avoid row-specific ancestral checkpoints unless explicitly modeled.

---

# 21. Partial-Pooling Model

Start with a transparent hierarchical normal model.

Possible implementation in NumPyro:

[
y_{ijs}
\sim
\mathcal N(
\mu+\alpha_i+\beta_j+\gamma_{ij},
\sigma
).
]

Use posterior means or regularized estimates for graph and curriculum selection.

Raw paired effects remain available.

If NumPyro adds too much complexity initially, approximate with regularized mixed-effects regression and migrate later.

---

# 22. Transfer Predictor

Implement baseline models first.

```text
global mean
row mean
column mean
row + column
symmetric estimate
semantic cosine similarity
embedding kernel
family size/frequency
```

Then:

```text
ridge regression
matrix factorization
low-rank latent model
```

Only advance to more sophisticated models if predictive performance justifies it.

---

# 23. Ontology Representation

Use developmental phenotype:

[
\phi_i=
[T_{i,*},T_{*,i}].
]

For low-rank representation:

[
\phi_i=[U_i,V_i].
]

Merge/split proposals operate on these phenotypes, not semantic labels alone.

---

# 24. Ontology Search

Keep version 1 narrow.

### Merge proposal

Nearest developmental-phenotype neighbors.

### Split proposal

Unsupervised clusters within a family.

### Acceptance

Use frozen validation protocol and complexity penalty.

Do not search repeatedly against the final test set.

---

# 25. Held-Out Evaluation Engine

Implement split strategies as reusable functions:

```python
leave_pair_out(...)
leave_source_out(...)
leave_target_out(...)
leave_family_out(...)
```

Each produces immutable train/test intervention IDs.

Store the split definition and seed.

---

# 26. Active Selection

Later stage only.

Version 1 acquisition:

```python
score = uncertainty * importance / estimated_cost
```

Batch selection adds a diversity penalty.

Every candidate carries expected:

* GPU seconds;
* dollars;
* information value;
* checkpoint reuse.

---

# 27. Curriculum Compiler

Input:

* shrunk transfer model;
* ontology;
* uncertainty;
* interaction warnings.

Output:

```json
{
  "phases": [
    {"families": ["D1", "D2"], "tokens": 100000},
    {"families": ["D1", "D2", "D3"], "tokens": 150000},
    {"families": ["D3", "D4", "D5"], "tokens": 200000},
    {"families": ["ALL"], "tokens": 300000}
  ]
}
```

Also automatically emit the reversed curriculum.

---

# 28. Fresh Validation Generator

Generate new run families:

```text
discovered
reversed
uniform
random
semantic/manual
```

with fresh initialization seeds.

Discovery checkpoints are never reused.

---

# 29. GCP Executor

Preserve the successful operational model of the existing script.

Its current command surface is already simple and useful:

`up`, `run`, `logs`, `status`, `fetch`, `down`.

Evolve it to:

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

### `plan`

Read experiment manifest and report:

* candidate runs;
* unique prefix runs;
* checkpoint reuse;
* GPU hours;
* storage;
* estimated spend;
* reserved spend;
* remaining budget.

### `submit`

Reserve projected cost and launch the immutable manifest.

### `resume`

Resume only incomplete artifacts.

### `cost`

Read actual and reserved spend from the experiment ledger.

---

# 30. Preserve Existing GCP Safety

Keep the three-layer protection already encoded in the prior runner:

1. arm shutdown before launch;
2. power off even if the job fails;
3. manually verify status afterward.

Continue deleting rather than merely stopping disposable research VMs.

The prior script correctly notes that stopped instances can continue incurring disk costs.

---

# 31. GCS Persistence

Change output persistence from “fetch before deleting” to incremental remote storage.

The previous runner warns that results left only on the VM must be fetched before deletion or written to GCS during the run.

For this project:

```text
manifest
boundary checkpoint
evaluation artifact
run completion record
cost ledger
```

should upload to GCS as soon as created.

The VM disk becomes cache rather than source of truth.

---

# 32. Multi-GPU Strategy

For small models, parallelize experiments rather than one model.

If a VM contains four GPUs:

```text
GPU0 → experiment A
GPU1 → experiment B
GPU2 → experiment C
GPU3 → experiment D
```

The existing runner already supports GPU pinning with `CUDA_VISIBLE_DEVICES`.

Retain that model initially.

Do not use multi-GPU sharding merely because JAX supports it.

Experiment-level replication is the primary scaling axis.

---

# 33. Host Memory Protection

Preserve:

* memory gating;
* launch staggering;
* bounded retry.

The previous script includes these because simultaneous corpus-loading peaks caused OOM kills in prior work.

For this project, additionally expose the memory threshold through config.

---

# 34. Spot / Preemption Strategy

Design runs to tolerate preemption.

Because boundary checkpoints are uploaded to GCS:

```text
preemption
→ new VM
→ restore parent checkpoint
→ resume incomplete RunSpec
```

No scientific branch should depend on a VM surviving.

---

# 35. Budget System

Configuration:

```yaml
budget:
  max_total_usd: 75
  max_batch_usd: 15
  max_run_usd: 2
  max_accelerator_hours: 100
  max_wall_hours: 8
  warn_fraction: 0.75
```

Maintain:

```text
spent
reserved
remaining
```

A batch cannot launch unless:

[
spent+reserved+estimated_{batch}
\le
budget.
]

Reservation is released when a job completes or is cancelled.

---

# 36. Cost Ledger

`cost.parquet`:

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
reservation_usd
```

Cost per scientific output:

[
$/\text{paired effect}
]

[
$/\text{transfer cell}
]

[
$/\text{information gain}
]

should be derivable.

---

# 37. Colab

Colab is an alternate execution frontend for the same package.

Notebooks should contain almost no unique research logic.

Example:

```python
from dsi.experiments import wp
from dsi.executor import local

specs = wp.generate(config)
local.run(specs)
```

Use Colab for:

* quickstart;
* capacity pilots;
* debugging JAX compilation;
* reproduction;
* exploratory plots.

Do not rely on Colab for large confirmatory matrices.

---

# 38. Hugging Face

Use Hugging Face as the dissemination layer.

Publish:

* synthetic dataset;
* final checkpoints;
* developmental graphs;
* selected Parquet artifacts;
* model card;
* reproducibility notebook;
* interactive research artifact.

The public UI should operate primarily over precomputed artifacts so that continuous GPU hosting is unnecessary.

---

# 39. Interactive Interface Architecture

Keep the UI outside the scientific core.

```text
JAX experiments
      ↓
GCS / local artifacts
      ↓
Parquet + JSON
      ↓
small query layer
      ↓
interactive page
```

Start with a lightweight prototype.

Possible early stack:

* Gradio or Streamlit.

Only build a bespoke frontend after Stage 6/7 demonstrates a compelling phenomenon.

---

# 40. UI Data Contract

Expose:

```text
runs.parquet
evaluations.parquet
learning_curves.parquet
transfer.parquet
nulls.parquet
developmental_graph.json
checkpoint_metadata.parquet
activation_summaries.parquet
curriculum.json
cost.parquet
```

The UI never imports training internals.

---

# 41. UI Views

## Developmental map

Linked to actual intervention evidence.

## Model interrogation

Modes:

```text
Aligned
Conflict
W-only
P-only
Counter-evidence
Custom
```

## Internal state

Compact:

* layer × signal;
* W/P probe response;
* representation similarity;
* selected feature activation.

## Developmental timeline

Checkpoint scrubber.

All panels share selection state.

---

# 42. Paper Figure Contract

Generate every paper figure from code.

```text
fig_01_overview.py
fig_02_capacity.py
fig_03_null_power.py
fig_04_wp_design.py
fig_05_wp_trajectories.py
fig_06_wp_diagnostics.py
fig_07_transfer_matrix.py
fig_08_asymmetry.py
fig_09_developmental_graph.py
fig_10_prediction_baselines.py
fig_11_curriculum_validation.py
```

No manually edited experimental numbers.

---

# 43. Testing Strategy

## Unit tests

* RNG key uniqueness;
* RunSpec canonicalization;
* artifact hashing;
* AULC estimation;
* threshold censoring;
* merge/split scoring;
* budget reservation.

## Integration tests

* same RunSpec twice on same hardware gives metrics within tolerance;
* same RunSpec across hardware gives distributionally comparable output;
* checkpoint resume matches uninterrupted training within tolerance;
* GCS artifact round-trip works.

Do not promise bitwise reproducibility across hardware.

---

# 44. Data and Schema Versioning

Every persisted table includes:

```text
schema_version
```

Every dataset/eval suite has immutable lineage ID.

Changing a schema does not mutate old artifacts.

Readers migrate old artifacts explicitly.

---

# 45. Stage-Based Compute Plan

### Stage 0

CPU estimator tests.

Target spend:

[
\approx $0.
]

### Stage 1

Small capacity pilot.

### Stage 2

Noise/power calibration.

### Stage 3

W/P confirmatory runs.

### Stage 4

Adjacent-scale replication.

### Stage 5

Small transfer matrix.

### Stage 6

Held-out prediction and ontology revision.

### Stage 7

Fresh curriculum validation.

### Stage 8

Polished interactive artifact.

Every stage requires explicit exit criteria before additional compute is authorized.

---

# 46. Implementation Sequence

## Milestone A — Local functional core

Build:

* model;
* optimizer;
* synthetic W/P generator;
* functional trainer;
* eval;
* RunSpec;
* artifact writer.

## Milestone B — Capacity + statistics

Build:

* capacity sweep;
* estimator simulation;
* null calibration;
* power planner.

## Milestone C — W/P science

Run:

* W→P;
* P→W;
* washout;
* competence controls;
* multi-seed inference.

## Milestone D — Cloud execution

Adapt the existing GCP runner for:

* manifests;
* GCS;
* budget reservation;
* resumability.

## Milestone E — Transfer matrix

Build:

* structured corpus;
* pair generator;
* pooled estimator;
* baseline predictors.

## Milestone F — Discovery

Build:

* developmental phenotypes;
* merge/split;
* held-out prediction;
* asymmetry tests.

## Milestone G — Control

Build:

* compiler;
* reverse compiler;
* fresh validation.

## Milestone H — HCI artifact

Build the polished linked-view interface only after the scientific phenomenon is established.

---

# 47. Minimal Core Loop

Conceptually:

```python
model_config = calibrate_capacity(corpus)

noise = calibrate_null(
    model_config=model_config,
    corpus=corpus,
)

design = choose_powered_design(
    sigma=noise.sigma_pair,
    delta_min=config.delta_min,
)

wp_results = run_wp_experiment(
    design=design,
    model_config=model_config,
)

if not wp_results.persistent_path_dependence:
    report_negative_result()
    stop_or_revise()

families = propose_families(corpus)

observations = run_initial_transfer_wave(
    families=families,
    design=design,
)

while budget.remaining():

    transfer_model = fit_partial_pooling_model(
        observations
    )

    prediction = evaluate_held_out(
        transfer_model,
        baselines=BASELINES,
    )

    if not prediction.beats_semantic_baseline:
        break

    families = revise_ontology(
        families,
        transfer_model,
    )

    if prediction.good_enough:
        break

    batch = select_next_batch(
        uncertainty=prediction.uncertainty,
        cost=budget.cost_model,
    )

    observations += execute(batch)

curriculum = compile_curriculum(
    transfer_model,
    families,
)

results = fresh_validate(
    discovered=curriculum,
    reversed=reverse(curriculum),
    uniform=uniform(),
    random=random_schedule(),
)
```

---

# 48. Final Architecture

The complete system remains intentionally layered:

```text
                    SCIENCE

candidate corpus structure
          ↓
RunSpec generation
          ↓
controlled interventions
          ↓
effect measurements
          ↓
statistical model
          ↓
held-out prediction
          ↓
ontology revision
          ↓
curriculum compiler

────────────────────────────────────

                 EXECUTION

Local / Colab / GCP
          ↓
JAX + Equinox + Optax
          ↓
Orbax checkpoints
          ↓
GCS / local artifacts
          ↓
Parquet

────────────────────────────────────

                INTERACTION

paper figures
notebooks
interactive developmental explorer
Hugging Face release
```

The project should remain simple enough that the scientific argument is visible in the code.

Infrastructure should never become the experiment.
