# Developmental System Identification for Pretraining

Can the developmental structure that makes one training order preferable to
another be experimentally discovered rather than assumed?

The project treats pretraining as a developmental process and escalates
through three levels: establishing that training order causes persistent
differences, inferring predictive structure among corpus families, and
compiling a curriculum from that structure and validating it on fresh
models.

- [research.md](research.md) — the scientific program and gates A–J
- [technical.md](technical.md) — implementation plan
- [report.md](report.md) — what has been measured, and when
- [RISKS.md](RISKS.md) — threats to validity and known failure modes
- [docs/experiments/](docs/experiments/) — per-experiment records

Three objects stay distinct throughout: **intervention measurements**, the
**developmental model** fitted to them, and the **derived curriculum**. A
curriculum is never discovered first and given a graph afterwards.

## Invalidators are not falsifiers

A broken apparatus licenses no conclusion in either direction. A sound
apparatus whose claim fails is a reportable negative result. The two are kept
apart everywhere, because conflating them is how a broken pipeline gets
written up as "no effect found".

## Gates

| Gate | Criterion | State |
|---|---|---|
| A | Estimator calibration | passing |
| B | Minimal adequate model selected | passing (regime frozen) |
| C | Null calibration and powered seed count | **failed — statistical invalidator** |
| D | Persistent W/P differentiation after washout | blocked by C |
| E | Adjacent-scale replication | not started |
| F | Directional transfer signal exceeds null and simple baselines | synthetic: viability shown; natural: **failed** |
| G | Held-out intervention prediction succeeds | **failed on natural corpus** |
| H | Ontology revision improves predictive compression | not started |
| I | Derived curriculum beats baselines and reverse | not started |
| J | Pipeline transfers unchanged to unseen natural corpus | pipeline transfers; prediction does not |

Gate A measures bias ≤ 7e-5 against a 2e-3 tolerance and interval coverage of
0.947–0.951 against a nominal 0.95.

Gate C is the live blocker. The apparatus calibrates cleanly through the
common tail, but the frozen neutral-default *endpoint* has within-history
variance too large for feasible confirmatory inference, so Claim 1 was never
run. That is an invalidator, not a negative result. See
[report.md](report.md).

Gates F and G were attempted on a natural corpus and **failed** their frozen
criteria: relational features lost to the global mean out of sample, and the
fallback source-only explanation then failed prospectively on pairs frozen
before they ran. Transfer itself is measurable; predicting it is what fails.

```bash
uv venv && uv pip install -e ".[dev]" && .venv/bin/python -m pytest
```

## Layout

```text
src/dsi/
  stats.py       Stage 0 estimators; no model dependency, validated first
  rng.py         role constants and fold_in derivation; defines the paired unit
  specs.py       frozen PhaseSpec/EvalSpec/RunSpec and canonical hashing
  data.py        synthetic W/P task: two independently learnable sources
  model.py       conventional decoder-only transformer
  train.py       functional training; constant LR by default
  eval.py        diagnostic suite: aligned, w_only, p_only, conflict
  artifacts.py   versioned Parquet
  corpus.py      arbitrary-corpus intake: dedup, split-before-proposal, audit
  layer2.py      synthetic compositional families over shared primitives
  natural.py     train-only vocabulary and deterministic chunking
  calibrate.py   Gate B adequacy criteria
  power.py       null calibration and seed planning
```

Deliberately absent until earned: Hydra and Orbax, partial pooling,
curriculum compilation, and any UI. The GCP executor exists in
`scripts/gcp.py` and is used for wave execution.

## The paired experimental unit

The unit of analysis is the **pair**, not the run. One observation of `T_ij`
costs two training runs and does not decompose into them.

```text
treatment:  θ₀(s) --[ D_i, m_s ]--> --[ D_j, m_t ]--> L_j^(i)(t)
control:    θ₀(s) --[ N,   m_s ]--> --[ D_j, m_t ]--> L_j^N(t)

T_ij^(s) = ∫₀^{m_t} [ L_j^N(t) − L_j^(i)(t) ] dt
```

Both arms hold the base checkpoint, target example ordering, evaluation
protocol, token budgets and optimizer schedule identical, and differ in
exactly one thing: the identity of the source-phase corpus. An identity-null
pair differs in exactly the same place — an independent draw of the same
condition — which is what makes σ_pair measured from nulls the right
yardstick for treatment pairs. `rng.py` enforces this and `tests/test_rng.py`
asserts it, because a pairing bug is silent: the runs complete, the numbers
look plausible, and every downstream conclusion is wrong.

`RunSpec` carries one `seed_family` rather than four independent seeds. Both
arms of a pair pass the same value; `arm` forces divergence on the source
stream alone, and is `None` for transfer pairs and distinct integers for
identity-null pairs.

## Two things worth knowing early

**Positive means faster.** Every estimator is written as `control − treatment`
over losses. Inverting it would invert every edge in the developmental graph,
so the convention is asserted in tests rather than trusted.

**The t=0 evaluation is mandatory.** Evaluating before any target tokens are
seen is what separates a head start carried in from the source phase from a
genuinely faster acquisition rate. AULC conflates the two, and the
decomposition is unrecoverable after the run. `EvalSpec` refuses offsets that
omit `0.0` even though nothing yet consumes the measurement.

## Smoke configuration

`scripts/run_wp_local.py` and the `n_digits=2` task configuration are
**smoke-only**. They exist to exercise the machinery end to end, not to
produce evidence. The regime is uncalibrated: model scale, task difficulty,
phase duration and seed count are all placeholders that Gates B and C
replace.

The recorded smoke run fails its competence gate and is retained as an
**invalidated exploratory run**. Training on the second family erases the
first, so the apparent order effect is recency — the confound the washout
phase exists to control. Nothing measured in that regime, including its
`P→W` retention gradient, may be used to tune the confirmatory design.
