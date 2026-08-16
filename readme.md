# Developmental System Identification for Pretraining

Can the developmental structure that makes one training order preferable to
another be experimentally discovered rather than assumed?

The project treats pretraining as a developmental process and escalates
through three levels: establishing that training order causes persistent
differences, inferring predictive structure among corpus families, and
compiling a curriculum from that structure and validating it on fresh
models.

- [research.md](research.md) — the scientific program, stages and gates A–I
- [technical.md](technical.md) — implementation plan

Three objects stay distinct throughout: **intervention measurements**, the
**developmental model** fitted to them, and the **derived curriculum**. A
curriculum is never discovered first and given a graph afterwards.

## Status: Milestone A

| Exit criterion | State |
|---|---|
| Gate A: estimators recover known effects with calibrated coverage | passing |
| RNG sharing / divergence contracts | passing |
| `RunSpec` canonicalization stable | passing |
| Target-phase t=0 evaluation wired into the interface | enforced by `EvalSpec` |
| One local W→P / P→W paired experiment writing versioned Parquet | running |

89 tests. Gate A measures bias ≤ 7e-5 against a 2e-3 tolerance and interval
coverage of 0.947–0.951 against a nominal 0.95.

```bash
uv venv && uv pip install -e ".[dev]" && .venv/bin/python -m pytest
```

```bash
.venv/bin/python scripts/run_wp_local.py --seed-families 4 --steps 250
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
```

Deliberately absent until their milestone: Hydra and Orbax (B/D), the GCP
executor and budget system (D), partial pooling (E), transfer, discovery and
curriculum compilation (E–G), and any UI (H).

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

## Two things worth knowing early

**Positive means faster.** Every estimator is written as `control − treatment`
over losses. Inverting it would invert every edge in the developmental graph,
so the convention is asserted in tests rather than trusted.

**The t=0 evaluation is mandatory.** Evaluating before any target tokens are
seen is what separates a head start carried in from the source phase from a
genuinely faster acquisition rate. AULC conflates the two, and the
decomposition is unrecoverable after the run. `EvalSpec` refuses offsets that
omit `0.0` even though nothing yet consumes the measurement.
