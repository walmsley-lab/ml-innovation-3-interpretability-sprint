# E4b prospective prototype — frozen protocol

**Status: frozen before any outcome exists.** This document is hashed and
committed before the first source state is trained.

## 0. What this is, and what it is not

This is a **feasibility prototype of the prospective loop**, not a compressed
confirmatory E4b. Its purpose is to demonstrate that the loop —
*fresh states → fused endpoint readouts → complete `V(S,D)` matrix → fit on
development seeds → freeze predictions → reveal held-out outcomes* — closes
end to end without leakage.

**No claim of predictive developmental-state identification may be made from
this prototype, regardless of the numerical outcome.** With 9 development
states and 3 held-out states, it cannot establish predictive validity. That
requires the preregistered, powered E4b in `e4b_design.md`, which remains
unrun and ungated by this prototype.

The prototype cannot license Experiment 6 (`f9da9fe23b1b2400`). Nothing here
changes the frozen evidence ladder.

## 1. States — 12, all fresh

`4 seeds x 3 history arms`, using seeds never previously trained in this
project (500s, 700s, and 950s are all consumed).

| element | specification |
|---|---|
| seeds | **800, 801, 802, 803** |
| arms | `A` = `IND`, `A_prime` = `IND_R`, `BG` = `BG` |
| source length | **4000 steps** — the protocol-defined endpoint |
| architecture | `d_model` 128, 4 layers, 4 heads, `d_ff` 512, batch 64 |
| source data key | `10_000 + seed`, per existing convention |

Identical in every respect to `discover_mediator.py`'s frozen defaults, so the
states are commensurable with the existing population.

**Intermediate (<4000-step) checkpoints are not E4b states and are never
substituted for one.**

## 2. Split — by seed, not by state

Splitting by individual state would leak: three arms of one seed share an
initialization, so a sibling in the development set reveals its held-out
partner.

| set | seeds | states |
|---|---|---|
| development | 800, 801, 802 | 9 |
| **prospective reserve** | **803** | **3** |

The reserve seed is untouched by any fitting decision.

## 3. Fused endpoint readout

At the 4000-step endpoint the source runner atomically emits, in one unit:
checkpoint, behavioral readouts (`zero_shot_BIND`, `zero_shot_FACT`), internal
readouts (137 basis-invariant `state_features`, `retrieval_max`,
`retrieval_mean`, `M_scalar`), and provenance (arm, seed, steps, config).

Reserve states are generated and read out in advance. Their `V(S,D)`
continuations sit behind a hard release barrier (§6).

## 4. Candidate corpora — 4, declared now

`D` = **`BIND`**, **`BINDT`**, **`FACT`**, **`BIND+FACT`** (even-split mixture).

Three pure streams plus one mixture, so the argmax has a non-degenerate choice.
Continuations restore **weights only** with a fresh optimizer state, so the
measured quantity is `V(W_t, D)`, consistent with every prior `V(S,D)` cell in
this project.

| element | specification |
|---|---|
| continuation length | 2000 steps |
| probe interval | 250 steps |
| continuation data key | **777**, shared by every cell, so cells differ only in incoming state and corpus |

Matrix: `12 states x 4 corpora = 48 cells` (36 development, 12 reserve).

## 5. Objective and comparison

**Objective:** `mean` final accuracy across `BIND`, `FACT`, `BINDT` — the
common yardstick, never the capability a cell's own corpus trains. `mean` is
the E4b-frozen primary; `min` is not used, having been documented as
near-degenerate before any of this ran.

Four pre-declared readouts, compared on identical data:

1. **behavior-only** — `zero_shot_BIND` accuracy and loss, `zero_shot_FACT`
   accuracy and loss (4 features);
2. **internal-only** — the 137 state features, standardized and reduced to
   **2 principal components fit on development states only**;
3. **behavior + internal** — the union (4 + 2);
4. **state-blind global-best** — pick the corpus with the highest mean
   development objective, ignoring the state. **This is the competitor to
   beat.** Beating a random baseline is not a result.

**Predictor: 1-nearest-neighbour in standardized readout space.** For a query
state, predict the corpus that was best for the nearest development state.
Chosen because it has no hyperparameters to tune at n=9; any fitted model with
free capacity would be tuned on 9 points and mean nothing. Standardization and
PCA are fit on development states only and applied unchanged to the reserve.

**Reported:** top-1 hit count and mean regret against the per-state oracle.
Point estimates only, with the sample size stated beside them. **No
significance claim is made from 3 held-out states.**

## 6. Order of operations — the firewall

1. Generate 9 development source states; fuse readouts at the endpoint.
2. Fan out each development state's 4 continuations as soon as its checkpoint
   lands. Development work has scheduling priority over reserve generation.
3. Generate 3 reserve source states and their readouts concurrently. **Run no
   reserve continuation.**
4. When all 9 development rows are complete, fit the ladder and **freeze**: one
   scripted operation writes predictor parameters, the standardizer and PCA
   basis, the predicted best corpus for each of the 3 reserve states, the
   protocol hash, and a timestamp.
5. **Only then** release the 12 reserve continuation cells.
6. Score the frozen predictions against the revealed reserve outcomes.

Step 5 must not begin before step 4 has written its artifact. The analyzer
refuses to score if the freeze artifact is absent or its protocol hash does not
match this document.

## 7. Paper language

> We additionally implemented a small prospective prototype in which checkpoint
> measurements were used to forecast future corpus value before continuation
> outcomes were revealed. Because the pilot contains only 12 states and three
> held-out test states, it demonstrates feasibility of the experimental loop
> rather than predictive validity; establishing the latter requires the
> preregistered, powered E4b study.
