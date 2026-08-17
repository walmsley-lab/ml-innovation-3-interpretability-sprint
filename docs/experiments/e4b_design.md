# E4b — powered predictive state readout (design only, not run)

**Design document. No E4b data exists. Nothing here is executed during the
sprint.** E4 is not expanded post hoc; this is its properly powered successor.

## 1. The question, stated so it cannot be satisfied by history recognition

> Do internal state measurements predict **held-out conditional data value**
> `V(S,D)` — which corpus is best *from this state* — rather than merely
> distinguishing which history produced the state?

That distinction is the entire point. E4 established that gradient geometry
separates histories cleanly (gnorm 0.54 vs 0.29; cos +0.73 vs +0.19) and still
did not resolve whether it predicts value. A representation that classifies
history perfectly and predicts value at chance is **not** a developmental-state
readout, and the project has now produced two such markers.

## 2. Why E4 was inconclusive — the constraint to fix

Not the geometry: 76 states were measured. The binding limit was the **overlap
with complete `V(S,D)` rows — 13 states**. E4b's central design requirement is
therefore that *every measured state has a complete V(S,D) row*, generated as
one pipeline rather than two that happen to intersect.

## 3. Power

The E4 bootstrap CI at n=13 spanned `[0.00000, +0.00285]` on a point estimate
of `+0.00105`. Half-width ≈ 0.0014 at n=13. Under a √n scaling, resolving an
effect of that size needs roughly **n ≥ 50 complete states**; detecting a
practically interesting effect (regret reduction ≥ 25% of the global-best
baseline) needs fewer, around **n ≥ 30**.

**Target: 60 states**, each with a complete row over the candidate corpora,
which also leaves room for a confirmatory holdout (§6).

## 4. Frozen before any outcome

| element | specification |
|---|---|
| **primary objective** | `mean` **collapse-robust** accuracy across capabilities (§4a). **Not bare `final`** — final-probe accuracy is corrupted by post-mastery collapse. **Not `min`** — `min` sat at the chance floor in every prior use, and E4's only apparent win came from it |
| **secondary** | `min`, reported but not decisive |
| **competitor** | state-blind **global-best**. Beating `random` is not a result |
| **validation** | leave-one-**state**-out. Row-wise splitting leaks, because other corpora on the same state reveal its value profile |
| **primary statistic** | mean regret against global-best, with bootstrap 95% CI over states |
| **success** | CI on the regret difference **excludes zero** in favour of the readout. Point estimates do not suffice — this is stated here precisely because E4 showed how misleading a bare point estimate is |

### 4a. The objective must be collapse-robust — required, with evidence

**A bare final-probe statistic is disqualified.** Continuations in this
substrate routinely master the target and then destabilise, so `final`
measures optimiser stability at one arbitrary probe as much as it measures
data value.

Two independent observations, both recorded before E4b is run:

* **P2 validation (48 continuations).** Every continuation reached BIND
  accuracy 1.0, and 21 of the 22 low-AULC units were post-mastery collapses:
  accuracy 1.0 falling to ~0.02–0.05 with loss rising from ~0.01 to ~4. The
  AULC ceiling was 0.905 with 26 of 48 units sitting on it.
* **Prospective prototype, cell `A__seed800 x BIND`** (protocol
  `018ddd863114f662`). Accuracy 1.000 at loss 0.001–0.002 held for 1750 of
  2000 steps, then collapsed **on the final probe** to 0.143 at loss 3.682.
  Under `mean` final accuracy that cell scores 0.0592, against 0.3385 and
  0.3418 for the `BINDT` and `FACT` cells from the same state. A cell that
  held the capability for 87% of its run is scored as though it never learned
  it.

The second case is the sharp one: the corpus that *taught the target best*
scores worst, purely because an instability event landed on the last probe.
An E4b inheriting `final` would rank candidate corpora partly by where each
run's instability happened to fall, and a state readout could then post a
respectable regret by predicting optimiser noise rather than data value.

**Requirement.** Before E4b runs, its objective must be specified as a
statistic that is robust to post-mastery collapse, and that specification
must be frozen and hashed with the rest of the protocol. Candidates, to be
chosen and fixed *before* outcomes exist:

1. mean accuracy over the last *k* probes, `k` fixed in advance;
2. post-convergence best, with the convergence criterion fixed in advance;
3. `final`, but with collapse detected by a prespecified rule and reported as
   a separate outcome rather than averaged into the objective.

Whichever is chosen, **collapse frequency is reported alongside the
objective, never silently absorbed into it.** An instability that changes
which corpus looks best is a result about the substrate and belongs in the
open, not smoothed away.

**This does not amend any frozen protocol.** The prospective prototype keeps
`mean` final accuracy as frozen and will be reported under it; changing an
objective after outcomes exist is the researcher degree of freedom this
project refuses. The requirement above binds E4b, which is unrun.

## 5. Candidate representations, declared in advance

Fitted and reported as separate pre-declared families, never merged after
seeing which wins:

1. **gradient/update geometry** — norms, layerwise mass entropy, gradient-weight
   alignment, cross-corpus gradient alignment;
2. **activation/representation** — the 137 basis-invariant features;
3. **retrieval marker** — the on-distribution attention statistic;
4. **combined 1+2+3**, with a PCA-dimension-matched variant so the combination
   cannot win on capacity alone.

## 6. Confirmatory split

40 states for development — feature choice, hyperparameters, everything. **20
held in reserve**, untouched until a single representation and single
predictor are frozen and hashed. Reported on the reserve set. One shot.

## 7. What a pass licenses, and what it does not

**Licenses:** Experiment 6, the frozen prospective tournament
(`f9da9fe23b1b2400`), which stays gated until then.

**Does not license:** any causal claim. A predictive readout shows the
information is present and legible, not that the measured quantity carries the
effect. Causal work needs an intervention that first demonstrates it can move
the state — a lesson already paid for.
