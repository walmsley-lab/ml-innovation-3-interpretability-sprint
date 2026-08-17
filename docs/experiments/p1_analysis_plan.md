# P1 analysis plan — frozen before any continuation outcome exists

Governs the hidden-futures test of `downstream_protocols.md` Protocol A.
Pair list frozen at `8392ad37bb953ebf`; stopping rule at `01c89adc9b66b9b6`.

**No continuation from the frozen pair list had been run or inspected when this
was written.**

## 1. What is being tested

For pairs of states matched on **present observables only**, do futures diverge
under an identical subsequent training run?

    Behaviour(S1) ~ Behaviour(S2)  now,  and  V(S1,D) != V(S2,D)  after

## 2. The estimand

Within-pair absolute divergence in `V(W, BIND)`:

    div(i,j) = | V_i - V_j |

reported separately for **final accuracy**, **`t=0`**, and **`rate_only`**.
These are never summed: head start and acquisition rate carry opposite signs in
most of this project's measurements and are individually more reproducible than
their sum.

## 3. The null, predeclared

**Random pairing within arm.** For each arm, all same-arm pairs that were *not*
matched form the reference distribution. This holds arm constant, so the
comparison isolates matching quality rather than history.

Significance by **permutation**: 10,000 reshuffles of the matched/unmatched
labels within arm, two-sided, on the difference in mean divergence.

## 4. What is reported, in full

* The **complete divergence distribution** over all frozen pairs — not a
  summary statistic alone. Histogram plus quantiles.
* **Effect size**: difference in mean divergence, and Cliff's delta as a
  non-parametric effect size robust to the skew expected here.
* **Uncertainty**: bootstrap 95% CI over pairs, 10,000 resamples.
* **Pre-continuation matching distance beside post-continuation divergence**,
  per pair, so a reader can judge whether "matched" pairs were actually matched
  and whether divergence tracks residual mismatch. A scatter of distance
  against divergence is part of the primary output, not an appendix.
* Results **per arm** as well as pooled, since arms differ in how tightly
  their states cluster.

## 5. Prohibited

* Selecting showcase pairs by outcome. Individual pairs may be shown **only**
  after the aggregate result is reported, and only for visualization.
* Dropping pairs after seeing divergence.
* Reporting a subset of the three metrics because one is more favourable.
* Re-deriving `epsilon` or the pair list.

## 6. Time-bounded subsample rule, decided now

The 71 pairs have a **fixed run order**, generated with seed `20260817` and
recorded in `artifacts/p1_frozen_pairs.json` before any continuation ran.
Continuations execute in that order.

At the research cutoff, **whatever prefix has completed is the reported
sample**. Because the order was fixed in advance and is independent of every
outcome, any prefix is an outcome-blind random subsample by construction — no
decision is required at cutoff, and no post-hoc selection is possible.

If the full set does not complete, the report states plainly: *"a time-bounded
prefix of N of 71 pairs, ordered by a seed fixed before any outcome existed."*
A partial run is reported as partial. It is not presented as the full set, and
the missing pairs are not characterized.

## 6b. Observed instability, recorded before the aggregate result

1 of the first 19 completed continuations ends >0.3 below its own trajectory
maximum: accuracy reaches 1.000, holds, and dips at a single late checkpoint.
The other 18 are clean. This was noticed while building the demo figure and is
recorded here because it affects how the three metrics should be read, not
because it changes them.

`final` is therefore the most fragile of the three — a single late dip moves it
by ~0.9 — while `rate_only` and the area under the curve are robust to one bad
checkpoint. **All three remain reported as frozen.** No metric is dropped, and
none is reweighted; the reader is simply told which is brittle and why.

The same fragility applies to *visual* pair selection: ranking pairs by
final-point divergence preferentially surfaces the unstable unit. Figure panels
therefore rank on AULC divergence. This affects only which pairs are drawn, not
the aggregate statistic, which uses every frozen pair.

## 7. Readings, fixed now

| outcome | reading |
|---|---|
| matched pairs diverge more than unmatched, CI excludes zero | behaviour underdetermines future learnability — the claim |
| no difference | behaviour is sufficient at this resolution — a real, reportable negative |
| divergence tracks residual matching distance | the apparent effect is imperfect matching, not hidden state. **This is the most dangerous confound and is why the scatter is primary** |
