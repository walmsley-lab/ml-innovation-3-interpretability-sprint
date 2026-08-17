# Lane D — readiness dose-response, frozen before any outcome exists

**No dose-response continuation had been run or inspected when this was
written.** Frozen prospectively, so this is not exploratory-by-default; it
becomes exploratory only if any element below is changed after outcomes appear.

## 1. Question

Does readiness for a target the model **cannot perform** accumulate gradually
with exposure to the source, rather than appearing at a moment?

This complements the temporal-replay negative (T4). That experiment searched a
narrow window for a *localized* change in future learnability and found none.
This asks the different question of whether the quantity grows *smoothly across
the whole source phase*, which a narrow-window search could not have detected.

## 2. Design

* **Exposure levels:** source steps at **0, 25, 50, 75, 100%** of the frozen
  4000-step source phase — 0, 1000, 2000, 3000, 4000.
* **Condition:** **disjoint surface only** (`IND#h1` source, `BINDT#h2` target).
  Disjoint because it is the stronger B₂ condition and removes shared-token
  reuse as an explanation.
* **Target:** `BINDT` — retrieval composed with a derangement, so zero-shot
  transfer is blocked by construction.
* **Seeds:** 4 fresh source trajectories, seeds **990–993**, arm `A`.
* **Continuation:** identical from every checkpoint — 2000 steps, batch 64, one
  frozen data key, so branches differ only in exposure level.
* Total: 4 seeds x 5 levels = **20 continuations**.

## 3. Outcomes

**Primary:** `rate_only` on the target — area under the learning curve minus the
`t=0` value. Chosen because it isolates acquisition speed from any head start,
and because `final` is fragile here (1 in 19 units shows a late instability dip
that moves it by ~0.9).

**Control:** `t=0` competence on the target. It should remain **at or near
chance across every exposure level** — that is what makes the primary a
readiness measure rather than a competence measure. If `t=0` rises with
exposure, the derangement is not blocking transfer as designed and the primary
outcome cannot be interpreted as readiness.

## 4. Trend test, fixed now

**Spearman rank correlation** between exposure level and `rate_only`, pooled
across seeds, with the per-seed correlations also reported. A pre-specified
linear trend on exposure is reported as a secondary.

**Changepoint, sigmoid, and any other nonlinear shape are prohibited.** T1
already showed how easily a shape can be read into data whose sampling
resolution cannot support it. The question here is monotonic association, and
only that is tested.

## 5. Readings, fixed now

| outcome | reading |
|---|---|
| `rate_only` rises monotonically with exposure while `t=0` stays at chance | **readiness accumulates before competence** — the clearest available demonstration of development rather than endpoint comparison |
| `rate_only` flat | readiness is not a graded function of source exposure at this resolution; the B₂ effect is closer to all-or-nothing across the levels tested |
| `t=0` rises with exposure | the derangement is not blocking zero-shot transfer; the primary is uninterpretable and must be reported as such |
| trend present but non-monotonic | reported as observed. **No shape is fitted** |

## 6. Status

Frozen prospectively. If the design is honoured exactly, this is a
pre-registered result at n=4 seeds; the sample is small and the report must say
so. Any deviation after outcomes are seen demotes it to exploratory.

It does **not** enter the frozen six-experiment evidence ladder either way.
