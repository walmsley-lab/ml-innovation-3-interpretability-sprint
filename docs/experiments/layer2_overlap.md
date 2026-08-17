# r = 0.20 overlap diagnostic — result

Manifest `cd73c7444818dcfcabeda2dd80bb0c2c7cec9114788559a9125d2c3fb52f3f10`,
15/15 units, every family receiving exactly 600 steps in every arm.

## Retention: restored, partially

Per-family final accuracy:

| arm | F1 | F4 | F5 | F6 | min |
|---|---|---|---|---|---|
| `predicted_best` | 1.000 | 0.755 | 0.696 | 0.552 | **0.529** |
| `exact_reverse` | 0.839 | 0.611 | 0.649 | 0.947 | **0.567** |
| `balanced_shuffled` | 1.000 | 0.951 | 0.998 | 0.997 | **0.951** |

Against the ceiling scout's min-accuracy of **0.106**, overlap lifts the
sequential arms to **0.53–0.57**. Failure mode 1 was real and `r = 0.20`
substantially addresses it. Retention is **not** fully restored — interleaving
still holds 0.951 — but the catastrophic collapse is gone.

## The order effect: absent

| comparison | value |
|---|---|
| `predicted_best` − `exact_reverse` (min acc) | **−0.0383** |
| pooled between-seed sd | 0.2556 |
| **effect / noise** | **−0.15** |

The predicted-best ordering does not beat its exact reverse. It is
fractionally worse, deep inside noise. This holds with retention restored and
with order fully decoupled from dose.

Note the seed instability: the sequential arms have sd 0.27 and 0.24 against
interleaving's **0.016**. Ordering is not merely unhelpful here, it is
erratic.

## Efficiency: interleaving wins decisively

| metric | best | reverse | interleaved |
|---|---|---|---|
| fixed-budget min acc | 0.5291 | 0.5674 | **0.9510** |
| fixed-budget mean acc | 0.7506 | 0.7614 | **0.9867** |
| mean familywise AUC | 0.5299 | 0.5238 | **0.8478** |
| joint steps to 0.90 on **all** families | 1/5 seeds | **0/5 seeds** | **5/5 seeds, median 950** |

Interleaving reaches the joint capability threshold in every seed in a median
of 950 of the 2,400 available steps. The best pairwise-derived ordering
reaches it in one seed out of five; the reverse ordering never does.

`best − interleaved` on the primary metric is **−0.4219**.

## Barrier verdict — Branch 2

**Retention restored, `predicted_best` ≈ `exact_reverse`.** Per the branch
fixed before the result was visible: this is evidence that **local pairwise
transfer does not straightforwardly compose into useful multi-stage
orderings**, and the static ordering construction is **not** to be patched to
make it work.

Keeping the two claims separate, as required:

* **Mechanistic.** Pairwise transfer effects are real and measurable
  (S/N 4.37 in the pilot). They do **not** compose additively into a
  four-stage ordering. That is a finding about composition, **not** evidence
  that developmental structure is absent.
* **Practical.** The ordering programme has no utility at this dose. Every
  efficiency metric favours interleaving, most starkly the joint
  steps-to-threshold, and the sequential arms are 15x less stable across
  seeds.

## What this licenses

Not another ordering variant. The next experiment is the
**resource-constrained scheduling benchmark**, where strong interleaving is
below ceiling so an efficiency difference can be detected at all — at this
dose interleaving reaches 0.987 mean and 0.951 min, leaving no headroom.

The candidate methods stand as recorded in `RESULTS.md`: uniform interleaving,
randomized interleaving, simple heuristic schedules, and a crude
state-dependent adaptive scheduler. No pairwise-derived static ordering
survives this diagnostic to be carried forward.
