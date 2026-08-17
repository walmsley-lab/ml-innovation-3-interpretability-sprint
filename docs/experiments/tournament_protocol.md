# What-next tournament — protocol, reserved and frozen in advance

**Written before the trigger fired, before any prediction was made, and before
any tournament continuation was run.** Nothing here may change once the first
prediction artifact is hashed.

## 0. Trigger

This experiment is **not licensed** until both hold:

1. **State × Data interaction** — the V(S,D) matrix shows a genuine ordering
   reversal, `argmax_D V(S₁,D) ≠ argmax_D V(S₂,D)`, not merely a state main
   effect or a data main effect (`RESULTS.md` X2c).
2. **Prospective predictor** — telemetry predicts conditional data value on
   held-out states better than chance (`RESULTS.md` X2d).

Without (1) there is nothing to choose between. Without (2) there is no basis
for choosing. Running the tournament on either alone would produce a
demonstration that looks like adaptive scheduling and isn't.

## 1. The claim this can earn

> Inspecting a model's internal state lets us predict which of several
> candidate corpora will be most valuable to train on next, and acting on that
> prediction beats state-blind alternatives at equal compute.

It cannot earn anything about closed-loop control, repeated selection, or
natural corpora. One step, one choice, one measurement.

## 2. States

**Fresh held-out states, seeds 800–807.** Disjoint from every seed used for
discovery (500–505), V(S,D) fitting, validation (700–711), confirmation
(100–105), C1 (200–203), B₂ (400–413) and H2.3 (600–607).

States are produced by the standard source phase across arms `A`, `A_prime`
and `BG`, giving a spread of incoming developmental states rather than eight
near-identical ones. A tournament run only from `A` states would not test
conditionality.

## 3. Candidate corpora

Five, fixed here:

| id | stream | why it is in the set |
|---|---|---|
| `D1` | `BIND` | the target capability |
| `D2` | `BINDT` | retrieval composed with a derangement |
| `D3` | `FACT` | parametric recall, the negative-control capability |
| `D4` | `BIND+FACT` | mixture |
| `D5` | `BG` | background; the do-nothing-useful control |

`D5` is included deliberately: a selector that cannot tell useful data from
filler is not a selector.

## 4. Objective — the common yardstick

Every branch is scored by the **same** frozen objective, never by the
capability its own corpus happens to train:

    V = min over {BIND, FACT, BINDT} of final accuracy

Min-across-capabilities, not mean. This is the project's own frozen efficiency
metric, adopted because a mean hides a destroyed capability (`RESULTS.md`). Mean
is reported as a secondary, and the primary is fixed as the min.

## 5. Equal compute

Every branch gets **identical token budget** (2000 target steps, batch 64) and
the **same data key**, so branches differ in incoming state and corpus and in
nothing else. Budget equality is asserted per unit, not assumed.

## 6. The procedure, in order

1. Train the eight held-out states. Extract telemetry from each.
2. **Predict.** From telemetry alone, and without running any continuation,
   produce for every state a full ranking of `D1…D5` and a single
   `argmax` choice. Write predictions to disk, hash them, timestamp them.
3. **Only then** run all 8 × 5 = 40 continuations.
4. Score against the frozen objective and compare to baselines.

Step 2 completing before step 3 is the whole design. A prediction produced
after any continuation is a postdiction.

## 7. Baselines — all state-blind

The state-aware selector must beat every one of these:

| baseline | rule |
|---|---|
| **global-best** | the single corpus with the highest mean `V` across the *fitting* states; the strongest and most important competitor |
| **random** | uniform over `D1…D5` |
| **static curriculum** | a fixed corpus order applied regardless of state |
| **current-loss** | pick the corpus matching the capability with the worst current loss |
| **current-competence** | pick the corpus matching the weakest current capability |

**Global-best is the one that matters.** If a single corpus is best from every
state, a state-aware scheduler has nothing to add, and beating random or static
would be an empty win. The last two are the honest heuristics a practitioner
would actually reach for, and they use *behaviour* rather than internal state —
so beating them is what isolates the contribution of telemetry.

## 8. Success and failure, fixed now

**Success:** the state-aware choice achieves higher mean `V` across held-out
states than **every** baseline, including global-best, and its per-state
`argmax` matches the realized best corpus more often than global-best does.

**Failure modes, all reportable:**

* state-aware ties global-best → a single corpus is universally best; no
  conditional value to exploit;
* state-aware beats random but not global-best → the selector has learned the
  data main effect, not the interaction. **This is the most likely false
  positive and must not be reported as adaptive scheduling;**
* predictor is at chance on held-out states → telemetry does not carry
  conditional data value, whatever the matrix showed.

## 9. Prohibited

* Changing corpora, objective, budget, baselines or success criteria after
  seeing any tournament outcome.
* Refitting the predictor on tournament states.
* Reporting the mean objective if the min was pre-specified as primary.
* Selecting which held-out states to report.

## 10. Cost, under the 64-core ceiling

8 states × 5 corpora = 40 continuations at 2000 steps, plus 24 source-phase
runs to produce the states. Roughly comparable to one V(S,D) matrix. Cores come
from H2.3 replication first and B₂ second, per the standing priority.
