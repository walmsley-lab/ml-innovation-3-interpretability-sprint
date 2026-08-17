# Two reserved downstream protocols, frozen in advance

Written before either experiment ran and before the V(S,D) gate resolved.
One-shot scouts. Neither may be redesigned after seeing its own outcome; if a
design flaw appears, the honest move is to report the flaw and the null, not to
re-cut the analysis.

The third reserved protocol — the prospective what-next tournament — is in
`tournament_protocol.md` (`f9da9fe23b1b2400`).

---

# Protocol A — behaviour-matched hidden futures

## A.0 What this is

Lane B's question, moved to a substrate that works. The W/P task was recorded
unsuitable (`RESULTS.md` B1b) because `P_first` could not reach competence on W,
so behaviourally-matched pairs never existed. The micro-world does not have
that defect: `BIND`, `FACT` and `BINDT` are all learnable and difficulty-matched
by construction, and 32 states already exist on disk.

> **Can two models that look the same right now have measurably different
> futures under identical subsequent training?**

## A.1 The matching problem, stated honestly

`A` and `A′` states are **not** behaviourally matched on the target capability:
zero-shot `BIND` is 0.145 versus 0.011. Matching cross-arm on a vector that
includes zero-shot `BIND` is therefore impossible, and pretending otherwise
would be the flaw that sinks the experiment.

Two designs follow, and the cleaner one is primary.

### Primary — within-arm matching (clean)

Among states from the **same arm**, find pairs matched on the **complete**
observable vector: zero-shot `BIND`, zero-shot `FACT`, zero-shot `BINDT`, and
source-stream loss. Give both the identical `BIND` continuation. Ask whether
future learnability diverges.

This has no arm confound. Any divergence is state information not visible in
behaviour, which is exactly the claim.

### Secondary — cross-arm matching on non-target behaviour (striking, weaker)

Match `A` against `A′` states on every measured competence **except the target
capability** — `FACT`, `BINDT`, source-stream loss — and test whether `BIND`
learnability diverges.

**This is the weaker design and must be reported as such.** Excluding the
target from the match is a real limitation: the models differ visibly on the
thing being predicted. It is included because the effect size is likely to be
large and because it is the version a reader will find intuitive, not because
it is the better test.

## A.2 Frozen parameters

* **States:** the 32 already on disk (discovery 500–505, validation 700–711).
  No new source training.
* **Matching rule:** components standardized across the population; a pair is
  matched if the maximum absolute standardized difference over the matching
  vector is ≤ `epsilon`, with `epsilon` set to the 10% quantile of same-arm
  distances and frozen before any continuation is scored.
* **Continuation:** `BIND`, 2000 steps, batch 64, **same data key** for both
  members of a pair. Identical by construction, not by check.
* **Outcome:** `V(W, BIND)` — final accuracy, with `t=0` and `rate_only`
  reported separately and never summed.
* **Primary statistic:** within-pair absolute divergence in `V`, compared
  against the divergence between **randomly paired** states as the null.

## A.3 Readings, fixed now

| outcome | reading |
|---|---|
| matched pairs diverge more than random pairs | behaviour underdetermines future learnability — the claim |
| matched pairs diverge no more than random | behaviour is sufficient at this resolution; a real and reportable negative |
| too few matched pairs at frozen `epsilon` | not testable at this population size. **Do not loosen `epsilon`** |

## A.4 Cost

Zero new source training. 2 × (number of matched pairs) continuations at 2000
steps. Expected well under one V(S,D) matrix.

---

# Protocol B — temporal replay of the ~270-step window

## B.0 What this tests, and what it does not

`T1` found that acquisition of zero-shot `B` **competence** during `A` training
is better described by a changepoint than by smooth growth (held-out RMSE
0.0171 vs 0.0331 linear, break at step 270 ± 8 across folds).

`T2` — whether the same moment marks a change in **future learnability**
`V(S_t, B)` — is untested, and is the question that would make it
developmentally meaningful. Competence and learnability can dissociate; that
dissociation is the entire reason `BINDT` exists.

**This protocol tests T2 prospectively.** The window comes from T1 and is not
re-fitted here.

## B.1 Frozen parameters

* **Window:** source steps **150 to 450**, centred on the T1 break at 270 with
  a margin of roughly ±3 fold-standard-deviations plus room to see the
  approach and the settle.
* **Spacing:** every **20 steps** → 16 checkpoints per trajectory.
* **Seeds:** 3 fresh source trajectories, seeds **900–902**, arm `A` only
  (the arm where T1 found the break).
* **Continuation:** every checkpoint gets the **identical** `BIND` continuation
  — 2000 steps, batch 64, same data key. Branches differ only in `t`.
* **Outcome:** `V(S_t, BIND)`, with `t=0` and `rate_only` separated.
* **Comparison:** the same three frozen descriptions as T1 — linear, sigmoid,
  changepoint — compared by **held-out fit**, leave-one-seed-out. Not by eye.

## B.2 Readings, fixed now

| outcome | reading |
|---|---|
| changepoint wins on `V(S_t,B)` **and** the break lands near 270 | competence and learnability change together; a genuine localized developmental transition |
| changepoint wins but the break is elsewhere | there is a transition, but T1's competence break is not it |
| linear wins | **learnability grows smoothly through a competence changepoint.** A dissociation, and arguably the more interesting outcome: the visible behavioural jump is not a developmental boundary |
| margin between descriptions < 5% | not distinguishable at this resolution. Report that, claim no shape |

**Language discipline.** "Changepoint is the better held-out description" —
never "phase transition", unless the data forces it and even then reluctantly.

## B.3 Cost

3 trajectories × 450 source steps (cheap, short) + 3 × 16 = **48
continuations** at 2000 steps. Comparable to one V(S,D) matrix.

## B.4 The one thing that must not happen

The window must not be re-centred after seeing `V(S_t,B)`. If the break lands
outside 150–450, that is the result — a T1/T2 dissociation — and it is
reported, not chased.

---

# Priority, per the standing decision rule

| gate outcome | order |
|---|---|
| interaction **and** prediction pass | tournament → hidden futures → temporal replay |
| interaction exists, prediction fails | hidden futures → temporal replay |
| interaction fails | hidden futures → temporal replay |

B₂ breadth comes after these unless already near-complete at negligible
opportunity cost. Enough wall-clock is reserved for submission integration
regardless of which branch fires.
