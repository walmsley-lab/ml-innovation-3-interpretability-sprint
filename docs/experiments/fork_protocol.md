# Matched-State Counterfactual Fork — frozen before any fork outcome exists

A sprint extension, logically independent of P1. **No fork continuation had been
run when this was written.**

## 1. The question

P1 asks whether behaviourally matched states diverge under *the same* future.
The `V(S,D)` matrix asks whether data value depends on state. This joins them:

> Do two models that look the same now nevertheless **require different future
> experience**?

    [ V(S1,D1) - V(S1,D2) ] - [ V(S2,D1) - V(S2,D2) ]

a paired difference-in-differences. The decisive outcome is a genuine
**ordering reversal**: `V(S1,D1) > V(S1,D2)` while `V(S2,D2) > V(S2,D1)`.

This is **not** the tournament. It does not claim we can read the state and
choose correctly. It asks only whether the choice *should* depend on state.

## 2. Frozen selections

**Corpora `D1 = BIND`, `D2 = BINDT`.** Chosen from the prior `V(S,D)` matrix as
the pair with the strongest ordering reversal: 10 of 13 discovery states prefer
BIND, 3 prefer BINDT, span 0.628. The other two pairings are 12–1 for FACT and
so offer little contrast. **That prior matrix is discovery; this is a fresh test
on previously unseen, behaviourally matched states.**

**16 pairs**, the lowest present-state matching distances from the frozen P1
list (`8392ad37bb953ebf`). Selected by **matching quality alone** — no outcome,
divergence, or fork result was consulted. Recorded in `artifacts/fork_frozen.json`.

**Design:** 16 pairs × 2 states × 2 corpora = **64 continuations**, 2000 steps,
batch 64, one frozen data key per corpus so every branch sees identical data.
More independent matched pairs at one seed, rather than fewer pairs at many —
declared as a sprint extension, not a fully powered standalone claim.

## 3. Metric and analysis, fixed now

**Metric:** AULC of the target capability, matching the robust choice used
elsewhere. `final` is *not* primary — 1 in 19 units shows a late instability dip
that moves `final` by ~0.9.

**Reported in this order:**

1. aggregate interaction across all 16 pairs, with bootstrap 95% CI (10,000
   resamples over pairs);
2. fraction of pairs showing the predicted interaction sign;
3. count of genuine ordering reversals;
4. dependence on matching distance — does the interaction track residual
   mismatch rather than hidden state;
5. outlier sensitivity — recomputed dropping units >0.3 below their own
   trajectory maximum;
6. **only then** one illustrative pair, chosen for legibility and labelled as
   illustrative.

## 4. Readings, fixed now

| outcome | reading |
|---|---|
| interaction CI excludes zero, reversals present | **matched-looking models require different future experience** — a fixed curriculum cannot know what comes next from behaviour alone |
| pairs diverge but **not** corpus-specifically | hidden **general plasticity** difference, not state-dependent data preference. Refines the thesis; does not invalidate it |
| no interaction | data value does not depend on state *among behaviourally matched pairs*, whatever the unmatched matrix showed |

## 5. Independence from P1

A null here does **not** alter P1 or H1–H4. P1 tests divergence under one
future; this tests whether divergence is corpus-specific. The second can be null
while the first is positive — that is precisely the general-plasticity reading
above, and it is informative rather than contradictory.
