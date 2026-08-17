# P1 stopping rule — frozen before any continuation outcome exists

Written while the state population is still generating and **before any fresh
hidden-futures continuation has been run or inspected**. Its purpose is to
remove the degree of freedom "keep generating states until the result becomes
significant", which would invalidate the prospective claim entirely.

Governs Protocol A of `downstream_protocols.md` (`8fc78c4087e2f87b`).

## 1. Stopping condition — matching quality only

Generation stops at the **first** poll where both hold, computed from
**current-state observables only**:

1. **`N_pairs >= 12`** eligible within-arm matched pairs at the frozen
   `epsilon`, and
2. **`epsilon <= 0.30`** in standardized units, so the pairs are genuinely
   matched rather than admitted by a loose threshold.

`epsilon` remains the 10% quantile of same-arm standardized distances, as
already frozen. It is never raised to manufacture pairs.

If the state population is exhausted before `N_pairs >= 12`, the result is
reported as **not testable at this population size**. It is not rescued by
loosening `epsilon`.

## 2. What may and may not be looked at

**May** be inspected while deciding to stop: the number of eligible pairs,
`epsilon`, the matching distances, and the matched states' *current* observable
vectors.

**May not** be inspected: any continuation outcome, any divergence statistic,
any `V(W,B)` value for a candidate pair. Those do not exist yet for the fresh
continuations and must not be consulted from prior runs to guide selection.

## 3. Procedure once the threshold is met

1. Freeze the pair list. Write it to disk, hash it, timestamp it.
2. Run **all** selected continuations. No subsetting, no dropping.
3. Report the pre-continuation matching distances **beside** the future
   divergence, so a reader can judge how well matched the pairs actually were.
4. Report every frozen eligible pair. No post-outcome pair selection.

## 5. Primary statistic and readings

Within-pair absolute divergence in `V(W, BIND)`, against randomly-paired states
as the null, permutation-tested. `t=0` and `rate_only` reported separately and
never summed.

| outcome | reading |
|---|---|
| matched pairs diverge more than random | behaviour underdetermines future learnability |
| matched pairs diverge no more than random | behaviour is sufficient at this resolution — a real, reportable negative |
| fewer than 12 eligible pairs | not testable at this population size |

## 4. What this rule costs

It makes a null result possible. That is the point: a stopping rule that can
only end in significance is not a stopping rule.
