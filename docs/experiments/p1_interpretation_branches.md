# P1 interpretation branches — frozen before the aggregate is inspected

The analysis plan (`a228f36586623fa4`) fixes *how* P1 is computed. This fixes
*what it means*, so the headline cannot drift after the number appears.

**No P1 aggregate had been computed or inspected when this was written.**
Individual continuation traces had been seen only while debugging a figure's
pair-ranking, which touched no aggregate statistic.

## The four conditions

Evaluated in order, on the frozen pair list, against the predeclared
within-arm null:

1. **Excess divergence** — matched pairs diverge more than the null.
2. **Interval** — bootstrap 95% CI over pairs excludes zero.
3. **Outlier independence** — the result survives dropping units whose final
   accuracy falls >0.3 below their own trajectory maximum (1 in 19 of the first
   completed units showed such a late dip).
4. **Not residual mismatch** — the relationship survives accounting for
   pre-continuation matching distance. If divergence tracks residual distance,
   the effect is imperfect matching rather than hidden state.

## The branches

### Strong positive — all four hold

> **Behaviourally similar present states can have measurably different
> developmental futures.**

Promoted to a headline claim. Panel 1 becomes the demo's lead. The digital-minds
framing is licensed: present behavioural evaluation can miss consequential
latent differences.

### Weak / mixed — (1) holds, any of (2)–(4) fails

> **Suggestive evidence for hidden developmental differences, but
> behaviour-matched divergence is unresolved.**

Reported as suggestive, never as the headline. Which specific condition failed
is stated explicitly — an unstable CI, outlier dependence and residual-mismatch
confounding are different weaknesses and must not be collapsed into "mixed".
Panel 1 stays in the demo, labelled unresolved.

### Null — (1) fails

> **Training history changes future learnability in controlled contrasts, but
> we did not establish that those differences remain hidden among
> behaviourally matched checkpoints.**

The confirmed core (H1–H4) is untouched: it rests on *unmatched* contrasts,
which this does not test. What is withdrawn is only the stronger claim that the
differences survive behavioural matching. The digital-minds framing becomes
correspondingly cautious, and E4b gains importance — if behavioural matching
does not hide the difference, the question becomes whether internal telemetry
sees variation that behavioural matching does not.

## Prohibited after seeing the aggregate

* Re-running with a different `epsilon`, pair list, null, or metric.
* Reporting a subset of `final` / `t=0` / `rate_only`.
* Choosing the branch by which reads better.
* Presenting a mixed result as positive by emphasising condition (1) alone.

## Reporting order, fixed

Aggregate first. Then outlier robustness. Then the matching-distance scatter.
**Illustrative pairs last, and only after the aggregate is stated** — chosen for
legibility, labelled as illustrative, never as evidence.
