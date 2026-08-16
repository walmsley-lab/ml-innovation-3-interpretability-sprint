# Empirical results and chronology

What has actually been measured, in the order it happened. Conclusions live
here; the scientific program is in [research.md](research.md), the system
specification in [technical.md](technical.md), and threats to validity in
[RISKS.md](RISKS.md).

Detailed experimental records:

* [docs/experiments/layer2_synthetic.md](docs/experiments/layer2_synthetic.md) — synthetic Layer-2 design and pilot (sections 0–13)
* [docs/experiments/natural_corpus.md](docs/experiments/natural_corpus.md) — natural-corpus Stage-5 work (sections 14–22)

---

## Layer 1 — synthetic W/P task

**Gate A passed.** Estimator bias ≤ 7e-5 against a 2e-3 tolerance; interval
coverage 0.947–0.951 against a nominal 0.95.

**Gate B passed after recalibration.** The original block-sequential design
produced catastrophic interference, and the overlap calibration then revealed
that the regime did not reliably learn the rule at all (seed 1001 solo
A_W = 0.529). Seed replication was made permanent in B1, and the frozen
regime is d_model 64, 4 layers, lr 3e-3, 1,200 steps, n_cues 512, overlap
floor r = 0.20 — worst-case coexistence 0.969. Common-tail adequacy passed
with T1 maintenance, worst post-tail coexistence 0.947.

**Gate C failed — a statistical invalidator, not a negative result.** Under
identity-null pairing the W-first history gave a null sd of 12.87 in
log-odds; two runs differing only in an independent source draw produced
neutral W-choice rates of 0.000 and 0.996. Claim 1 as specified is
unrunnable at any feasible seed count. The apparatus was calibrated through
the tail; the *endpoint* is what could not carry inference.

An exploratory wave (11 seeds per history, not confirmatory) found the
W-first neutral-choice distribution broad across the range and the P-first
distribution concentrated toward P with occasional excursions.

## Layer 2 — synthetic compositional families

Four families frozen (F1, F4, F5, F6); the two AGGREGATE families were
dropped under the hard-stop rule.

**Transfer is measurable**: signal-to-noise 4.37. `F6→F5` remains
high-variance rigor-pass debt at 8× the median seed sd.

**Held-out prediction**: structural features 0.271 RMSE against additive's
0.674 — 59.8% better, with additive worse than the global mean. Recorded as
**pilot-level relational predictive structure, not primitive-level causal
structure**, because only 2 of 12 directed pairs are primitive-disjoint.

## Stage 5 — natural corpus (20 Newsgroups)

Pipeline run in the required order with official labels excluded from
everything that could influence discovery. Four of six proposed families
cleared the 200,000-token support gate; 12 directed pairs frozen as 9
observed and 3 untouched, before any transfer ran.

**Measurability passed** the prespecified S/N ≥ 2.0 threshold at 2.31.

**The primary relational gate failed.** Relational features lost to the
global mean out of sample (LOPO 0.1166 vs 0.0709), so the adaptive step was
not licensed and no untouched pair was spent.

**Prospective validation refuted the fallback explanation.** Predictions for
the 3 untouched pairs were hashed and timestamped before they ran.
Source-only scored 0.0711 against the global mean's 0.0699. Additive was a
demonstrated interpolation artifact — best LOPO, worst prospective at
0.1434, overshooting by ~3× on the two diagnostic cells. **The LOPO ranking
inverted prospectively**, so at n=9–12 model selection itself is unreliable.

**The `t=0` decomposition is load-bearing.** Head start and rate-only carry
opposite signs in 7 of 9 observed pairs and are each 3–4× more reproducible
than their sum (S/N 8.79 and 7.18 against 2.31).

## H3 — common control

A confound was found first: with four families, `N_ij` is the complement of
`{i,j}` and the centroid cosine is symmetric, so the cosine feature was a
**one-to-one function of the control composition**.

Under a common target-specific control, verified invariant from recorded
stream hashes:

* gross transfer structure **survives but weakened** — AULC S/N 2.002 against
  a 2.0 threshold;
* AULC relational prediction remains **effectively absent** (1.5% gain);
* component relational gains **survive the control repair almost unchanged**
  — 19.8% and 19.2%, against 20.2% and 18.7% before.

This promotes the component/state-space response to **leading hypothesis**. It
does not establish prospective relational structure: the evidence is a LOPO
advantage at n=12, and that metric was shown to invert at this scale.

## Current state

* Frozen and unrun: the WikiText k=8 protocol — 7 usable families after
  excluding a residual cluster, dose fixed at the 20NG level, equal-family
  control weighting, hashed 13/5/3 unordered-pair pools, a 25% prospective
  material-improvement rule.
* [RESEARCH_V2_CANDIDATE.md](RESEARCH_V2_CANDIDATE.md) is a **draft under
  review**, not canonical, and does not revise any result above.
