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

## V2 language micro-world — the confirmed finding (2026-08-17)

A second experimental layer, in a language micro-world under ordinary
next-token prediction rather than an answer-position objective. Full design in
[docs/experiments/v2_bridge_design.md](docs/experiments/v2_bridge_design.md);
frozen amendments in
[v2_1_spec.md](docs/experiments/v2_1_spec.md) (`f92f5831bece0d91`); live record
in [campaign_2026_08_17.md](docs/experiments/campaign_2026_08_17.md); claim
status in [CLAIMS.md](CLAIMS.md).

**Training history produces a large, selective advantage in future
learnability.** Confirmatory, fresh seeds 100–105, frozen criteria:

| capability | A | A′ | BG |
|---|---|---|---|
| **B** (`BIND`) at `t=0` | **0.1322 ± 0.0149** | 0.0156 ± 0.0087 | 0.0156 ± 0.0051 |
| **C** (`FACT`) at `t=0` | 0.0176 ± 0.0025 | 0.0146 ± 0.0044 | 0.0130 ± 0.0063 |

Chance is 0.0156; both controls sit on it exactly for B. AULC A−A′ = **+0.5265**
on B (effect/noise +1.75) against **+0.0395** on C (+0.36) — thirteen times
smaller on the control capability. The control is non-vacuous: the background
arm reaches `FACT` 0.410 against a frozen 0.30 competence gate.

`A` and `A′` are matched by construction on unigram, bigram and positional
statistics, on entity recurrence, and on distinct entities per document, all
verified against a same-stream null. They differ in one property: whether a
recurring entity keeps its value.

**It is not memorised content.** With **zero shared entity tokens** between
source and target (disjoint sub-pools, verified), the head start is fully
retained — 0.1567 ± 0.0334 disjoint against 0.1431 ± 0.0157 shared, 111%, while
disjoint controls stay at chance.

**It persists across a 32× capacity increase** (exploratory, seeds 0–2): A at
`t=0` = 0.126 (786k params), 0.157 (6.3M), 0.097 (25.2M), with A′ and BG at
chance at every scale.

### The mechanistic hypothesis was falsified

The pre-registered mediator — an off-distribution prefix-matching score `M`,
measured on random repeated-token sequences so it could not be a function of
the corpus it explains — **failed its confirmatory gate**: amplitude 1.63 and
selectivity 0.76 against a ≥2.0 requirement. The compound G1 gate therefore
FAILS, even though two of its four criteria passed strongly.

A striking single-seed transient in preflight (`M` 0.0284 → 0.1100 → 0.0370 on
A, with A′ and BG flat) **did not reproduce** — not on fresh confirmatory
seeds, and not at any of three scales, where at 1× the *background* arm showed
the largest peak. The `A_peak` statistic used to chase it is biased toward
noisier arms, which is how that happened.

**The phenomenon is more robust than our explanation for it.** That is the
honest state of the result, and the falsification is reported as a property of
the methodology rather than a failed branch: the apparatus was built to be able
to kill its own hypothesis, and it did.

### What remains open

Breadth (does the effect extend beyond shared computation), the identity of the
actual mediator, whether data value is state-conditional, and whether internal
telemetry can predict it. Each has a running experiment and a frozen criterion;
none is claimed.

## Current state

* Frozen and unrun: the WikiText k=8 protocol — 7 usable families after
  excluding a residual cluster, dose fixed at the 20NG level, equal-family
  control weighting, hashed 13/5/3 unordered-pair pools, a 25% prospective
  material-improvement rule.
* [RESEARCH_V2_CANDIDATE.md](RESEARCH_V2_CANDIDATE.md) is a **draft under
  review**, not canonical, and does not revise any result above.
