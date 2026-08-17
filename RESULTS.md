# RESULTS

What has actually been measured. This is the canonical results document: it
merges the former `RESULTS.md` barrier log, `report.md` chronology, and the
`RESULTS.md` ledger into one place.

**Nothing here has been rewritten to fit the current framing.** Failed and
inconclusive gates are preserved with their original numbers, because several
of them — the block-sequential collapse, the composition failure, Gate C, the
falsified induction mechanism, the failed state-aware predictor — are among the
most useful things the project knows.

Every quantitative figure is recomputable: `PYTHONPATH=src python
scripts/audit_claims.py`.

---

## The progression in one screen

    static ordering failed              block-sequential 0.383 vs interleaved 0.987
      |                                 pairwise transfer does not compose (-0.15)
      v
    developmental-state hypothesis
      |
      v
    history-dependent learnability      CONFIRMED  0.1322 vs chance 0.0156, C null
      |                                 survives disjoint content (111%)
      |                                 survives 32x scale (exploratory)
      v
    state-conditioned data value        SUPPORTED  interaction 0.381-0.464,
      |                                 ranking reversals across states
      v
    prediction of that conditionality   FAILED     regret 0.047 vs global-best 0.021
      |
      v
    adaptive training                   NOT LICENSED

**The bottleneck is state inference, not the phenomenon.**

---

# Part I — Claim ledger

Every claim the project might make, with its current evidential status. A claim
moves to **Supported** or **Falsified** only when its *frozen* criterion has
been evaluated on the data it was written for. Nothing moves on a trend, a
smoke run, or a preflight.

Status values: **Supported** · **Tentative** (evidence exists, criterion not
yet met) · **Falsified** · **Untested** · **Apparatus** (a property of the
instrument, not a scientific result).

Last updated: 2026-08-17, during the overnight campaign.

**Submission priority order** (standing): confirmed core → State×Data
interaction → prospective state prediction → one-step what-next win → causal
mechanism → breadth refinement.

---

## Established before this campaign

| # | Claim | Status | Evidence |
|---|---|---|---|
| E1 | Block-sequential curricula destroy capability that interleaving retains | **Supported** | Ceiling scout at identical compute and allocation: interleaved 0.9867 mean / 0.9510 min vs block-sequential 0.3827 / 0.1062 |
| E2 | Overlap at `r = 0.20` restores retention | **Supported** | Overlap diagnostic: min acc 0.106 → 0.53–0.57 |
| E3 | Pairwise transfer composes into useful multi-stage orderings | **Falsified** | Overlap diagnostic Branch 2: best − reverse = −0.0383 against pooled seed sd 0.2556, effect/noise −0.15, with retention restored and dose decoupled from order |
| E4 | Directed transfer is measurable | **Supported** | Synthetic S/N 4.37; 20NG 2.31; WikiText 5.95–8.32 |
| E5 | Relational features predict held-out transfer better than additive | **Tentative** | Synthetic pilot 0.271 vs 0.674 RMSE, but only 2 of 12 pairs primitive-disjoint. WikiText prospective +30.0% / +37.7% but failed jackknife robustness on one component. 20NG: inverted |
| E6 | The `t = 0` decomposition is load-bearing | **Supported** | Head start and rate-only carry opposite signs in 7 of 9 pairs and are 3–4× more reproducible than their sum |
| E7 | The frozen neutral-default endpoint supports inference | **Falsified** | Gate C: null sd 12.87 in log-odds; two runs differing only in a source draw gave 0.000 and 0.996 |

## Headline — confirmed

| # | Claim | Status | Evidence |
|---|---|---|---|
| **H1** | **Training history produces a large, selective advantage in future learnability** | **Supported** | G1 confirmatory, frozen V2.1 (`f92f5831bece0d91`), fresh seeds 100–105. `B` at `t=0`: A **0.1322±0.0149** vs A′ **0.0156±0.0087** and BG **0.0156±0.0051** — both controls *exactly* at the 0.0156 chance floor. AULC A−A′ = **+0.5265**, effect/noise **+1.75** |
| **H2** | **The advantage is specific to the target, not generic** | **Supported** | Same runs. `C` at `t=0`: A 0.0176, A′ 0.0146, BG 0.0130 — no advantage. AULC A−A′ on C = **+0.0395**, effect/noise **+0.36**, i.e. 13× smaller than on B. The control is non-vacuous: BG reaches FACT 0.410 against a frozen 0.30 competence gate |
| **H3** | **The advantage is not memorised content transfer** | **Supported** | C1 discrimination, fresh seeds 200–203, **zero shared entity tokens** between source and target. Disjoint `t=0` **0.1567±0.0334** vs shared **0.1431±0.0157** — 111% retained; disjoint controls at chance (A′ 0.0181, BG 0.0137). Token identity is irrelevant to the effect |
| **H4** | **The advantage persists across a 32× capacity increase** | **Tentative** | Exploratory scale scout, seeds 0–2, original V2 setup. A `t=0` = 0.126±0.016 (786k), 0.157±0.042 (6.3M), 0.097±0.037 (25.2M); A′/BG at chance at every scale. Exploratory: 3 seeds, unrepaired target |

## Lane A — the mechanistic hypothesis, falsified

**This falsification is a result, not a failed branch.** The apparatus was built
to be able to kill its own hypothesis, and it did — twice, on independent fresh
seeds, after a single-seed preflight spike that looked convincing.

| # | Claim | Status | Evidence |
|---|---|---|---|
| A0 | The substrate does not manufacture the effect | **Apparatus — passed** | P1/P2/P6/P7 pass. Short-range predictors cannot distinguish A from A′ (positional gap 0.0003, unigram 0.0005, bigram 0.0009, all inside null); distinct entities 2.89 vs 2.89; binding oracle 1.0000 vs 0.0153 at chance 0.0156 |
| A1 | The phase boundary alone produces no B advantage (P5) | **Apparatus — passed** | BIND accuracy across a `BG→BG` boundary: jump 0.0000, on both CPU and GPU paths |
| A2 | Training on A raises M more than A′ (P3) | **Falsified (original criterion)** | Sustained-emergence criterion: rise 0.0110 on A against a required 0.05; ratio 3.08 passed but amplitude failed. Recorded as FAIL, not reinterpreted. V2.1 re-specifies M as a transient event and runs on fresh seeds |
| A2b | The transient M event is A-selective | **Falsified** | **G1 confirmatory, fresh seeds 100–105, frozen V2.1: amplitude 1.63 (gate ≥2.0) and selectivity 0.76 (gate ≥2.0) — both FAIL.** Independently, exploratory scale scout seeds 0–2: selectivity 0.96 (1×), 0.61 (8×), 0.90 (32×); at 1× `BG` shows the *largest* peak. The preflight's 3.9× single-seed spike did not reproduce anywhere. Also recorded: `A_peak = max over trajectory` is biased toward noisier arms, which is how BG won at 1× |
| **A2c** | **The G1 gate passes** | **Falsified** | The frozen G1 gate required all four criteria. Criteria 2 (A→B advantage) and 3 (C unmoved) **pass strongly**; criterion 1 (A raises M more than A′) **fails at 0.76**. The compound gate therefore **FAILS**, and expansion to 12 seeds is not licensed under the mechanism hypothesis. Not reinterpreted |
| A3 | B is learnable but does not ceiling (P4) | **Apparatus — passed** | 0.861 at 4000 target steps; recalibrated to TARGET_STEPS=2000 giving BG BIND 0.516, FACT 0.510 |
| A3b | C is learnable, so the specificity control is non-vacuous | **Apparatus — passed** | FACT was previously never trained (chance 0.008–0.021). With target `BIND+FACT`, FACT reaches 0.510 at step 2000 and 0.864 by 6000 |
| A4 | A→B beats A′→B, and C does not move (H2.1) | **Supported** | Promoted to H1/H2 above |
| A4b | A confers a `t=0` head start on B, and it survives scale | **Tentative — positive** | Exploratory scale scout: A `t=0` = 0.126±0.016 (1×), 0.157±0.042 (8×), 0.097±0.037 (32×) against A′ 0.012–0.014 and BG 0.017–0.021, with chance 0.0156. A′/BG at chance at every scale. AULC much noisier (effect/noise ≈1.9). **Strongest alternative explanation: A and B share the computation by construction, so this may be direct task transfer, not readiness — C1 scout running** |
| A5 | M rises before B accelerates (H2.2) | **Moot** | Requires an A-selective M. A2b falsified |
| A5b | The mediator of the confirmed transfer is unidentified | **Open — the central question** | The effect (H1–H4) is real; `M` does not explain it. Finding the actual mediator is now the primary scientific target |
| A5c | The on-distribution retrieval score distinguishes A from A′ | **Supported for A vs A′ only — NOT established against background** | A vs A′: discovery +0.1157 (p=0.0012), **validation +0.1267 (p=0.0002)** on fresh seeds 700–711. **A vs BG, which had never been tested: discovery +0.1154 (p=0.0012) but validation +0.0465 (p=0.3870) — not significant.** The background arm's retrieval rose to 0.1284 on fresh seeds against A's 0.1749. So the marker separates A from the value-rebinding control but **does not cleanly separate A from background** out of sample. Claim narrowed accordingly: it is an A-vs-A′ discriminator, not a general A-selectivity marker. Not causal (A6b inconclusive), not predictive of conditional data value (X2d falsified), and measured on-distribution |
| A6b | Retrieval-head ablation preferentially removes the A→B advantage | **INCONCLUSIVE — insufficient intervention efficacy. NOT a mechanistic falsification.** | H2.3 corrected, seeds 600–603, both conditions, 20 units. Arm × ablation interaction: `t=0` −0.0181, AULC +0.0094, final −0.2441 — no positive interaction. **But the intervention did not work:** zero-shot BIND on A went 0.1387 → 0.1611 under retrieval ablation (no reduction), vs 0.1450 → 0.1431 under matched-random. Weights are provably zeroed (P7), so the capability is redundant across remaining heads. **Necessity was never tested at adequate strength.** Must not be cited as evidence that retrieval is not the mediator |
| A6c | Top-k head ablation is an adequate causal intervention in this substrate | **Falsified** | Removing the 4 highest-scoring heads of 16 leaves zero-shot BIND unchanged. Any future causal test needs a stronger intervention — many more heads, whole-layer ablation, or activation patching — and this row is a prerequisite for interpreting any ablation null here, including our own |
| A6 | M ablation preferentially removes the A→B advantage (H2.3) | **Not licensed** | Ablating a mediator that is not selective would test nothing. Runner built; awaits a candidate mediator |
| A7 | The effect requires ≥2-layer composition at matched capacity (H2.4) | **Untested** | Deferred to V2-c |
| A8 | Promoting M accelerates B — `do(M+)` (sufficiency) | **Not licensed** | Same reason as A6 |
| **A9** | **The transfer is broader than computation overlap (readiness)** | **Untested — critical** | B₂ factorial running: target `BINDT` = retrieval ∘ derangement, so zero-shot is blocked by construction. Crossed with disjoint entity pools. Decides whether H1–H4 is computation-specific transfer or developmental readiness |

## Lane B — sprint latent state

| # | Claim | Status | Evidence |
|---|---|---|---|
| B0 | State features are basis-invariant and comparable across inits | **Apparatus — passed** | 137 features, deterministic, no raw activation coordinates; verified 127/137 differ between histories sharing one init |
| B1 | The pilot regime supports a preference measurement | **Falsified** | 0 of 24 models cleared the competence floor; W_COMPETENCE 0.256 against chance 0.25. Config error (below frozen budget), corrected |
| B1b | The W/P substrate supports a behaviourally-matched latent-state test | **Falsified** | With the frozen overlap regime, corrected common tail (T1) and a symmetric 2000-step budget: competence pass rate **1/6**. `P_first` reaches W = 0.238–0.293 against a 0.60 floor in every seed — W is only learnable when it gets a pure phase, which `P_first` never provides. `W_first` is itself seed-fragile (0.977 / 0.469 / 0.508), consistent with RISKS.md §2b. **Substrate recorded as unsuitable; Lane B stopped, not redesigned.** |
| B2 | Internal state predicts post-intervention outcome better than behaviour (S2) | **Untested — blocked** | Blocked by B1b. The pipeline (generate → extract → analyze, 137 basis-invariant features, hash-verified protocol) is built and validated; it needs a substrate where both skills coexist |
| B3 | State information survives behavioural matching (S1) | **Untested** | Gated behind B2 and the 20-matched-pair floor |
| B4 | Behaviour at `t` determines behaviour at `t+Δ` | **Untested** | The null B2 is written against |

## Exploratory — developmental shape (off critical path)

| # | Claim | Status | Evidence |
|---|---|---|---|
| T1 | Acquisition of zero-shot `B` competence during `A` training is better described by a localized change than by smooth growth | **Tentative — exploratory** | Held-out (leave-one-seed-out) model comparison over 4 seeds, three descriptions frozen in advance: changepoint RMSE **0.0171 ± 0.0036** vs linear 0.0331 ± 0.0037 and sigmoid 0.0353 ± 0.0038 — changepoint better by **48%**. Fitted break at step **270 ± 8** across folds |
| T2 | That change is a transition in **future learnability** `V(S_t, B)` | **Untested** | T1 models *acquired competence*, not learnability. The two dissociate — a model can gain learnability without gaining zero-shot competence, which is what `BINDT` exists to test. `V(S_t,B)` needs dense multi-step checkpointing plus identical continuations; the discovery runs save one checkpoint at the end of the source phase, so no temporally-ordered states exist |
| T3 | Behaviour-matched states have divergent futures | **Not licensed — protocol reserved** | `docs/experiments/downstream_protocols.md` Protocol A, hashed before any continuation. Lane B's question moved to a substrate that works: `BIND`/`FACT`/`BINDT` are all learnable and difficulty-matched, and 32 states already exist. **Primary design is within-arm matching** on the complete observable vector (no arm confound); cross-arm matching on non-target behaviour is secondary and explicitly weaker, because `A` and `A′` differ visibly on the capability being predicted |

**Language discipline for T1:** "changepoint is the better held-out
description, with a tightly localized break" — not "phase transition". The
comparison was made on held-out fit rather than by inspecting a curve, which is
what makes it reportable at all, and 4 seeds is not many.

## Cross-cutting

| # | Claim | Status | Notes |
|---|---|---|---|
| X1 | Mediator-blindness explains why pairwise transfer failed to compose | **Untested — conjecture** | Explicitly a hypothesis V2 tests, not a diagnosis. Even a full V2 pass would not establish it; that needs its own experiment |
| X2a | **State main effect** — different incoming states have different future learnability | **Supported (implied)** | Already entailed by H1: the A state learns B far faster than the A′ or BG states. Recorded separately so it is never mistaken for the interaction |
| X2b | **Data main effect** — some corpora are more valuable than others | **Untested** | Ordinary curriculum knowledge. Necessary to measure so it can be partialled out, not interesting on its own |
| X2c | **State × Data interaction** — the identity of the best next corpus depends on incoming state | **SUPPORTED — but smaller than first reported** | **Corrected**: the first pass computed reversals over *all* states including incomplete ones, where a state with one measured corpus trivially has it as argmax. Restricted to the **12 complete states**: mean objective — state main **0.553**, data main 0.066, interaction **0.381** (interaction is *second* to state main, not largest); 2 distinct argmax corpora (BIND, FACT), 42 sign reversals. Min objective — 0.396 / 0.140 / **0.464**, 3 distinct argmax, 56 reversals, but min sits at the chance floor (0.005–0.011) so its reversals may be noise. **The reversal is real; the interaction is substantial but not dominant** |
| X2d | Telemetry **predicts** which corpus is best, prospectively | **FALSIFIED** | Completed balanced matrix, leave-one-state-out over 12 complete states. State-aware selection is **worse than the state-blind global-best rule** on both objectives: min — 7/12 top-1 and regret 0.0034 vs global-best 7/12 and 0.0028; mean — 10/12 and 0.0473 vs global-best 11/12 and **0.0207**, i.e. more than 2× the regret. Both beat random (0.0076 / 0.0758), so the predictor learned the **data main effect** and not the conditionality. **The interaction is real (X2c) and our telemetry cannot exploit it.** No tuning or redesign was attempted |
| X2d-prev | *(superseded)* interim read on partial matrix | **Superseded** | Leave-one-state-out on 7 complete states: state-aware **ties global-best exactly** (5/7 top-1, mean regret 0.0011 each) against random 0.0050. Reading: *no exploitable conditionality in the presently measured states* — a statement about the world, not a failing predictor. All regrets are tiny (<0.002), so the corpora barely differ in value from these states. **Provisional: the balance gate has not been satisfied** (7 complete states, per-corpus spread 5; thresholds are 8 and 4). Final call awaits the balanced matrix |
| X2e | Acting on the prediction beats state-blind selection at equal compute | **NOT LICENSED — trigger failed** | Requires X2c **and** X2d. X2c passed, X2d falsified, so the tournament is declined per the standing decision rule. Protocol remains frozen at `f9da9fe23b1b2400` and unrun. Running it anyway would demonstrate a selector we already know loses to global-best | `docs/experiments/tournament_protocol.md`, hashed before any prediction exists. Requires X2c **and** X2d. Fresh states 800–807, 5 candidate corpora, predictions hashed before continuations, min-across-capabilities objective. **The competitor that matters is global-best**: beating random or static while tying global-best means the selector learned the data main effect, not the interaction, and must not be reported as adaptive scheduling |
| X3 | Developmental coordinates can be inferred rather than supplied | **Untested** | Milestone D |
| X4 | State-aware selection beats strong interleaving | **Untested** | Milestone E. Note E1: interleaving is the baseline to beat |

---

## Audit trail

`scripts/audit_claims.py` recomputes headline numbers from raw units. It exists
because three claim rows once failed to write silently while being reported as
recorded, and it has since caught two substantive drifts: the inflated
distinct-argmax count in X2c (incomplete states counted as reversals) and the
untested A-vs-BG comparison in A5c. **Run it before quoting any number.**

## Rules for this ledger

* A claim never moves on a trend. It moves when its frozen criterion is
  evaluated on the data it was written for.
* Preflight and audit outcomes are **Apparatus**, never Supported. They say the
  instrument works, not that the world is a certain way.
* A falsified claim stays in the table. E3 and E7 are among the most useful
  things the project knows, and deleting them would make the record look like a
  string of successes.
* A claim that failed for a *configuration* reason (B1) is recorded as
  falsified for that configuration, with the cause named. It is not quietly
  re-run into a pass.
* **A falsified hypothesis with a surviving effect is the most valuable state
  the ledger can be in.** H1–H4 are confirmed while A2b/A2c are falsified: the
  phenomenon is real and our explanation for it was wrong. Reporting the effect
  without the falsification would be the failure mode this ledger exists to
  prevent.

## Calibration debt, recorded

The frozen `B` headroom band was 0.40–0.60; G1's BG arm finished at **0.240
± 0.367**. Calibration used two seeds (900–901, giving 0.516) and did not
capture between-seed variance. The miss is toward *more* headroom, so it does
not threaten H1–H4, but the calibration was undersized and any future budget
freeze needs more seeds.

---

# Part II — Chronological record


What has actually been measured, in the order it happened. Conclusions live
here; the scientific program is in [research.md](RESEARCH.md), the system
specification in [technical.md](docs/technical.md), and threats to validity in
[RISKS.md](docs/RISKS.md).

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
status in [CLAIMS.md](RESULTS.md).

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

---

# Part III — Barrier log


> **Resuming from another machine or session?** Start with
> [HANDOFF.md](docs/archive/HANDOFF.md): what is running, how to restore state, what is
> established, what has been controlled for, and where it can go next.

## Research objective

**Discover measurable structure in how models acquire, transfer, retain, and
interfere with information, then use that structure to organize or represent
training data so models reach greater capability with fewer tokens, less
corpus, and less compute.**

Stage 5 and developmental ontology discovery remain central. But pairwise
transfer matrices, DAGs, semantic families, overlap schedules and static
curricula are **candidate mechanisms and representations, not objectives we
are obligated to vindicate**. Any of them may be discarded without loss if a
cheaper or better-performing representation appears.

### From curriculum ordering to developmental scheduling

The ceiling scout already taught us something the original framing did not
anticipate: **presentation structure alone produced an enormous effect at
identical compute and identical corpus allocation** — interleaved 0.9867 mean
and 0.9510 min accuracy against block-sequential 0.3827 and 0.1062.

That means retention, recurrence, interference and timing are part of the
developmental problem, not merely prerequisites to an ordering. The eventual
intervention may not look like `A -> B -> C -> D` at all. It may be a
time-varying mixture, rehearsal, recurrence, selective exposure, or
state-dependent allocation.

The hypothesis is therefore broadened from **curriculum ordering** to
**developmental scheduling**.

### The Stage-5 target, stated as an outcome

Given a natural corpus and a model state: infer useful developmental units
and relationships from limited measurements, predict which training exposures
or representations carry the highest marginal learning value, organize
training accordingly, and **prospectively demonstrate higher capability at
equal compute, or equal capability with fewer tokens**.

### Ontology, kept alive but grounded

Current families are **provisional measurement units**. A model-native
developmental ontology should be inferred from how examples or families
behave during learning — what they prepare for, interfere with, reinforce,
require, or change representationally. Semantic similarity is one candidate
explanatory signal among several.

If reliable interactions emerge, the question becomes whether **cheap
observables** predict them: gradient alignment, representation similarity,
loss trajectories, learning-speed signatures, probes, activation changes. The
long-run method needs a small calibration set, not exhaustive pairwise
training.

### Rule for experiment selection

> Before proposing a substantial new run: **if this succeeds, does it
> materially improve our ability to predict or control useful learning per
> token?**

If not, it is diagnostic-only or deprioritized.

---

Updated at each frozen barrier. Completed stage, hashes, gate result,
interpretation, risks, next licensed action. Nothing else.

---

## Barrier: WikiText Stage-5 confirmatory gate — **FAIL** (2026-08-16)

**Completed stage.** Natural-corpus Stage 5 on WikiText-103, 7 usable
families, common-control design. 108 units: 78 development (26 directed
pairs x 3 seeds), 30 confirmatory (10 pairs x 3 seeds), 21 shared controls.
All valid; control invariant PASS on every target x seed group.

**Frozen artifacts.**

| object | hash |
|---|---|
| pair pools (13/5/3 unordered) | `497b9c3fc66e8adbca96ac2eef41e9e2ada14ffcdc0d78bb4cbbf589c42b3c27` |
| confirmatory predictions | `2ff3cf08f744dc9a3c8a98984e026e50e0148186fb0bec2368b1f985cc0b71a5` |
| Layer-2 scout manifest | `1deb0ae83655d08dbf930c3ddd9109c6441bedb03941bde349a6a9796ec5ebac` |

**Gate result: FAIL.** Both primary components clear the 25% material
threshold against the best simpler model and the seed-noise floor.
`rate_only` is jackknife-robust across all ten leave-one-pair-out subsets
(29.3%-60.6%). `head_start` is not: dropping `0->4`, `4->0` or `4->3` gives
21.8%, 18.8%, 22.8%. The criterion required every condition on both
components under one model.

**Scientific interpretation.** The relational model was best in development
LOPO **and best prospectively on both components**, beating the best simpler
model by 30.0% (head start) and 37.7% (rate-only). The 20NG inversion — where
the development-best model became prospectively worst — **did not recur**.
The formal gate nonetheless fails because one component's margin leans on
family 4. The honest statement is that a jackknife-robust 25% improvement was
not demonstrated at n=10 confirmatory pairs, not that structure is absent.

This gate is **closed and will not be refit or reinterpreted**. Family-4
sensitivity is a post-hoc diagnostic only, and any ontology-revision work
stays separate from confirmatory claims.

**Current risks.**
- Confirmatory pool underpowered at 10 directed pairs; a real effect could
  fail this criterion by variance alone.
- `head_start` margin concentrated in one family.
- Relational model is p=15 at ridge 1e-4 on 26 development pairs; it
  generalized here, but the parameterization remains fragile.
- Adaptive pool (3 unordered / 6 directed) is **untouched and must stay so**
  unless a gate licenses it.

**Next licensed action.** The Layer-2 ceiling scout, already frozen and now
running. Stage 5 was **not** reached; no adaptive intervention was selected
or launched.

**Against the four questions.**

* *Phenomenon* — **yes, on WikiText.** Development S/N 5.95 and 6.09 on the
  primary components, confirmatory 7.62 and 8.32, under a repaired control,
  with matched composition and asserted exposure equality.
* *Prediction* — **partial.** The relational model beat every simple baseline
  prospectively on both components (30.0%, 37.7%) against frozen predictions,
  and the 20NG inversion did not recur. It failed the frozen robustness
  condition on one component. This is the closest the project has come, and
  it is not a pass.
* *Utility* — **untested.** No curriculum has been compiled or evaluated.
* *Efficiency* — **unmeasured.** 26 development pairs were used to predict 10
  held-out; the minimum sufficient fraction is unknown and is the target of a
  later sample-efficiency program.

---

---

## Barrier: Layer-2 ceiling scout — **FAIL** (2026-08-16)

**Completed stage.** 15/15 units, manifest `1deb0ae83655d08d`, allocation
identical across arms and families.

**Gate result: FAIL.** `predicted_best` 0.3827 vs `exact_reverse` 0.3731 on
final mean accuracy — a +0.0096 difference against pooled between-seed sd of
0.0187, so effect/noise +0.51. Best does **not** beat balanced either: 0.383
against **0.987**.

**Scientific interpretation.** Sequential block curricula retain only the
final family. Every family reaches ceiling at the end of its own phase and
then collapses by 0.69-0.89; only the last survives. The two sequential arms
differ in outcome only in *which* family is last, so their gap is recency,
not order. The balanced arm reaching 0.951-0.998 on all four families shows
the task is fully learnable at this dose — the failure is specific to
block-sequential presentation, and reproduces the Layer-1 catastrophic
interference finding that overlap `r = 0.20` was introduced to fix.

A second failure mode survives even if forgetting is fixed: the pairwise
matrix measured *immediate acquisition of the next family* over two phases,
while the scout composed it into a four-phase ordering scored on *retention
of all families*. Transitivity was never tested.

**Current risks.** The pairwise-graph approach may not compose into
multi-phase curricula at all; that is now an open and testable question
rather than an assumption.

**Next licensed action.** Proposed and **not run**: re-run the same three
arms with Layer-1's validated overlap floor `r = 0.20`, changing that one
thing only. It separates "forgetting swamped the effect" from "pairwise
transfer does not compose". 15 units.

Sharpened by the efficiency framing: the comparator is the **interleaved
arm**, not the reverse arm. The reverse arm remains as the order control, but
the question that matters is whether any ordering beats interleaving at equal
compute. Efficiency metrics (tokens- and steps-to-threshold, accuracy at
fixed budget, min across families, retention) are frozen below and would be
recorded per unit.

**Against the four questions.**

* *Phenomenon* — **no**, for block-sequential Layer-2 curricula at this dose.
  The dominant effect is recency, not order.
* *Prediction* — untested here; the scout never reached a prediction stage.
* *Utility* — **no.** The compiled curriculum lost heavily to plain
  interleaving (0.383 vs 0.987). Any curriculum claim must beat the balanced
  control, and this one does not come close.
* *Efficiency* — the scout cost 15 units to establish that the regime has no
  headroom, which is the cheapest possible way to have learned it.

---

---

## Barrier: r = 0.20 overlap diagnostic — **Branch 2** (2026-08-16)

**Completed stage.** 15/15 units, manifest `cd73c7444818dcfc`, every family at
exactly 600 steps in every arm.

**Gate result.** Retention restored (min acc 0.106 -> 0.53-0.57), order effect
absent (best - reverse = -0.0383 against pooled seed sd 0.2556, effect/noise
**-0.15**), and interleaving wins every efficiency metric (min acc 0.951 vs
0.529; joint steps-to-threshold reached 5/5 seeds at median 950 steps vs 1/5
and 0/5).

**Scientific interpretation.** Overlap at `r = 0.20` substantially fixes the
catastrophic forgetting the ceiling scout exposed, so failure mode 1 was real.
With retention restored and order decoupled from dose, **the ordering effect
is still absent**. Per the branch fixed in advance: local pairwise transfer
does **not** straightforwardly compose into useful multi-stage orderings, and
the static ordering construction is not to be patched further.

Claims kept separate. *Mechanistic*: pairwise transfer is real and measurable
but does not compose additively — a finding about composition, not evidence
that developmental structure is absent. *Practical*: the ordering programme
has no utility at this dose, and the sequential arms are ~15x less stable
across seeds than interleaving (sd 0.27 vs 0.016).

**Current risks.** Interleaving may be near-optimal at this scale; until a
regime exists where it is below ceiling, no method can demonstrate an
efficiency gain over it.

**Next licensed action.** The resource-constrained scheduling benchmark. No
pairwise-derived static ordering carries forward.

**Against the four questions.**

* *Phenomenon* — **yes, but not the hypothesized one.** The reproducible
  effect at equal compute is presentation structure (interleaved vs blocked),
  not ordering.
* *Prediction* — pairwise predictions do not compose into multi-stage
  outcomes. WikiText prospective prediction remains the only partial success.
* *Utility* — **no.** Interleaving wins every frozen efficiency metric.
* *Efficiency* — interleaving reaches joint 0.90 across all families in a
  median 950 steps, 5/5 seeds; the best ordering manages 1/5.

---

## Post-barrier branches (now resolved — Branch 2 fired)

Fixed before the result is visible:

* **Retention not restored** — diagnose retention. Do **not** expand the graph
  programme.
* **Retention restored, predicted-best ~ reverse** — evidence that local
  pairwise transfer does **not** straightforwardly compose into useful
  multi-stage orderings. Do **not** keep patching the static ordering
  construction to make it work.
* **Retention restored, predicted-best > reverse beyond noise** — pairwise
  developmental information has some compositional value. This is **not** yet
  practical utility.

In the latter two cases the next major experiment orients to useful learning
per token, in a **deliberately resource-constrained regime where strong
interleaving is below ceiling**, comparing at equal budget on the already
frozen metrics with primary emphasis on min-across-families and joint
steps/tokens-to-threshold. Candidate methods: uniform interleaving,
randomized interleaving, simple heuristic schedules, any pairwise-derived
schedule that survives this diagnostic, and a **crude adaptive/closed-loop
scheduler** that periodically observes competence, learning rate and
forgetting across families and allocates the next window toward underlearned,
high-value or endangered families.

The scheduler is not to be optimized for sophistication. The first question is
only whether **state-dependent allocation** yields measurable sample- or
compute-efficiency gains over strong interleaving.

## Ontology programme — the north star, not yet licensed

The four-question ladder is the **evidence ladder** for the ontology
programme, not a replacement for it. The end state remains a **model-native
developmental ontology**: units defined by how learning transfers,
interferes, accelerates and changes representations, rather than by human
semantic labels.

Standing constraints, recorded now so they are not rediscovered later:

* The current semantic families and any first recovered graph are
  **provisional measurement units**, not the ontology.
* Ontology optimization does **not** begin from partial or failed results.
  Reliable interaction structure must exist first; as of this barrier it does
  not.
* Once stable non-additive, held-out-predictive structure exists, the next
  question is which observables explain it — representation similarity,
  gradient alignment, loss trajectories, learning speed, probe/activation
  change, semantic structure, or combinations.
* **Every ontology revision must make frozen prospective predictions before
  being considered better.** This is the WikiText standard, applied to
  ontologies.
* The deliverable is a predictive and actionable developmental ontology. A
  graph on its own is not the deliverable, and neither is a curriculum. The
  deliverable is a developmental model of training that yields **more
  capability from less data and compute**.

### Efficiency is the terminal metric

Candidate ontologies and representations are judged by concrete training
outcomes, not by effect size:

* same target performance with **fewer tokens**;
* a given accuracy in **fewer optimization steps**;
* intelligent ordering beating shuffled/interleaved **at equal compute**;
* a better representation needing **less corpus**;
* identifying redundant or low-value data that can be **omitted** without
  harming capability;
* higher final accuracy **at a fixed token/compute budget**.

Once reliable structure exists, subsequent experiments measure sample and
compute efficiency, not merely effect size.

### The scout already answered one efficiency question

Not the one it asked. At **identical token and compute budget**, and
identical aggregate family allocation:

    balanced / interleaved   0.9867 mean acc, 0.9510 min acc
    block-sequential         0.3827 mean acc, 0.1062 min acc

That is a very large capability difference at fixed budget, produced purely
by **presentation structure**. It is the strongest training-efficiency effect
this project has measured, and it points the opposite way from the
hypothesis under test: blocking destroys capability that interleaving
retains.

The consequence for design is concrete. **Interleaved is the baseline to
beat, not a control to beat.** Any ordering technique must show an advantage
*over* interleaving at equal compute, and no result that merely beats
block-sequential ordering is interesting.

### Efficiency metrics to freeze before the next experiment

Recorded now so they are fixed in advance rather than chosen after:

* **tokens-to-threshold** — tokens to reach accuracy tau on each family, and
  on all families jointly;
* **steps-to-threshold** — the same in optimizer steps;
* **accuracy at fixed budget** — final mean and **min** across families, since
  a mean hides a destroyed family;
* **area under the learning curve** per family;
* **retention** — accuracy at the end of the curriculum against accuracy at
  the end of that family's own exposure.

The min-across-families metric is the one that matters most: the scout's
sequential arms looked half-decent on the mean and catastrophic on the min.

## Infrastructure

`dsi-cpu-bench` and `dsi-cpu-w2`, both c3-standard-22, us-west1-a, verified
environment-identical (image `v20260807`, jax 0.11.0, matching vocab, pools
and corpus-cache hashes). Throughput ~3.06 trajectories/min per worker, flat
in concurrency. `pdp-gpu` TERMINATED and preserved.

---

## Barrier: V2 language micro-world — the pivot to developmental state (2026-08-17)

**Nothing above is revised.** The static-ordering failures stand exactly as
recorded: block-sequential curricula destroy capability (0.383 vs 0.987), and
pairwise transfer does **not** compose into useful orderings (effect/noise
−0.15 with retention restored). Those results are what motivated this layer,
and they remain the reason the ordering programme was abandoned.

### The progression, stated plainly

    static ordering failed
      -> developmental-state hypothesis
      -> history-dependent future learnability   [CONFIRMED]
      -> state-conditioned data value            [SUPPORTED]
      -> prediction of that conditionality       [FAILED]
      -> adaptive training                       [not licensed]

### Confirmed

**Training history produces a large, selective difference in future
learnability.** Fresh confirmatory seeds, frozen criteria. Target capability
`B` at `t=0`: source arm **0.1322 ± 0.0149** against both controls at
**0.0156** — exactly the chance floor. Negative-control capability `C` shows
nothing (AULC advantage +0.0395 against +0.5265 on `B`), and `C` is genuinely
learnable, so the specificity test is not vacuous.

**Not memorised content.** With **zero shared entity tokens** between source
and target, the effect is fully retained (111%).

**Exploratory scale persistence.** Survives a 32× parameter increase
(786k → 25.2M), 3 seeds, original setup. Exploratory, not promoted.

### Falsified and inconclusive — kept visible

* **The pre-registered mechanism was falsified.** An off-distribution
  induction-style probe failed its confirmatory gate (selectivity 0.76 against
  a ≥2.0 requirement). A striking single-seed preflight transient did not
  reproduce anywhere.
* **Causal mediation is INCONCLUSIVE, not falsified.** The retrieval-head
  ablation produced no interaction, but it also failed to reduce the capability
  at all (zero-shot `B` 0.139 → 0.161), because the capability is redundant
  across heads. Necessity was never tested at adequate strength. **Top-k head
  ablation is a demonstrably inadequate intervention in this substrate.**
* **State-aware data selection FAILED against global-best.** Leave-one-state-out
  over 12 complete states: regret 0.0473 against global-best's 0.0207. It beats
  random, so it learned the data main effect and not the conditionality.
  **Adaptive scheduling is not licensed.**

### Supported, with corrected magnitudes

**State × Data interaction.** Balanced 48-cell matrix, 12 states measured on
every corpus. Interaction share 0.381 (mean objective) and 0.464 (min);
2–3 distinct argmax corpora across states; 42–56 sign reversals. **The identity
of the best next corpus depends on incoming state.** Correction on the record:
a first pass reported interaction 0.479 as the largest component; restricted to
complete states it is 0.381 and *second* to the state main effect (0.553).

**Retrieval state marker, narrowed.** Distinguishes the source arm from the
value-rebinding control prospectively on fresh seeds (+0.1267, permutation
p = 0.0002). **It does not cleanly separate the source arm from background out
of sample (p = 0.387)** — a comparison that had not been run until audit.
A marker, not a mediator.

**Temporal changepoint — exploratory only.** Held-out model comparison finds a
changepoint better than linear (RMSE 0.0171 vs 0.0331) with a break at step
270 ± 8. This models *acquired competence*, not future learnability, and the
two can dissociate.

### Active at the time of writing

* **Prospective hidden futures** — do states matched on present observables
  diverge under identical future training? Pairs frozen from current-state
  observables before outcomes exist.
* **True temporal replay** — dense checkpoints across the 150–450 window with
  identical continuations, measuring `V(S_t, B)` rather than a competence proxy.

### The honest summary

> Training-data value is conditional on model state. We cannot yet read that
> state well enough to exploit the conditionality.

That is a specific technical bottleneck — **state inference**, not phenomenon —
and it is where the programme continues.
