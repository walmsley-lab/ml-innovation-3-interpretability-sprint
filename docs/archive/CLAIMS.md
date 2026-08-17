# Claim ledger

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
