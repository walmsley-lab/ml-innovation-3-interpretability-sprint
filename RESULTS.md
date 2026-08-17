# RESULTS

What has actually been measured. This is the canonical results document: it
merges the former `STATUS.md` barrier log, `report.md` chronology, and the
`CLAIMS.md` ledger into one place.

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
| A4b | A confers a `t=0` head start on B, and it survives scale | **Tentative — positive** | Exploratory scale scout: A `t=0` = 0.126±0.016 (1×), 0.157±0.042 (8×), 0.097±0.037 (32×) against A′ 0.012–0.014 and BG 0.017–0.021, with chance 0.0156. A′/BG at chance at every scale. AULC much noisier (effect/noise ≈1.9). **Strongest alternative explanation: A and B share the computation by construction, so this may be direct task transfer, not readiness. **Resolved:** the C1 disjoint-pool scout subsequently confirmed the effect survives with zero shared entity tokens (H3), ruling out content transfer |
| A5 | M rises before B accelerates (H2.2) | **Moot** | Requires an A-selective M. A2b falsified |
| A5b | The mediator of the confirmed transfer is unidentified | **Open — the central question** | The effect (H1–H4) is real; `M` does not explain it. Finding the actual mediator is now the primary scientific target |
| A5c | The on-distribution retrieval score distinguishes A from A′ | **Supported for A vs A′ only — NOT established against background** | A vs A′: discovery +0.1157 (p=0.0012), **validation +0.1267 (p=0.0002)** on fresh seeds 700–711. **A vs BG, which had never been tested: discovery +0.1154 (p=0.0012) but validation +0.0465 (p=0.3870) — not significant.** The background arm's retrieval rose to 0.1284 on fresh seeds against A's 0.1749. So the marker separates A from the value-rebinding control but **does not cleanly separate A from background** out of sample. Claim narrowed accordingly: it is an A-vs-A′ discriminator, not a general A-selectivity marker. Not causal (A6b inconclusive), not predictive of conditional data value (X2d falsified), and measured on-distribution |
| A6b | Retrieval-head ablation preferentially removes the A→B advantage | **INCONCLUSIVE — insufficient intervention efficacy. NOT a mechanistic falsification.** | H2.3 corrected, seeds 600–603, both conditions, 20 units. Arm × ablation interaction: `t=0` −0.0181, AULC +0.0094, final −0.2441 — no positive interaction. **But the intervention did not work:** zero-shot BIND on A went 0.1387 → 0.1611 under retrieval ablation (no reduction), vs 0.1450 → 0.1431 under matched-random. Weights are provably zeroed (P7), so the capability is redundant across remaining heads. **Necessity was never tested at adequate strength.** Must not be cited as evidence that retrieval is not the mediator |
| A6c | Top-k head ablation is an adequate causal intervention in this substrate | **Falsified** | Removing the 4 highest-scoring heads of 16 leaves zero-shot BIND unchanged. Any future causal test needs a stronger intervention — many more heads, whole-layer ablation, or activation patching — and this row is a prerequisite for interpreting any ablation null here, including our own |
| A6 | M ablation preferentially removes the A→B advantage (H2.3) | **Not licensed** | Ablating a mediator that is not selective would test nothing. Runner built; awaits a candidate mediator |
| A7 | The effect requires ≥2-layer composition at matched capacity (H2.4) | **Untested** | Deferred to V2-c |
| A8 | Promoting M accelerates B — `do(M+)` (sufficiency) | **Not licensed** | Same reason as A6 |
| **A9** | **The transfer is broader than computation overlap — developmental readiness** | **SUPPORTED** | B₂ factorial complete, 24 units, 4 seeds per arm x 3 arms x 2 surface conditions, target `BINDT` (retrieval ∘ derangement). **Zero-shot is blocked by construction and confirmed blocked**: the source arm sits at or below chance at `t=0` (0.0122 disjoint, 0.0156 shared, chance 0.0156), so no answer can be transferred. Yet the source arm learns the target far faster — `rate_only` advantage over the matched control **+0.5438** (disjoint surface, effect/noise 1.86) and **+0.3967** (shared, 1.63); final accuracy 1.0000 against A′ 0.26/0.51 and BG 0.19/0.32. **The advantage survives the disjoint-surface condition** (zero shared entity tokens) and is if anything larger there. Per the pre-declared reading table, an advantage in `rate_only` with `t=0` at or below chance is **readiness**, not task transfer. **Caveats:** n=4 per arm; effect/noise 1.6–1.9 rather than overwhelming; the control arm has large between-seed variance (sd 0.14–0.21 on `rate_only`) |

## Lane B — sprint latent state

| # | Claim | Status | Evidence |
|---|---|---|---|
| B0 | State features are basis-invariant and comparable across inits | **Apparatus — passed** | 137 features, deterministic, no raw activation coordinates; verified 127/137 differ between histories sharing one init |
| B1 | The pilot regime supports a preference measurement | **Falsified** | 0 of 24 models cleared the competence floor; W_COMPETENCE 0.256 against chance 0.25. Config error (below frozen budget), corrected |
| B1b | The W/P substrate supports a behaviorally-matched latent-state test | **Falsified** | With the frozen overlap regime, corrected common tail (T1) and a symmetric 2000-step budget: competence pass rate **1/6**. `P_first` reaches W = 0.238–0.293 against a 0.60 floor in every seed — W is only learnable when it gets a pure phase, which `P_first` never provides. `W_first` is itself seed-fragile (0.977 / 0.469 / 0.508), consistent with `docs/risks.md` §2b. **Substrate recorded as unsuitable; Lane B stopped, not redesigned.** |
| B2 | Internal state predicts post-intervention outcome better than behavior (S2) | **Untested — blocked** | Blocked by B1b. The pipeline (generate → extract → analyze, 137 basis-invariant features, hash-verified protocol) is built and validated; it needs a substrate where both skills coexist |
| B3 | State information survives behavioral matching (S1) | **Untested** | Gated behind B2 and the 20-matched-pair floor |
| B4 | Behaviour at `t` determines behavior at `t+Δ` | **Untested** | The null B2 is written against |

## Exploratory — developmental shape (off critical path)

| # | Claim | Status | Evidence |
|---|---|---|---|
| T1 | Acquisition of zero-shot `B` competence during `A` training is better described by a localized change than by smooth growth | **Tentative — exploratory, and its precision claim is withdrawn (see T1-corr)** | Held-out (leave-one-seed-out) comparison over 4 seeds, three descriptions frozen in advance: changepoint RMSE 0.0171 ± 0.0036 vs linear 0.0331 ± 0.0037 and sigmoid 0.0353 ± 0.0038. The traces were sampled every **250 steps**, so the fitted break location is an interpolation between the first two samples and carries no better than 250-step resolution. **The defensible statement is that competence rises somewhere within the first 250 steps.** Models *acquired competence*, not future learnability — see T4, which tested learnability directly and found no localized change |
| T2 | That change is a transition in **future learnability** `V(S_t, B)` | **Untested** | T1 models *acquired competence*, not learnability. The two dissociate — a model can gain learnability without gaining zero-shot competence, which is what `BINDT` exists to test. `V(S_t,B)` needs dense multi-step checkpointing plus identical continuations; the discovery runs save one checkpoint at the end of the source phase, so no temporally-ordered states exist |
| T3 | Behaviour-matched states have divergent futures (P1) | **FALSIFIED — well-powered null** | All **71 frozen pairs** complete (contiguous prefix of the pre-seeded run order, so no subsample), 51 states, against a within-arm unmatched null of 376 pairs. Matched pairs do **not** diverge more: `final` +0.0193 (perm p=0.704), `rate_only` −0.0171 (p=0.504). Survives outlier removal (p=0.951, 0.701). `t=0` shows matched pairs diverging *less* (−0.0032, p=0.005) but that is **circular** — zero-shot accuracy is part of the matching vector, so matched pairs have similar `t=0` by construction; the correlation between matching distance and `t=0` divergence is +0.802, confirming it. The informative metrics are `final` and `rate_only`, neither of which is matched on, and both are flat. **Per the frozen null branch (`8bbbadd1529efb1e`): training history changes future learnability in controlled contrasts, but we did not establish that those differences remain hidden among behaviorally matched checkpoints.** H1–H4 are untouched — they rest on unmatched contrasts, which P1 does not test |
| T3b | Behavior-matched states prefer *different* future corpora (Fork) | **FALSIFIED — null** | Matched-State Counterfactual Fork, protocol `facef66229928f4a`, all **16 frozen pairs** complete (4 branches each: 2 states x 2 corpora, BIND vs BINDT, the pair with the strongest prior ordering reversal). Mean state x data interaction **+0.0888, 95% CI [−0.1173, +0.3162]** — includes zero; sd 0.4488 is 5x the mean. **8/16 pairs share the aggregate sign, exactly chance**, so the 8 apparent ordering reversals are what random sign assignment produces. Dropping pairs touching an unstable unit gives +0.0295, CI [−0.1707, +0.2368]. Correlation with matching distance −0.304. **The single most extreme pair looks dramatic (0.613/0.083 against 0.024/0.528) and is the maximum of a noise distribution; per the protocol it is not shown as a result.** |
| **T3-joint** | *(joint reading of P1 + Fork)* Present behavior fails to capture future-relevant developmental variation | **FALSIFIED** | Two independent, adequately powered nulls: matched states neither diverge under a shared future (P1, 71 pairs vs a 376-pair null) nor prefer different futures (Fork, 16 pairs, CI spanning zero at chance sign agreement). **The honest reading is the opposite of the original hypothesis: present behavioral measurements capture more future-relevant developmental variation than we expected.** Matching on a modest behavioral vector — zero-shot accuracy and loss on two capabilities — was sufficient to remove the differences in future learning that unmatched contrasts show clearly (H1–H4, A9). This does not weaken those results; it bounds them. The developmental differences are real and consequential, and they are **legible in behavior** rather than hidden behind it |
| T4 | **Future learnability `V(S_t,B)` changes locally around the ~270 window** | **FALSIFIED** | Protocol B, 3 fresh seeds (900–902), 48 checkpoints across 150–450 every 20 steps, each given an identical `BIND` continuation. Held-out model comparison: **linear 0.1333**, sigmoid 0.1416, changepoint 0.1425 — linear wins, and by only 5.8%, which the protocol pre-declared as *not distinguishable*. Correlation of `V(S_t,B)` with `t` is **r = −0.199** against a spread of 0.36–0.91 (sd 0.133): the variance is training instability, not development. **No localized change in future learnability in this window** |
| T1-corr | *(correction to T1)* The competence break was located to ±8 steps | **Withdrawn — false precision** | T1 fitted traces sampled every **250 steps** (points at 0, 250, 500…), so the changepoint at 270 was *interpolated between the first two samples*. The ±8 measured across-fold agreement on that interpolation, not measurement precision. Independently, P2 shows zero-shot competence already at ~0.13 by step 150 and flat thereafter, so the rise completed before the window opened. **The honest statement is that the rise occurs somewhere within the first 250 steps.** The window was not re-centred after seeing this — the protocol pre-declared that a break outside 150–450 is reported, not chased |

**Language discipline for T1:** "changepoint is the better held-out
description, with a tightly localized break" — not "phase transition". The
comparison was made on held-out fit rather than by inspecting a curve, which is
what makes it reportable at all, and 4 seeds is not many.

## Experiment 4 — gradient/update geometry

| # | Claim | Status | Evidence |
|---|---|---|---|
| E4a | Gradient/update geometry distinguishes history-conditioned states | **Supported as a state/history marker** | Measured on 76 saved states with identical future minibatches. Between-arm separation is large and consistent: gradient norm on `BIND` 0.54 (A) vs 0.29 (A′); cos(∇BIND, ∇FACT) +0.73 vs +0.19. **This is history discrimination — the same thing the retrieval marker already does. It is not evidence of predicting conditional value** |
| E4b | Gradient geometry predicts state-conditioned data value `V(S,D)` | **Formally: primary gate met. Scientifically: INCONCLUSIVE. Not promoted.** | *As written*, the frozen gate asked whether the readout beats global-best on the pre-specified `min` objective. On the point estimate it does: regret **0.0017 vs 0.0027**, top-1 9/13 vs 7/13. **The frozen gate did not require uncertainty separation, and none is retroactively claimed.** It is nonetheless recorded as inconclusive because (i) the advantage is 0.0010 in absolute terms, (ii) `min` had *already* been flagged in X2c as sitting at the chance floor with possibly-noise reversals, before this experiment ran, (iii) a bootstrap over 13 states gives 95% CI [0.00000, +0.00285], reaching zero, and (iv) the `mean` objective points the **opposite** way (regret 0.0436 vs 0.0191, CI [−0.07362, +0.00000]), also unresolved. Two objectives disagreeing, both with intervals touching zero, at n=13, is not a readout result |
| E4c | Experiment 6 (prospective tournament) is licensed | **Not licensed** | Held gated by decision, not by the letter of the gate. Firing it on a metric documented as near-degenerate *before* the result existed would exploit a known weakness. **E4 is not expanded post hoc**; whether a prospectively powered E4b is warranted — with substantially greater geometry × complete-`V(S,D)` overlap than the 13 states available here — is decided after the frozen experiments finish |

The binding constraint was never the geometry (76 states measured) but the
overlap with complete `V(S,D)` rows: only **13** states have both.

### Late exploratory lanes B and C — post-hoc, not on the frozen ladder

Both reuse existing artifacts; neither retrains, and neither may promote a
claim. They are recorded because they bear on *why* E4b is needed.

| # | Question | Status | Evidence |
|---|---|---|---|
| LB | Does internal geometry add predictive value over the behavioral readout? | **Exploratory clean negative** | Behavior-only, geometry-only, and behavior+geometry are indistinguishable within noise (spread 0.0015 against a 0.0019 meaningful-margin threshold; top-1 8/13, 11/13, 8/13; mean regret 0.0451, 0.0436, 0.0448). Combined is *worse* than geometry alone, which is the signature of noise rather than of incremental value — the 0.0003 gap is **not** described as incremental value. All three underperform the state-blind global-best baseline (0.0191). **Interpretation:** internal geometry can identify training history, but neither geometry nor the current behavioral readout predicts `V(S,D)` well enough to beat a simple state-blind baseline in this pilot. Exploratory support for the *need* for a powered E4b; not part of the frozen ladder |
| LC | Is P1's null an artifact of how much behavior was matched on? | **Exploratory — no** | Recovery of the abandoned sufficiency analysis. The original defect was subset-specific epsilon (matching strictness varied ~50× across subsets, so richer subsets got looser matched sets). Fixed by holding matching pressure constant: four **nested** behavioral feature sets, standardized once, same fixed **K = 71** closest within-arm pairs for every subset, P1's eligibility constraints and same-arm null preserved. `t = 0` excluded as circular (it is a matching variable). **Anchor: the top rung reproduces frozen P1 exactly** (final `+0.0193`, rate_only `−0.0171`), so the ladder measures P1's quantity. One pre-specified trend test across the ladder: Spearman ρ = −0.738 (final, p = 0.26), −0.800 (rate_only, p = 0.20) — not significant. Coarsening from four behavioral numbers to one does not make hidden divergence appear |

Lane C caveat: per-rung permutation tests are reported for transparency and are
**not** the pre-specified analysis. Across eight rung-by-outcome cells a
Bonferroni threshold is p < 0.006; the one nominally low cell (p = 0.034) does
not clear it, is *positive* in direction, and is non-monotone in the ladder.
Lane C does **not** show behavior is sufficient to predict the future — the
question it was built to answer needs a design containing divergence to
titrate, which this data does not have.

Lane D (dose-response) and Lane A (B₂ replication) are reported in their own
sections; Lane A carries robustness evidence only.

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
here; the scientific program is in [RESEARCH.md](RESEARCH.md), the system
specification in [docs/technical.md](docs/technical.md), and threats to validity in
[docs/risks.md](docs/risks.md).

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

**Measurability passed** the pre-specified S/N ≥ 2.0 threshold at 2.31.

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
status in [the claim ledger](RESULTS.md).

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
* [docs/archive/RESEARCH_V2_CANDIDATE.md](docs/archive/RESEARCH_V2_CANDIDATE.md) is a **draft under
  review**, not canonical, and does not revise any result above.

---

# Part III — Detailed historical records

Not duplicated here. The canonical status of every claim is Part I; this
section points at the underlying records.

| record | contents |
|---|---|
| [docs/archive/barrier_log.md](docs/archive/barrier_log.md) | per-barrier log: completed stage, hashes, gate result, interpretation, risks, next licensed action |
| [docs/experiments/](docs/experiments/) | per-experiment designs and frozen protocols, each hashed before the runs it governs |
| [docs/archive/](docs/archive/) | superseded plans, the original ordering programme, earlier root documents |

Frozen protocol hashes, for verification:

| object | sha256 (first 16) |
|---|---|
| V2.1 spec | `f92f5831bece0d91` |
| what-next tournament (unrun, trigger failed) | `f9da9fe23b1b2400` |
| downstream protocols | `8fc78c4087e2f87b` |
| P1 stopping rule | `01c89adc9b66b9b6` |
| latent-state protocol | `afd0bf5174cd8073` |
