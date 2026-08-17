# RESEARCH

The scientific programme, as it now stands. This merges the state-space framing
that was drafted as a candidate revision and has since been **earned by
evidence**, with the parts of the original staged plan that survive.

The original plan — the fixed transfer-graph programme with stages 0–8 and
gates A–J — is preserved verbatim at
[docs/archive/research_v1_ordering_programme.md](docs/archive/research_v1_ordering_programme.md).
It is superseded, not deleted: its failures are what licensed this framing, and
`RESULTS.md` records them with their original numbers.

---

## 1. The object of study

The original plan treated the object of discovery as a **fixed scalar transfer
graph** `T_ij`, from which a developmental ontology and then a curriculum would
be derived. That programme failed empirically, twice and specifically:

* block-sequential curricula destroy capability at equal compute (0.383 mean /
  0.106 min, against interleaved 0.987 / 0.951);
* with retention repaired and dose decoupled from order, pairwise transfer
  **does not compose** into useful multi-stage orderings (effect/noise −0.15).

The object is therefore not a graph over datasets. It is a **developmental
state-space system**:

    z_{t+D} = F(z_t, u_t)
    y_t     = G(z_t)

`z_t` the learner's developmental state; `u_t` the intervention — which data,
in what mixture, at what budget; `y_t` what is observable; `F` the transition
model, **the object to identify**; `G` the observation model, which is not the
identity and may be substantially lossy.

Under this framing `T_ij` is a derived, local, scalar projection of `F`, and
three things follow that the fixed-graph framing cannot express: a scalar
summary can cancel the signal it summarizes; a transfer effect may be
state-dependent, `T_ij(z)`; and order effects are the ordinary behaviour of a
non-commuting transition operator rather than an anomaly.

**What has since been earned.** The state-dependence is no longer a conjecture.
`V(S,D)` shows a substantial State × Data interaction with ranking reversals:
the identity of the best next corpus depends on the incoming state. That is
direct evidence against a static notion of "good training data" and is the
clearest vindication of this framing over the fixed-graph one.

**What has not been earned.** Reading `z` well enough to act on it. A
state-aware selector built from current telemetry **loses to a state-blind
global-best rule** on held-out states. The transition model `F` is not
identified, and no adaptive claim is licensed.

## 2. The evidence ladder

Each rung is reachable only from the one below. Status as of the current
record.

| # | rung | status |
|---|---|---|
| A1 | history changes future learnability | **confirmed** |
| A2 | the effect is selective, not general learning ability | **confirmed** |
| A3 | not explained by memorised content | **confirmed** |
| A4 | survives a substantial change of scale | exploratory (32×) |
| A5 | developmental radius — how far does it generalize | open (`B₂`, target-distance ladder) |
| A6 | an internal state predicts future learnability | marker replicated; **not** predictive of conditional value |
| A7 | behaviourally matched models have different futures | **pending prospective test** |
| A8 | when the relevant state emerges | pending true `V(S_t,D)` replay |
| A9 | the state is causal | **inconclusive** — the intervention had no efficacy |
| B1 | corpus properties that produce the state | not started |
| B2 | causal perturbation: corpus feature → Δstate → Δlearnability | not started |
| C1 | state × data interaction exists | **supported** |
| C2 | telemetry predicts conditional data value | **falsified with current features** |
| C3 | one-step prospective data choice beats baselines | not licensed (needs C2) |
| C4 | repeated / closed-loop adaptive scheduling | not licensed (needs C3) |

**The gap is C2.** The conditional training signal exists and state inference is
inadequate. That is a far more specific bottleneck than "curriculum matters",
and it is where the programme continues.

## 3. Standing methodological commitments

These were paid for, mostly the hard way.

* **Freeze before outcomes.** Every prediction artifact is hashed and
  timestamped before the outcomes it predicts exist, and re-verified at
  scoring. Current frozen objects: V2.1 spec `f92f5831bece0d91`, tournament
  protocol `f9da9fe23b1b2400` (unrun), downstream protocols
  `8fc78c4087e2f87b`, Lane B protocol `afd0bf5174cd8073`.
* **`t = 0` evaluation is mandatory.** Head start and acquisition rate carry
  opposite signs in most pairs and are 3–4× more reproducible than their sum. A
  scalar that mixes them can cancel a real effect.
* **Report the efficacy check before the interaction.** An intervention that
  does not measurably move the capability cannot test necessity. Removing the
  top 4 of 16 heads left zero-shot competence unchanged; the resulting null is
  inconclusive, not negative.
* **Null-calibrate rather than assert tolerances.** Absolute thresholds set
  below the sampling noise floor cannot pass regardless of the truth.
* **Machine-check the ledger.** `scripts/audit_claims.py` recomputes headline
  numbers from raw units. It has caught an inflated reversal count and an
  untested comparison, both after they had been reported as results.
* **A falsified hypothesis with a surviving effect is the most valuable state
  the record can be in.** It means the phenomenon is real and the explanation
  was wrong.

## 4. Mechanism hypotheses, ranked by current plausibility

The evidence argues **against** the relevant state being a small set of
attention heads: the capability is redundant across heads, and top-k ablation
is demonstrably inadequate here. Prioritized alternatives:

1. **Gradient / update geometry.** Take identical future minibatches and
   compare gradient and update geometry across history-conditioned checkpoints,
   then ask whether those quantities prospectively predict measured `V(S,D)`.
   This addresses why the same data has different value from different weight
   states more directly than another head search.
2. Representation / feature preparation.
3. Distributed routing across heads and MLPs.
4. Parameter-subspace readiness.
5. Interference and retention geometry.

## 5. Deferred work, with triggers


## Fan-out plan — triggered by B₂

Three tracks, scientifically distinct enough to run concurrently. The **only**
hard dependency: no corpus attribution or adaptive selection until a state
signal exists that is worth attributing to or acting on. Design work for those
can proceed in parallel; launches cannot.

### Track 1 — Mechanism discovery (runs either way)
`M` is falsified, but H1–H4 stand: something carries the transfer. Search the
source-phase state with broader telemetry already built or cheap to add —
activation subspaces and effective rank (`dsi.state`, done), gradient alignment
between source and target batches, layerwise update magnitude, attention and
MLP feature statistics — then intervene on the best candidate with the ablation
runner (`run_v2_units.py --ablation`, built and idle).
*Why it runs regardless:* the effect is confirmed and unexplained. This is the
highest-value question in both branches.

### Track 2 — Developmental generalization
**If B₂ positive:** transition-window replay plus a state-dependence probe.
**If B₂ negative:** a "distance from A" target ladder — a family of targets
ordered by how far their computation sits from A's — to find where transfer
disappears. That boundary *is* the developmental radius, and it is a result in
its own right.
**If B₂ mixed:** the ladder becomes the primary experiment; mixed is the most
informative outcome because it means the boundary is inside our reach.

### Dense replay around the localized window — RETURN AFTER the stretch-goal gates resolve
*Idea.* Checkpoint densely around source step ~270 (where the held-out
changepoint fit puts the break, sd 8 across folds), clone each checkpoint, give
every clone the identical `B` continuation, and measure `V(S_t, B)`.
*Why it matters.* T1 shows a localized change in *competence*. Whether the
same moment marks a change in *learnability* is the question that would make it
developmentally meaningful, and it is untested.
*Cost.* Cheap — the window is narrow and `continue_from_state.py` is the
primitive. Perhaps 10 checkpoints x 1 corpus x 3 seeds.
*Return when.* V(S,D) interaction and predictor validation have resolved, and
the tournament has either run or been declined. **Not before** — it is
exploratory and must not take cores from a licensed prospective demo.

### Stronger causal intervention — RETURN AFTER submission
*Idea.* An intervention capable of **demonstrably moving the candidate state**
before any mediation is interpreted. Top-k head ablation is not it: removing
the 4 highest-scoring heads of 16 left zero-shot BIND unchanged (A6c), because
the capability is redundantly distributed.
*Candidates.* Broader coordinated ablation (many heads, or whole layers);
activation / residual-stream patching; transplantation (machinery already
built and verified in `dsi.mechanism.transplant_heads`).
*The gating rule, which is the real lesson.* **Report the efficacy check
first.** An intervention that does not move the capability cannot test
necessity, and a null interaction from such an intervention is uninformative
rather than negative. H2.3 is recorded as inconclusive for exactly this reason.
*Return when.* After the submission window. Nothing in the confirmed core
(H1–H4) depends on it.

### Optimizer-state continuation — RETURN AFTER a state predictor survives confirmation
*Idea.* Persist optimizer state alongside weights and re-run the same branches
with moments restored, matched against the weights-only continuations already
measured.
*Why it matters.* `continue_from_state.py` measures `V(W_t, D)` — weights only.
If readiness resides partly in optimizer/history state it is invisible to that
instrument, and the difference between the two continuations localizes where
readiness lives. That is a real result, not a robustness check.
*Cheap setup now.* `dsi.checkpoint` would need an opt_state path; the branches
themselves are already defined.

### Track 3 — Intervention (gated)
Corpus attribution, then the one-step candidate-corpus tournament.
*RETURN AFTER* Track 1 or 2 yields a state signal that prospectively predicts
future learning. Designs may be written now; runs are not licensed until then.

### Execution pattern
```
B2 result -> fan out Tracks 1 and 2 concurrently
          -> Track 3 launches only when a predictive state signal exists
```

## The discrimination-first chain (current priority order)

Inserted after the scale scout showed a large, scale-persistent `t=0` advantage
for `A` that our candidate mechanism `M` does **not** explain. The ordering
below exists so a demonstration is not impressive for the wrong reason.

### 1. C1 discrimination — RUNNING
*Idea.* Source with identical copy structure over a **disjoint entity
sub-pool** (`IND#h1` → `BIND#h2`, zero shared entity tokens) against a
same-pool source (`IND#h2` → `BIND#h2`).
*Why it matters.* `A` and `B` share the underlying computation by
construction, so "A helps B" may be ordinary task transfer rather than
developmental readiness. This is currently the highest-value blocker: it
decides whether the whole effect is interesting.
*Reading.* Disjoint retains the head start → the mechanism generalizes across
token identity (readiness). Disjoint collapses to chance → content transfer,
and the effect is much less interesting.

### 2. Dense transition-window replay — RETURN AFTER C1 shows readiness survives
*Idea.* Checkpoint tightly around the surviving latent change; give pre-peak,
peak and post-peak checkpoints identical future `B` continuation.
*Why it matters.* Tests whether crossing an internal event changes the model's
future **before** ordinary behaviour distinguishes the checkpoints.
*Caveat now known.* Do **not** reuse `A_peak = max over trajectory` — it is
biased toward noisier arms, which is how `BG` scored the highest peak at 1x.
A replay statistic must be robust to that.

### 3. Corpus attribution — RETURN AFTER a real latent shift is established
*Idea.* Which human-readable data slices advance or delay the transition.
*Why it matters.* It is the step that makes the state actionable rather than
descriptive, and the first thing a reader can inspect directly.

### 4. One-step candidate-corpus tournament — RETURN AFTER a state signal
predicts future learning
*Idea.* Given the observed state, predict which of several candidate pools
produces the largest useful gain; verify prospectively.
*Why it matters.* The first visible bridge from interpretability to
intervention.

## Triggered by V2 results

### `do(M+)` sufficiency
*Idea.* Install or advance `M` by intervening on the mechanism itself — head
grafting from a donor checkpoint, freezing in place, or an auxiliary
prefix-matching objective — then test for accelerated `B` with source exposure
held constant.
*Why it matters.* H2.3 tests necessity only. Without sufficiency, "M is
required for the transfer" must not be reported as "M produces the transfer",
and the causal chain is incomplete.
*Return when.* H2.3 necessity passes (G2).
*Cheap setup now.* None that is free of contamination. The grafting confound —
a donor's heads arrive with a compatible residual basis, so the transplant
imports more than `M` — is a design problem to solve on paper, and can be
thought about at zero compute cost.

### Architecture sweep (depth × width × heads)
*Idea.* V2-c and V2-d, then width and head-count, measured sequentially rather
than factorially: emergence time of `M`, transfer magnitude, intervention
effect, retention, gradient alignment, representation similarity per config.
*Why it matters.* Turns "does order matter" into "which properties of
transformer learning make order matter" — considerably more explanatory.
*Return when.* The basic mechanism replicates at 12 seeds (G1b) **and** H2.3
passes. Pull V2-c forward early only if the ablation interaction is ambiguous,
since depth-1 can discriminate mechanisms when ablation cannot.
*Cheap setup now.* Parameter-matching table for depth {1,2,4} at equal
non-embedding params — arithmetic only, no runs.

### Second mediator, second capability pair
*Idea.* Replicate the identification apparatus on a different mechanism and a
different capability pair.
*Why it matters.* One mediator generalizes to nothing. This is what converts a
case study into a method.
*Return when.* The full V2 chain (G1b + G2) passes.
*Cheap setup now.* Candidate list only.

### State-dependent transfer: `T_AB(S1) ≠ T_AB(S2)`
*Idea.* Run the same `A→B` transition from two different prior states
(`X→A→B` vs `Y→A→B`), holding `A` and `B` exposure fixed. If `A`'s effect on
`B` depends materially on incoming state, the static transfer graph is the
wrong abstraction and the object is a transition system
`P(S_{t+1} | S_t, D_t)`.
*Why it matters.* This is the cleanest available explanation for the L0
composition failure (overlap diagnostic, Branch 2, effect/noise −0.15), and it
is the prerequisite abstraction for closed-loop scheduling.
*Return when.* A mediator is identified (G2 passes) — the experiment needs a
state representation worth conditioning on.
*Cheap setup now.* The design reuses the V2 substrate almost unchanged; the
arm structure can be drafted at no compute cost.

## Lane B redesign — RETURN AFTER V2.1 / G1 resolves

*Idea.* A substrate that can host a behaviourally-matched preference test:
two skills of comparable acquisition difficulty that reliably coexist, so both
orderings reach competence.
*Why it matters.* The Digital Minds claim — behaviourally matched models with
different histories diverge under identical future experience — is still worth
testing. Only the W/P substrate is disqualified, not the question.
*What killed it.* W needs a pure phase to be learnable; `P_first` never
provides one, so competence pass rate was 1/6 (B1b). Difficulty matching is a
**gate**, not a nicety — the same lesson the Layer-2 `t90` calibration encodes.
*Return when.* V2.1 / G1 resolves. The micro-world is a candidate host: `BIND`
and `FACT` are already difficulty-matched by construction and both learnable.
*Cheap setup now.* None needed — the entire Lane B pipeline (generation,
basis-invariant state extraction, hash-verified frozen analysis) is built and
substrate-agnostic. It needs a substrate, not code.

## Triggered by sprint results

### Richer latent-state representations
*Idea.* Beyond residual-stream activations: gradient alignment, curvature
proxies, function/task vectors, sparse features.
*Why it matters.* The sprint experiment's `P_int` is deliberately simple. If
state predicts future divergence at all, what *kind* of state does it best.
*Return when.* S2 passes — internal state beats the behavioural baseline.
*Cheap setup now.* The frozen probe input set is shared, so extraction
plumbing built for the sprint experiment already covers most of this.

### Behavioural-matching at larger populations
*Idea.* If cross-history matched pairs are too few at the frozen `epsilon`,
scale the population rather than loosening `epsilon`.
*Why it matters.* S1 is untestable below a minimum pair count, and loosening
the matching rule to manufacture pairs would invalidate the claim.
*Return when.* S2 passes and S1 is pair-limited rather than signal-limited.
*Cheap setup now.* None; it is a compute decision, and compute is not the
constraint.

## Triggered by mechanism existing at all

### MDA / training-data attribution
*Idea.* Attribute an identified mechanism back to the training examples that
produced it: "feature X appeared at step 42k; MDA identifies the examples
disproportionately associated with it."
*Why it matters.* It is the bridge from "a mechanism exists" to "this data
built it", and it is the piece that would make developmental scheduling
actionable rather than descriptive.
*Return when.* There is a real mechanism worth attributing — G2 passes.
*Cheap setup now.* None. Attribution before a mechanism exists is attribution
to nothing.

### Closed-loop / adaptive scheduling
*Idea.* Estimate state online and choose the next training window to move it
deliberately; the resource-constrained benchmark where strong interleaving is
below ceiling, versus uniform, randomized, heuristic and adaptive schedulers on
the frozen efficiency metrics.
*Why it matters.* The north star. Also the one branch with a licensed prior
result: interleaving wins every efficiency metric at equal compute
(min acc 0.951 vs 0.529), so it is the baseline to beat.
*Return when.* Internal state prospectively predicts transitions — i.e. after
state-dependent transfer, not merely after a mediator is identified.
*Cheap setup now.* The efficiency metrics are already frozen in `RESULTS.md`
and need no further work.

### Broader natural-corpus work
*Idea.* Return to WikiText/20NG-scale corpora with a mechanism-based rather
than label-based state representation.
*Why it matters.* The eventual claim has to survive outside a controlled
substrate.
*Return when.* The mechanism or state representation survives the controlled
bridge (G2 + sprint S2).
*Cheap setup now.* **Nothing.** The frozen WikiText adaptive pool (3 unordered
/ 6 directed) stays untouched and unlicensed. Corpus caches already exist and
are verified byte-identical; no further preparation is useful.

### Interactive research artifact / UI
*Idea.* `research.md` §42, `technical.md` §55–57.
*Why it matters.* Communication, not science.
*Return when.* There are results worth navigating. Figures for the paper are
on the critical path; a UI is not.
*Cheap setup now.* The figure contract (`technical.md` §58) is the useful
subset and is covered by paper work.

## Deliberately not revisited

* Extending the L0 synthetic ontology or adding families to rescue a result.
  The four frozen families and the 12-pair pilot are terminal.
* Refitting or reinterpreting the WikiText Stage-5 gate. Closed.
* Patching the static ordering construction. Branch 2 fired; no
  pairwise-derived static ordering carries forward.
