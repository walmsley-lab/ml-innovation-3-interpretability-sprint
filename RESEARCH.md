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

    z_{t+Δ} = F(z_t, u_t)
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
| A6a | some internal measurement distinguishes history-conditioned states | **marker replicated** prospectively (p = 0.0002 vs the matched control; not separated from background, p = 0.387) |
| A6b | internal state predicts **conditional corpus value** — which `D` is best from this `S` | **falsified with current telemetry** — loses to a state-blind baseline |
| A7 | behaviourally matched models have different futures | **pending prospective test** |
| A8 | when the relevant state emerges | **no localized change detected** in the tested window; not resolved at finer resolution |
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

## 4. Mechanism hypotheses, in priority order for the next experiment

What the evidence establishes is **negative**: the relevant state is probably
not a small set of attention heads, because the capability is redundant across
heads and top-k ablation is demonstrably inadequate here. It does **not**
positively identify any alternative. The ordering below is therefore by
expected information per experiment, not by demonstrated plausibility.

1. **Gradient / update geometry — the highest-priority next hypothesis.** Take identical future minibatches and
   compare gradient and update geometry across history-conditioned checkpoints,
   then ask whether those quantities prospectively predict measured `V(S,D)`.
   This addresses why the same data has different value from different weight
   states more directly than another head search. Untested.
2. Representation / feature preparation.
3. Distributed routing across heads and MLPs.
4. Parameter-subspace readiness.
5. Interference and retention geometry.

## 5. Open questions and next experiments

Current only. Superseded execution plans — several with triggers conditioned on
gates that have since resolved differently — are archived at
[docs/archive/execution_plans.md](docs/archive/execution_plans.md).

### Active

| question | experiment | status |
|---|---|---|
| Can behaviourally matched models have different futures? | pairs frozen from present observables, identical fresh continuation | running; stopping rule frozen at `01c89adc9b66b9b6` |
| Does *future learnability* change locally during early training? | dense checkpoints across 150–450, identical continuations, `V(S_t,B)` | **complete — no localized change found** |

### Next, in priority order

1. **State inference.** The single blocking problem. Current telemetry loses to
   a state-blind baseline, so build a representation that predicts held-out
   conditional data value — and require it to beat global-best before anything
   downstream is attempted. Evidence argues against a small set of attention
   heads and toward gradient/update geometry: take identical future minibatches,
   compare gradient and update geometry across history-conditioned checkpoints,
   and ask whether those quantities prospectively predict measured `V(S,D)`.
2. **A causal intervention that works.** Any mediation test must first
   demonstrate it can move the capability. Top-k head ablation demonstrably
   cannot. Candidates: broad coordinated ablation, whole-layer ablation,
   activation/residual patching, transplantation.
3. **Developmental radius.** A family of targets ordered by distance from the
   source, to find where transfer disappears. A graded boundary is as
   informative as a cliff.
4. **Corpus causes.** Factor the source into human-readable properties, perturb
   one at a time, and measure `corpus feature → Δstate → Δlearnability`.

### Gated, with explicit triggers

| deferred | RETURN AFTER |
|---|---|
| prospective what-next tournament (frozen, unrun, `f9da9fe23b1b2400`) | a predictor beats global-best on held-out states |
| `do(M+)` sufficiency | a necessity test with demonstrated intervention efficacy |
| second mediator / capability pair | a first mediator is causally established |
| optimizer-state continuation — does readiness live in weights or history? | a state predictor survives confirmation |
| training-data attribution | a causal mediator exists to attribute to |
| closed-loop scheduling | one-step prospective data choice wins |
| broader natural-corpus work | the mechanism survives the controlled bridge |

### Not revisited

The static-ordering programme. Block-sequential curricula destroy capability,
pairwise transfer does not compose, and no pairwise-derived static ordering
carries forward. The L0 synthetic ontology is terminal and is not extended to
rescue a result.
