# Research ladder: from the sprint to closed-loop developmental pretraining

Five milestones. Each is a claim about **developmental state**, and each is
only reachable from the one below it. Nothing downstream is built yet; the
ladder exists so we know the path and avoid architectural dead ends.

```
A  state exists          history leaves latent information predicting future divergence
B  state is causal       a measurable mechanism mediates learnability, survives intervention
C  state is conditional  identical data acts differently from different incoming states
D  state is discoverable coordinates inferred, not supplied, predict held-out state x data
E  state is controllable state-aware selection beats strong conventional baselines
```

**Priority is by falsification value, not by ladder order.** The four results
that would most radically change the programme, in order:

1. behaviour-matched histories fail to diverge predictably (kills A)
2. `M` fails to mediate transfer (kills B)
3. causal manipulation of `M` fails (kills B's causal half)
4. identical training does not depend on incoming state (kills C, and with it
   the transition-system abstraction that D and E assume)

---

## Milestone A — state exists

**Experiment.** The sprint latent-state experiment
(`docs/experiments/sprint_latent_state_design.md`): 128 paired inits × 2
histories (`W→P`, `P→W`) to a matching checkpoint `t*`, then two frozen future
interventions applied identically. Predict the preference endpoint from
behaviour at `t*` versus internal state at `t*`.

**Prerequisite.** Checkpoint persistence (done) + activation extraction.

**Frozen criterion.** `P_int` beats `P_beh` on held-out models under both
interventions, with the dimension-matched variant also beating it, and the
combined predictor showing incremental value for internal state. Fails if
`P_beh ≈ P_int`, or if both are at chance.

**Cost.** 256 models to `t*` + 512 continuations; ~1 h on the two C3 workers.

**Primary risk.** The endpoint may be unpredictable by anything — Gate C's
within-history sd is 12.87 in log-odds. Also: matching on behaviour is
conditioning, and `P_int` can win on capacity alone.

**Implementation.** Activation extraction on a frozen probe set; the matching
rule and `epsilon` frozen on a development split of models.

**Parallel-preparable.** The behavioural predictor's feature set, the frozen
probe inputs, and the matching rule — all before any model is trained.

**Licenses.** "Models matched on current observable behaviour differ
internally in ways that predict divergence under identical future experience.
Behavioural evidence at a point in time does not determine future behaviour."

**If it fails.** Either behaviour is sufficient (a real answer to the sprint's
question) or the endpoint is noise (which closes the Gate C story). Neither is
a dead end; both are reportable. B is unaffected — it is a different substrate.

## Milestone B — state is causal

**Experiments.** (1) V2 G1 scout, 3 arms × 6 seeds, expand to 12 if earned.
(2) H2.3 arm × ablation interaction.

**Prerequisite.** V2 preflight P1–P7. P1, P2, P6, P7 **pass as of now**; P3
(the `M` probe detects the mechanism), P4 (`B` learnable, not ceilinged) and
P5 (phase boundary alone produces nothing) need training.

**Frozen criterion.** Scout: `A` produces more `M` than `A'`; `A→B` beats
`A'→B`; `C` does not move; `M` precedes `B` acceleration. Ablation:
`Δ = [B_A − B_A']_intact − [B_A − B_A']_do(M−) > 0` **and** the matched-random
interaction ≈ 0.

**Cost.** 18 runs to the first decision, 36 to replicate, 72 short for
ablation. ~1–2 h wall-clock each with seed-level parallelism.

**Primary risk.** The micro-world manufactures the effect; `M` correlates
without causing; tiny-model optimizer/capacity dynamics masquerade as
developmental state.

**Implementation.** Micro-world generator (done), `M` probe and ablation
(done), telemetry stack, GPU executor.

**Parallel-preparable.** Ablation code smoke-tested before the scout returns,
so H2.3 launches in minutes rather than starting implementation.

**Licenses.** "In a controlled LM setting, a mechanism emerges during a
training history, precedes the future-learning advantage, and its targeted
disruption selectively attenuates that advantage."

**If it fails.** At the scout: the known-mechanism bridge does not materialize
under an LM objective — substantive, and it retro-illuminates L0 as
substrate-specific. At the ablation: stop telling a mechanistic story; the
transfer result survives without a mediator.

## Milestone C — state is conditional and general

**Experiments.** (1) State dependence: the same `A→B` transition from two
deliberately different incoming states, `X→A→B` vs `Y→A→B`, `A` and `B`
exposure held fixed. (2) A second mediator on a different capability pair.

**Prerequisite.** B's causal half passes (H2.3).

**Frozen criterion.** `T_AB(S_X) ≠ T_AB(S_Y)` beyond the seed-noise floor,
with `A` and `B` dose asserted equal per unit.

**Cost.** Exploratory scout: 2 states × 3–4 seeds ≈ 8 runs, ~30 min. Confirmatory
later.

**Primary risk.** State difference confounded with competence difference at the
`A` entry point; the two prior states must be matched on `B`-relevant
competence or the result is trivial.

**Implementation.** Reuses the V2 substrate almost unchanged.

**Parallel-preparable.** The arm structure, at zero compute cost. **Worth a
cheap exploratory probe even before B completes** — if transfer is obviously
state-dependent, that alone explains why pairwise transfer predicts locally and
fails to compose globally (overlap diagnostic, Branch 2, effect/noise −0.15).

**Licenses.** "Transfer is a function of incoming state, so the object is a
transition system `P(S_{t+1} | S_t, D_t)`, not a graph over datasets."

**If it fails.** The static graph abstraction survives, and the L0 composition
failure needs a different explanation — retention dynamics or no compositional
structure at this scale. D and E would need rethinking, since both assume a
state to condition on.

## Milestone D — state is discoverable and predictive

**Experiment.** Infer developmental coordinates from training dynamics rather
than supplying them, then predict held-out `(state × data)` transitions.

**Prerequisite.** C — a state worth discovering coordinates for.

**Frozen criterion.** Inferred coordinates beat supplied ones on held-out
transitions, under the WikiText standard: predictions hashed and timestamped
before the outcomes exist.

**Cost.** Not sized. Depends entirely on what C's state representation looks
like.

**Primary risk.** The discovery step inherits the seed fragility Layer 1
found; coordinates that do not survive re-initialization are not coordinates.

**Implementation.** Not started, deliberately.

**Licenses.** "A model-native developmental ontology exists and predicts
better than a supplied one."

**If it fails.** Supplied ontologies remain the practical option; the
programme continues but loses its automation story.

## Milestone E — state is controllable

**Experiment.** State-aware data selection versus strong interleaving in a
deliberately resource-constrained regime where interleaving is below ceiling.

**Prerequisite.** D — prospective state × data prediction.

**Frozen criterion.** Already frozen in `STATUS.md`: min-across-families and
joint steps/tokens-to-threshold at equal compute. **Interleaving is the
baseline to beat, not a control to beat** — it reached 0.951 min accuracy
against block-sequential's 0.529.

**Cost.** Not sized.

**Primary risk.** Interleaving may simply be near-optimal at this scale,
leaving no headroom for any method.

**Licenses.** The north star: developmental-state-guided pretraining.

**If it fails.** The science stands; the engineering application does not.

---

## Four reusable concepts

Current V2 code should be plumbed toward these as it grows, so later
experiments share them rather than each inventing its own. **Only refactor
where useful** — none of this is worth a rewrite today.

**`StateSnapshot`** — everything known about a model at one moment: the
checkpoint (`dsi.checkpoint`, done), behaviour on frozen eval sets, and
mechanistic / representation / optimization telemetry. Milestone A's `P_int`
and Milestone B's `M` are both *views* of a snapshot, which is why they should
share a type rather than each defining its own extraction.

**`Intervention`** — a training-data action (a stream mixture over a budget) or
a mechanistic action (`ablate_heads`, done; later `do(M+)`). Making data and
mechanism interventions the same kind of object is what lets the ablation arm
and the curriculum arm be analyzed by one code path.

**`TransitionRunner`** — `(StateSnapshot, Intervention) -> trajectory`. Every
experiment on this ladder is an instance: the sprint experiment is one
snapshot and two interventions; V2 G1 is three interventions from one
initialization; state dependence is one intervention from two snapshots.

**`TransitionAnalyzer`** — behavioural and mechanistic change, learning
velocity, state similarity, and the predictive baseline ladder (global mean →
source/target-only → additive → structural/state). The baseline ladder already
exists conceptually in the L0 work and should be one implementation, since
every milestone re-runs it.

Building these four now costs little and prevents the dead end where Milestone
C's analysis cannot reuse Milestone B's telemetry.

## Deferred, with triggers

Full detail in `BACKLOG.md`. Summary:

| deferred | RETURN AFTER |
|---|---|
| `do(M+)` sufficiency | B's necessity passes (H2.3) |
| architecture sweep (depth × width × heads) | the basic mechanism replicates at 12 seeds |
| second mediator | full V2 chain passes |
| state-dependent transfer (full study) | a mediator is identified; cheap scout may run earlier |
| automatic state/ontology discovery | state dependence exists (C) |
| MDA / training-data attribution | a causal mediator exists (B) |
| closed-loop controller | prospective state × data prediction (D) |
| broader natural-corpus work | mechanism survives the controlled bridge (B + A) |
| UI | there are results worth navigating |
