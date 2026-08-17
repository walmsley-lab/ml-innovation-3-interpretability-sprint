# Sprint experiment: behaviourally matched, developmentally different

**Status: design proposal. Nothing implemented, nothing frozen.** Runs on the
existing Layer-1 W/P substrate at the frozen regime. Revises no result.

---

## 1. The question, and why Gate C makes it askable

**Does current behaviour underdetermine future behaviour, and does internal
state carry the missing information?**

Gate C is usually filed as a failure. For this experiment it is the enabling
result. Under identity-null pairing, the W-first history gave a null sd of
**12.87 in log-odds**, and two runs differing only in an independent source
draw produced neutral W-choice rates of **0.000 and 0.996** (`RISKS.md` §2c).
That killed "history determines the preference endpoint" — Claim 1 was never
run.

Read forward instead of backward, it establishes the precondition this
experiment needs:

> There is a large amount of future behavioural divergence in this substrate,
> and **training history does not explain it.** Something else does, or nothing
> does.

So the sprint question is not "can we recover history from the weights". It is
the sharper and more sprint-relevant one: **behavioural evidence at time `t`
does not determine behaviour at `t+Δ`; does internal state?**

This matters for the framing. A version of this experiment built on "history
remains detectable when behaviour is matched" is at direct risk from Gate C,
because within-history variance dwarfs between-history variance. The version
built on "state predicts future divergence" is *motivated* by Gate C.

## 2. Two claims, ranked

**S2 — primary. Prospective divergence.** Internal state at `t*` predicts
behaviour after an identical future intervention **better than behaviour at
`t*` does**, on held-out models.

**S1 — secondary. History detection.** Among pairs matched on behaviour at
`t*`, internal state carries information about which history produced each
model.

S2 is primary because it is the claim the sprint is about — the limits of
behavioural evidence — and because it does not require history to be the
cause. S1 is the more headline-friendly claim and the more fragile one; it is
reported as secondary and only if S2 holds.

## 3. Design

### Population

The frozen Layer-1 regime (`d64/l4`, lr 3e-3, 1,200 steps, `n_cues` 512,
overlap floor `r = 0.20`), unchanged.

**128 paired initializations × 2 histories = 256 models** trained to the end
of the common tail. `W→P` and `P→W` share an initialization per pair, asserted
on the embedding fingerprint exactly as the existing paired-RNG contract
requires (`HANDOFF.md` §4). This is load-bearing: shared inits decorrelate
initialization from history, so an S1 probe cannot succeed by reading the init
seed instead of the history.

Weights are saved at `t*` — the matching checkpoint. **This requires
checkpoint persistence, which does not exist in the repo today.**

### The future intervention

Each `t*` checkpoint is continued under **two** frozen interventions, applied
identically to every model:

- `I_conflict` — the conflict set plus counter-evidence, at fixed dose;
- `I_continue` — neutral continuation training at matched token budget.

Fitting on one and evaluating on the other is a generalization test that a
single intervention cannot provide: it asks whether the state features predict
*divergence in general* or only one specific perturbation.

**Outcome `y`:** the frozen preference endpoint — neutral-default W-choice rate
in log-odds, the same estimand Gate C characterized, plus retained competence
as a covariate.

### Predictors, and a baseline strong enough to be fair

The additive-baseline lesson applies directly: a strawman behavioural baseline
would manufacture this result.

**Behaviour-only (`P_beh`)** gets the *full* observable picture at `t*`, not
three scalars: per-condition held-out accuracies, clean-rule and cue-isolation
results, current loss, logit margins and entropy, and the complete response
distribution over the frozen preference eval set.

**Internal-state (`P_int`)** gets residual-stream activations per layer at the
answer position and mean-pooled, attention-pattern statistics, and
representation geometry (effective rank, CKA to a reference), all extracted on
one **frozen probe input set identical across every model**.

Three fairness controls, frozen in advance:

1. identical ridge/CV protocol and identical held-out split for both;
2. a **dimension-matched** variant of `P_int` (PCA to `dim(P_beh)`) reported
   alongside the full one — `P_int` has orders of magnitude more features and
   would otherwise win on capacity alone;
3. a **combined** predictor `P_beh + P_int`, so the reported quantity is the
   *incremental* value of internal state over behaviour, not a horse race.

### Matched pairs, and the control that makes them mean something

Pair types, all constructed under one frozen matching rule:

| type | members | role |
|---|---|---|
| **cross-history matched** | `W→P` vs `P→W`, matched on behaviour at `t*` | the S1 test |
| **same-history matched** | `W→P` vs `W→P`, matched on behaviour | **the control that decides the interpretation** |
| unmatched | random pairs | sanity floor |

The same-history control is the crux. If internal state predicts future
divergence within a *single* history just as well as across histories, then
the finding is **"state predicts future behaviour"** — which is the S2 claim,
is genuinely sprint-relevant, and is *not* "history is hidden in the state".
Reporting the cross-history result without this control would overclaim in
exactly the direction the project is most tempted to.

**Matching rule:** all behavioural components standardized, pair admitted if
the maximum absolute standardized difference ≤ `epsilon`. `epsilon` is frozen
from a **development split of models** and never tuned against outcomes.

## 4. Confounds

**M1. Matching is conditioning.** Selecting pairs on behaviour conditions on a
variable downstream of both history and state. *Guard:* matching rule and
`epsilon` frozen on a development split; a disjoint test split of models is
never seen during rule selection; the rule is hashed before test-split pairs
are formed.

**M2. Probe capacity.** `P_int` can overfit its way to a win. *Guard:* the
dimension-matched variant and the incremental-over-behaviour framing above.

**M3. Init leakage into S1.** *Guard:* shared initialization across histories
within a pair, asserted on the embedding fingerprint.

**M4. The endpoint may be pure noise.** Gate C's variance is enormous; it is
possible that *nothing* predicts `y`. *Guard:* this is the stop condition, not
a confound to be worked around — see §5.

**M5. Multiplicity.** Many probe variants, layers, poolings. *Guard:* one
primary feature set and one primary predictor frozen; all else secondary and
labelled exploratory.

**M6. Competence drift.** A model that simply lost competence will look
"divergent". *Guard:* retained competence enters as a covariate and models
failing the competence gate are excluded by a frozen rule, before outcomes.

## 5. Stop conditions

- **`P_int` at chance on held-out models under both interventions** → stop.
  The endpoint carries no predictable structure. This is a clean negative that
  *closes* the Gate C story rather than leaving it ambiguous, and it is worth
  reporting.
- **Fewer than a pre-registered minimum of cross-history matched pairs at the
  frozen `epsilon`** → S1 is not testable at this population size. Report S2
  only; do **not** loosen `epsilon` to manufacture pairs.
- **`P_beh` ≈ `P_int`** → behaviour is sufficient. The sprint claim fails and
  that is a real answer to the sprint's question.

## 6. Cost

| stage | units | where | wall-clock |
|---|---|---|---|
| SP-0 preflight | ~8 models | local / C3 | minutes |
| SP-a population to `t*` | 256 | 2 × C3 already running | ~25 min |
| SP-b two interventions | 512 continuations | 2 × C3 | ~35 min |

≈ **1 hour of compute on machines that are already billing.** The cost of this
experiment is entirely engineering: checkpoint persistence and activation
extraction. Neither exists yet, and both are shared with V2.

## 7. What a pass would and would not license

**Would:** "In a controlled substrate, models matched on current observable
behaviour differ internally in ways that predict their divergence under an
identical future intervention. Behavioural evidence at a point in time is not
sufficient to determine future behaviour."

**Would not:** that this generalizes beyond the W/P substrate; that the latent
differences are "preferences" in any richer sense; that history *caused* the
difference (that is S1, secondary, and controlled by the same-history pairs);
or anything about model welfare, sentience, or identity. The claim is about
**the limits of behavioural evidence**, which is the sprint's stated
methodological problem, and it should be stated at that width and no wider.
