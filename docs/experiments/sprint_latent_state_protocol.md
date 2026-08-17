# Lane B frozen protocol

**Written before any divergence result has been computed.** The generator
(`scripts/sprint_population.py`) and the extractor
(`scripts/sprint_extract_state.py`) deliberately produce behaviour, checkpoints,
state features and outcomes without ever forming a pair, fitting a predictor,
or computing a divergence. Nothing in this file may change once the analysis
script has been run for the first time.

The failure this exists to prevent is documented in the project's own record:
at 20 Newsgroups the development-best model became prospectively worst, so at
n≈10–26 model selection is unreliable. A matching rule or predictor chosen
after seeing outcomes would reproduce that failure with no way to detect it.

## 1. Primary claim

**S2.** Internal state at `t*` predicts the outcome after an identical future
intervention **better than behaviour at `t*` does**, on held-out models.

**S1, secondary.** Among behaviour-matched pairs, internal state carries
information about which history produced each model. Reported only if S2 holds.

## 2. Primary outcome

`y` = `NEUTRAL_CONFLICT.heldout.logodds` — the mean of
`log P(W-answer) − log P(P-answer)` on the neutral conflict condition, held-out
split, measured after the intervention.

Log-odds rather than a choice rate: a rate thresholds the decision and discards
the margin. This is the same estimand Gate C characterized, which is what makes
its variance known in advance rather than discovered during analysis.

## 3. Predictors

Both fitted with **identical** ridge/CV protocol, identical folds, identical
standardization, on the identical held-out split of models.

| name | features |
|---|---|
| `P_beh` | the complete `t*` behavioural vector: every condition × every split × {loss, accuracy, follows_w, logodds} |
| `P_int` | the 137 basis-invariant state features from `StateProbeSpec` |
| `P_int_matched` | `P_int` reduced by PCA to `dim(P_beh)` |
| `P_combined` | `P_beh + P_int` |

`P_beh` gets the *complete* observable picture, not a handful of scalars. A
weak behavioural baseline would manufacture this result the way a
near-saturated additive model did at 20NG.

`P_int` contains **no raw activation coordinates**. Across differently
initialized models those are uninterpretable — coordinate *k* means something
different in each init — so every feature is invariant to a change of basis in
the residual stream: norms, participation ratios, example-Gram spectra, and
attention statistics.

**The headline quantity is `P_combined` versus `P_beh`** — the *incremental*
value of internal state over behaviour, not a horse race.

## 4. Matching rule

Behaviour components standardized across the population. A pair is **matched**
if the maximum absolute standardized difference across all components is
≤ `epsilon`.

`epsilon` is set to the value that admits **10% of randomly drawn same-history
pairs** on a **development split** of models, and is then frozen. It is never
tuned against outcomes, and it is never loosened to manufacture pairs.

Pair types:

| type | members | role |
|---|---|---|
| cross-history matched | `W_first` vs `P_first` | the S1 test |
| **same-history matched** | `W_first` vs `W_first` | **decides the interpretation** |
| unmatched | random | sanity floor |

The same-history control is the crux. If internal state predicts divergence
within a single history just as well as across histories, the finding is
"state predicts future behaviour" — still the sprint claim, still about the
limits of behavioural evidence — and specifically **not** "history is hidden in
the state". Reporting the cross-history result without this control would
overclaim in the direction this project is most tempted by.

## 5. Splits

Models are partitioned by **pair index** into development (40%) and test (60%)
before any analysis. `epsilon` and any hyperparameter are chosen on
development; every reported number comes from test. A pair's two histories
never straddle the split.

## 6. Generalization across interventions

Predictors are fitted on `I_conflict` and evaluated on `I_continue`, and the
reverse. This asks whether state predicts divergence *in general* or only one
specific perturbation — something a single intervention cannot distinguish.

## 7. Stop conditions

* `P_int` at chance on held-out models under **both** interventions → stop. The
  endpoint carries no predictable structure. A clean negative that closes the
  Gate C story.
* Fewer than **20** cross-history matched pairs at the frozen `epsilon` → S1 is
  not testable at this population size. Report S2 only.
* `P_beh ≈ P_int` → behaviour is sufficient. The sprint claim fails, and that
  is a real answer to the sprint's question.

## 8. Covariates and exclusions

* Models failing the competence gate at `t*` are excluded by a rule fixed here:
  either explicit-mode accuracy below 0.60 on the held-out split.
* Retained competence enters as a covariate, so a model that merely lost
  competence does not read as "divergent".

## 9. What a pass licenses

"In a controlled substrate, models matched on current observable behaviour
differ internally in ways that predict their divergence under an identical
future intervention. Behavioural evidence at a point in time is not sufficient
to determine future behaviour."

It does **not** license claims beyond the W/P substrate, that the latent
differences are "preferences" in any richer sense, that history *caused* the
difference (that is S1), or anything about welfare, sentience or identity.
