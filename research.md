# Research Plan: Developmental System Identification for Pretraining

## 1. Research Objective

The central question is:

> **Can the developmental structure that makes one training order preferable to another be experimentally discovered rather than assumed?**

Most pretraining pipelines optimize corpus composition, quality, weighting, and total token allocation. Temporal structure is often treated as secondary or as a manually designed curriculum.

This project tests a stronger hypothesis:

[
\boxed{
\text{earlier training changes the conditions under which later learning occurs}
}
]

If so, then:

[
A\rightarrow B
]

and

[
B\rightarrow A
]

need not be equivalent developmental histories even when both contain the same data and total training budget.

The project treats pretraining as a **developmental system-identification problem**:

[
\boxed{
\text{perturb training}
\rightarrow
\text{observe learning}
\rightarrow
\text{infer developmental structure}
\rightarrow
\text{predict unseen interventions}
\rightarrow
\text{control training}
}
]

The research escalates through three scientific levels:

1. **Controlled developmental causality**
2. **Developmental system identification**
3. **Developmental control**

A final portability stage tests whether the same pipeline can operate on a previously unseen natural-language corpus without corpus-specific scientific logic.

---

# 2. Core Scientific Separation

Three objects must remain distinct throughout the work.

## 2.1 Intervention measurements

These are observed experimental quantities.

Examples:

[
T_{ij}^{(s)}
]

paired transfer effects,

learning curves,

conflict log-odds,

tokens-to-threshold.

## 2.2 Developmental model

This is an inferred representation fitted to the measurements.

Examples:

* transfer matrix;
* partial-pooled interaction model;
* low-rank developmental factors;
* revised corpus families;
* sparse developmental map.

## 2.3 Derived curriculum

This is a downstream control policy generated from the fitted developmental model.

Thus:

[
\boxed{
\text{measurements}
\neq
\text{model}
\neq
\text{curriculum}
}
]

The system must never discover a curriculum first and retrofit a graph afterward.

---

# 3. Invalidators vs. Falsifiers

A core methodological distinction is between:

## Invalidators

The experimental apparatus is not trustworthy.

Examples:

* biased estimator;
* broken null calibration;
* competence gate failure;
* excessive divergence;
* RNG leakage;
* improper pairing.

An invalidated experiment licenses no positive or negative scientific conclusion.

## Falsifiers

The apparatus is sound and the scientific claim fails.

Negative outcomes at properly powered, valid stages are reportable scientific results.

This distinction is enforced throughout the stage gates.

---

# 4. Stage 0 — Estimator Validation

Before training any model, validate the statistical machinery using synthetic learning curves with known effects.

Construct curves with planted:

[
\Delta_{\mathrm{true}}.
]

Verify:

[
\mathbb E[\hat{\Delta}]
\approx
\Delta_{\mathrm{true}}
]

and verify nominal confidence-interval coverage.

Test:

* AULC calculation;
* paired differences;
* confidence intervals;
* equivalence tests;
* threshold-crossing logic;
* censoring logic.

### Gate A

Estimator tests must recover known effects within prespecified tolerance and achieve acceptable interval coverage.

Failure is an **invalidator**.

This stage runs on CPU and should occur before any GPU usage.

---

# 5. Stage 1 — Model Capacity Calibration

The experimental learner should be the **smallest adequate model**, not the largest affordable model.

Let (N) denote model parameter count.

Select:

[
N^*=\min_N N
]

subject to competence constraints:

[
A_W(N)\ge\tau_W,
]

[
A_P(N)\ge\tau_P,
]

[
A_G(N)\ge\tau_G.
]

For a multi-family corpus:

[
\min_j A_j(N)\ge\tau_D.
]

## 5.1 Capacity × token sweep

Model size must be varied jointly with training budget.

A small model may appear inadequate only because it is under-trained.

Run a coarse grid over:

[
N\times \text{tokens}.
]

Candidate scales may be geometrically spaced, for example:

[
1M,;2M,;4M,;8M,;16M,;32M.
]

Exact values remain empirical.

## 5.2 Developmental resolution

An overpowered model can also be undesirable if capabilities emerge nearly instantly.

Define:

[
R(N)=t_{90}-t_{10}.
]

A useful model should have:

[
R(N)\ge R_{\min}.
]

This ensures enough temporal resolution for checkpoint-level observation.

### Gate B

Choose the smallest model that passes competence, generalization, and developmental-resolution thresholds.

Architecture and nuisance hyperparameters are then frozen for confirmatory experiments.

---

# 6. Stage 2 — Null Calibration and Power Planning

The number of seeds must be justified empirically.

## 6.1 Identity-null experiments

Run structurally identical paired interventions:

[
N_1\rightarrow B
]

versus

[
N_2\rightarrow B.
]

The expected effect is:

[
\Delta_{\mathrm{null}}=0.
]

These estimate the real noise floor of the paired estimator.

They test:

* stochastic optimizer variance;
* data realization variance;
* RNG correctness;
* hardware nondeterminism;
* estimator bias;
* pairing quality.

## 6.2 Execution null vs. data null

Two null notions may be useful.

### Execution null

As much as possible is held identical to test execution correctness.

### Data null

Two independently sampled but matched neutral corpora test variation in the neutral-generating process.

These should not be conflated.

## 6.3 Power

Estimate:

[
\sigma_{\mathrm{pair}}.
]

Prespecify a minimum scientifically meaningful effect:

[
\delta_{\min}.
]

Choose paired seed count (n) to achieve a target statistical power.

Seed count is therefore a result of the pilot, not a convention.

### Gate C

Identity-null distributions must be centered appropriately and power analysis must define (n) before confirmatory training begins.

Failure is an **invalidator**.

---

# 7. Stage 3 — Controlled W/P Experiment

The first causal experiment uses two independently learnable signals.

* (W): underlying rule, utility relation, or world structure
* (P): predictive preference/cue

Aligned examples satisfy:

[
W(x)=P(x).
]

Primary curricula:

[
W\rightarrow P
]

and

[
P\rightarrow W.
]

Controls include:

[
W\text{-only},
]

[
P\text{-only},
]

and a balanced mixed baseline.

---

# 8. Competence Gates

Conflict behavior is uninterpretable unless both strategies were learnable.

Require:

[
A_W\ge\tau_W
]

and:

[
A_P\ge\tau_P.
]

A model that follows (P) because it failed to learn (W) is not evidence of path dependence.

Competence failure is an **invalidator**, not a negative result.

---

# 8a. Two Layers of Experimental Environment

The W/P task is **Layer 1**: the causal primitive, and the environment in
which blocker diagnostics are run. It is the smallest setting in which
training order can be manipulated at all, and it is where the catastrophic
interference described in `RISKS.md` was found.

It is not sufficient for the program. With two families the transfer matrix
has a single off-diagonal pair, so Claims 2 through 5 are untestable in
principle, not merely in practice: there is no interaction degree of freedom
for `gamma_ij` to occupy, nothing to merge or split, and no allocation to hold
fixed while ordering varies. The additive `alpha_i + beta_j` baseline, the
sharpest competitor to the developmental claim, cannot even be fitted.

**Layer 2** is a six-family synthetic corpus over one shared world, with
overlapping latent primitives, matched acquisition difficulty, a persistent
background distribution in every phase, and curricula expressed as
time-varying mixture weights. The design proposal is in
[`DESIGN_LAYER2.md`](DESIGN_LAYER2.md).

Layer 2 does not replace Layer 1 and does not begin until the Layer-1
diagnostics conclude. A redesign is not a reason to abandon an unfinished
diagnostic.

---

# 8b. Invalidated Task Construction, and the Capability/Preference Separation

The first implementation of the W/P task was **unidentifiable**, and the
error was found only after Gate B had failed against it. The finding is
recorded here rather than quietly corrected, because the sequence matters:
the gate exposed an impossible criterion rather than being relaxed after the
outcome was seen.

## The defect

In that construction the training families ``W`` and ``P`` presented
**identically distributed model-visible inputs**. Digits were drawn from the
same table, the cue class was independently uniform in both, and the model
was never told which family an example came from. Only the target-generating
function changed, from (W(x)) to (P(x)).

Verified empirically: drawing both families from the same RNG key produced
byte-identical visible inputs, differing only in the answer token.

## The consequence

Let (f) be any deterministic predictor of the answer from the visible input.
On inputs where (W(x)\neq P(x)) at most one of the two conditions can be
scored correct, and only where (W(x)=P(x)) can both be. With (K) balanced,
independent answer classes:

[
A_W+A_P
\le
1+\Pr[W=P]
=
1+\frac{1}{K},
]

and therefore

[
\min(A_W,A_P)\le\frac{1}{2}\left(1+\frac{1}{K}\right).
]

At (K=4) this is (\min(A_W,A_P)\le 0.625), against a prespecified
(\tau_{\mathrm{retention}}=0.80). **Simultaneous W/P behavioural competence
was impossible at any model capacity, learning rate, or training duration.**
Every capacity sweep run against it was measuring an unsatisfiable criterion.

This also explains the diagnostic results obtained before the defect was
identified: retention pinned near chance at every scale, and a strongly
negative W-vs-P gradient cosine under answer-only loss, because the two
objectives demand *different outputs on the same inputs*.

The threshold is **not** lowered in response. An impossible criterion is
repaired by making the task well-posed, not by moving the bar to whatever the
broken task could reach.

## The separation the construction was missing

Three things were conflated that must be measured separately.

**Capability.** Can this checkpoint execute (W)? Can it execute (P)? These
are questions about what the model retains, and answering them requires the
request to be part of the input.

**Preference.** When the input does not specify which strategy to apply, and
the two disagree, which does the model produce? This is a behavioural default.

**History.** Which training order produced that capability profile and that
default.

The invalidated construction tried to read capability off a stimulus that
also had to carry preference, which is why it could measure neither.

## The corrected construction

An explicit mode token makes the request part of the input:

* ``USE_W`` — execute the rule; used for W training and W competence;
* ``USE_P`` — execute the cue map; used for P training and P competence;
* ``NEUTRAL`` — no strategy requested.

``NEUTRAL`` with (W=P) is ordinary aligned training and privileges neither
strategy. ``NEUTRAL`` with (W\neq P) is reserved exclusively for the
preference measurement.

The estimands become:

[
A_W=\Pr[\text{correct}\mid \texttt{USE\_W}],
\qquad
A_P=\Pr[\text{correct}\mid \texttt{USE\_P}],
]

both evaluated on the **same checkpoint**, and

[
\mathrm{Pref}=\Pr[\text{answer}=W(x)\mid \texttt{NEUTRAL},\,W\neq P].
]

Under this construction an oracle that reads the mode achieves (A_W=A_P=1),
so the competence gates are jointly satisfiable and the (0.625) ceiling is
gone. Competence and preference are now separately identified: the mode token
is the only explicit task identifier, and the content tokens remain
distribution-matched across the two explicit families.

## Gate B does not see the phenomenon

Gate B is confined to capability coexistence. It trains and evaluates only
the explicit modes:

[
\texttt{W\_EXPLICIT}\rightarrow\texttt{P\_EXPLICIT}
\quad\text{and}\quad
\texttt{P\_EXPLICIT}\rightarrow\texttt{W\_EXPLICIT},
]

asking whether one checkpoint retains both capabilities in both orders. The
neutral conflict condition is not generated, not trained on, and not
evaluated at this gate, so no regime can be selected on the preference or
order effect that the project exists to measure. The aligned neutral tail
enters only after corrected Gate B succeeds.

## Interpretation limit

The neutral conflict measurement is, initially, a **behavioural default**
only. Observing that a model answers with (W) under ``NEUTRAL`` does not
establish that it executed the W computation; it may have learned a third
solution that coincides with (W) on these inputs. Whether neutral computation
resembles the W route, the P route, or neither is a separate mechanistic
question, and is not settled by the behavioural measurement.

---

# 9. Diagnostic Evaluations

## Clean-rule evaluation

Remove the cue and test (W).

## Cue-isolation evaluation

Make (W) uninformative and test (P).

## Conflict evaluation

Construct:

[
W(x)\neq P(x).
]

Measure:

* categorical preference;
* forced-choice log-probabilities;
* log-odds:

[
\ell=
\log p(y_W)-\log p(y_P).
]

## Counter-evidence

Provide later evidence favoring one strategy and measure persistence/revision.

---

# 10. Recency and Learning-Rate Controls

Simple:

[
W\rightarrow P
]

versus:

[
P\rightarrow W
]

can be confounded by recency or LR-schedule position.

Therefore include shared-tail conditions:

[
W\rightarrow P\rightarrow M
]

and:

[
P\rightarrow W\rightarrow M,
]

where (M) is identical balanced mixed training.

Final diagnostics occur after the common tail.

This tests the stronger proposition:

> different developmental histories can produce persistent differences even after identical subsequent experience.

Where practical:

* use constant LR during the developmental window;
* or explicitly counterbalance LR position;
* test at least one early-vs.-late timing intervention.

### Claim 1 falsifier

After competence gates and common washout, if the confidence interval lies entirely inside:

[
(-\delta_{\min},+\delta_{\min}),
]

persistent path dependence is rejected at that scale.

A transient effect that disappears after washout falsifies persistence but not transient path dependence.

---

# 11. Stage 4 — Adjacent-Scale Replication

Replicate the Stage 3 headline result at nearby calibrated model sizes.

For example:

[
N_{\mathrm{small}},N^*,N_{\mathrm{large}}.
]

Measure:

* effect direction;
* effect magnitude;
* developmental timing;
* saturation behavior.

### Gate E

Failure to reproduce does not necessarily falsify path dependence outright.

It bounds the claim to a specific scale regime and must be reported as such.

---

# 12. Transfer Effects and Order Effects

These are distinct estimands.

## Transfer effect

[
\Delta^{\mathrm{transfer}}_{A\rightarrow B}
===========================================

## Y_B(A\rightarrow B)

Y_B(N\rightarrow B).
]

This asks:

> Does prior exposure to (A) alter later acquisition of (B)?

## Order effect

[
\Delta^{\mathrm{order}}_{A,B}
=============================

## Y(A\rightarrow B)

Y(B\rightarrow A).
]

This asks:

> Does reversing the order of the same training sources alter the outcome?

Neither is a substitute for the other.

They should be stored and analyzed separately.

---

# 13. Stage 5 — Generic Corpus Intake

The system should not require manually supplied (D_1,\ldots,D_k).

It must accept a generic text corpus:

[
X={x_1,\ldots,x_N}.
]

The supported first-paper ingestion contract is:

* document identifier;
* text;
* optional ordinary metadata.

The claim is therefore initially:

> arbitrary text corpora satisfying the ingestion contract

rather than arbitrary multimodal data.

---

# 14. Train / Validation / Test Split Before Discovery

The raw corpus must first be split:

[
X=
X_{\mathrm{train}}
\cup
X_{\mathrm{val}}
\cup
X_{\mathrm{test}}.
]

The initial family proposer is fitted only on:

[
X_{\mathrm{train}}.
]

Then it is frozen and used to assign validation and test material.

This prevents corpus-structure leakage from held-out documents.

---

# 15. Provisional Family Proposal

The initial decomposition is:

[
\mathcal P_0(X)
===============

{D_1,\ldots,D_k}.
]

Possible proposers include:

* TF-IDF / LSA clustering;
* frozen text embeddings;
* topic models;
* ordinary dataset metadata;
* semantic classifiers;
* LLM-generated descriptions.

LLM labels or semantic clusters are proposals, not ground truth.

The principle is:

[
\boxed{
\text{the proposer proposes;
training interventions adjudicate}
}
]

---

# 16. Hidden Synthetic Ground Truth

For synthetic or semi-synthetic Corpus A, the generator may contain a planted latent structure.

That information must be quarantined from discovery.

Hidden metadata is used only for evaluation.

Discovery code must not use planted labels to:

* propose merges;
* propose splits;
* fit relationships;
* select experiments;
* derive curricula.

---

# 17. Generic Acquisition Metric

An arbitrary corpus may lack task-specific labels.

A universal family-level acquisition signal is held-out next-token loss:

[
L_j(t).
]

This makes the transfer protocol applicable without hand-authored task labels.

At target-phase start, measure:

[
L_j(0).
]

This allows later separation of immediate transfer from altered learning rate.

---

# 18. Immediate vs. Acquisition-Rate Transfer

For source (D_i) and target (D_j), define the immediate target offset:

[
H_{ij}
======

L_j^{N}(0)-L_j^{(i)}(0).
]

Then separately analyze target-phase improvement.

This prevents AULC from silently conflating:

1. a head start before target training;
2. faster acquisition during target training.

The (t=0) evaluation is mandatory from the first transfer implementation.

---

# 19. Standardized Pairwise Transfer Protocol

For ordered pair:

[
D_i,D_j,\quad i\neq j,
]

and seed-family index (s):

[
\theta_0^{(s)}
\rightarrow
\begin{cases}
D_i\rightarrow D_j\
N\rightarrow D_j
\end{cases}.
]

Within the pair, hold fixed:

* parent checkpoint;
* source token count;
* target token count;
* optimizer update count;
* LR schedule position;
* target example ordering;
* evaluation suite/version;
* relevant RNG streams;
* code version;
* data version.

The intentional difference is the source-phase corpus.

---

# 20. Unit of Analysis

The unit of analysis is the **pair**, not the individual run.

For pair (s):

[
T_{ij}^{(s)}
============

\int_0^{m_t}
\left[
L_j^N(t)
--------

L_j^{(i)}(t)
\right]dt.
]

The (n) paired effects are the observations for cell (ij).

Raw arms are not treated as independent replicates.

---

# 21. Transfer Matrix

Define:

[
T_{ij}
======

\mathbb E_s[T_{ij}^{(s)}].
]

Maintain separate objects:

[
T
]

effect estimates,

[
U
]

uncertainty,

and:

[
N
]

replication counts.

Do not collapse them into graph edges.

---

# 22. Directionality

Decompose:

[
T=S+A,
]

with:

[
S=\frac{T+T^\top}{2},
]

and:

[
A=\frac{T-T^\top}{2}.
]

The antisymmetric component (A) captures directional structure.

Compare its magnitude against identity-null expectations.

### Claim 2a falsifier

If antisymmetric structure is indistinguishable from the null noise floor, strong directional developmental claims are unsupported.

---

# 23. Partial Pooling

Raw cell maxima are subject to winner's curse.

Use a simple hierarchical model:

[
y_{ijs}
\sim
\mathcal N(\mu_{ij},\sigma^2),
]

with:

[
\mu_{ij}
========

\mu+\alpha_i+\beta_j+\gamma_{ij}.
]

Interpretation:

* (\alpha_i): general source usefulness;
* (\beta_j): target susceptibility;
* (\gamma_{ij}): pair-specific developmental interaction.

Shrunk estimates are preferred for downstream selection.

Raw measurements remain available.

---

# 24. Load-Bearing Baseline: Additive Model

The strongest simple competitor is:

[
\mu+\alpha_i+\beta_j.
]

If:

[
\gamma_{ij}
]

adds no held-out predictive value, then the system has discovered source rankings and target difficulty, but not meaningful relational developmental structure.

This comparison should be treated as a headline result.

---

# 25. Other Prespecified Prediction Baselines

Compare against:

* global mean;
* source mean;
* target mean;
* additive source + target;
* symmetric transfer estimate;
* family size;
* family frequency;
* semantic cosine similarity;
* embedding-kernel regression.

### Claim 2b falsifier

If semantic/statistical baselines predict held-out transfer as well as or better than the intervention-derived developmental model, the developmental-ontology claim is weakened or rejected.

---

# 26. Developmental Phenotype

A family must be characterized as both source and target.

Define:

[
\phi_i=
[
T_{i,*},
T_{*,i}
].
]

For a low-rank model:

[
T\approx UV^\top,
]

use:

[
\phi_i=[U_i,V_i].
]

This captures:

* what learning (D_i) changes;
* what changes learning (D_i).

---

# 27. Ontology Revision

Candidate operations:

* retain;
* merge;
* split;
* introduce latent aggregate.

## Merge

Merge families with sufficiently similar developmental phenotypes when doing so does not degrade held-out predictive performance.

## Split

Propose splits from discovery-visible structure only:

* unsupervised embedding clusters;
* statistical heterogeneity;
* semantic proposal methods.

Never use hidden generator labels for discovery.

## Acceptance

Use a frozen validation protocol plus complexity penalty.

Possible criteria:

* prespecified (\Delta)RMSE improvement;
* information criterion;
* regularized held-out objective.

Search attempts should be recorded to prevent repeated test-set optimization.

---

# 28. Held-Out Intervention Prediction

Use:

## Leave-one-pair-out

## Leave-one-source-out

## Leave-one-target-out

## Held-out-family evaluation

Metrics:

* RMSE;
* MAE;
* rank correlation;
* uncertainty calibration.

### Claim 3 falsifier

The developmental model fails if it does not beat prespecified baselines by the predefined margin on held-out interventions.

---

# 29. Higher-Order Probe

Before assuming pairwise additivity is sufficient for curriculum compilation, test at least one interaction:

[
A+C\rightarrow B.
]

Compare with predictions from:

[
A\rightarrow B
]

and:

[
C\rightarrow B.
]

If strongly mispredicted, the pairwise compiler's additivity assumption is falsified.

The compiler must then account for interaction terms or explicitly narrow its scope.

---

# 30. Active Experiment Selection

Only introduce active selection after the developmental predictor works.

Version 1:

[
e^*=
\arg\max_e U_e.
]

Budget-aware version:

[
e^*
===

\arg\max_e
\frac{
U_e I_e
}{
C_e
},
]

where:

* (U_e): predictive uncertainty;
* (I_e): scientific importance;
* (C_e): estimated compute cost.

Batch selection should avoid redundant experiments.

Full expected information gain is deferred.

---

# 31. Curriculum Compilation

The first compiler should remain transparent.

Input:

* shrunk transfer model;
* uncertainty;
* developmental phenotype;
* higher-order warnings.

Output:

[
\pi_{\mathrm{discovered}}.
]

No optimality claim is made.

The compiler produces a schedule predicted to be favorable under the identified developmental system.

---

# 32. Stage 7 — Fresh-Model Validation

Train entirely fresh models under:

1. discovered curriculum;
2. **reversed discovered curriculum**;
3. uniform mixture;
4. randomized schedule;
5. semantic/manual baseline.

No discovery checkpoint is reused.

## Primary control metric

Let:

[
C(\pi)
======

\min
{
t:S(\theta_t^\pi)\ge\tau
}.
]

If threshold is not reached by (T_{\max}), the run is censored.

Report:

* probability of reaching (\tau);
* threshold-time distribution;
* AULC;
* final score;
* retention;
* generalization.

### Claim 5 falsifiers

Claim 5 fails if:

* discovered does not beat uniform/random;
* discovered and reversed perform equivalently.

If both discovered and reversed beat uniform, the advantage may come from mixture allocation rather than order.

---

# 33. Validation Separation

Maintain three distinct evaluation layers:

[
\boxed{
\text{discovery evaluations}
}
]

[
\boxed{
\text{held-out intervention evaluations}
}
]

[
\boxed{
\text{curriculum target evaluations}
}
]

No target suite used to construct (T) should silently become the final curriculum benchmark.

---

# 34. Censoring Analysis

Tokens-to-threshold requires explicit treatment of non-reaching runs.

Do not report only a mean threshold time.

Report jointly:

* probability of reaching threshold;
* threshold time among successful runs;
* optionally survival-style curves.

Threshold (\tau) should be prespecified and chosen so baseline runs are neither trivially successful nor almost always censored.

---

# 35. Mechanistic Measurements

Mechanistic interpretability is supplementary.

Potential measurements include:

* hidden-state activations;
* linear probe scores;
* representation similarity;
* selected sparse features;
* attribution;
* activation patching;
* targeted ablations.

Initial mechanistic question:

> Does internal representation state improve prediction of future transfer?

Do not infer:

[
A\rightarrow R\rightarrow B
]

from probe decodability alone.

---

# 36. Influence-Based Baselines

Where practical, compare full training interventions against cheaper influence approximations.

These may serve as:

* competitor methods;
* prioritization heuristics;
* amortization tools for (T).

They do not replace the intervention protocol unless validated.

---

# 37. Corpus A — Identifiable Benchmark

The first multi-family corpus should contain hidden planted developmental structure.

Purposes:

* validate system identification;
* validate ontology revision;
* validate directional recovery;
* measure error against known latent structure.

The discovery system must not see ground truth.

---

# 38. Corpus B — Blind Natural-Corpus Demonstration

The paper should include a second, previously unseen heterogeneous natural-language corpus.

The same frozen pipeline should perform:

[
X_B
\rightarrow
\mathcal P_0
\rightarrow
T
\rightarrow
M_{\mathrm{dev}}
\rightarrow
\mathcal P_1
\rightarrow
\pi.
]

No corpus-specific scientific logic should be introduced.

Questions:

1. Does the system autonomously propose usable families?
2. Does developmental structure predict unseen transfer?
3. Does it beat semantic and additive baselines?
4. Is meaningful directionality present?
5. Does ontology revision improve prediction?
6. Does the derived curriculum improve fresh training?
7. Does reversing it remove or weaken the advantage?

---

# 39. Gate J — Portability

The strongest generic-corpus claim requires success on Corpus B.

A portability claim is supported only if the frozen pipeline works without corpus-specific developmental coding.

At minimum:

[
\text{developmental predictor}

>

\text{semantic/additive baselines}
]

on held-out intervention prediction by a prespecified margin.

Stronger support also requires:

[
C(\pi_{\mathrm{discovered}})
<
C(\pi_{\mathrm{uniform}})
]

with the reverse control behaving consistently with the ordering hypothesis.

If Corpus A succeeds and Corpus B fails, portability is not established.

That negative result remains scientifically useful.

---

# 40. Compute and Cost

Every run records:

* tokens processed;
* accelerator seconds;
* dollar cost;
* checkpoint reuse;
* storage;
* retry/failure status.

Eventually active experiment selection may optimize:

[
\frac{
\text{expected information gain}
}{
\text{dollar}
}.
]

Each research stage must justify the next stage's compute.

---

# 41. Scientific Gates

## Gate A

Estimator calibration.

## Gate B

Minimal adequate model selected.

## Gate C

Null calibration and powered seed count.

## Gate D

Persistent W/P differentiation after washout.

## Gate E

Adjacent-scale replication.

## Gate F

Directional transfer signal exceeds null and simple baselines.

## Gate G

Held-out intervention prediction succeeds.

## Gate H

Ontology revision improves predictive compression.

## Gate I

Derived curriculum beats baselines and reverse.

## Gate J

Pipeline transfers unchanged to unseen natural corpus.

Negative outcomes at any gate are acceptable results.

---

# 42. Interactive Research Artifact

The project should include one single-page scientific interface linking:

[
\boxed{
\text{corpus}
+
\text{developmental history}
+
\text{stimulus}
+
\text{behavior}
+
\text{internal state}
}
]

## Corpus view

Show:

* corpus size;
* provisional families;
* representative documents;
* semantic descriptions;
* family evolution.

## Developmental map

Nodes represent developmental families.

Edges show measured transfer with:

* effect estimate;
* uncertainty;
* replication count;
* source/target exposure;
* actual paired intervention.

## Model interrogation

Allow:

* aligned;
* conflict;
* W-only;
* P-only;
* counter-evidence;
* custom input.

Compare:

* curricula;
* checkpoints;
* model sizes.

## Internal state

Use restrained scientific views:

* layer × signal heatmap;
* probe responses;
* representation similarity;
* selected activation summaries.

## Timeline

A checkpoint scrubber lets users inspect when behaviors and representations emerge.

The polished UI is built only after the scientific result is established.

A minimal explorer may be created earlier to validate artifact contracts.

---

# 43. Lessons From the Predecessor Project

The abandoned predecessor reinforces several design rules.

## Preserve

* staged scientific gates;
* provenance-first results;
* competence before interpretation;
* cheap tests before expensive training.

## Avoid

* runtime artifacts coupled to Git paths;
* proliferating shell/orchestration scripts;
* premature implementation of future conceptual abstractions;
* visually ambitious interfaces before measurements stabilize;
* normalized metrics whose denominator behavior has not been audited;
* theorizing from anomalous individual seeds.

These lessons support greater simplicity, not greater architectural complexity.

---

# 44. Final Claim Hierarchy

### Claim 1

Training order can produce persistent developmental differences.

### Claim 2

Controlled interventions reveal directional transfer beyond noise.

### Claim 3

That structure is not reducible to semantic similarity or source/target difficulty.

### Claim 4

The fitted developmental model predicts unseen interventions.

### Claim 5

Corpus units can be revised based on developmental consequence.

### Claim 6

A curriculum derived from that model improves fresh-model learning and outperforms its reverse.

### Claim 7

The complete pipeline can ingest and organize a previously unseen natural text corpus without corpus-specific scientific logic.

The strongest contribution is therefore:

[
\boxed{
\text{causal demonstration}
\rightarrow
\text{system identification}
\rightarrow
\text{ontology revision}
\rightarrow
\text{control}
\rightarrow
\text{portability}
}
]

The central empirical criterion remains:

[
\boxed{
\text{Can a model of development predict what happens when training history is changed?}
}
]
