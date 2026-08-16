# Research Plan: Developmental System Identification for Pretraining

## 1. Research Objective

The central question is:

> **Can the developmental structure that makes one training order preferable to another be experimentally discovered rather than assumed?**

The project treats pretraining as a developmental process rather than only a static mixture-optimization problem.

The long-term loop is:

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

The project proceeds through three scientific levels:

1. **Controlled developmental causality**
   Establish that developmental order can cause persistent differences in learning.

2. **Developmental system identification**
   Infer predictive relationships among candidate corpus components and revise those components when the initial ontology is inadequate.

3. **Developmental control**
   Use the discovered structure to derive a curriculum and test whether it improves fresh-model learning.

The objective is not to recover a unique metaphysical ontology of knowledge. The practical criterion is:

[
\boxed{
\text{Does the inferred developmental representation predict what happens when training history changes?}
}
]

---

## 2. Core Scientific Commitments

The project should preserve several distinctions throughout.

### Observation is not interpretation

Measured transfer effects are primary data.

A graph is an interpretation of those measurements.

A curriculum is a downstream control decision.

These remain separate objects:

[
\boxed{\text{intervention measurements}}
]

[
\boxed{\text{developmental model}}
]

[
\boxed{\text{derived curriculum}}
]

### Semantic structure is not developmental structure

A semantic ontology describes what examples appear to concern.

A developmental ontology describes distinctions that predict different consequences for later learning.

Semantic clustering, metadata, or LLM-generated descriptions may propose candidate units, but they are not accepted as ground truth.

### Transfer and ordering are different estimands

For families (A) and (B), distinguish:

**Transfer effect**

[
\Delta^{\mathrm{transfer}}_{A\rightarrow B}
===========================================

## Y_B(A\rightarrow B)

Y_B(N\rightarrow B),
]

where (N) is an explicit matched control.

**Order effect**

[
\Delta^{\mathrm{order}}_{A,B}
=============================

## Y(A\rightarrow B)

Y(B\rightarrow A).
]

The first asks whether exposure to (A) changes acquisition of (B).

The second asks whether the ordering of (A) and (B) matters.

Neither should be used as shorthand for the other.

---

# 3. Stage 0 — Estimator and Experimental-System Validation

Before training scientific models, validate the measurement machinery itself.

Generate synthetic learning curves with known implanted effect

[
\Delta_{\mathrm{true}}.
]

Pass these through the exact estimators that will later compute:

* area-under-learning-curve effects;
* endpoint effects;
* confidence intervals;
* paired differences;
* censoring metrics.

Verify:

[
\mathbb E[\hat{\Delta}]
\approx
\Delta_{\mathrm{true}}
]

and that nominal confidence intervals achieve approximately their expected coverage.

This catches statistical and pairing bugs before GPU compute is spent.

### Exit criterion

The estimator must recover known synthetic effects within prespecified tolerance and demonstrate calibrated interval coverage.

---

# 4. Stage 1 — Model Capacity Calibration

The experimental model should be the **smallest scientifically adequate learner**, not the largest model affordable.

Let (N) be parameter count.

Select

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
A_G(N)\ge\tau_G,
]

and for a multi-family corpus:

[
\min_j A_j(N)\ge\tau_D.
]

The calibration sweep varies both model capacity and token budget to distinguish true capacity limitation from under-training.

Candidate scales may be geometrically spaced, e.g.

[
1\text{M},2\text{M},4\text{M},8\text{M},16\text{M},32\text{M},
]

subject to actual task difficulty.

### Developmental resolution

A model that instantly solves the task is also undesirable.

Define a learning window

[
R(N)=t_{90}-t_{10},
]

where (t_{10}) and (t_{90}) mark 10% and 90% of observed capability acquisition.

The chosen model must have enough temporal resolution for checkpoint-level developmental analysis.

### Generalization controls

Calibration should measure:

* train performance;
* held-out surface forms;
* held-out compositional structures;
* worst-family competence;
* learning speed;
* cost per run.

### Scale robustness

Once Stage I produces a headline result, replicate it at 2–3 adjacent model scales to guard against selecting a single artificially favorable regime.

---

# 5. Stage 2 — Noise and Power Calibration

This is mandatory before choosing the number of seeds for the primary experiments.

### Identity-null experiments

Run conditions that are scientifically identical but stochastically independent:

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

These runs estimate the actual noise floor of the measurement procedure.

They test:

* pairing quality;
* stochastic optimizer variance;
* RNG handling;
* hardware nondeterminism;
* estimator bias;
* residual control mismatch.

### Estimate paired noise

From the pilot, estimate

[
\sigma_{\mathrm{pair}}.
]

Prespecify the smallest scientifically meaningful effect

[
\delta_{\min}.
]

Then choose paired seed count (n) to achieve a desired power target.

Seed count should therefore be justified statistically rather than chosen arbitrarily.

### Multiplicity

The transfer matrix contains many comparisons.

Raw per-cell winners should not be selected directly.

Use:

* hierarchical partial pooling;
* shrinkage;
* or prespecified multiplicity control such as Benjamini–Hochberg for confirmatory scans.

One metric must be declared primary.

The preferred primary transfer metric is learning-speed improvement, such as AULC.

Endpoint performance and tokens-to-threshold are secondary.

### Exit criterion

Null cells must be centered near zero within calibrated uncertainty, and the required paired seed count must be determined before the confirmatory intervention matrix begins.

---

# 6. Stage 3 — Controlled W/P Developmental Experiment

Use a synthetic task with two independently learnable information sources:

* (W): underlying rule, utility, or world structure;
* (P): predictive preference cue.

Ordinary aligned examples satisfy:

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

and a balanced mixed condition.

### Competence gates

Before conflict behavior is interpreted:

[
A_W\ge\tau_W
]

and

[
A_P\ge\tau_P.
]

A model that follows (P) because it never learned (W) does not constitute path dependence.

### Diagnostic evaluations

#### Clean-rule

Cue removed; (W) remains informative.

#### Cue isolation

(W) is uninformative; (P) determines the correct response.

#### Conflict

[
W(x)\neq P(x).
]

#### Counter-evidence

Introduce later evidence designed to favor one strategy and measure revision.

---

# 7. Recency and Learning-Rate Controls

Simple

[
W\rightarrow P
]

versus

[
P\rightarrow W
]

can be confounded by training position, learning-rate schedule, and recency.

Therefore add a common post-curriculum washout phase:

[
W\rightarrow P\rightarrow M
]

versus

[
P\rightarrow W\rightarrow M,
]

where (M) is identical balanced mixed training.

Final diagnostics occur after this shared experience.

This tests the stronger claim:

> Different developmental histories leave persistent differences even after identical subsequent training.

Where feasible:

* use constant learning rate across the controlled developmental window;
* or counterbalance schedule position;
* conduct at least one early-vs-late timing probe.

### Stage I headline criterion

The primary result is persistent differentiation after competence controls and shared washout.

A negative result is scientifically valid and should be reported.

---

# 8. Stage 4 — Scale Robustness

Replicate the principal W/P comparison at nearby calibrated scales.

For example:

[
N_{\mathrm{small}},
N^*,
N_{\mathrm{large}}.
]

Ask whether:

* direction is preserved;
* effect magnitude changes;
* developmental timing shifts;
* saturation suppresses the phenomenon.

The goal is not universal scaling-law discovery but protection against a single-scale artifact.

---

# 9. Stage 5 — Structured Multi-Family Corpus

Construct a synthetic or semi-synthetic corpus with approximately 6–12 candidate families.

The generator may contain planted latent relationships, but these are **quarantined from the discovery system**.

Ground-truth generative metadata is reserved strictly for evaluation.

The discovery process may initially use:

* unsupervised embeddings;
* surface statistical features;
* dataset metadata that would exist naturally;
* external LLM-generated semantic proposals, clearly marked as priors.

It must never use hidden generator labels to decide merges, splits, or relationships.

---

# 10. Standardized Transfer Protocol

For each ordered pair

[
D_i,D_j,\quad i\neq j,
]

compare:

[
D_i\rightarrow D_j
]

against

[
N_i\rightarrow D_j.
]

Hold constant:

* source-phase tokens;
* target-phase tokens;
* optimizer updates;
* checkpoint pool;
* target examples;
* evaluation protocol;
* relevant RNG streams.

Repeat across the statistically determined paired seed count.

### Primary transfer metric

Let

[
L_j^{(i)}(t)
]

be target loss after exposure to (D_i).

Define

[
T_{ij}
======

\int_0^{m_t}
\left[
L_j^{N}(t)
----------

L_j^{(i)}(t)
\right]dt.
]

Positive values indicate faster acquisition of (D_j).

Negative values indicate interference.

### Null cells

Continue running identity-null cells throughout the matrix to monitor drift in experimental noise.

### Neutral controls

Because (N) carries causal meaning, test robustness to at least two plausible neutral-prefix constructions in a targeted ablation.

A leave-one-out mixture prefix is a useful additional control.

---

# 11. Order-Specific Comparisons

For selected scientifically important pairs, also compare:

[
D_i\rightarrow D_j
]

against

[
D_j\rightarrow D_i.
]

These order comparisons are co-primary where the claim concerns path dependence.

They should not be replaced by neutral-prefix transfer effects.

---

# 12. Partial-Pooled Transfer Model

Avoid constructing (T) purely from independent raw cell means.

A simple hierarchical model can be used:

[
y_{ijs}
\sim
\mathcal N(\mu_{ij},\sigma^2),
]

with

[
\mu_{ij}
========

\mu+\alpha_i+\beta_j+\gamma_{ij}.
]

Here:

* (\alpha_i): source-family tendency;
* (\beta_j): target susceptibility;
* (\gamma_{ij}): pair-specific interaction.

Pair effects receive shrinkage toward the overall structure.

This reduces winner's curse and makes noisy cells less likely to dominate downstream graph construction.

The raw measurements remain available alongside shrunk estimates.

---

# 13. Developmental Transfer Matrix

Maintain separately:

[
T
]

for estimated effects,

[
U
]

for uncertainty,

and

[
N
]

for replication counts.

Do not collapse all three into one graph edge.

### Directionality

Measure concordance between

[
T
]

and

[
T^\top.
]

If transfer is nearly symmetric, the developmental-order interpretation weakens.

Strong patterns such as

[
T_{AB}\gg T_{BA}
]

provide more direct evidence of directional developmental structure.

---

# 14. Required Prediction Baselines

The developmental representation must outperform simple alternatives.

Prespecify at least:

* global mean;
* source mean;
* target mean;
* source + target additive model;
* symmetric estimate

[
(T+T^\top)/2;
]

* family size;
* family frequency;
* semantic embedding similarity;
* embedding-kernel regression.

The central Stage II comparison becomes:

> **Does intervention-derived developmental structure predict held-out transfer better than semantic similarity and simple statistical baselines?**

If not, the ontology-discovery claim should be weakened.

---

# 15. Developmental Phenotypes

Do not represent family (D_i) only by its outgoing transfer row.

Define:

[
\phi_i=
[
T_{i,*},
T_{*,i}
].
]

This captures both:

> What does learning (D_i) change?

and

> What changes the learning of (D_i)?

If a factor model is used:

[
T\approx UV^\top,
]

then a compact phenotype is:

[
\phi_i=[U_i,V_i].
]

---

# 16. Ontology Revision

Candidate operations:

* retain;
* merge;
* split;
* introduce a latent aggregate.

### Merge

Families whose developmental phenotypes are redundant may be merged.

### Split

A family may be split using information available to the discovery system, such as unsupervised embedding structure.

The split is accepted only if it improves held-out intervention prediction enough to justify added complexity.

### Complexity control

Use a fixed criterion, such as:

* held-out (\Delta)RMSE threshold;
* information criterion;
* regularized validation objective.

Avoid repeatedly trying ontology changes until one improves a test set.

---

# 17. Held-Out Intervention Validation

Use several levels of generalization.

### Leave-one-pair-out

Hide individual

[
D_i\rightarrow D_j.
]

### Leave-one-target-out

Hide all

[
*\rightarrow D_j.
]

### Leave-one-source-out

Hide all

[
D_i\rightarrow *.
]

### Held-out-family

Withhold a complete family from ontology fitting.

Metrics include:

* RMSE;
* MAE;
* rank correlation;
* uncertainty calibration.

This is the core system-identification criterion.

---

# 18. Higher-Order Probe

Pairwise transfer assumes substantial additivity.

Before compiling complex curricula, include at least one deliberately designed synergy test:

[
A+C\rightarrow B.
]

Compare against:

[
A\rightarrow B
]

and

[
C\rightarrow B.
]

If pairwise prediction fails substantially, the curriculum compiler must acknowledge interaction structure rather than treating the graph as additive.

---

# 19. Active Experiment Selection

Only activate this stage after held-out prediction is demonstrably useful.

Version 1:

[
e^*
===

\arg\max_e U_e,
]

where (U_e) is predictive uncertainty.

A budget-aware heuristic is:

[
e^*
===

\arg\max_e
\frac{
U_e \cdot I_e
}{
C_e
},
]

where:

* (I_e): estimated scientific importance;
* (C_e): expected monetary or accelerator cost.

Batch selection should include diversity so that one wave does not contain many redundant interventions.

A full Bayesian expected-information-gain system remains a later extension.

---

# 20. Stage 6 — Curriculum Compilation

Use a deliberately transparent compiler first.

Strong positive transfer relationships may define coarse phases such as:

[
D_1+D_2
]

then

[
D_1+D_2+D_3+D_4
]

then

[
D_3+D_4+D_5
]

then full mixture.

The compiler should avoid claiming optimality.

It generates a curriculum predicted by the discovered developmental model.

---

# 21. Stage 7 — Fresh-Model Validation

Train completely fresh models under:

1. discovered curriculum;
2. **reversed discovered curriculum**;
3. uniform mixture;
4. randomized schedule;
5. semantic/manual baseline.

The reversed curriculum is critical.

If both the discovered and reversed schedules outperform uniform equally, the apparent advantage may come from mixture allocation rather than developmental ordering.

### Target separation

Distinguish:

[
\text{discovery evaluations}
]

from

[
\text{held-out intervention evaluations}
]

from

[
\text{curriculum target evaluations}.
]

### Primary control metric

For target score (S) and threshold (\tau):

[
C(\pi)
======

\min
{
t:S(\theta_t^\pi)\ge\tau
}.
]

If threshold is not reached by (T_{\max}), treat the observation as censored.

Report:

* probability of reaching threshold;
* threshold-time distribution;
* AULC;
* final score;
* generalization;
* retention.

---

# 22. Mechanistic Observation

Mechanistic interpretability is supplementary rather than foundational.

At checkpoints collect selected:

* hidden-state activations;
* probe scores;
* representation similarity;
* feature activity;
* attribution measurements;
* targeted activation interventions where justified.

The first question is:

> Does internal state improve prediction of future transfer?

A stronger later question is:

> Does intervention on representation (R) causally alter acquisition of downstream behavior (B)?

Do not infer

[
A\rightarrow R\rightarrow B
]

from probe decodability alone.

---

# 23. Influence-Based Baselines

Compare expensive intervention measurements against cheaper data-influence approximations such as influence-function or datamodel-style proxies where practical.

This tests whether:

[
T_{ij}
]

can be approximated cheaply enough to prioritize which full training interventions deserve actual execution.

Such methods are competitors and possible amortization tools rather than replacements assumed in advance.

---

# 24. Compute and Cost as Scientific Variables

Track for every experiment:

* tokens;
* accelerator seconds;
* dollar cost;
* shared-prefix reuse;
* storage;
* failed/retried runs.

Eventually experiment selection can optimize:

[
\frac{
\text{information gained}
}{
\text{dollar}
}.
]

The scientific program should scale only when each stage justifies the next stage's compute.

---

# 25. Stage Gates

Each stage receives quantitative exit criteria.

Example progression:

### Gate A

Estimator calibration passes.

### Gate B

Smallest adequate model identified.

### Gate C

Null noise calibrated and statistical power established.

### Gate D

W/P produces persistent, replicated differentiation after common washout.

### Gate E

Effect reproduces at adjacent model scale.

### Gate F

Transfer matrix contains directional signal exceeding semantic/statistical baselines.

### Gate G

Developmental model predicts held-out interventions.

### Gate H

Ontology revision improves predictive compression.

### Gate I

Derived curriculum beats baselines and reversed-order control in fresh models.

A failed gate is an interpretable negative result, not a reason to silently alter the experiment.

---

# 26. Required Figures

1. **Discovery-to-control overview**
2. **W/P branching and washout design**
3. **Capacity and learning-window calibration**
4. **Null distribution and power calibration**
5. **W/P learning trajectories**
6. **Diagnostic conflict differentiation**
7. **Transfer matrix with uncertainty**
8. **Directional asymmetry / (T) vs. (T^\top)**
9. **Developmental interaction map**
10. **Held-out prediction versus semantic/statistical baselines**
11. **Fresh curriculum validation including reversed curriculum**

An appendix may contain mechanistic trajectory visualizations.

---

# 27. Interactive Research Artifact

The public artifact should make the developmental process navigable.

One page should link:

[
\boxed{
\text{history}
+
\text{stimulus}
+
\text{behavior}
+
\text{internal state}
}
]

### Developmental map

Click nodes and edges to inspect:

* measured transfer;
* uncertainty;
* controls;
* seed count;
* source/target exposure;
* underlying experimental branches.

### Model interrogation

Support:

* aligned;
* conflict;
* W-only;
* P-only;
* counter-evidence;
* custom synthetic examples.

Allow comparison across:

* curricula;
* checkpoints;
* model sizes.

### Internal-state view

Use restrained views such as:

* layer × signal heatmaps;
* probe values;
* representation similarity;
* selected feature activations.

### Developmental scrubber

Allow users to move through checkpoints and observe when behaviors and representations emerge.

The polished bespoke interface is deferred until the scientific result is established, but a minimal explorer should be developed early enough to validate the artifact contract.

---

# 28. Final Scientific Claim Structure

The project should only escalate its claims as evidence accumulates.

### Claim 1

Training order can produce persistent developmental differences.

### Claim 2

Pairwise intervention effects contain directional structure not reducible to semantic similarity.

### Claim 3

A model fitted to those effects predicts interventions it has not observed.

### Claim 4

Developmental units can be revised according to predictive consequences rather than semantic labels alone.

### Claim 5

A curriculum derived from the discovered system improves fresh-model sample efficiency and outperforms its reversed ordering.

The framework remains valuable even if the project stops at an earlier claim.

The strongest form of the program is:

[
\boxed{
\text{causal demonstration}
\rightarrow
\text{system identification}
\rightarrow
\text{control}
}
]
