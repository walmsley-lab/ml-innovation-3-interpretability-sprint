# Candidate research-trajectory revision (DRAFT — NOT MERGED)

**Status: draft for joint review. `research.md` is unmodified and remains
canonical.** Nothing here revises a gate, a claim, or a result.

**This revision does not rescue today's Stage-5 attempt.** The natural-corpus
Stage-5 test of 2026-08-16 **failed** under its frozen criterion: held-out
relational prediction of `T_aulc`, where the relational model (LOPO 0.1166)
was beaten by the global mean (0.0709), and the provisional source-only
explanation then failed prospectively (0.0711 against the global mean's
0.0699) on 3 pairs frozen before the run. That failure stands as recorded in
`DESIGN_LAYER2.md` 16-17 and is not reinterpreted by anything below. A
reframing that converted a failed test into a passed one by redescription
would be the exact move this project's invalidator/falsifier separation
exists to prevent.

---

## 1. The reframing

The canonical plan treats the object of discovery as a **fixed scalar
transfer graph**: a matrix `T_ij` of pairwise effects, from which a
developmental ontology and then a curriculum are derived.

The candidate revision treats the object of discovery as a **developmental
state-space system**:

    z_{t+Δ} = F(z_t, u_t)
    y_t     = G(z_t)

* `z_t` — the learner's developmental state.
* `u_t` — the intervention applied over the interval: which data, in what
  mixture, at what budget.
* `y_t` — what is observable: losses, learning curves, competence measures.
* `F` — the developmental transition model. **This is the object to
  identify.**
* `G` — the observation model, which is not the identity and may be
  substantially lossy.

Under this framing a pairwise transfer effect is not the fundamental object.
`T_ij` is a **derived, local, scalar projection** of `F`: the response to one
intervention `u = D_i` measured at one operating point `z`, summarized
through one functional of `y`. Three things follow that the fixed-graph
framing cannot express.

**A scalar summary can destroy the signal it summarizes.** Today's AULC is
the integral of a difference between two curves whose immediate and
rate-wise parts oppose each other in 7 of 9 observed pairs, and each part is
3-4x more reproducible than their sum (S/N 8.79 and 7.18 against 2.31). A
projection that cancels is not evidence of absent structure.

**A transfer effect may be state-dependent.** `T_ij` measured from one base
checkpoint need not equal `T_ij` measured from another. The fixed-graph
framing has no place to put that dependence; the state-space framing makes it
`T_ij(z)`.

**Order effects become a property of `F`, not an anomaly.** `A -> B` differing
from `B -> A` is the ordinary behaviour of a non-commuting transition
operator, rather than a phenomenon requiring separate machinery.

## 2. Provisional state representation

The state should distinguish **at least** two components:

* **immediate competence** — what the learner can do at the moment the
  intervention ends;
* **plasticity / acquisition rate** — how quickly it acquires what comes
  next.

These are distinguished because today's data shows them **moving in opposite
directions in most pairs**, so a representation collapsing them is known in
advance to be inadequate.

**They are observable proxies, not proven latent state variables.** The `t=0`
head start and the baseline-corrected rate effect are functionals of `y`, and
promoting them to coordinates of `z` is a hypothesis this project has not
tested. Two specific reasons for caution, both live in today's data:

* endpoint effects are small and weakly reproducible (S/N 1.35), so the arms
  largely converge, and a rate advantage mirroring a starting deficit may be
  **mechanical regression toward a shared asymptote** rather than a plasticity
  difference;
* the measured head start also reflects the control's composition, which in
  the current design varies with the pair.

A serious state representation must be tested for whether it **predicts**
transitions, not merely whether it describes them. Candidate coordinates
beyond the two above — representational geometry, gradient alignment,
effective rank, curvature — are listed as candidates, not commitments.

## 3. Learner-dependent intervention families

The canonical plan proposes families from **corpus semantics** (TF-IDF, LSA,
k-means) and then measures transfer between them.

The candidate revision makes family discovery **learner-dependent** in the
long-term formulation: a developmental family is a set of examples whose
inclusion induces a **coherent transition in learner state**. Candidate
characterizations — gradient structure, representation change, loss-profile
similarity, learning-dynamics clustering — replace or supplement lexical
similarity.

**What today's evidence does and does not support.** It does not establish
that semantic ontology differs from developmental ontology, and that claim is
**not** made here. What is established is narrower and sufficient to motivate
the hypothesis: *the TF-IDF/LSA partition of 20 Newsgroups, together with
centroid-cosine and unigram-KL relational features, failed to predict natural
AULC transfer out of sample.* A confound compounds it — with four families
the symmetric cosine is a one-to-one function of the control composition, so
its failure and its modest exploratory success are both uninterpretable as
evidence about ontology until the control design is fixed.

## 4. Graphs are retained, as a layer rather than the object

Dependency graphs remain valuable and are **not** discarded. They move from
being the fundamental object to being an **interpretability and control
layer** over `F`: a sparse, human-readable projection `T_ij(z)` at an
operating point, used to explain what the transition model has learned and to
constrain curriculum search. A graph that changes with `z` is informative
about development rather than embarrassing to the theory.

## 5. Revised stages

| # | stage | exit condition |
|---|---|---|
| 1 | **Causal developmental apparatus** | order effects are real, measurable, and survive null calibration |
| 2 | **Multi-intervention measurement** | many interventions measured under one protocol, exposure-matched, components preserved |
| 3 | **Predictive synthetic system identification** | a transition model predicts held-out interventions in a substrate with known ground truth |
| 4 | **Automatic intervention-family discovery on arbitrary corpus** | learner-dependent families that are distinguishable, learnable, and support-adequate |
| 5 | **Adaptive developmental system identification** | held-out predictive transition model **and** a model-selected unseen intervention |
| 6 | **Closed-loop active identification** | sequential selection improves the model faster than a fixed design |
| 7 | **Developmental control** | a compiled curriculum beats baselines and reverse on a fresh model |
| 8 | **Scale / state / architecture transfer** | the identified structure survives a change of learner |

Mapping to work already done: Layer 1 sits in stage 1 (calibrated through the
tail, Gate C invalidated the neutral-default endpoint). The synthetic Layer-2
pilot sits in stage 3 at viability level. Today's natural pilot **attempted
stage 5 directly from stage 4's output and failed**, which the revised
ordering explains: stage 4's families were semantic rather than
learner-dependent, and no stage-3-quality transition model existed for them.

## 6. What this revision would cost

It is more expensive than the fixed-graph plan. Identifying `F` needs
interventions from **multiple operating points**, not one base checkpoint per
pair, and the number of measurements grows with the state coverage required.
The honest reading is that the fixed-graph plan is a **special case** worth
retaining wherever it suffices, and the revision should be adopted only where
the projection is demonstrably lossy. Today's component-versus-composite
result is one such demonstration; it is not yet many.

## 7. Open questions for review

* How much of `z` must be observable for `F` to be identifiable at all?
* Does a learner-dependent family definition remain stable across seeds, or
  does it inherit the seed fragility Layer 1 found?
* Is there a cheap sufficient statistic for "same developmental state", or
  does state comparison require full checkpoints?
* Does the revision change any **frozen** object? It must not.
