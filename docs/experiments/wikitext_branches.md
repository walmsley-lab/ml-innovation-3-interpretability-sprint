# Predeclared interpretation branches (written before outcomes)

Declared before any WikiText development result was inspected, so that when a
gate lands the reading is a lookup rather than a fresh design cycle. Nothing
here adds a feature, model, seed, corpus, probe or protocol change to the
running experiment.

Branch selection is determined by the frozen gates, not chosen afterwards.

---

## Branch A — weak development signal

**Trigger.** S/N < 2.0 on **both** primary components in the complete
development pool.

**Reading.** Stop before model selection. This is the prespecified
measurability stop, and the diagnosis is the same fork used at 20NG:

* excessive within-pair seed variance → measurement/statistical invalidator,
  licensing no conclusion in either direction;
* small between-pair spread with reproducible cells → a **valid weak- or
  no-relational-structure result at this scale**, reportable as a negative.

**Permitted next action.** Report. Do not fit, do not run confirmatory, do
not re-cut the response. The natural-corpus arm of the project has then
returned two independent negatives (20NG and WikiText), which is itself the
finding.

**What it would NOT license.** Concluding that developmental structure does
not exist in natural corpora — only that it is not measurable by this
apparatus at this dose and scale.

## Branch B — development succeeds, confirmatory fails

**Trigger.** Development S/N >= 2.0 and a ladder rung beats the simpler
models in development LOPO, but the frozen 25% criterion fails on the
confirmatory pool.

**Reading.** **Stop before adaptive selection.** This is the most likely
branch given everything measured so far, and the one most at risk of being
written up wrongly.

It must be reported as: *this predictor, against this criterion, did not
generalize prospectively.* It is **not** "no structure". The continuous
prospective RMSE table against global, source-only, target-only and additive
is the reportable result, and the gate outcome is one line within it.

Two sub-readings, distinguished by the continuous table and **not** by
choosing a better-looking metric afterwards:

* **relational close to but above threshold** — the effect may be real and
  the apparatus underpowered; the honest statement is that a 25% material
  improvement was not demonstrated at n=10 confirmatory pairs;
* **relational at or worse than the simpler models** — main effects explain
  what structure exists, consistent with the 20NG result, and the
  learner-dependent ontology hypothesis strengthens.

**Permitted next action.** Report, and record which sub-reading holds. No
re-selection on a secondary metric, no ladder extension, no adaptive step.

## Branch C — confirmatory passes

**Trigger.** The frozen criterion passes on **both** components under one
model, jackknife-robust and above the noise floor.

**Reading.** A developmental predictor has generalized prospectively to
unseen natural-corpus relationships. That licenses exactly one further step
and no more.

**Permitted next action.** Score the untouched adaptive reserve under the
frozen acquisition rule, reject leverage-ineligible candidates, freeze and
hash the selection, launch it. **Stage 5 is reached at launch.**

**What it would still NOT license.** Any claim about latent state, causal
ontology, or curriculum control; any claim that `head_start` and `rate_only`
are state variables rather than observable proxies; or generalization beyond
this corpus and scale.

---

## Reporting skeleton, common to all branches

Populated as results land, so no branch requires new analysis code:

1. development measurability table — spread, median and max seed sd, S/N, per
   response;
2. flagged cells (within-pair sd > 4x median), excluded from fitting;
3. development ladder — parameters, ridge penalty, LOPO RMSE per rung;
4. frozen confirmatory predictions with hash and timestamp;
5. **continuous** prospective RMSE against every rung, always shown;
6. gate outcome with each condition reported separately;
7. adaptive diagnostics — leverage, cap, Mahalanobis, eligibility;
8. what the result does and does not establish.
