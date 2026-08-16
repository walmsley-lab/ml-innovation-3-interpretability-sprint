# Stage-6 sketch: active identification versus random selection

**Isolated from Stage 5.** No code, artifact, or result in the running
experiment depends on anything here, and nothing here may be implemented
until Stage 5 completes. It exists so that a Stage-5 pass is not followed by
a design cycle.

## The question

Stage 5 asks whether a fitted developmental model can predict an unseen
intervention and select one. Stage 6 asks the harder question:

> Does choosing interventions **actively** identify the developmental system
> faster than choosing them at random?

A model that predicts well may still gain nothing from choosing its own next
experiment; that is an empirical question and it has its own null.

## Minimal design

Two arms over the same universe and budget, differing only in how the next
intervention is chosen:

* **active** — `argmax` of the frozen acquisition rule over eligible unrun
  pairs;
* **random** — a uniform draw from the same eligible set, same seed
  discipline.

Both refit after each acquisition and are scored on the **same** held-out
pairs, sequestered before either arm starts. The comparison is the learning
curve of held-out error against number of interventions.

## What makes it a real test

* **The random arm is the null**, and it must be run, not assumed. Active
  selection that merely matches random is a negative result worth reporting.
* **Multiple random draws** are needed; a single random sequence is one
  sample of a high-variance process.
* **The sequestered evaluation set** cannot be touched by either arm.
* **Eligibility must apply to both arms**, or active selection is compared
  against a random arm that is allowed to make choices active selection is
  forbidden.

## Known problems to solve before it is worth running

* **Cost.** Sequential refitting means the wave cannot be one parallel batch;
  the serial depth is the number of acquisitions.
* **Small universes.** With 42 directed pairs, an active arm exhausts the
  informative candidates quickly, and the curves converge for reasons that
  have nothing to do with selection quality.
* **The metric.** Held-out error against acquisition count conflates model
  improvement with candidate difficulty.

## Prerequisite

Stage 6 is not licensed by a Stage-5 launch. It requires the Stage-5
**outcome** — the adaptive intervention run and compared against its frozen
prediction — because an active loop whose single closed step missed badly
does not warrant a sequential study.
