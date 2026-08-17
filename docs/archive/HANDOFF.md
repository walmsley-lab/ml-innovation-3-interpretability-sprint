# HANDOFF

Entry point for resuming this project from a different machine or session.
Read [STATUS.md](STATUS.md) first for the research objective and the barrier
log; this file covers how to pick up, what is established, what has been
controlled for, and where it can go next.

---

## 0. Where the project actually is (2026-08-17)

**Read this before section 1, which describes a superseded run.**

The static-ordering programme failed and those results stand unrevised. The
project pivoted to developmental state in a language micro-world under ordinary
next-token prediction. Full barrier record in [STATUS.md](STATUS.md), claim-by-
claim status in [CLAIMS.md](CLAIMS.md).

    static ordering failed -> developmental-state hypothesis
      -> history-dependent learnability [CONFIRMED]
      -> state-conditioned data value   [SUPPORTED]
      -> prediction of it               [FAILED]
      -> adaptive training              [not licensed]

**Confirmed:** training history produces a large, selective difference in future
learnability (`t=0` 0.1322 vs both controls at the 0.0156 chance floor); it
survives disjoint content entirely; exploratorily it survives 32x scale.

**Falsified / inconclusive, and kept visible:** the pre-registered
induction-style mechanism failed its gate; causal mediation of the surviving
retrieval marker is **inconclusive** because the ablation had no measurable
efficacy; state-aware data selection **lost to global-best**, so adaptive
scheduling is not licensed.

**The honest one-liner:** data value is conditional on model state, and we
cannot yet read that state well enough to exploit it. The bottleneck is state
inference, not the phenomenon.

### Reproducing the current results

```bash
uv venv && uv pip install -e ".[dev]"
PYTHONPATH=src python scripts/preflight_microworld.py      # substrate checks P1/P2/P6/P7
PYTHONPATH=src python scripts/audit_microworld_shortcuts.py  # A/A' shortcut audit
PYTHONPATH=src python scripts/audit_claims.py             # recompute every headline number
```

Key frozen objects, all hashed before the runs they govern:

| object | sha256 (first 16) |
|---|---|
| V2.1 spec | `f92f5831bece0d91` |
| what-next tournament protocol (unrun) | `f9da9fe23b1b2400` |
| downstream protocols (hidden futures, replay) | `8fc78c4087e2f87b` |
| Lane B latent-state protocol | `afd0bf5174cd8073` |

### Two lessons that cost the most

1. **An intervention that does not move the capability cannot test necessity.**
   Removing the top 4 of 16 heads left zero-shot competence unchanged. Report
   the efficacy check before the interaction.
2. **Machine-check the ledger.** `scripts/audit_claims.py` caught an inflated
   reversal count (incomplete states counted as argmax) and an untested
   A-vs-background comparison, both after they had been reported as results.

---

## 1. What is running right now

**The `r = 0.20` overlap diagnostic**, manifest
`cd73c7444818dcfcabeda2dd80bb0c2c7cec9114788559a9125d2c3fb52f3f10`, 15 units
(3 arms x 5 seeds), launched on both workers and **not yet complete**.

| worker | shards | tmux session |
|---|---|---|
| `<worker-a>` | 0–7 | `ovl` |
| `<worker-b>` | 8–14 | `ovl` |

Both are `c3-standard-22` in `us-west1-a` and verified environment-identical.
Per-unit outputs are idempotent and atomically written, so re-running a shard
is safe and finished units are never recomputed.

```bash
# progress
gcloud compute ssh <worker-a> --zone us-west1-a --ssh-flag="-o ConnectTimeout=15" \
  --command 'ls ~/work/artifacts/layer2_overlap/units | wc -l'

# pull results from both workers and merge locally
./scripts/fetch_overlap.sh

# apply the frozen barrier (refuses to run on an incomplete manifest)
PYTHONPATH=src python scripts/analyze_layer2_scout.py --path artifacts/layer2_overlap
```

**Do not change the manifest, seeds, arms, thresholds or analysis.** Bring the
complete frozen barrier result back before deciding anything.

## 2. Restoring state on a new machine

```bash
git clone https://github.com/walmsley-lab/ml-innovation-3-interpretability-sprint.git
cd ml-innovation-3-interpretability-sprint
uv venv && uv pip install -e ".[dev]"

# artifacts are gitignored; restore them from GCS
gcloud storage rsync -r gs://<ARTIFACT_BUCKET>/dsi-artifacts artifacts

# the WikiText corpus is re-downloadable; the corpus cache in
# artifacts/corpus_v2/cache is what the runners actually use
```

SSH to the workers is intermittently flaky under load — retry with
`--ssh-flag="-o ConnectTimeout=15"`; it has always been transient, never a
dead VM. Verify with `gcloud compute instances list` before concluding
anything is wrong.

**Cost note:** both C3 workers bill ~$1/hr each while running. `<gpu-worker>` is
TERMINATED and its disk must be preserved.

## 3. What is established

**Layer 1 (synthetic W/P).** Calibrated through the common tail: d64/l4,
lr 3e-3, 1,200 steps, `n_cues` 512, overlap floor `r = 0.20` (worst-case
coexistence 0.969). **Gate C failed as a statistical invalidator** — the
frozen neutral-default endpoint has within-history variance too large for
feasible inference (null sd 12.87 in log-odds). Claim 1 was never run.

**Layer 2 (synthetic compositional).** Pairwise transfer is measurable,
S/N 4.37. Structural features beat additive by 59.8% held-out. Recorded as
**viability, not primitive-level structure** — only 2 of 12 directed pairs are
primitive-disjoint.

**Ceiling scout — FAILED, and it is the most important result so far.**
Block-sequential curricula retain only the last family: each family hits
ceiling at the end of its own phase then collapses by 0.69–0.89. The two
orderings differ only in *which* family is last, so their gap is recency, not
order. But the control arm is the finding:

| at identical compute and allocation | mean acc | min acc |
|---|---|---|
| interleaved | **0.9867** | **0.9510** |
| block-sequential | 0.3827 | 0.1062 |

**Presentation structure alone is worth more than any ordering effect we have
measured.** This is why the objective broadened from curriculum ordering to
developmental scheduling, and why **interleaving is the baseline to beat, not
a control to beat**.

**WikiText Stage 5 — gate FAILED, but informatively.** The relational model
was best in development LOPO *and* best prospectively on both components
(30.0% and 37.7% better than the best simpler model) against predictions
frozen and hashed before the confirmatory pool ran. The 20NG inversion —
where the development-best model became prospectively worst — **did not
recur**. It failed only jackknife robustness on `head_start`, where all three
failing drops involve family 4. `rate_only` passed everything.

This gate is **closed**: not to be refit or reinterpreted. Family-4
sensitivity is post-hoc diagnostic only.

## 4. What has been controlled for

Worth reading before proposing anything — several of these were confounds
discovered mid-flight, and re-introducing one would silently invalidate a
result.

* **Pairing.** Both arms share one initialization, asserted per unit on an
  embedding fingerprint.
* **Exposure.** Matched exactly in LM tokens, asserted per unit, never assumed.
* **`t = 0` evaluation** is mandatory. It has repeatedly been load-bearing:
  head start and rate-only carry opposite signs in 7 of 9 pairs and are 3–4x
  more reproducible than their sum.
* **Control composition.** With four families, the complementary control
  `N_ij` was a one-to-one function of the symmetric cosine feature — the
  feature could not be distinguished from the background. Repaired with a
  common `N_j`; the component signal survived the repair almost unchanged.
* **Execution environment.** The complementary matrix was re-run on the VM so
  H3 was a within-environment comparison. Cross-environment check: corr 0.973,
  mean shift 0.0010, rms below the seed-noise floor.
* **Corpus cache.** Verified byte-identical to the uncached path on all seven
  family streams and the control, manifests included.
* **Order vs dose.** In the running overlap diagnostic, own-phase steps are
  compensated so every family receives exactly 600 steps in every arm.
  Uncompensated, position would have determined exposure and order would have
  been confounded with dose.
* **Pool leakage.** WikiText pools are partitioned on **unordered** pairs, so
  a reverse-direction observation cannot leak pair identity.
* **Model selection.** LOPO is a **selection device only**. Success is
  prospective, on a pool frozen before it ran, requiring a 25% improvement
  over the **best simpler model** (not the global mean), jackknife-robust and
  above the seed-noise floor, on both components under one model.
* **Freezing.** Every prediction artifact is hashed and timestamped before the
  outcomes it predicts exist, and re-verified at scoring.

Two methodological lessons paid for the hard way:

1. **LOPO at n≈10–26 can invert prospectively.** At 20NG the best-LOPO model
   became the worst prospectively. Development fit is not evidence.
2. **A near-saturated model looks best in-sample.** Additive with 7 parameters
   on 8 points had in-sample RMSE 0.0039 and was wrong by ~3x out of sample.

## 5. Where this can go next

The running diagnostic has three branches, fixed in advance in
[STATUS.md](STATUS.md): retention not restored → diagnose retention, do not
expand the graph programme; retention restored but arms tie → pairwise
transfer does not compose into useful orderings, and the static ordering
construction should **not** be patched to make it work; retention restored and
best > reverse beyond noise → compositional value exists, but that is still
not practical utility.

In the latter two cases the next major experiment is a **deliberately
resource-constrained regime where strong interleaving is below ceiling** —
because at the current dose interleaving reaches 0.987 and there is no
headroom to demonstrate an efficiency gain over it. Candidates: uniform
interleaving, randomized interleaving, simple heuristic schedules, any
surviving pairwise-derived schedule, and a **crude adaptive scheduler** that
observes competence, learning rate and forgetting and allocates the next
window toward underlearned, high-value or endangered families. The first
question is only whether state-dependent allocation beats strong interleaving
on the frozen efficiency metrics, primary among them min-across-families and
joint steps/tokens-to-threshold.

Longer run: if reliable interactions emerge, test whether **cheap
observables** — gradient alignment, representation similarity, loss
trajectories, learning-speed signatures, probes, activation change — predict
them, so the method needs a small calibration set rather than exhaustive
pairwise training. Any revised ontology must make **frozen prospective
predictions** before being considered better.

**Selection rule for any new run:** if it succeeds, does it materially improve
our ability to predict or control useful learning per token? If not, it is
diagnostic-only or deprioritized.

## 6. Honest open risks

* The whole pairwise-graph programme may not compose into multi-stage
  schedules. The running diagnostic is the first direct test.
* Interleaving may simply be near-optimal at this scale, leaving no headroom
  for any structured method. The constrained regime exists to find out.
* WikiText's relational advantage rests on p=15 at ridge 1e-4 over 26 pairs.
  It generalized once. That is one success, not a robust method.
* The confirmatory pool (10 directed pairs) is small enough that a real effect
  can fail the frozen criterion on variance alone — which is arguably what
  happened.
* Synthetic Layer-2 conclusions rest on 4 families with only 2 of 12 directed
  pairs primitive-disjoint.
