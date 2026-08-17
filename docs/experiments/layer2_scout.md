# Layer-2 ceiling scout (manifest frozen, not yet run)

`sha256 1deb0ae83655d08dbf930c3ddd9109c6441bedb03941bde349a6a9796ec5ebac`
— frozen before any scout unit ran, and **not launched** until the WikiText
confirmatory scoring barrier has passed.

## The ordering is a viability-derived ceiling candidate

The `predicted_best` ordering is derived from the synthetic Layer-2 transfer
pilot. That pilot is recorded as **viability, not primitive-level structural
estimate**, because only **2 of its 12 directed pairs are primitive-disjoint**,
so its structural features rest on almost no contrast.

The ordering is therefore a **ceiling candidate**: the best curriculum this
evidence can nominate, used to ask whether *any* order effect is reachable at
this scale. It is not an estimate of the true developmental ordering, and a
scout result — positive or negative — says nothing about primitive-level
structure.

## Design

| arm | presentation | predicted transition sum |
|---|---|---|
| `predicted_best` | F6 → F5 → F4 → F1 | −0.4487 |
| `exact_reverse` | F1 → F4 → F5 → F6 | −1.4397 |
| `balanced_shuffled` | all four interleaved | — |

Predicted best-minus-reverse gap: **+0.9910**.

15 units, 3 arms × 5 seeds (5000–5004), 600 steps per family, 2,400 steps
total per unit. **Aggregate family allocation and total training budget are
identical across arms** — asserted per unit, since an allocation difference
would confound order with dose. The arms differ only in presentation order.

Evaluation is held-out accuracy and loss on **every** family, plus a per
family learning curve, so a curriculum that wins on one family while
destroying another is visible rather than averaged away.

## Stop rule, frozen

If `predicted_best` does not beat `exact_reverse` by more than the
between-seed noise, **stop the expansion and diagnose**. No curriculum,
hyperparameter, threshold or seed may be adapted on partial scout results.

If the scout does show a substantial effect, the permitted next step is to
generate and freeze the full pairwise source × target × seed transfer
manifest and shard it — not to tune the curriculum.

## Execution architecture

Per-unit outputs are idempotent and written atomically, so any idle worker
can steal any shard with no coordination and no risk of torn or duplicated
results. Scientific decisions happen only at synchronization barriers after a
complete manifest, never on partial results.

---

# Scout result: FAIL — recency dominates, and there is no headroom

Manifest `1deb0ae83655d08d`, 15/15 units, allocation identical across arms
and families (asserted per unit and re-checked at aggregation).

## The frozen stop rule

| metric | best | reverse | balanced | diff | pooled seed sd | effect/noise |
|---|---|---|---|---|---|---|
| `final_mean_acc` | 0.3827 | 0.3731 | **0.9867** | +0.0096 | 0.0187 | **+0.51** |
| `final_min_acc` | 0.1062 | 0.1025 | **0.9510** | +0.0037 | 0.0144 | +0.26 |
| `final_mean_loss` | 5.5006 | 5.2522 | **0.0393** | +0.2483 | 0.6834 | −0.36 |

`predicted_best` does not beat `exact_reverse` by more than between-seed
noise. **Do not launch the full pairwise matrix.**

`best beats balanced`: **False**, and not narrowly — 0.383 against 0.987.

## Why: sequential block curricula retain only the last family

Per-family final accuracy:

| arm | F1 | F4 | F5 | F6 |
|---|---|---|---|---|
| `predicted_best` (F6→F5→F4→**F1**) | **1.000** | 0.125 | 0.109 | 0.297 |
| `exact_reverse` (F1→F4→F5→**F6**) | 0.307 | 0.124 | 0.103 | **0.960** |
| `balanced_shuffled` | 1.000 | 0.951 | 0.998 | 0.997 |

Every family is learned to ceiling **at the end of its own phase** and then
collapses:

    predicted_best  F6 1.000 -> 0.297 (-0.703)   F5 1.000 -> 0.109 (-0.891)
                    F4 0.956 -> 0.125 (-0.830)   F1 1.000 -> 1.000 (+0.000)
    exact_reverse   F1 1.000 -> 0.307 (-0.693)   F4 0.947 -> 0.124 (-0.824)
                    F5 0.950 -> 0.103 (-0.848)   F6 0.960 -> 0.960 (+0.000)

**Only the final family survives.** The two sequential arms differ in outcome
only in which family happens to be last, and their 0.0096 gap is the
difference between retaining F1 and retaining F6, plus noise. There is no
order signal because there is nothing left to carry one.

The balanced arm reaching 0.951–0.998 on **all four** families settles that
this is not a capability limit: the task is fully learnable at this dose and
model size. The failure is specific to block-sequential presentation.

## Most likely failure modes, in order

1. **Catastrophic forgetting / recency dominance.** Already established in
   Layer 1 (`docs/risks.md` 2) and now reproduced in Layer 2. Layer 1's validated
   remedy was overlapping curricula at floor `r = 0.20`, which restored
   worst-case coexistence to 0.969. The scout used pure block-sequential
   phases with no overlap and reproduced the original pathology.
2. **A composition/measurement mismatch.** The pairwise `T_ij` matrix
   measured *immediate acquisition of the next family*, over two phases,
   evaluated on the target. The scout composed those pairwise numbers into a
   four-phase ordering and evaluated *retention of all families at the end*.
   Nothing in the pilot ever tested transitivity, or that pairwise
   acquisition effects compose across three transitions.
3. **No common tail.** Layer 1 required a common tail for post-curriculum
   competence; the scout has none, so the last phase is unopposed.

Failure mode 1 is sufficient on its own to explain the result. Failure mode 2
would still apply even if forgetting were controlled, and is the more
interesting problem.

## Smallest next diagnostic experiment — proposed, NOT run

**Re-run these exact three arms with Layer-1's validated overlap floor
`r = 0.20`**, changing one thing and nothing else: each phase mixes in 20%
of previously-seen families, with aggregate family allocation and total
budget held identical across arms as before. Same families, same seeds, same
15-unit structure, same evaluation.

It is the smallest experiment that separates the two live explanations:

* if order effects appear once forgetting is controlled, failure mode 1 was
  the whole story and the pairwise programme is viable under overlap;
* if the arms remain indistinguishable while all families are retained, the
  composition mismatch (mode 2) is real, and pairwise transfer does not
  compose into multi-phase curricula — which would be a substantive negative
  about the entire pairwise-graph approach.

Either outcome is informative, and it costs 15 units.

**Not run.** No curriculum, threshold, seed or family set may be adapted on
this result beyond that single controlled change.
