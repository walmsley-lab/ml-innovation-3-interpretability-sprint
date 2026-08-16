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
