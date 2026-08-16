# STATUS

Updated at each frozen barrier. Completed stage, hashes, gate result,
interpretation, risks, next licensed action. Nothing else.

---

## Barrier: WikiText Stage-5 confirmatory gate — **FAIL** (2026-08-16)

**Completed stage.** Natural-corpus Stage 5 on WikiText-103, 7 usable
families, common-control design. 108 units: 78 development (26 directed
pairs x 3 seeds), 30 confirmatory (10 pairs x 3 seeds), 21 shared controls.
All valid; control invariant PASS on every target x seed group.

**Frozen artifacts.**

| object | hash |
|---|---|
| pair pools (13/5/3 unordered) | `497b9c3fc66e8adbca96ac2eef41e9e2ada14ffcdc0d78bb4cbbf589c42b3c27` |
| confirmatory predictions | `2ff3cf08f744dc9a3c8a98984e026e50e0148186fb0bec2368b1f985cc0b71a5` |
| Layer-2 scout manifest | `1deb0ae83655d08dbf930c3ddd9109c6441bedb03941bde349a6a9796ec5ebac` |

**Gate result: FAIL.** Both primary components clear the 25% material
threshold against the best simpler model and the seed-noise floor.
`rate_only` is jackknife-robust across all ten leave-one-pair-out subsets
(29.3%-60.6%). `head_start` is not: dropping `0->4`, `4->0` or `4->3` gives
21.8%, 18.8%, 22.8%. The criterion required every condition on both
components under one model.

**Scientific interpretation.** The relational model was best in development
LOPO **and best prospectively on both components**, beating the best simpler
model by 30.0% (head start) and 37.7% (rate-only). The 20NG inversion — where
the development-best model became prospectively worst — **did not recur**.
The formal gate nonetheless fails because one component's margin leans on
family 4. The honest statement is that a jackknife-robust 25% improvement was
not demonstrated at n=10 confirmatory pairs, not that structure is absent.

This gate is **closed and will not be refit or reinterpreted**. Family-4
sensitivity is a post-hoc diagnostic only, and any ontology-revision work
stays separate from confirmatory claims.

**Current risks.**
- Confirmatory pool underpowered at 10 directed pairs; a real effect could
  fail this criterion by variance alone.
- `head_start` margin concentrated in one family.
- Relational model is p=15 at ridge 1e-4 on 26 development pairs; it
  generalized here, but the parameterization remains fragile.
- Adaptive pool (3 unordered / 6 directed) is **untouched and must stay so**
  unless a gate licenses it.

**Next licensed action.** The Layer-2 ceiling scout, already frozen and now
running. Stage 5 was **not** reached; no adaptive intervention was selected
or launched.

---

## In flight

**Layer-2 ceiling scout**, manifest `1deb0ae83655d08d`, 15 units (3 arms x 5
seeds), distributed across both C3 workers. Frozen stop rule: if
`predicted_best` does not beat `exact_reverse` by more than between-seed
noise, stop the expansion and diagnose. No curriculum, hyperparameter,
threshold or seed may be adapted on partial results.

## Infrastructure

`dsi-cpu-bench` and `dsi-cpu-w2`, both c3-standard-22, us-west1-a, verified
environment-identical (image `v20260807`, jax 0.11.0, matching vocab, pools
and corpus-cache hashes). Throughput ~3.06 trajectories/min per worker, flat
in concurrency. `pdp-gpu` TERMINATED and preserved.
