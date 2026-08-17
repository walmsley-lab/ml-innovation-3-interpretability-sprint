# V2.1 frozen spec — transient-event mechanism model

**Supersedes three elements of the V2 preflight. Everything else in
`v2_bridge_design.md` is unchanged and still binding.**

This spec is written **after** seeing the V2 preflight traces and **before**
any confirmatory seed has been run. That ordering is the whole point: the
preflight motivated the changes, and it is therefore disqualified as evidence
for them. G1 runs on entirely fresh seeds.

---

## 0. The original result stands as a failure

**P3 under the original sustained-emergence criterion: FAIL.** Recorded, not
reinterpreted.

| quantity | value |
|---|---|
| `M` at init | 0.0284 |
| `M` at end of source phase, arm A | 0.0394 |
| `M` at end of source phase, arm A′ | 0.0320 |
| rise on A | 0.0110 (required ≥ 0.05) |
| rise ratio A / A′ | 3.08 (required ≥ 2.0) |

The ratio condition passed and the amplitude condition failed. Under the
frozen criterion — which required both — this is a **FAIL**, and it appears as
such in `RESULTS.md`.

## 1. What the trace showed, as an exploratory observation

`M` across the source phase, sampled every 750 steps:

```
A        0.0284  0.1100  0.0491  0.0346  0.0426  0.0370
A_prime  0.0284  0.0364  0.0361  0.0323  0.0327  0.0325
BG       0.0284  0.0344  0.0294  0.0303  0.0303  0.0287
```

Arm A produces a large early excursion that then decays; A′ and BG stay near
baseline throughout. **This is exploratory.** One seed, one configuration, and
the observation that motivated this spec. It is not evidence that the effect is
real, reproducible, or consequential.

What it does establish is that the **emergence model was mis-specified**. The
frozen `MProbeSpec` defined emergence as "first checkpoint at which `M` crosses
a threshold *and stays above it*". That definition was chosen to be robust to a
single noisy checkpoint. It cannot detect a transient event by construction, so
a mechanism that forms and then reorganizes is invisible to it regardless of how
large the excursion is.

## 2. The transient-event statistic, pre-registered

Let `M(t)` be the mediator score at source-phase telemetry checkpoints
`t = 0, p, 2p, …`, with `p` the probe interval, and let `M_0 = M(0)` be the
value at initialization.

```
amplitude   A_peak  =  max_t M(t)  -  M_0
peak time   t_peak  =  argmax_t M(t)
area        A_area  =  p * sum_t max(0, M(t) - M_0)
```

`A_peak` is primary. `t_peak` and `A_area` are secondary and are recorded for
every unit, but no gate depends on them.

**Criterion for P3 under V2.1**, both required:

1. **Amplitude** — `max_t M(t) ≥ 2 × M_0` on arm A, averaged over confirmatory
   seeds. A doubling of prefix-matching mass over the model's own
   initialization value.
2. **Selectivity** — `A_peak(A) / A_peak(A′) ≥ 2.0`.

### Why these numbers, and the honest caveat

The selectivity ratio of 2.0 is **unchanged from the original spec**, so it is
not a post-hoc choice.

The amplitude condition is stated **relative to the model's own baseline**
rather than as an absolute score, because the baseline is architecture- and
initialization-dependent — an absolute 0.05 means different things at d128 and
d512, which the scale scout makes immediately relevant. A factor of 2 is the
smallest scale-free statement of "the mechanism strengthened materially" and
was chosen for that reason rather than by fitting the observed trajectory.

**The caveat, stated plainly:** any threshold chosen now is chosen by someone
who has seen one trajectory, and the observed excursion (3.9× baseline) would
clear it. That is why this criterion is not self-validating and why **G1 must
run on fresh seeds**. The open question is reproducibility, which the preflight
cannot answer.

Width is **not** used as a gate. A width or duration measure has many defensible
definitions and none of them is mechanistically privileged, so including one
would add researcher degrees of freedom without adding evidence.

### The prospective question

Beyond the gate, the scientific question `M` now poses is:

> Does the **timing and magnitude of the transient event** prospectively predict
> `B` acquisition across seeds?

Tested as a partial correlation between `A_peak` (and separately `t_peak`) and
`B` acquisition speed, **controlling for a general learning-speed covariate**
(background loss trajectory and `C` acquisition). Without that control, "fast
seeds are fast" would pass this automatically — the same guard as Lane B's C6.

## 3. Repair to the negative control `C`

**The defect:** the target phase trained on `BIND` only. `FACT` was evaluated
but never trained, so `C` sat at chance (0.008–0.021 against a 0.0156 floor) by
construction. "C did not move" was therefore vacuous and tested nothing.

**The repair:** the target phase is the mixture `BIND+FACT`, so both the target
capability and the negative control are in the training data and both are
learnable in principle. The specificity claim becomes falsifiable: a generic
effect would lift both, and only a mechanism-specific one lifts `BIND` alone.

**Competence gate on `C`, frozen:** `FACT` accuracy in the `BG` arm at the end
of the target phase must exceed **0.30** for `C` to serve as the specificity
control. Below that, `C` is reported as *not learnable at this budget* and the
specificity claim is explicitly weakened rather than quietly asserted.

## 4. Repair to `B` headroom

**The defect:** the `BG` arm reached 0.861 with a trajectory running
`0.166 → 0.826 → 0.842 → 0.904`. Near-ceiling leaves nothing for an advantage
to occupy — the ceiling scout's lesson, which cost 15 units to learn the first
time.

**The repair:** the target-phase budget is calibrated so the `BG` arm finishes
in **0.40–0.60** BIND accuracy. Calibration runs on **dedicated seeds 900–901**,
which are never used for any confirmatory unit, and the resulting budget is
frozen here before G1 launches.

**Calibrated target budget: `TARGET_STEPS = 2000`.**

From seeds 900–901, `BG` arm, target phase `BIND+FACT`:

| target step | BIND | FACT |
|---|---|---|
| 1500 | 0.497 | 0.366 |
| 1750 | 0.506 | 0.480 |
| **2000** | **0.516** | **0.510** |
| 2250 | 0.558 | 0.570 |
| 4000 | 0.978 | 0.834 |

2000 places `BG` mid-band on `B` (0.516, target 0.40–0.60) and well clear of the
`C` competence gate (0.510 against 0.30). The C repair is confirmed effective:
`FACT` reaches 0.864 by step 6000, so it is genuinely learnable and the
specificity test is no longer vacuous.

Source-phase budget is **unchanged at 4000 steps**; the observed transient
occurred near step 750 and shortening the source phase would risk truncating
the very event under test.

One instability is recorded, not smoothed: the calibration trace dips at steps
2750 (BIND 0.164) and 5500 (0.730). Training is not monotone here. 2000 sits
before the first dip, but seed-level variance in G1 should be read with this in
mind.

## 5. Firewall

* The V2 preflight runs that motivated this spec are **not evidence for it**
  and are not reused in any V2.1 analysis.
* G1 confirmatory seeds start at **100** and are disjoint from every seed used
  in preflight (0), calibration (900–901), and the scale scout (0–2).
* The **scale scout keeps the original V2 setup** and is not modified, stopped,
  or re-tuned in light of the transient-`M` observation. It is exploratory,
  it cannot validate V2.1, and it answers a different question — whether the
  qualitative behaviour survives capacity.
* Source-phase budget, `A`/`A′`/`BG` construction, model config, batch size,
  optimizer, the `M` probe's off-distribution probe set, and all Lane B
  material are **unchanged**.

## 6. Queued on success

If V2.1 reproduces the transient event on fresh seeds, the **dense
transition-window scout** is immediately licensed on independent seeds:
checkpoint tightly around the rise and decay, then give pre-peak, peak and
post-peak checkpoints the identical future `B` continuation. The question is
whether crossing the internal event changes the model's future **before**
ordinary behaviour distinguishes the checkpoints. Recorded in `RESEARCH.md` with
that trigger.
