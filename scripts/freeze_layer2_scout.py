"""Freeze the Layer-2 ceiling-scout manifest before anything runs.

The question is deliberately narrow and cheap:

> Does the ordering our fitted pairwise model predicts is best actually beat
> its own exact reverse, and beat a balanced mixture?

If the best-predicted curriculum cannot beat its reverse, no larger
curriculum search is worth running, whatever the pairwise matrix says. That
is what makes this a **ceiling scout** rather than a curriculum experiment.

Three arms, identical in aggregate family allocation and total training
budget, differing only in the **order** in which the families are presented:

* ``predicted_best`` — the ordering maximizing summed predicted transfer over
  consecutive transitions, under the frozen Layer-2 pilot matrix;
* ``exact_reverse`` — that ordering reversed, so every transition is inverted
  while allocation is untouched;
* ``balanced_shuffled`` — all families interleaved uniformly throughout, the
  no-curriculum control.

**What this cannot establish.** The ordering is derived from the synthetic
Layer-2 transfer pilot, which is recorded as viability rather than
primitive-level structure: only 2 of its 12 directed pairs are
primitive-disjoint. A positive scout result licenses expansion, not a
structural claim.

Seeds and arms are frozen here. Nothing about the curricula, budgets,
thresholds or seeds may be adapted on partial results.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/layer2_scout")
SEEDS = (5000, 5001, 5002, 5003, 5004)
STEPS_PER_PHASE = 600          # matches the frozen Layer-2 pilot phase length
PILOT = Path("artifacts/layer2_transfer/units")


def pilot_matrix() -> dict:
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(PILOT / "*.json"))]
    if not rows:
        raise SystemExit("no Layer-2 pilot units found; the scout needs the frozen matrix")
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["source"], r["target"]), []).append(r["T_aulc"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    T = pilot_matrix()
    families = sorted({k[0] for k in T} | {k[1] for k in T})

    def score(order):
        return sum(T[(order[i], order[i + 1])] for i in range(len(order) - 1))

    orders = sorted(itertools.permutations(families), key=score, reverse=True)
    best = list(orders[0])
    reverse = best[::-1]

    arms = {
        "predicted_best": {"type": "sequential", "order": best,
                           "predicted_transition_sum": score(tuple(best))},
        "exact_reverse": {"type": "sequential", "order": reverse,
                          "predicted_transition_sum": score(tuple(reverse))},
        "balanced_shuffled": {"type": "interleaved", "order": list(families),
                              "predicted_transition_sum": None},
    }

    units = [{"arm": arm, "seed": seed, "unit_id": f"{arm}__seed{seed}"}
             for arm in arms for seed in SEEDS]

    payload = {
        "families": families,
        "arms": arms,
        "seeds": list(SEEDS),
        "steps_per_phase": STEPS_PER_PHASE,
        "total_steps": STEPS_PER_PHASE * len(families),
        "allocation": "identical across arms: every family receives "
                      f"{STEPS_PER_PHASE} steps of training in every arm; the arms "
                      "differ only in presentation order",
        "units": units,
        "n_units": len(units),
        "derived_from": "artifacts/layer2_transfer (frozen Layer-2 pilot matrix)",
        "limits": "The pilot is recorded as viability, not primitive-level "
                  "structure: only 2 of 12 directed pairs are primitive-disjoint. "
                  "A positive scout licenses expansion, not a structural claim.",
        "stop_rule": "If predicted_best does not beat exact_reverse by more than "
                     "the between-seed noise, stop the expansion and diagnose. "
                     "No curriculum, hyperparameter, threshold or seed may be "
                     "adapted on partial scout results.",
        "frozen_before_any_scout_run": True,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scout_manifest.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen manifest is never rewritten")
    path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"families: {families}")
    for name, arm in arms.items():
        s = arm["predicted_transition_sum"]
        print(f"  {name:>18s}  {' -> '.join(arm['order'])}"
              + (f"   predicted {s:+.4f}" if s is not None else "   (interleaved)"))
    gap = arms["predicted_best"]["predicted_transition_sum"] - \
        arms["exact_reverse"]["predicted_transition_sum"]
    print(f"\npredicted best-minus-reverse gap: {gap:+.4f}")
    print(f"{len(units)} units = 3 arms x {len(SEEDS)} seeds")
    print(f"sha256 {payload['sha256']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
