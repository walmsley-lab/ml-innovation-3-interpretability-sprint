"""Freeze the r=0.20 overlap diagnostic before anything runs.

One change from the ceiling scout and nothing else: each phase mixes in
previously-seen families at Layer-1's validated overlap floor `r = 0.20`,
which is the setting that restored worst-case coexistence to 0.969 in Layer 1
after pure block-sequential phases produced catastrophic interference.

**The allocation trap, and how it is closed.** Layer-1 overlap draws from
*previously seen* families, so a family early in the ordering receives extra
exposure in every later phase while the last family receives none. Left
uncorrected, per-family exposure would depend on position, and because the
two sequential arms place different families in each position, order would be
confounded with dose — the exact confound the scout was careful to avoid.

Own-phase steps are therefore compensated so that **every family receives
exactly 600 steps of exposure in every arm**:

    position 1: 380 own + 220 received as overlap = 600
    position 2: 500 own + 100 received            = 600
    position 3: 560 own +  40 received            = 600
    position 4: 600 own +   0 received            = 600

Total 2,400 steps per unit in every arm, identical to the scout. Only the
*timing of concentration* differs.

**What this tests.** Whether pairwise developmental effects compose under
retention-preserving overlap. Beating the reverse arm is the mechanistic
question; it is **not** sufficient evidence of practical value. The practical
baseline is the interleaved arm at equal compute, and the two claims are
kept separate throughout.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/layer2_overlap")
SEEDS = (5000, 5001, 5002, 5003, 5004)
TOTAL_STEPS = 2400
OVERLAP_R = 0.20
PILOT = Path("artifacts/layer2_transfer/units")
THRESHOLD = 0.90        # tokens/steps-to-threshold target, frozen in advance


def schedule_for(order, n_families: int, total_steps: int, r: float):
    """Steps per (phase, family), compensated so every family totals equally."""
    per_phase = total_steps // n_families
    overlap = int(round(per_phase * r))
    plan = []
    received = {f: 0 for f in order}
    for k, fam in enumerate(order):
        previous = order[:k]
        share = overlap // len(previous) if previous else 0
        for p in previous:
            received[p] += share
        plan.append({"phase": k + 1, "family": fam, "previous": list(previous),
                     "overlap_steps_total": overlap if previous else 0,
                     "overlap_steps_each": share})
    for k, fam in enumerate(order):
        plan[k]["own_steps"] = per_phase - received[fam]
    totals = {f: plan[k]["own_steps"] + received[f] for k, f in enumerate(order)}
    return plan, totals


def main() -> None:
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(PILOT / "*.json"))]
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["source"], r["target"]), []).append(r["T_aulc"])
    T = {k: float(np.mean(v)) for k, v in acc.items()}
    families = sorted({k[0] for k in T} | {k[1] for k in T})

    def score(o):
        return sum(T[(o[i], o[i + 1])] for i in range(len(o) - 1))

    best = list(max(itertools.permutations(families), key=score))
    reverse = best[::-1]

    arms = {}
    for name, order in (("predicted_best", best), ("exact_reverse", reverse)):
        plan, totals = schedule_for(order, len(families), TOTAL_STEPS, OVERLAP_R)
        if len(set(totals.values())) != 1:
            raise AssertionError(f"{name} allocation not equal: {totals}")
        arms[name] = {"type": "sequential_overlap", "order": order, "plan": plan,
                      "per_family_total_steps": totals,
                      "predicted_transition_sum": score(tuple(order))}
    arms["balanced_shuffled"] = {
        "type": "interleaved", "order": list(families),
        "per_family_total_steps": {f: TOTAL_STEPS // len(families) for f in families},
        "predicted_transition_sum": None}

    allocations = [tuple(sorted(a["per_family_total_steps"].items())) for a in arms.values()]
    if len(set(allocations)) != 1:
        raise AssertionError("aggregate allocation differs across arms")

    payload = {
        "families": families, "arms": arms, "seeds": list(SEEDS),
        "total_steps": TOTAL_STEPS, "overlap_r": OVERLAP_R,
        "threshold": THRESHOLD,
        "allocation": "identical across arms AND families: every family receives "
                      f"{TOTAL_STEPS // len(families)} steps in every arm; own-phase "
                      "steps are compensated for overlap received later, so order is "
                      "not confounded with dose",
        "units": [{"arm": a, "seed": s, "unit_id": f"{a}__seed{s}"}
                  for a in arms for s in SEEDS],
        "n_units": len(arms) * len(SEEDS),
        "question": "Do pairwise developmental effects compose under retention-"
                    "preserving overlap?",
        "claim_separation": "Beating exact_reverse is the MECHANISTIC result and is "
                            "not sufficient evidence of practical value. The practical "
                            "baseline is balanced_shuffled at equal compute. Failing to "
                            "beat interleaving does NOT mean no developmental structure "
                            "exists; the two claims are reported separately.",
        "frozen_metrics": ["steps_to_threshold", "tokens_to_threshold",
                           "fixed_budget_mean_acc", "fixed_budget_min_acc",
                           "familywise_auc", "retention"],
        "primary_metric": "fixed_budget_min_acc",
        "derived_from": "artifacts/layer2_transfer (viability-derived ceiling candidate, "
                        "not a primitive-level structural estimate)",
        "frozen_before_any_run": True,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "overlap_manifest.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen manifest is never rewritten")
    path.write_text(json.dumps(payload, indent=2) + "\n")

    for name, arm in arms.items():
        print(f"{name}:")
        if arm["type"] == "sequential_overlap":
            for p in arm["plan"]:
                print(f"  phase {p['phase']} {p['family'].split('_')[0]:>3s}: "
                      f"own {p['own_steps']:>3d} + overlap {p['overlap_steps_total']:>3d} "
                      f"over {len(p['previous'])} previous")
        print(f"  per-family totals: "
              f"{ {k.split('_')[0]: v for k, v in arm['per_family_total_steps'].items()} }")
    print(f"\nallocation identical across arms and families: PASS")
    print(f"{payload['n_units']} units   sha256 {payload['sha256']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
