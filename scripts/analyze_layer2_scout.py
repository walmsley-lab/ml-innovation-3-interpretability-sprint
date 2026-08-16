"""Mechanically aggregate the Layer-2 ceiling scout and apply its stop rule.

No model fitting, no design decisions. The manifest and its stop rule were
frozen before any unit ran; this script reads the units, computes the effect
sizes and the between-seed noise, and reports whether the rule passes.

The comparison the scout exists for is `predicted_best` against
`exact_reverse`: identical family allocation, identical budget, opposite
order. `balanced_shuffled` is reported alongside because an ordering that
beats its reverse while losing to a plain interleaved mixture would mean
something quite different from a curriculum effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/layer2_scout")
ARMS = ("predicted_best", "exact_reverse", "balanced_shuffled")
METRICS = ("final_mean_acc", "final_min_acc", "final_mean_loss")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=OUT)
    args = ap.parse_args()

    manifest = json.loads((args.path / "scout_manifest.json").read_text())
    units: dict = {}
    for file in sorted((args.path / "units").glob("*.json")):
        row = json.loads(file.read_text())
        if row.get("manifest_sha256") != manifest["sha256"]:
            raise AssertionError(f"{file.name} was produced under a different manifest")
        units.setdefault(row["arm"], []).append(row)

    total = sum(len(v) for v in units.values())
    print(f"manifest {manifest['sha256'][:16]}  units {total}/{manifest['n_units']}")
    if total < manifest["n_units"]:
        print("INCOMPLETE — the stop rule is applied only to a complete manifest.")
        return

    # Allocation equality is asserted per unit at run time; re-check here.
    for arm, rows in units.items():
        for row in rows:
            if len(set(row["allocation_per_family"].values())) != 1:
                raise AssertionError(f"{arm} seed {row['seed']} has unequal allocation")
    print("allocation identical across arms and families: PASS\n")

    stats = {}
    for metric in METRICS:
        print(f"{metric}:")
        for arm in ARMS:
            v = np.array([r[metric] for r in units[arm]])
            stats[(arm, metric)] = (float(v.mean()), float(v.std(ddof=1)))
            print(f"  {arm:>18s}  mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
                  f"n={len(v)}")
        print()

    # The frozen rule: best minus reverse, against between-seed noise.
    report = {"manifest_sha256": manifest["sha256"], "arms": {}, "comparisons": {}}
    for arm in ARMS:
        report["arms"][arm] = {m: stats[(arm, m)] for m in METRICS}

    print("frozen stop rule — predicted_best vs exact_reverse")
    verdict = {}
    for metric in METRICS:
        b, bsd = stats[("predicted_best", metric)]
        r, rsd = stats[("exact_reverse", metric)]
        bal, _ = stats[("balanced_shuffled", metric)]
        pooled = float(np.sqrt((bsd ** 2 + rsd ** 2) / 2))
        diff = b - r
        # Loss is better when lower; accuracy better when higher.
        signed = -diff if "loss" in metric else diff
        ratio = signed / pooled if pooled else float("nan")
        verdict[metric] = {"best": b, "reverse": r, "balanced": bal,
                           "difference": diff, "pooled_seed_sd": pooled,
                           "effect_over_noise": ratio,
                           "best_beats_reverse": bool(signed > pooled),
                           "best_beats_balanced": bool(
                               (bal - b) > 0 if "loss" in metric else (b - bal) > 0)}
        print(f"  {metric:>18s}  best {b:.4f}  reverse {r:.4f}  balanced {bal:.4f}"
              f"   diff {diff:+.4f}  pooled seed sd {pooled:.4f}"
              f"   effect/noise {ratio:+.2f}")
    report["comparisons"] = verdict

    primary = verdict["final_mean_acc"]
    passes = primary["best_beats_reverse"]
    report["stop_rule_passes"] = bool(passes)
    print(f"\nSTOP RULE ({'PASS' if passes else 'FAIL'}): predicted_best "
          f"{'beats' if passes else 'does NOT beat'} exact_reverse by more than "
          f"between-seed noise on final_mean_acc.")
    if not passes:
        print("  Do not launch the full pairwise matrix. Diagnose.")
        print(f"  best beats balanced: {primary['best_beats_balanced']}")
    (args.path / "scout_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.path / 'scout_result.json'}")


if __name__ == "__main__":
    main()
