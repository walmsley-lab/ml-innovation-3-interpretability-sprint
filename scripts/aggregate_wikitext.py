"""Continuous validation and response aggregation during the WikiText wave.

Runs while trajectories land. Every mechanical step that does not require the
complete outcome set happens here, so that when the last development artifact
arrives the model fit begins immediately rather than after another pass of
processing.

It validates rather than merely reads. Each unit is checked for exposure
equality, shared initialization, and the presence of the sample manifests
that identify exactly which chunks the intervention used. A unit failing any
check is reported and excluded, not silently averaged in.

**It does not fit or select models.** The development pool is the fitting
pool, and model selection on an incomplete outcome set would let the pool's
composition depend on which trajectories happened to finish first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESPONSES = ("head_start", "T_aulc_rate_only", "T_aulc", "endpoint")
PRIMARY = ("head_start", "T_aulc_rate_only")


def validate(row: dict) -> list:
    problems = []
    if row.get("lm_tokens_source") != row.get("lm_tokens_control"):
        problems.append(f"exposure mismatch {row.get('lm_tokens_source')} vs "
                        f"{row.get('lm_tokens_control')}")
    for key in ("source_manifest", "control_manifest", "target_manifest"):
        manifest = row.get(key)
        if not manifest or not manifest.get("stream_hash"):
            problems.append(f"missing {key} stream hash")
    curve = row.get("curve_control") or []
    if not curve or curve[0]["offset"] != 0.0:
        problems.append("missing t=0 evaluation")
    for response in RESPONSES:
        if response not in row:
            problems.append(f"missing response {response}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("artifacts/wikitext_transfer"))
    ap.add_argument("--pool", default="development")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pools = json.loads(Path("artifacts/corpus_v2/frozen_pools.json").read_text())
    expected = {tuple(p) for p in pools["directed"][args.pool]}

    units, invalid, control_hashes = {}, [], {}
    for file in sorted((args.path / "units").glob("*.json")):
        row = json.loads(file.read_text())
        pair = (row["source"], row["target"])
        if pair not in expected:
            continue
        problems = validate(row)
        if problems:
            invalid.append((file.name, problems))
            continue
        units.setdefault(pair, []).append(row)
        # The common control must be identical across sources into a target.
        key = (row["target"], row["seed"])
        control_hashes.setdefault(key, set()).add(
            row["control_manifest"]["stream_hash"])

    drift = {k: v for k, v in control_hashes.items() if len(v) > 1}
    complete = {p: v for p, v in units.items() if len(v) >= 3}
    n_units = sum(len(v) for v in units.values())

    print(f"pool {args.pool}: {len(expected)} directed pairs expected")
    print(f"  units valid {n_units}, invalid {len(invalid)}")
    print(f"  pairs with all 3 seeds: {len(complete)}/{len(expected)}")
    print(f"  control invariant: {'PASS' if not drift else f'FAIL {drift}'}")
    for name, problems in invalid[:5]:
        print(f"    INVALID {name}: {'; '.join(problems)}")

    summary = {"pool": args.pool, "expected_pairs": len(expected),
               "valid_units": n_units, "invalid": invalid,
               "complete_pairs": len(complete), "control_invariant_ok": not drift,
               "responses": {}}
    for response in RESPONSES:
        means = {f"{s}->{t}": float(np.mean([r[response] for r in v]))
                 for (s, t), v in sorted(complete.items())}
        sds = {f"{s}->{t}": float(np.std([r[response] for r in v], ddof=1))
               for (s, t), v in sorted(complete.items())}
        if means:
            spread = float(np.std(list(means.values()), ddof=1)) if len(means) > 1 else 0.0
            median_sd = float(np.median(list(sds.values())))
            summary["responses"][response] = {
                "means": means, "sds": sds, "spread": spread,
                "median_seed_sd": median_sd,
                "max_seed_sd": float(max(sds.values())),
                "sn": spread / median_sd if median_sd else None,
                # Prespecified: a cell above 4x the median is flagged and
                # excluded from fitting rather than silently carried.
                "flagged_cells": [k for k, v in sds.items() if v > 4 * median_sd],
            }
            if not args.quiet:
                sn = summary["responses"][response]["sn"]
                print(f"  {response:>18s}  spread {spread:.4f}  med sd {median_sd:.4f}"
                      f"  S/N {sn:.2f}" if sn else f"  {response:>18s}  (1 pair)")

    (args.path / f"{args.pool}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if len(complete) == len(expected) and not drift and not invalid:
        print(f"\n{args.pool.upper()} POOL COMPLETE AND VALID — ready for model fit")


if __name__ == "__main__":
    main()
