"""Official model selection on the complete development pool, then freeze.

Two steps that must happen in this order and only once:

1. **Fit and select** the frozen ladder on the **complete** development pool.
   The script refuses to run on a partial pool. Mechanical aggregation runs
   continuously during the wave, but the official comparison waits for every
   development relationship, or the fitting set's composition depends on
   which trajectories happened to finish first.
2. **Freeze and hash** predictions for the untouched confirmatory pool,
   before any confirmatory trajectory runs.

Both primary components are frozen together under the same model, and the
continuous performance of every ladder rung is recorded alongside the gate,
so a formal gate failure can never be reported as "no structure".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wikitext_model import MODELS, design, features, fit_ladder, ridge_fit  # noqa: E402

from dsi.artifacts import code_version, utc_now  # noqa: E402

CORPUS = Path("artifacts/corpus_v2")
PRIMARY = ("head_start", "T_aulc_rate_only")
SECONDARY = ("T_aulc", "endpoint")


def load_pool(path: Path, pool: str, pools: dict) -> dict:
    expected = {tuple(p) for p in pools["directed"][pool]}
    units: dict = {}
    for file in sorted((path / "units").glob("*.json")):
        row = json.loads(file.read_text())
        pair = (row["source"], row["target"])
        if pair in expected:
            units.setdefault(pair, []).append(row)
    missing = expected - set(units)
    incomplete = {p for p, v in units.items() if len(v) < 3}
    return {"units": units, "expected": expected, "missing": missing,
            "incomplete": incomplete}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("artifacts/wikitext_transfer"))
    ap.add_argument("--allow-flagged", action="store_true")
    args = ap.parse_args()

    pools = json.loads((CORPUS / "frozen_pools.json").read_text())
    dev = load_pool(args.path, "development", pools)
    if dev["missing"] or dev["incomplete"]:
        raise SystemExit(
            f"development pool incomplete: {len(dev['missing'])} pairs missing, "
            f"{len(dev['incomplete'])} with fewer than 3 seeds. Official model "
            "selection waits for the complete frozen pool.")

    confirmatory = [tuple(p) for p in pools["directed"]["confirmatory"]]
    if any((args.path / "units" / f"f{s}__to__f{t}__seed3000.json").exists()
           for s, t in confirmatory):
        raise SystemExit(
            "a confirmatory unit already exists; predictions must be frozen "
            "before any confirmatory trajectory runs")

    pairs = sorted(dev["units"])
    feats = features()
    payload = {"recorded_at": utc_now(), "code_version": code_version(),
               "development_pairs": [list(p) for p in pairs],
               "confirmatory_pairs": [list(p) for p in confirmatory],
               "features": {f"{a}->{b}": feats[(a, b)]
                            for a, b in pairs + confirmatory},
               "frozen_before_any_confirmatory_trajectory": True,
               "responses": {}}

    print(f"development pool complete: {len(pairs)} directed pairs, "
          f"{sum(len(v) for v in dev['units'].values())} units\n")

    for response in PRIMARY + SECONDARY:
        y = np.array([np.mean([r[response] for r in dev["units"][p]]) for p in pairs])
        sds = np.array([np.std([r[response] for r in dev["units"][p]], ddof=1)
                        for p in pairs])
        ladder = fit_ladder(pairs, y)
        best = min(MODELS, key=lambda k: ladder[k]["lopo_rmse"])
        print(f"{response}:  (development LOPO, selection device only)")
        for kind in MODELS:
            row = ladder[kind]
            print(f"  {kind:>12s}  p={row['n_parameters']:>2d}  "
                  f"ridge={row['penalty']:<7g}  LOPO {row['lopo_rmse']:.4f}"
                  f"{'   <- best' if kind == best else ''}")

        entry = {"development_mean": {f"{a}->{b}": float(v) for (a, b), v in zip(pairs, y)},
                 "development_seed_sd": {f"{a}->{b}": float(v) for (a, b), v in zip(pairs, sds)},
                 "median_seed_sd": float(np.median(sds)),
                 "ladder": ladder, "selected_by_lopo": best, "predictions": {}}
        # Predictions from EVERY rung are frozen, so the confirmatory
        # comparison against the simpler models is fixed in advance too.
        for kind in MODELS:
            X = design(pairs, kind)
            beta = ridge_fit(X, y, ladder[kind]["penalty"])
            Xc = design(confirmatory, kind)
            entry["predictions"][kind] = {
                f"{a}->{b}": float(v) for (a, b), v in zip(confirmatory, Xc @ beta)}
        payload["responses"][response] = entry
        print()

    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = args.path / "frozen_confirmatory_predictions.json"
    if out.exists():
        raise SystemExit(f"{out} exists; a frozen prediction artifact is never rewritten")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"sha256 {payload['sha256']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
