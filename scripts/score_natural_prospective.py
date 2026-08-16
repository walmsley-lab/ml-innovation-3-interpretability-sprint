"""Score the frozen prospective predictions against the 3 untouched pairs.

Reads the frozen artifact and the observed outcomes. Fits nothing on the
untouched pairs and rewrites no frozen object: the whole point is that every
number compared here was committed to disk, hashed and timestamped, before
any of these three pairs was run.

Two sections, kept apart on purpose:

* **Primary.** The prospective AULC comparison across the frozen ladder, with
  RMSE, MAE, sign accuracy and per-pair errors. `1->4` and `4->1` are the
  diagnostic cells: `source_only` and `additive` diverge most there, so they
  separate a genuine LOPO advantage from an interpolation artifact.
* **Exploratory.** Component-level analysis over all 12 pairs. Labelled
  exploratory throughout and **cannot rescue the failed primary AULC gate**.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_natural_transfer import _design, family_features, load_units  # noqa: E402

OUT = Path("artifacts/natural_transfer")
METRICS = ("T_aulc", "head_start", "T_aulc_rate_only", "endpoint")
MODELS = ("global", "source_only", "target_only", "additive", "cosine", "relational")


def verify_freeze(frozen: dict) -> str:
    """Re-derive the artifact hash so the comparison is auditable."""
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(body, indent=2, sort_keys=True).encode()).hexdigest()
    return ("verified" if digest == frozen.get("sha256")
            else f"MISMATCH recomputed {digest} against recorded {frozen.get('sha256')}")


def score_primary(frozen: dict, units: dict, untouched: list, metric: str) -> dict:
    observed = {f"{a}->{b}": float(np.mean([r[metric] for r in units[(a, b)]]))
                for a, b in untouched}
    seed_sd = {f"{a}->{b}": float(np.std([r[metric] for r in units[(a, b)]], ddof=1))
               for a, b in untouched}
    rows = {}
    for kind in MODELS:
        predictions = frozen["metrics"][metric]["models"][kind]["predictions"]
        errors = {k: predictions[k] - observed[k] for k in observed}
        values = np.array(list(errors.values()))
        signs = sum(1 for k in observed
                    if np.sign(predictions[k]) == np.sign(observed[k]))
        rows[kind] = {
            "lopo_rmse_frozen": frozen["metrics"][metric]["models"][kind]["lopo_rmse"],
            "prospective_rmse": float(np.sqrt(np.mean(values ** 2))),
            "prospective_mae": float(np.mean(np.abs(values))),
            "sign_accuracy": f"{signs}/{len(observed)}",
            "predictions": predictions, "errors": errors,
        }
    return {"observed": observed, "seed_sd": seed_sd, "models": rows}


def lopo(pairs: list, features: dict, y: np.ndarray, kind: str):
    X = _design(pairs, features, kind)
    if X.shape[1] >= len(pairs):
        return None
    predictions = []
    for i in range(len(pairs)):
        keep = [j for j in range(len(pairs)) if j != i]
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        predictions.append(float(X[i] @ beta))
    return float(np.sqrt(np.mean((np.array(predictions) - y) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="T_aulc")
    args = ap.parse_args()

    frozen = json.loads((OUT / "frozen_predictions_prospective.json").read_text())
    units = load_units(OUT)
    untouched = [tuple(p) for p in frozen["untouched_pairs"]]
    observed_pairs = [tuple(p) for p in frozen["observed_pairs"]]
    missing = [p for p in untouched if p not in units]
    if missing:
        print(f"untouched pairs not yet run: {missing}")
        return

    print(f"frozen artifact {frozen['recorded_at']}  hash {verify_freeze(frozen)}")
    print(f"primary question: {frozen['primary_question']}")
    print(f"primary model/metric: {frozen['primary_model']} / {frozen['primary_metric']}\n")

    result = score_primary(frozen, units, untouched, args.metric)
    print(f"=== PRIMARY: prospective {args.metric} on the 3 untouched pairs ===\n")
    print("observed:")
    for name, value in result["observed"].items():
        print(f"  {name}  {value:+.4f} +- {result['seed_sd'][name]:.4f}")
    print(f"\n{'model':>12s} {'LOPO(frozen)':>13s} {'prosp RMSE':>11s} {'MAE':>8s} "
          f"{'sign':>6s}   per-pair error")
    for kind in MODELS:
        row = result["models"][kind]
        errors = "  ".join(f"{k} {v:+.4f}" for k, v in row["errors"].items())
        print(f"{kind:>12s} {row['lopo_rmse_frozen']:>13.4f} "
              f"{row['prospective_rmse']:>11.4f} {row['prospective_mae']:>8.4f} "
              f"{row['sign_accuracy']:>6s}   {errors}")

    print("\ndiagnostic cells (source_only vs additive divergence):")
    for name in ("1->4", "4->1"):
        obs = result["observed"][name]
        s_pred = result["models"]["source_only"]["predictions"][name]
        a_pred = result["models"]["additive"]["predictions"][name]
        print(f"  {name}  observed {obs:+.4f}   source_only {s_pred:+.4f} "
              f"(err {s_pred - obs:+.4f})   additive {a_pred:+.4f} (err {a_pred - obs:+.4f})")

    print("\n=== EXPLORATORY: component analysis over all 12 pairs ===")
    print("Hypothesis generation only. Cannot rescue the failed primary AULC gate.\n")
    features = family_features()
    all_pairs = observed_pairs + untouched
    print(f"{'metric':>18s} " + " ".join(f"{k:>12s}" for k in MODELS))
    table = {}
    for metric in METRICS:
        y = np.array([np.mean([r[metric] for r in units[p]]) for p in all_pairs])
        table[metric] = {k: lopo(all_pairs, features, y, k) for k in MODELS}
        cells = " ".join(f"{table[metric][k]:>12.4f}" if table[metric][k] is not None
                         else f"{'-':>12s}" for k in MODELS)
        print(f"{metric:>18s} {cells}")

    print("\nbest model per metric (12-pair LOPO, exploratory):")
    for metric in METRICS:
        scored = {k: v for k, v in table[metric].items() if v is not None}
        best = min(scored, key=scored.get)
        gain = 100 * (scored["global"] - scored[best]) / scored["global"]
        print(f"  {metric:>18s}  {best:>12s}  {scored[best]:.4f}  "
              f"({gain:+.1f}% vs global)")

    (OUT / "prospective_scores.json").write_text(json.dumps(
        {"primary": result, "exploratory_12pair_lopo": table,
         "frozen_hash_check": verify_freeze(frozen)}, indent=2) + "\n")
    print(f"\nwrote {OUT / 'prospective_scores.json'}")


if __name__ == "__main__":
    main()
