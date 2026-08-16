"""Freeze prospective predictions for the 3 untouched natural pairs.

Written and committed **before** any untouched pair is run. Every rung of the
ladder is frozen on every metric, so the prospective comparison cannot be
made against a model chosen after the outcomes are visible.

The primary question this sets up is narrow and stated in advance: **does the
source-only AULC pattern generalize prospectively?** The other rungs and the
other metrics are frozen alongside it so that the comparison is available,
not so that a better-scoring one can be promoted afterwards. The failed
primary AULC relational gate stays failed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_natural_transfer import _design, family_features, load_units  # noqa: E402

from dsi.artifacts import code_version, utc_now  # noqa: E402

FROZEN = Path("artifacts/natural_pilot")
OUT = Path("artifacts/natural_transfer")
METRICS = ("T_aulc", "head_start", "T_aulc_rate_only", "endpoint")
MODELS = ("global", "source_only", "target_only", "additive", "cosine", "relational")


def main() -> None:
    units = load_units(OUT)
    pairs = json.loads((FROZEN / "frozen_pairs.json").read_text())
    observed = [tuple(p) for p in pairs["observed_pairs"]]
    untouched = [tuple(p) for p in pairs["heldout_pairs"]]
    if any(p in units for p in untouched):
        raise AssertionError("an untouched pair has already been run; freeze is void")
    if not all(p in units for p in observed):
        raise AssertionError("not all 9 observed pairs are present")

    features = family_features()
    payload = {"frozen_before_any_untouched_pair_was_run": True,
               "recorded_at": utc_now(), "code_version": code_version(),
               "observed_pairs": [list(p) for p in observed],
               "untouched_pairs": [list(p) for p in untouched],
               "features": {f"{a}->{b}": features[(a, b)]
                            for a, b in observed + untouched},
               "conclusion_preserved_verbatim": [
                   "The primary AULC relational prediction FAILED: the relational "
                   "model (LOPO 0.1166) is beaten by the global mean (0.0709).",
                   "source-only is the current provisional explanation: LOPO 0.0498 "
                   "with 4 parameters, against additive 0.0483 with 7.",
                   "head_start and rate_only analyses are secondary and exploratory. "
                   "They may not be used to retroactively rescue the failed primary "
                   "gate.",
               ],
               "primary_question": "does the source-only AULC pattern generalize "
                                   "prospectively to the 3 untouched pairs?",
               "primary_metric": "T_aulc", "primary_model": "source_only",
               "note": "The failed primary AULC relational gate is preserved as failed. "
                       "head_start and rate_only relational structure is exploratory "
                       "hypothesis generation only and may not rescue the primary gate.",
               "metrics": {}}

    for metric in METRICS:
        y = np.array([np.mean([r[metric] for r in units[p]]) for p in observed])
        entry = {"observed_mean": float(y.mean()), "observed_sd": float(y.std(ddof=1)),
                 "models": {}}
        for kind in MODELS:
            X = _design(observed, features, kind)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            lopo = []
            for i in range(len(observed)):
                keep = [j for j in range(len(observed)) if j != i]
                b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
                lopo.append(float(X[i] @ b))
            entry["models"][kind] = {
                "n_parameters": int(X.shape[1]),
                "coefficients": beta.tolist(),
                "design_matrix_observed": X.tolist(),
                "design_matrix_untouched": _design(untouched, features, kind).tolist(),
                "lopo_rmse": float(np.sqrt(np.mean((np.array(lopo) - y) ** 2))),
                "predictions": {f"{a}->{b}": float(_design([(a, b)], features, kind)[0] @ beta)
                                for a, b in untouched},
            }
        payload["metrics"][metric] = entry

    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path = OUT / "frozen_predictions_prospective.json"
    if path.exists():
        raise AssertionError(
            f"{path} already exists; a frozen prediction artifact is never rewritten")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")
    print(f"sha256 (over payload before the hash field) {payload['sha256']}")
    print(f"recorded_at {payload['recorded_at']}\n")
    for metric in METRICS:
        print(f"{metric}:")
        for kind in MODELS:
            m = payload["metrics"][metric]["models"][kind]
            preds = "  ".join(f"{k} {v:+.4f}" for k, v in m["predictions"].items())
            print(f"  {kind:>12s} (LOPO {m['lopo_rmse']:.4f})   {preds}")
        print()


if __name__ == "__main__":
    main()
