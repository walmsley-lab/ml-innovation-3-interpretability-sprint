"""Apply the frozen gate, then score the untouched adaptive reserve.

Two stages, hard-separated:

* ``--gate`` scores the confirmatory pool against the predictions frozen
  before it ran, applying the criterion from ``success_criterion``. Continuous
  performance against every simpler model is always reported alongside the
  binary outcome, because a formal gate failure is not the same finding as
  "no structure" and must not be written up as one.
* ``--select`` scores the adaptive reserve. It **refuses to run unless the
  gate passed**, since selection driven by an unvalidated predictor tests
  nothing.

Candidate static diagnostics — features, leverage, Mahalanobis distance —
depend only on frozen inputs and are computed in either mode, so they can be
prepared during the confirmatory wave. Model-dependent scoring and the
selection itself happen only after a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from success_criterion import SIMPLER_MODELS, relational_claim_passes  # noqa: E402
from wikitext_model import MODELS, design, features, ridge_fit  # noqa: E402

from dsi.artifacts import code_version, utc_now  # noqa: E402

CORPUS = Path("artifacts/corpus_v2")
PRIMARY = ("head_start", "T_aulc_rate_only")
LEVERAGE_PERCENTILE = 95


def observed(path: Path, pairs, response: str) -> dict:
    out = {}
    for source, target in pairs:
        rows = [json.loads(f.read_text()) for f in
                (path / "units").glob(f"f{source}__to__f{target}__seed*.json")]
        if len(rows) >= 3:
            out[f"{source}->{target}"] = float(np.mean([r[response] for r in rows]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("artifacts/wikitext_transfer"))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()

    pools = json.loads((CORPUS / "frozen_pools.json").read_text())
    frozen = json.loads((args.path / "frozen_confirmatory_predictions.json").read_text())
    confirmatory = [tuple(p) for p in pools["directed"]["confirmatory"]]
    adaptive = [tuple(p) for p in pools["directed"]["adaptive"]]

    body = {k: v for k, v in frozen.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(body, indent=2, sort_keys=True).encode()).hexdigest()
    print(f"frozen predictions {frozen['recorded_at']}  hash "
          f"{'verified' if digest == frozen['sha256'] else 'MISMATCH'}")

    if args.gate:
        obs = {c: observed(args.path, confirmatory, c) for c in PRIMARY}
        missing = [c for c in PRIMARY if len(obs[c]) < len(confirmatory)]
        if missing:
            raise SystemExit(f"confirmatory pool incomplete for {missing}")

        pro = {c: {m: frozen["responses"][c]["predictions"][m] for m in MODELS}
               for c in PRIMARY}
        floor = {c: frozen["responses"][c]["median_seed_sd"] for c in PRIMARY}

        # Continuous performance ALWAYS reported, gate outcome or not.
        print("\ncontinuous prospective RMSE (reported regardless of the gate):")
        print(f"{'response':>18s} " + " ".join(f"{m:>12s}" for m in MODELS))
        for c in PRIMARY:
            pairs_c = sorted(obs[c])
            truth = np.array([obs[c][p] for p in pairs_c])
            cells = []
            for m in MODELS:
                pred = np.array([pro[c][m][p] for p in pairs_c])
                cells.append(f"{np.sqrt(np.mean((pred - truth) ** 2)):>12.4f}")
            print(f"{c:>18s} " + " ".join(cells))

        result = relational_claim_passes(pro, obs, floor)
        print(f"\nrule: {result['rule']}")
        for c, entry in result["components"].items():
            print(f"  {c}: best simpler = {entry['best_simpler_model']} "
                  f"({entry['rmse'][entry['best_simpler_model']]:.4f}), "
                  f"relational {entry['rmse']['relational']:.4f}, "
                  f"needs <= {entry['required_rmse']:.4f}  -> "
                  f"material {entry['material']}, jackknife {entry['jackknife_robust']}, "
                  f"above floor {entry['above_noise_floor']}")
        print(f"\nGATE: {'PASS' if result['passes'] else 'FAIL'}")
        if not result["passes"]:
            print("A gate failure is a failure of THIS predictor against THIS "
                  "criterion. It is not a finding of 'no structure'; the "
                  "continuous table above is the reportable result.")
        (args.path / "gate_result.json").write_text(json.dumps(
            {**result, "recorded_at": utc_now(), "code_version": code_version()},
            indent=2) + "\n")
        return

    # Static candidate diagnostics: frozen inputs only, safe to compute early.
    feats = features()
    dev_pairs = [tuple(p) for p in frozen["development_pairs"]]
    print(f"\nadaptive reserve: {len(adaptive)} untouched directed pairs")
    X = design(dev_pairs, "relational")
    cov = np.linalg.pinv(X.T @ X)
    cloud = np.array([[feats[p]["cosine"], feats[p]["kl"]] for p in dev_pairs])
    centre, spread = cloud.mean(axis=0), np.cov(cloud.T) + np.eye(2) * 1e-9
    inv = np.linalg.pinv(spread)
    dev_lev = np.array([design([p], "relational")[0] @ cov @ design([p], "relational")[0]
                        for p in dev_pairs])
    cap = float(np.percentile(dev_lev, LEVERAGE_PERCENTILE))

    diagnostics = {}
    for pair in adaptive:
        x0 = design([pair], "relational")[0]
        v = np.array([feats[pair]["cosine"], feats[pair]["kl"]]) - centre
        leverage = float(x0 @ cov @ x0)
        diagnostics[f"{pair[0]}->{pair[1]}"] = {
            "leverage": leverage, "leverage_cap": cap,
            "eligible": bool(leverage <= cap),
            "mahalanobis": float(np.sqrt(max(v @ inv @ v, 0.0))),
            "features": feats[pair]}
    print(f"leverage cap (p{LEVERAGE_PERCENTILE} of development) = {cap:.4f}")
    for name, d in sorted(diagnostics.items()):
        print(f"  {name}  leverage {d['leverage']:.4f}  "
              f"mahalanobis {d['mahalanobis']:.2f}  "
              f"{'ELIGIBLE' if d['eligible'] else 'INELIGIBLE (extrapolation)'}")

    if not args.select:
        (args.path / "adaptive_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n")
        print("\nstatic diagnostics staged. Selection requires --select and a passed gate.")
        return

    gate = json.loads((args.path / "gate_result.json").read_text())
    if not gate.get("passes"):
        raise SystemExit("gate did not pass; adaptive selection is not licensed")

    eligible = {k: v for k, v in diagnostics.items() if v["eligible"]}
    if not eligible:
        raise SystemExit("every candidate is leverage-ineligible; the model cannot "
                         "select a defensible intervention")

    obs_dev = frozen["responses"]
    scored = {}
    for name, d in eligible.items():
        s, t = (int(x) for x in name.split("->"))
        u = 0.0
        for c in PRIMARY:
            y = np.array([obs_dev[c]["development_mean"][f"{a}->{b}"] for a, b in dev_pairs])
            pen = obs_dev[c]["ladder"]["relational"]["penalty"]
            beta = ridge_fit(X, y, pen)
            resid = float(np.sqrt(np.mean((X @ beta - y) ** 2)))
            u += resid * np.sqrt(max(d["leverage"], 0.0))
        scored[name] = u
    selected = max(sorted(scored), key=lambda k: scored[k])
    payload = {"acquisition": scored, "selected": selected,
               "diagnostics": diagnostics, "gate_passed": True,
               "rule": "argmax U over leverage-eligible candidates; U summed over "
                       "both primary components; ties break on lower (source, target)",
               "recorded_at": utc_now(), "code_version": code_version()}
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = args.path / "adaptive_selection.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the selection is frozen once")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSELECTED {selected}   sha256 {payload['sha256']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
