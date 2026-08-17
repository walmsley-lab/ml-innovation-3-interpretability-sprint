"""Lane B, stage SP-c: the frozen analysis.

Implements `docs/experiments/sprint_latent_state_protocol.md`, sha256
``afd0bf5174cd8073cf78e2802b1d654e8df2c93f5790f7a0f026456c31648e19``, which was
written and hashed before any divergence result existed. Nothing in the
protocol may be changed now that this script has run.

Two analyses:

**S2, primary.** Does internal state at ``t*`` predict the post-intervention
outcome better than behaviour at ``t*`` does, on held-out models? The headline
quantity is ``P_combined`` against ``P_beh`` — the *incremental* value of state
over behaviour, not a horse race.

**S1, secondary.** Among behaviour-matched pairs, does internal state carry
information about which history produced each model? Reported only if S2 holds.

The script refuses to report a matched-pair result below the protocol's floor
of 20 cross-history matched pairs, and it prints power diagnostics beside every
number: at pilot population sizes a ridge fit on 137 features is noisy, and a
number without its n is an invitation to over-read it.

    PYTHONPATH=src python scripts/sprint_analyze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROTOCOL = Path("docs/experiments/sprint_latent_state_protocol.md")
PROTOCOL_SHA = "afd0bf5174cd8073cf78e2802b1d654e8df2c93f5790f7a0f026456c31648e19"

OUTCOME = "NEUTRAL_CONFLICT.heldout.logodds"
COMPETENCE = ("W_COMPETENCE.heldout.accuracy", "P_COMPETENCE.heldout.accuracy")
COMPETENCE_FLOOR = 0.60
DEV_FRACTION = 0.40
MIN_MATCHED_PAIRS = 20
MATCH_QUANTILE = 0.10          # epsilon admits this fraction of random same-history pairs
ALPHAS = np.logspace(-3, 4, 22)
HISTORIES = ("W_first", "P_first")


def _check_protocol() -> None:
    """Refuse to run against a protocol that has been edited."""
    if not PROTOCOL.exists():
        raise SystemExit(f"protocol not found at {PROTOCOL}; analysis is not licensed")
    actual = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if actual != PROTOCOL_SHA:
        raise SystemExit(
            f"protocol hash mismatch\n  expected {PROTOCOL_SHA}\n  actual   {actual}\n"
            "The frozen protocol has changed since this analysis was written. "
            "Editing it after outcomes exist is the failure this check prevents."
        )


def load_models(population: Path) -> list[dict]:
    """One row per (pair, history), joining behaviour, state and outcomes."""
    units = sorted((population / "units").glob("pair_*.json"))
    state_dir = population / "state"
    rows = []
    for unit_path in units:
        record = json.loads(unit_path.read_text())
        state_path = state_dir / unit_path.name
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        for history in HISTORIES:
            payload = record["histories"][history]
            rows.append({
                "pair": record["pair"],
                "history": history,
                "behaviour": payload["t_star_behaviour"],
                "state": state["histories"][history],
                "outcomes": {k: v[OUTCOME] for k, v in payload["outcomes"].items()},
                "competence": float(np.mean(
                    [payload["t_star_behaviour"][c] for c in COMPETENCE])),
                "competence_min": float(np.min(
                    [payload["t_star_behaviour"][c] for c in COMPETENCE])),
            })
    return rows


def _matrix(rows: list[dict], field: str) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0][field])
    return np.array([[r[field][n] for n in names] for r in rows]), names


def _score(x_train, y_train, x_test, y_test) -> dict:
    """Ridge with alpha chosen by CV on the training split only."""
    scaler = StandardScaler().fit(x_train)
    model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    resid = y_test - pred
    var = float(np.var(y_test))
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "r2": float(1.0 - np.mean(resid**2) / var) if var > 0 else float("nan"),
        "alpha": float(model.alpha_),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(x_train.shape[1]),
    }


def run_s2(rows, dev, test, out: dict) -> None:
    beh, beh_names = _matrix(rows, "behaviour")
    state, state_names = _matrix(rows, "state")

    # PCA for the dimension-matched variant is fitted on DEV ONLY. Fitting it
    # on the full population would leak test-set structure into the feature
    # construction, which is a subtle way to hand P_int an advantage.
    n_components = min(beh.shape[1], len(dev) - 1, state.shape[1])
    pca = PCA(n_components=n_components).fit(state[dev])
    state_matched = pca.transform(state)

    feature_sets = {
        "P_beh": beh,
        "P_int": state,
        "P_int_matched": state_matched,
        "P_combined": np.hstack([beh, state]),
    }

    y = {name: np.array([r["outcomes"][name] for r in rows])
         for name in rows[0]["outcomes"]}
    competence = np.array([r["competence"] for r in rows])

    out["outcome_variance"] = {k: float(np.var(v)) for k, v in y.items()}
    out["n_models"] = len(rows)
    out["n_dev"] = len(dev)
    out["n_test"] = len(test)
    out["s2"] = {}

    for fit_on in y:
        for eval_on in y:
            key = f"fit_{fit_on}__eval_{eval_on}"
            out["s2"][key] = {
                name: _score(x[dev], y[fit_on][dev], x[test], y[eval_on][test])
                for name, x in feature_sets.items()
            }

    # Secondary: outcome residualized on competence, so a model that merely
    # lost competence does not read as "divergent".
    out["s2_competence_residualized"] = {}
    for fit_on in y:
        c_dev = competence[dev][:, None]
        beta = np.linalg.lstsq(
            np.hstack([c_dev, np.ones_like(c_dev)]), y[fit_on][dev], rcond=None)[0]
        def resid(idx):
            c = competence[idx][:, None]
            return y[fit_on][idx] - np.hstack([c, np.ones_like(c)]) @ beta
        out["s2_competence_residualized"][fit_on] = {
            name: _score(x[dev], resid(dev), x[test], resid(test))
            for name, x in feature_sets.items()
        }


def run_matching(rows, dev, test, out: dict) -> None:
    beh, _ = _matrix(rows, "behaviour")
    scaler = StandardScaler().fit(beh[dev])
    z = scaler.transform(beh)
    history = np.array([r["history"] for r in rows])
    pair = np.array([r["pair"] for r in rows])

    def distance(i, j) -> float:
        return float(np.max(np.abs(z[i] - z[j])))

    # epsilon calibrated on DEV same-history pairs only.
    same_dev = [distance(i, j) for a, i in enumerate(dev) for j in dev[a + 1:]
                if history[i] == history[j]]
    if not same_dev:
        out["matching"] = {"error": "no same-history pairs in the development split"}
        return
    epsilon = float(np.quantile(same_dev, MATCH_QUANTILE))

    def collect(indices, want_cross: bool):
        found = []
        for a, i in enumerate(indices):
            for j in indices[a + 1:]:
                is_cross = history[i] != history[j]
                if is_cross != want_cross:
                    continue
                if distance(i, j) <= epsilon:
                    found.append((int(i), int(j)))
        return found

    cross = collect(test, True)
    same = collect(test, False)
    same_pair_cross = [(i, j) for i, j in cross if pair[i] == pair[j]]

    out["matching"] = {
        "epsilon": epsilon,
        "epsilon_source": f"{MATCH_QUANTILE:.0%} quantile of dev same-history distances",
        "n_cross_history_matched": len(cross),
        "n_same_history_matched": len(same),
        "n_same_pair_cross_history_matched": len(same_pair_cross),
        "floor": MIN_MATCHED_PAIRS,
        "testable": len(cross) >= MIN_MATCHED_PAIRS,
    }

    if len(cross) < MIN_MATCHED_PAIRS:
        out["matching"]["verdict"] = (
            f"{len(cross)} cross-history matched pairs is below the frozen floor of "
            f"{MIN_MATCHED_PAIRS}. S1 is not testable at this population size. "
            "Report S2 only. Do NOT loosen epsilon."
        )
        return

    state, _ = _matrix(rows, "state")

    def divergence_analysis(pairs, label):
        if len(pairs) < MIN_MATCHED_PAIRS:
            return {"n": len(pairs), "skipped": "below floor"}
        result = {"n": len(pairs)}
        for name in rows[0]["outcomes"]:
            y = np.array([r["outcomes"][name] for r in rows])
            dy = np.array([abs(y[i] - y[j]) for i, j in pairs])
            d_state = np.array([np.linalg.norm(state[i] - state[j]) for i, j in pairs])
            d_beh = np.array([np.linalg.norm(z[i] - z[j]) for i, j in pairs])
            result[name] = {
                "mean_abs_outcome_difference": float(dy.mean()),
                "corr_state_distance": float(np.corrcoef(d_state, dy)[0, 1]),
                "corr_behaviour_distance": float(np.corrcoef(d_beh, dy)[0, 1]),
            }
        return result

    out["matching"]["cross_history"] = divergence_analysis(cross, "cross")
    out["matching"]["same_history"] = divergence_analysis(same, "same")
    out["matching"]["note"] = (
        "If the same-history correlation matches the cross-history one, the "
        "finding is 'state predicts future behaviour', NOT 'history is hidden "
        "in the state'."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=Path, default=Path("artifacts/sprint_population"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    _check_protocol()
    rows = load_models(args.population)
    if not rows:
        print(f"no extracted units in {args.population}; run sprint_extract_state.py first")
        return 0

    kept = [r for r in rows if r["competence_min"] >= COMPETENCE_FLOOR]
    out: dict = {
        "protocol_sha256": PROTOCOL_SHA,
        "n_models_generated": len(rows),
        "n_excluded_competence": len(rows) - len(kept),
        "competence_floor": COMPETENCE_FLOOR,
    }

    if len(kept) < 8:
        out["verdict"] = (
            f"only {len(kept)} of {len(rows)} models clear the competence floor of "
            f"{COMPETENCE_FLOOR}. Too few to analyze; this is a regime problem, not a result."
        )
        print(json.dumps(out, indent=2))
        return 0

    pairs = sorted({r["pair"] for r in kept})
    n_dev_pairs = max(1, int(round(DEV_FRACTION * len(pairs))))
    dev_pairs = set(pairs[:n_dev_pairs])
    dev = np.array([i for i, r in enumerate(kept) if r["pair"] in dev_pairs])
    test = np.array([i for i, r in enumerate(kept) if r["pair"] not in dev_pairs])

    if len(dev) < 4 or len(test) < 4:
        out["verdict"] = "population too small to split; generate more pairs"
        print(json.dumps(out, indent=2))
        return 0

    run_s2(kept, dev, test, out)
    run_matching(kept, dev, test, out)

    print(f"protocol {PROTOCOL_SHA[:12]} verified\n")
    print(f"models {out['n_models_generated']}, excluded on competence "
          f"{out['n_excluded_competence']}, dev {out['n_dev']}, test {out['n_test']}")
    print(f"\nS2 — held-out R2 (higher is better). "
          f"Headline is P_combined vs P_beh.")
    for key, scores in out["s2"].items():
        print(f"\n  {key}")
        for name, s in scores.items():
            print(f"    {name:16s} r2 {s['r2']:+.3f}  rmse {s['rmse']:.4f}  "
                  f"p {s['n_features']:3d}  n_test {s['n_test']}")

    m = out["matching"]
    print(f"\nMatching — epsilon {m.get('epsilon', float('nan')):.4f} "
          f"({m.get('epsilon_source', 'n/a')})")
    print(f"  cross-history matched {m.get('n_cross_history_matched', 0)} "
          f"(floor {MIN_MATCHED_PAIRS}), same-history matched "
          f"{m.get('n_same_history_matched', 0)}")
    if "verdict" in m:
        print(f"  {m['verdict']}")

    dest = args.out or (args.population / "analysis.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")

    n_test = out.get("n_test", 0)
    if n_test < 40:
        print(f"\nPOWER WARNING: {n_test} test models against 137 state features. "
              "Treat every number above as a pipeline check, not a finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
