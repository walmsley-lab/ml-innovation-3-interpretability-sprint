"""Stage-5 gate: is natural transfer measurable, and is it predictable?

Two questions in order, and the second is not asked if the first fails.

**Measurability.** Between-pair spread of the mean effect against the
within-pair seed standard deviation. Gate C is the precedent for why this
comes first: an apparatus whose null noise spans the range of the metric
cannot support any conclusion, and running the predictive ladder on it would
be ceremony.

**Predictability.** A ladder fitted on the 9 observed pairs and scored on the
3 pairs frozen as held out before any natural transfer ran. The split unit is
the **pair** — every seed of a held-out pair is held out with it.

Features must exist for a pair that has never been run, so pair-identity
coefficients are excluded by construction. They are computed from the frozen
family memberships over **train documents only**; the family assignment is
not refitted here and neither is the proposer.

With 9 observed pairs the ladder stops at three free parameters. Anything
richer is fitted noise wearing the name of structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dsi.corpus import Corpus, Document, deduplicate, split_documents
from dsi.natural import whitespace_tokens

FROZEN = Path("artifacts/natural_pilot")
USABLE_FAMILIES = (1, 2, 3, 4)


def load_units(path: Path) -> dict:
    units: dict = {}
    for file in sorted((path / "units").glob("*.json")):
        row = json.loads(file.read_text())
        units.setdefault((row["source"], row["target"]), []).append(row)
    return units


def measurability(units: dict, metric: str = "T_aulc") -> dict:
    """Between-pair spread against within-pair seed noise."""
    per_pair = {p: [r[metric] for r in rows] for p, rows in units.items()}
    means = {p: float(np.mean(v)) for p, v in per_pair.items()}
    sds = {p: float(np.std(v, ddof=1)) for p, v in per_pair.items() if len(v) > 1}
    spread = float(np.std(list(means.values()), ddof=1))
    median_sd = float(np.median(list(sds.values()))) if sds else float("nan")
    return {"metric": metric, "means": means, "sds": sds,
            "between_pair_spread": spread, "median_within_pair_sd": median_sd,
            "signal_to_noise": spread / median_sd if median_sd else float("nan")}


def family_features(vocab_size: int = 20000) -> dict:
    """Directional and symmetric relationships between families.

    Computed from train documents under the frozen assignment. Two features
    only, and both exist for an unseen pair:

    * ``cosine`` — symmetric similarity of TF-IDF family centroids. The
      natural-corpus analogue of the synthetic shared-primitive count.
    * ``kl`` — ``KL(target || source)`` over unigram distributions, which is
      **asymmetric** and therefore able to express a directional effect at
      all. A symmetric feature alone cannot distinguish ``i->j`` from
      ``j->i``, and directionality is the whole question.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    frozen = json.loads((FROZEN / "assignments.json").read_text())
    from sklearn.datasets import fetch_20newsgroups
    raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    dedup, _ = deduplicate(Corpus("20ng", "v1", tuple(Document.of(t) for t in raw.data)))
    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))

    texts = {f: [] for f in USABLE_FAMILIES}
    for document in splits["train"].documents:
        family = frozen["train"].get(document.doc_id)
        if family in texts:
            texts[family].append(document.text)

    vectorizer = TfidfVectorizer(max_features=vocab_size, stop_words="english")
    matrix = vectorizer.fit_transform([" ".join(texts[f]) for f in USABLE_FAMILIES])
    centroids = np.asarray(matrix.todense())
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    cosine = centroids @ centroids.T

    unigram = {}
    for family in USABLE_FAMILIES:
        counts: dict = {}
        for text in texts[family]:
            for piece in whitespace_tokens(text):
                counts[piece] = counts.get(piece, 0) + 1
        unigram[family] = counts

    keys = sorted({k for c in unigram.values() for k in c})
    probs = {}
    for family, counts in unigram.items():
        vector = np.array([counts.get(k, 0) for k in keys], dtype=float) + 1.0
        probs[family] = vector / vector.sum()

    features = {}
    for a, source in enumerate(USABLE_FAMILIES):
        for b, target in enumerate(USABLE_FAMILIES):
            if source == target:
                continue
            kl = float(np.sum(probs[target] * np.log(probs[target] / probs[source])))
            features[(source, target)] = {"cosine": float(cosine[a, b]), "kl": kl}
    return features


def _design(pairs, features, kind: str) -> np.ndarray:
    rows = []
    for source, target in pairs:
        f = features[(source, target)]
        if kind == "global":
            rows.append([1.0])
        elif kind == "source_only":
            rows.append([1.0] + [1.0 if source == g else 0.0 for g in USABLE_FAMILIES[1:]])
        elif kind == "target_only":
            rows.append([1.0] + [1.0 if target == g else 0.0 for g in USABLE_FAMILIES[1:]])
        elif kind == "additive":
            row = [1.0]
            row += [1.0 if source == g else 0.0 for g in USABLE_FAMILIES[1:]]
            row += [1.0 if target == g else 0.0 for g in USABLE_FAMILIES[1:]]
            rows.append(row)
        elif kind == "cosine":
            rows.append([1.0, f["cosine"]])
        elif kind == "relational":
            rows.append([1.0, f["cosine"], f["kl"]])
        else:
            raise ValueError(kind)
    return np.array(rows)


def lopo_ladder(units: dict, observed: list, features: dict, metric: str) -> dict:
    """Leave-one-pair-out over the 9 observed identities.

    Model comparison happens entirely inside the observed pairs. With four
    families there are only 3 unobserved pairs in existence, and spending
    them on a batch test would leave the adaptive selector no candidate at
    all; docs/experiments/natural_corpus.md 15 records that role conflict.

    The held-out unit is the **pair**: all of its seeds leave with it, so no
    seed of a predicted pair is ever in the fit.
    """
    y = np.array([np.mean([r[metric] for r in units[p]]) for p in observed])
    results = {}
    for kind in ("global", "source_only", "target_only", "additive", "cosine", "relational"):
        X = _design(observed, features, kind)
        if X.shape[1] >= len(observed):
            results[kind] = {"n_parameters": int(X.shape[1]), "lopo_rmse": None,
                             "note": "not identifiable under leave-one-pair-out"}
            continue
        predictions = []
        for i in range(len(observed)):
            keep = [j for j in range(len(observed)) if j != i]
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            predictions.append(float(X[i] @ beta))
        residual = np.array(predictions) - y
        beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
        results[kind] = {
            "n_parameters": int(X.shape[1]),
            "in_sample_rmse": float(np.sqrt(np.mean((X @ beta_full - y) ** 2))),
            "lopo_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "lopo_mae": float(np.mean(np.abs(residual))),
            "lopo_predictions": {f"{a}->{b}": v for (a, b), v in zip(observed, predictions)},
        }
    results["_observed"] = {f"{a}->{b}": float(v) for (a, b), v in zip(observed, y)}
    return results


def freeze_predictions(units: dict, observed: list, untouched: list, features: dict,
                       metric: str, kind: str, lopo_rmse: float, path: Path) -> dict:
    """Fit on all 9 observed pairs and freeze predictions for the untouched 3.

    Written before any untouched pair is run, so a prediction cannot be
    revised after its outcome is visible.

    The acquisition score is the frozen rule from docs/experiments/natural_corpus.md 15:
    ``U_e = s_lopo * sqrt(x0' (X'X)^-1 x0)``, the predictive standard error at
    the candidate's feature location with the residual scale taken from
    leave-one-pair-out rather than in-sample. Importance and cost are uniform
    across the three candidates, so ``argmax U_e`` is the whole rule. Ties
    break on the lower ``(source, target)``.
    """
    if path.exists():
        return json.loads(path.read_text())
    y = np.array([np.mean([r[metric] for r in units[p]]) for p in observed])
    X = _design(observed, features, kind)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    covariance = np.linalg.pinv(X.T @ X)

    candidates = {}
    for pair in untouched:
        x0 = _design([pair], features, kind)[0]
        leverage = float(x0 @ covariance @ x0)
        candidates[f"{pair[0]}->{pair[1]}"] = {
            "prediction": float(x0 @ beta),
            "acquisition_U": float(lopo_rmse * np.sqrt(max(leverage, 0.0))),
            "features": features[pair],
        }
    selected = max(sorted(candidates), key=lambda k: candidates[k]["acquisition_U"])
    payload = {"metric": metric, "model": kind, "fitted_on_pairs": [list(p) for p in observed],
               "coefficients": beta.tolist(), "lopo_rmse": lopo_rmse,
               "candidates": candidates, "selected": selected,
               "acquisition_rule": "argmax U_e, U_e = s_lopo * sqrt(x0' (X'X)^-1 x0); "
                                   "I_e and C_e uniform; ties break on lower (source, target)",
               "frozen_before_any_untouched_pair_was_run": True}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("artifacts/natural_transfer"))
    ap.add_argument("--metric", default="T_aulc")
    args = ap.parse_args()

    units = load_units(args.path)
    pairs = json.loads((FROZEN / "frozen_pairs.json").read_text())
    heldout_pairs = [tuple(p) for p in pairs["heldout_pairs"]]

    print(f"{len(units)} pairs, {sum(len(v) for v in units.values())} units\n")
    for metric in ("T_aulc", "T_aulc_rate_only", "head_start", "endpoint"):
        m = measurability(units, metric)
        print(f"{metric:>18s}  spread {m['between_pair_spread']:.4f}  "
              f"median seed sd {m['median_within_pair_sd']:.4f}  "
              f"S/N {m['signal_to_noise']:.2f}")

    m = measurability(units, args.metric)
    print(f"\nper-pair {args.metric} (mean +- seed sd):")
    for pair in sorted(m["means"]):
        flag = "  HELD OUT" if pair in heldout_pairs else ""
        print(f"  f{pair[0]}->f{pair[1]}  {m['means'][pair]:+.4f} +- "
              f"{m['sds'].get(pair, float('nan')):.4f}{flag}")

    pairs_all = [tuple(p) for p in pairs["observed_pairs"]]
    observed = [p for p in pairs_all if p in units]
    untouched = [p for p in heldout_pairs]
    if len(observed) < len(pairs_all):
        print(f"\nonly {len(observed)}/{len(pairs_all)} observed pairs present; "
              "ladder not fitted")
        return
    if any(p in units for p in untouched):
        raise AssertionError(
            "an untouched pair has been run; it is the adaptive candidate pool "
            "and running it destroys the only unobserved intervention available")

    features = family_features()
    result = lopo_ladder(units, observed, features, args.metric)
    print(f"\nleave-one-pair-out over {len(observed)} observed pairs, "
          f"metric = {args.metric}")
    print(f"{'model':>12s} {'params':>7s} {'in-sample':>10s} {'LOPO RMSE':>10s} {'MAE':>8s}")
    for kind in ("global", "source_only", "target_only", "additive", "cosine", "relational"):
        r = result[kind]
        if r.get("lopo_rmse") is None:
            print(f"{kind:>12s} {r['n_parameters']:>7d} {'-':>10s} {'-':>10s} {'-':>8s}"
                  f"   {r['note']}")
            continue
        print(f"{kind:>12s} {r['n_parameters']:>7d} {r['in_sample_rmse']:>10.4f} "
              f"{r['lopo_rmse']:>10.4f} {r['lopo_mae']:>8.4f}")
    (args.path / "lopo_ladder.json").write_text(json.dumps(result, indent=2) + "\n")

    scored = {k: result[k]["lopo_rmse"] for k in ("global", "source_only", "target_only",
                                                  "additive", "cosine", "relational")
              if result[k].get("lopo_rmse") is not None}
    best = min(scored, key=scored.get)
    baseline = scored["global"]
    print(f"\nbest by LOPO: {best} ({scored[best]:.4f}), "
          f"global mean baseline {baseline:.4f}, "
          f"{100 * (baseline - scored[best]) / baseline:+.1f}% vs global")

    if best == "global":
        print("\nNo model beats the global mean out of sample. The adaptive step is "
              "not licensed: selection would be driven by a predictor with no "
              "demonstrated predictive value.")
        return

    frozen_path = args.path / "frozen_predictions.json"
    payload = freeze_predictions(units, observed, untouched, features, args.metric,
                                 best, scored[best], frozen_path)
    print(f"\nfrozen predictions for the 3 untouched pairs ({payload['model']} model):")
    for name, row in sorted(payload["candidates"].items()):
        mark = "  <- SELECTED" if name == payload["selected"] else ""
        print(f"  {name}  predicted {row['prediction']:+.4f}  "
              f"U_e {row['acquisition_U']:.4f}{mark}")
    print(f"\nwrote {frozen_path}")


if __name__ == "__main__":
    main()
