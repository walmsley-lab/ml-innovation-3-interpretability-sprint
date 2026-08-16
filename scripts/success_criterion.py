"""The frozen prospective success criterion. Written before any outcome.

Kept in one place, and in code rather than only in prose, so the gate cannot
drift between what was promised and what is computed.

The central rule: **a relational-developmental claim must beat the best
simpler model, not merely the global mean.** A relational model that beats
the global mean while a source-only model predicts better has demonstrated
that something about the source matters, which is a main effect, and nothing
about the relationship between source and target. Requiring only "beats
global" would let exactly that pass, and on this project it already nearly
did: source-only reached a 13.8% LOPO gain on the same matrix where the
relational model reached 1.5%.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MATERIAL_IMPROVEMENT", "SIMPLER_MODELS", "relational_claim_passes"]

MATERIAL_IMPROVEMENT = 0.25

# Every model that is NOT a claim about the source-target relationship.
# The relational claim must beat the best of these.
SIMPLER_MODELS = ("global", "source_only", "target_only", "additive")


def _rmse(predictions, observed) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(observed)) ** 2)))


def relational_claim_passes(prospective: dict, observed: dict, noise_floor: dict,
                            components=("head_start", "T_aulc_rate_only"),
                            relational: str = "relational") -> dict:
    """Evaluate the frozen criterion on a confirmatory pool.

    Args:
        prospective: ``{component: {model: {pair: prediction}}}``.
        observed: ``{component: {pair: observed mean}}``.
        noise_floor: ``{component: median within-pair seed sd}``.

    Every condition must hold, on **both** components, under the **same**
    model:

    1. relational RMSE <= 0.75 x the RMSE of the **best simpler model**;
    2. that margin survives leaving out any single confirmatory pair;
    3. the RMSE reduction exceeds the component's noise floor.
    """
    result = {"components": {}, "passes": True}
    for component in components:
        pairs = sorted(observed[component])
        truth = [observed[component][p] for p in pairs]
        rmses = {m: _rmse([prospective[component][m][p] for p in pairs], truth)
                 for m in SIMPLER_MODELS + (relational,)}
        best_simple = min(SIMPLER_MODELS, key=lambda m: rmses[m])
        target = rmses[best_simple] * (1.0 - MATERIAL_IMPROVEMENT)
        material = rmses[relational] <= target
        above_floor = (rmses[best_simple] - rmses[relational]) > noise_floor[component]

        jackknife = []
        for drop in range(len(pairs)):
            keep = [p for i, p in enumerate(pairs) if i != drop]
            kept_truth = [observed[component][p] for p in keep]
            r = _rmse([prospective[component][relational][p] for p in keep], kept_truth)
            b = min(_rmse([prospective[component][m][p] for p in keep], kept_truth)
                    for m in SIMPLER_MODELS)
            jackknife.append(r <= b * (1.0 - MATERIAL_IMPROVEMENT))
        robust = all(jackknife)

        entry = {"rmse": rmses, "best_simpler_model": best_simple,
                 "required_rmse": target, "material": bool(material),
                 "jackknife_robust": bool(robust), "above_noise_floor": bool(above_floor),
                 "passes": bool(material and robust and above_floor)}
        result["components"][component] = entry
        result["passes"] &= entry["passes"]

    result["passes"] = bool(result["passes"])
    result["rule"] = (
        f"relational RMSE <= {1 - MATERIAL_IMPROVEMENT:.2f} x best of "
        f"{SIMPLER_MODELS}, jackknife-robust, above the seed-noise floor, on "
        f"BOTH components under one model, prospectively on a frozen pool")
    return result
