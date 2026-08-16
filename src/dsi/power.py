"""Gate C: turn a measured noise floor into a justified seed count.

Seed count is a result of the pilot, not a convention. The planner takes the
paired standard deviation measured from identity-null runs, the smallest
effect worth detecting, and a power target, and returns the number of pairs
required plus what they will cost.

The direction of the dependency matters. ``delta_min`` is prespecified from
what would be scientifically meaningful; ``sigma_pair`` is measured; ``n``
falls out. Choosing ``n`` first and reporting whatever effect it can detect
inverts the logic and produces a design that is powered for nothing in
particular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize as _opt
from scipy import stats as _sps

__all__ = [
    "PowerPlan", "achieved_power", "required_pairs", "detectable_effect", "plan_power",
]

MAX_PAIRS = 10_000
"""Refuse rather than return an absurd design."""


def achieved_power(n_pairs: int, sigma_pair: float, delta_min: float, alpha: float) -> float:
    """Power of a two-sided paired t-test at ``n_pairs``.

    Uses the noncentral t distribution rather than a normal approximation.
    At the seed counts this project can afford — often fewer than ten pairs —
    the normal approximation overstates power by enough to matter, which
    would mean running an underpowered confirmatory experiment while
    believing it adequate.
    """
    if n_pairs < 2:
        return 0.0
    df = n_pairs - 1
    ncp = delta_min / sigma_pair * math.sqrt(n_pairs)
    crit = _sps.t.ppf(1.0 - alpha / 2.0, df)
    upper = _sps.nct.sf(crit, df, ncp)
    lower = _sps.nct.cdf(-crit, df, ncp)
    return float(upper + lower)


def required_pairs(
    sigma_pair: float, delta_min: float, *, alpha: float = 0.05, power: float = 0.90
) -> int:
    """Smallest number of pairs reaching the power target."""
    _validate(sigma_pair, delta_min, alpha, power)
    for n in range(2, MAX_PAIRS + 1):
        if achieved_power(n, sigma_pair, delta_min, alpha) >= power:
            return n
    raise ValueError(
        f"more than {MAX_PAIRS} pairs needed for power {power} at "
        f"sigma_pair={sigma_pair:.4g} and delta_min={delta_min:.4g}. Either the "
        "measurement is too noisy for the effect to be worth chasing, or "
        "delta_min is smaller than the apparatus can resolve."
    )


def detectable_effect(
    n_pairs: int, sigma_pair: float, *, alpha: float = 0.05, power: float = 0.90
) -> float:
    """The smallest effect ``n_pairs`` can detect at the power target.

    The honest thing to report when the budget fixes ``n`` and the question
    becomes what that buys.
    """
    if n_pairs < 2:
        raise ValueError(f"need at least two pairs, got {n_pairs}")
    _validate(sigma_pair, 1.0, alpha, power)

    # Bracket by doubling rather than with a fixed upper bound. The
    # noncentral t overflows to NaN at large noncentrality, so a wide
    # constant bracket makes the solver evaluate points it cannot use.
    hi = sigma_pair
    for _ in range(64):
        if achieved_power(n_pairs, sigma_pair, hi, alpha) >= power:
            break
        hi *= 2.0
    else:
        raise ValueError(
            f"{n_pairs} pairs cannot reach power {power} at any effect size "
            f"with sigma_pair={sigma_pair:.4g}"
        )
    return float(
        _opt.brentq(
            lambda d: achieved_power(n_pairs, sigma_pair, d, alpha) - power,
            1e-12, hi, xtol=1e-12,
        )
    )


@dataclass(frozen=True)
class PowerPlan:
    """A design, and what it costs."""

    sigma_pair: float
    delta_min: float
    alpha: float
    power_target: float
    required_pairs: int
    achieved_power: float
    runs: int
    tokens: int
    accelerator_seconds: float
    estimated_usd: float

    def render(self) -> str:
        return "\n".join(
            [
                f"sigma_pair        {self.sigma_pair:.4f}",
                f"delta_min         {self.delta_min:.4f}",
                f"alpha             {self.alpha:.3f}",
                f"power target      {self.power_target:.2f}",
                f"required_pairs    {self.required_pairs}",
                f"achieved power    {self.achieved_power:.3f}",
                f"estimated runs    {self.runs}",
                f"estimated tokens  {self.tokens:,}",
                f"accelerator sec   {self.accelerator_seconds:,.0f}",
                f"estimated cost    ${self.estimated_usd:,.2f}",
            ]
        )


def plan_power(
    sigma_pair: float,
    delta_min: float,
    *,
    alpha: float = 0.05,
    power: float = 0.90,
    arms_per_pair: int = 2,
    tokens_per_run: int = 0,
    seconds_per_run: float = 0.0,
    usd_per_accelerator_hour: float = 0.0,
) -> PowerPlan:
    """Full plan: required pairs plus the cost of running them."""
    n = required_pairs(sigma_pair, delta_min, alpha=alpha, power=power)
    runs = n * arms_per_pair
    seconds = runs * seconds_per_run
    return PowerPlan(
        sigma_pair=sigma_pair,
        delta_min=delta_min,
        alpha=alpha,
        power_target=power,
        required_pairs=n,
        achieved_power=achieved_power(n, sigma_pair, delta_min, alpha),
        runs=runs,
        tokens=runs * tokens_per_run,
        accelerator_seconds=seconds,
        estimated_usd=seconds / 3600.0 * usd_per_accelerator_hour,
    )


def _validate(sigma_pair: float, delta_min: float, alpha: float, power: float) -> None:
    if sigma_pair <= 0:
        raise ValueError(
            f"sigma_pair must be positive, got {sigma_pair}. A zero noise floor "
            "means the null calibration failed rather than that no seeds are needed."
        )
    if delta_min <= 0:
        raise ValueError(f"delta_min must be positive, got {delta_min}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie in (0, 1), got {power}")
