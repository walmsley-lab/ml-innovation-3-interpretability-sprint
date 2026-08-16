"""Estimators for developmental transfer effects.

This module is Stage 0 of the research plan and Gate A of the stage
sequence. It operates on learning curves as plain arrays and has no
dependency on model, data, or training code, which is why it is validated
before any accelerator time is spent.

Sign convention, used without exception throughout the project:

    A positive effect means the treatment arm acquired the target faster
    than the control arm.

Curves are losses, so "faster" means *lower*. Every estimator here is
therefore written as ``control - treatment``. Getting this backwards would
invert every developmental edge in the project, so it is asserted in tests
rather than left to discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps

__all__ = [
    "aulc",
    "transfer_effect",
    "endpoint_effect",
    "ThresholdCrossing",
    "tokens_to_threshold",
    "threshold_effect",
    "CensoredSummary",
    "summarize_crossings",
    "PairedEstimate",
    "paired_summary",
    "equivalence_verdict",
    "synthetic_curve",
]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _as_curve(t, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a (time, value) learning curve."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError(f"curve must be 1-D, got t.ndim={t.ndim}, y.ndim={y.ndim}")
    if t.shape != y.shape:
        raise ValueError(f"t and y must have equal length, got {t.shape} and {y.shape}")
    if t.size < 2:
        raise ValueError("curve needs at least two points to integrate")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains non-finite values")
    return t, y


def _require_origin(t: np.ndarray) -> None:
    """Require that the curve starts at zero target tokens.

    The target-phase t=0 evaluation is taken *before* any target tokens are
    seen. Without it, AULC conflates a head start carried in from the source
    phase with a genuinely faster acquisition rate, and the two cannot be
    separated after the fact. Estimators that depend on the decomposition
    refuse to run rather than silently return the conflated quantity.
    """
    if t[0] != 0.0:
        raise ValueError(
            "this estimator requires the target-phase t=0 evaluation "
            f"(curve starts at t={t[0]!r}). Evaluate before any target tokens; "
            "the offset/rate decomposition is unrecoverable otherwise."
        )


# --------------------------------------------------------------------------
# Area under the learning curve
# --------------------------------------------------------------------------


def aulc(t, y, *, normalize: bool = False) -> float:
    """Trapezoidal area under a learning curve.

    Args:
        t: Evaluation points, strictly increasing. For transfer measurements
            these are target-phase tokens.
        y: Curve values, conventionally target loss.
        normalize: Divide by the elapsed span, giving a time-averaged value
            in the units of ``y``. The research plan defines the transfer
            estimand as the raw integral, which is correct because the target
            budget is held constant within a cell; normalization is offered
            for reporting across cells with different budgets.
    """
    t, y = _as_curve(t, y)
    area = float(np.trapezoid(y, t))
    if normalize:
        area /= float(t[-1] - t[0])
    return area


def transfer_effect(
    t,
    control,
    treatment,
    *,
    normalize: bool = False,
    baseline_correct: bool = False,
) -> float:
    """The primary transfer metric for one matched pair.

    Implements the research plan's

        T = integral over the target phase of [ L_control(t) - L_treatment(t) ]

    Positive means the treatment arm acquired the target faster.

    Args:
        baseline_correct: Subtract the head start the treatment arm already
            held at zero target tokens, leaving only the difference in
            acquisition *rate*. Requires the t=0 evaluation.

            Which of the two quantities is the estimand is a
            prespecification decision, not a default: crediting the head
            start to transfer is defensible, but it should be a stated
            choice rather than an artifact of the integral. Both are
            reported so the decision can be made once and applied uniformly.
    """
    t, control = _as_curve(t, control)
    _, treatment = _as_curve(t, treatment)

    effect = aulc(t, control, normalize=normalize) - aulc(t, treatment, normalize=normalize)

    if baseline_correct:
        _require_origin(t)
        head_start = float(control[0] - treatment[0])
        if not normalize:
            head_start *= float(t[-1] - t[0])
        effect -= head_start

    return effect


def endpoint_effect(control, treatment) -> float:
    """Secondary metric: difference in final target loss. Positive favors treatment."""
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("curves must be non-empty")
    return float(control[-1] - treatment[-1])


# --------------------------------------------------------------------------
# Threshold crossing, with censoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdCrossing:
    """When a curve first reached a threshold, or that it never did.

    A censored observation is a real observation. It is never silently
    dropped and never imputed as the maximum time; both would bias the
    threshold-time distribution toward whichever arm failed more often.
    """

    time: float | None
    censored: bool

    def __post_init__(self) -> None:
        if self.censored != (self.time is None):
            raise ValueError("censored must be true exactly when time is None")


def tokens_to_threshold(t, y, threshold: float, *, direction: str = "below") -> ThresholdCrossing:
    """First time the curve crosses ``threshold``, linearly interpolated.

    Args:
        direction: ``"below"`` for losses (crossing downward), ``"above"``
            for scores. There is no default that is right for both, so the
            caller states which one the curve is.
    """
    if direction not in ("below", "above"):
        raise ValueError(f"direction must be 'below' or 'above', got {direction!r}")
    t, y = _as_curve(t, y)

    reached = y <= threshold if direction == "below" else y >= threshold
    if not reached.any():
        return ThresholdCrossing(time=None, censored=True)

    i = int(np.argmax(reached))
    if i == 0:
        return ThresholdCrossing(time=float(t[0]), censored=False)

    # Linear interpolation between the last unreached point and the first
    # reached one. Curves are sampled at checkpoints, so the true crossing
    # lies between evaluations; snapping to the checkpoint grid would
    # quantize the estimate at exactly the resolution the effect lives in.
    y0, y1 = float(y[i - 1]), float(y[i])
    t0, t1 = float(t[i - 1]), float(t[i])
    if y1 == y0:
        return ThresholdCrossing(time=t1, censored=False)
    frac = (threshold - y0) / (y1 - y0)
    return ThresholdCrossing(time=t0 + frac * (t1 - t0), censored=False)


def threshold_effect(
    t, control, treatment, threshold: float, *, direction: str = "below"
) -> float | None:
    """Paired tokens-to-threshold difference, or None if either arm is censored.

    Positive means the treatment arm reached the threshold sooner. Returns
    None rather than a number when either arm never crossed: the pair
    carries information about *whether* the threshold was reached, which
    belongs in the censoring summary, not in a difference.
    """
    c = tokens_to_threshold(t, control, threshold, direction=direction)
    x = tokens_to_threshold(t, treatment, threshold, direction=direction)
    if c.censored or x.censored:
        return None
    return float(c.time - x.time)


@dataclass(frozen=True)
class CensoredSummary:
    """Threshold outcomes reported as (probability of reaching, time among reachers).

    A single mean would hide the difference between an arm that reaches the
    threshold more often and one that reaches it faster. The research plan
    requires both be reported.
    """

    n: int
    n_reached: int
    p_reached: float
    times: np.ndarray


def summarize_crossings(crossings) -> CensoredSummary:
    times = np.array(
        [c.time for c in crossings if not c.censored],
        dtype=np.float64,
    )
    n = len(list(crossings))
    return CensoredSummary(
        n=n,
        n_reached=int(times.size),
        p_reached=(float(times.size) / n) if n else float("nan"),
        times=times,
    )


# --------------------------------------------------------------------------
# Paired inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedEstimate:
    """A paired effect with its interval. The pair is the unit of analysis."""

    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    alpha: float


def paired_summary(deltas, *, alpha: float = 0.05) -> PairedEstimate:
    """Mean paired difference with a Student-t interval.

    ``deltas`` are per-pair effects, one per seed family. Each is already a
    difference between matched arms, so the shared component of run-to-run
    variation has cancelled and the residual spread estimates sigma_pair,
    the quantity the power planner consumes.
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(f"deltas must be 1-D, got ndim={d.ndim}")
    if d.size < 2:
        raise ValueError("need at least two pairs to estimate an interval")
    if not np.all(np.isfinite(d)):
        raise ValueError("deltas contain non-finite values")

    n = int(d.size)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    crit = float(_sps.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return PairedEstimate(
        mean=mean,
        sd=sd,
        se=se,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        n=n,
        alpha=alpha,
    )


def equivalence_verdict(estimate: PairedEstimate, delta_min: float) -> str:
    """Classify a paired estimate against the smallest meaningful effect.

    The Claim 1 falsifier is an equivalence bound, not "p > 0.05". Because
    the design is powered against a prespecified ``delta_min``, an interval
    lying wholly inside (-delta_min, +delta_min) is a positive finding of
    no meaningful effect, and is reportable as such. An interval that merely
    happens to contain zero while extending past the bound is uninformative
    and must not be reported as a null.

    Returns one of ``"equivalent"``, ``"different"``, ``"inconclusive"``.
    """
    if delta_min <= 0:
        raise ValueError("delta_min must be positive")
    if estimate.ci_low > -delta_min and estimate.ci_high < delta_min:
        return "equivalent"
    if estimate.ci_low > 0.0 or estimate.ci_high < 0.0:
        return "different"
    return "inconclusive"


# --------------------------------------------------------------------------
# Simulation for estimator validation
# --------------------------------------------------------------------------


def synthetic_curve(
    t,
    *,
    floor: float = 0.2,
    span: float = 1.0,
    rate: float = 3.0,
    offset: float = 0.0,
    noise_sd: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A learning curve with an analytically known implanted effect.

    ``y(t) = floor + span * exp(-rate * t) - offset + noise``

    A constant ``offset`` is deliberately chosen for the bias and coverage
    tests because it makes the true normalized AULC effect exactly
    ``offset``, with no numerical-integration error to disentangle from
    estimator error. Rate effects are produced by varying ``rate`` directly.
    """
    t = np.asarray(t, dtype=np.float64)
    y = floor + span * np.exp(-rate * t) - offset
    if noise_sd:
        if rng is None:
            raise ValueError("noise_sd requires an explicit rng, for reproducibility")
        y = y + rng.normal(0.0, noise_sd, size=t.shape)
    return y
