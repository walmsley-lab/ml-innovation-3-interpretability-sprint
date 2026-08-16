"""Gate B: select the smallest scientifically adequate regime.

The experimental learner should be the smallest adequate model, not the
largest affordable one. "Adequate" is defined here by neutral criteria only:

* both sources are independently learnable;
* both survive sequential exposure;
* both generalize to held-out compositional structures;
* both acquire slowly enough to be observed developing.

Every one of these is symmetric in W and P. None of them can be improved by a
regime that favours one source over the other, which is the property that
makes the selection neutral with respect to the phenomenon under study.

Nothing in this module refers to the conflict condition or to any order
effect. The sweep that feeds it does not measure conflict at all, so
selecting a regime on the magnitude of the effect the project exists to test
is structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsi.stats import tokens_to_threshold

__all__ = [
    "LearningWindow",
    "learning_window",
    "RegimeCriteria",
    "RegimeCandidate",
    "evaluate_candidate",
    "select_regime",
]


@dataclass(frozen=True)
class LearningWindow:
    """Where a capability was acquired, and how long acquisition took.

    ``R = t90 - t10`` is the developmental resolution of research plan §5.2.
    A capability that appears between two consecutive checkpoints cannot be
    observed developing, and a regime that produces one is unusable for
    checkpoint-level analysis however good its final accuracy.
    """

    t10: float | None
    t90: float | None
    width: float | None
    floor: float
    ceiling: float
    censored: bool

    @property
    def resolved(self) -> bool:
        return not self.censored


def learning_window(t, y, *, min_gain: float = 0.05) -> LearningWindow:
    """Locate the 10% and 90% points of observed capability acquisition.

    Thresholds are placed on the *observed* range rather than on absolute
    accuracy, because the question is how quickly the capability appeared,
    not how good it eventually became. Competence is a separate criterion.

    Args:
        min_gain: A curve that never rises by this much has not acquired
            anything, and its window is censored rather than reported as
            zero. A flat curve at chance would otherwise produce ``width=0``
            and be indistinguishable from instant learning, which is the
            opposite failure.
    """
    values = list(y)
    if len(values) < 2:
        raise ValueError("a learning window needs at least two evaluation points")

    floor, ceiling = values[0], max(values)
    gain = ceiling - floor
    if gain < min_gain:
        return LearningWindow(None, None, None, floor, ceiling, censored=True)

    lo = tokens_to_threshold(t, y, floor + 0.10 * gain, direction="above")
    hi = tokens_to_threshold(t, y, floor + 0.90 * gain, direction="above")
    if lo.censored or hi.censored:
        return LearningWindow(None, None, None, floor, ceiling, censored=True)
    return LearningWindow(
        t10=lo.time, t90=hi.time, width=hi.time - lo.time,
        floor=floor, ceiling=ceiling, censored=False,
    )


@dataclass(frozen=True)
class RegimeCriteria:
    """Prespecified adequacy thresholds. Fixed before the sweep runs."""

    tau_w: float = 0.90
    """Competence on the rule, measured in isolation."""

    tau_p: float = 0.90
    """Competence on the cue, measured in isolation."""

    tau_generalization: float = 0.80
    """Worst-source accuracy on held-out compositional structures."""

    tau_retention: float = 0.80
    """Worst source, worst order, after the full sequential curriculum."""

    min_window: float = 0.0
    """Minimum t90 - t10, in the units of the acquisition curve's x axis."""

    def __post_init__(self) -> None:
        for name in ("tau_w", "tau_p", "tau_generalization", "tau_retention"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1], got {value}")
        if self.min_window < 0:
            raise ValueError(f"min_window must be non-negative, got {self.min_window}")


@dataclass(frozen=True)
class RegimeCandidate:
    """One measured point in the capacity sweep.

    ``retention_worst`` is deliberately a minimum over both sources *and*
    both orders. A statistic that distinguished the orders could encode an
    order preference and would stop being neutral; a minimum over both cannot.
    """

    label: str
    params: int
    tokens: int
    n_digits: int
    d_model: int
    n_layers: int
    learning_rate: float
    steps_per_phase: int

    acc_w: float
    acc_p: float
    generalization_worst: float
    retention_worst: float
    window_w: LearningWindow
    window_p: LearningWindow

    def failures(self, criteria: RegimeCriteria) -> tuple[str, ...]:
        """Which adequacy criteria this candidate misses, in fixed order."""
        reasons: list[str] = []
        if self.acc_w < criteria.tau_w:
            reasons.append(f"A_W={self.acc_w:.3f}<{criteria.tau_w}")
        if self.acc_p < criteria.tau_p:
            reasons.append(f"A_P={self.acc_p:.3f}<{criteria.tau_p}")
        if self.generalization_worst < criteria.tau_generalization:
            reasons.append(
                f"gen={self.generalization_worst:.3f}<{criteria.tau_generalization}"
            )
        if self.retention_worst < criteria.tau_retention:
            reasons.append(
                f"retention={self.retention_worst:.3f}<{criteria.tau_retention}"
            )
        for name, window in (("R_W", self.window_w), ("R_P", self.window_p)):
            if window.censored:
                reasons.append(f"{name}=censored")
            elif window.width < criteria.min_window:
                reasons.append(f"{name}={window.width:.3f}<{criteria.min_window:.3f}")
        return tuple(reasons)

    def is_adequate(self, criteria: RegimeCriteria) -> bool:
        return not self.failures(criteria)


def evaluate_candidate(
    candidate: RegimeCandidate, criteria: RegimeCriteria
) -> tuple[bool, tuple[str, ...]]:
    return candidate.is_adequate(criteria), candidate.failures(criteria)


def select_regime(
    candidates: list[RegimeCandidate], criteria: RegimeCriteria
) -> RegimeCandidate:
    """The smallest adequate candidate.

    Ordered by parameter count, then token budget, then label. The tie-breaks
    are deterministic so that the same sweep always selects the same regime,
    and so the choice cannot drift with dictionary ordering.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    eligible = [c for c in candidates if c.is_adequate(criteria)]
    if not eligible:
        raise ValueError(
            f"no candidate meets the adequacy criteria; {len(candidates)} evaluated. "
            "Gate B has not been passed, and widening the sweep is the response, "
            "not lowering the thresholds."
        )
    return min(eligible, key=lambda c: (c.params, c.tokens, c.label))
