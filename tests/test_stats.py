"""Gate A: the estimators must recover known effects with calibrated intervals.

Research plan Stage 0. These tests run on CPU and gate every downstream
stage. If they fail, no measurement made later in the project is
interpretable, so this file is the first thing that must pass.

Tolerances are prespecified here rather than tuned to observed output.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsi.stats import (
    PairedEstimate,
    ThresholdCrossing,
    aulc,
    endpoint_effect,
    equivalence_verdict,
    paired_summary,
    summarize_crossings,
    synthetic_curve,
    threshold_effect,
    tokens_to_threshold,
    transfer_effect,
)

# --- Prespecified tolerances ----------------------------------------------

DELTA_TRUE = 0.05  # implanted effect, in loss units
N_PAIRS = 8  # pairs per simulated experiment
N_TRIALS = 2000  # simulated experiments per coverage check
ALPHA = 0.05  # nominal 95% intervals

BIAS_TOL = 0.002  # |E[delta_hat] - delta_true| must be under this
COVERAGE_LO = 0.93  # binomial 3-sigma band around 0.95 at N_TRIALS=2000
COVERAGE_HI = 0.97

GRID = np.linspace(0.0, 1.0, 21)  # target-phase grid, includes t=0


# --- Integration ----------------------------------------------------------


def test_aulc_matches_closed_form_integral():
    """Trapezoidal AULC recovers a known integral to integration tolerance."""
    t = np.linspace(0.0, 1.0, 4001)
    y = synthetic_curve(t, floor=0.2, span=1.0, rate=3.0)
    # integral of 0.2 + exp(-3t) over [0,1] = 0.2 + (1 - e^-3)/3
    expected = 0.2 + (1.0 - np.exp(-3.0)) / 3.0
    assert aulc(t, y) == pytest.approx(expected, abs=1e-6)


def test_aulc_normalize_is_time_average():
    t = np.array([0.0, 2.0])
    y = np.array([1.0, 1.0])
    assert aulc(t, y) == pytest.approx(2.0)
    assert aulc(t, y, normalize=True) == pytest.approx(1.0)


def test_aulc_rejects_malformed_curves():
    with pytest.raises(ValueError, match="strictly increasing"):
        aulc([0.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="equal length"):
        aulc([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="at least two points"):
        aulc([0.0], [1.0])


# --- Sign convention ------------------------------------------------------


def test_positive_effect_means_treatment_learned_faster():
    """The sign convention is asserted, not trusted.

    Inverting this would invert every edge in the developmental graph.
    """
    control = synthetic_curve(GRID, offset=0.0)
    treatment = synthetic_curve(GRID, offset=DELTA_TRUE)  # lower loss

    assert np.all(treatment <= control)
    assert transfer_effect(GRID, control, treatment) > 0
    assert endpoint_effect(control, treatment) > 0
    assert transfer_effect(GRID, treatment, control) < 0


# --- Bias -----------------------------------------------------------------


def test_transfer_effect_recovers_implanted_offset_exactly():
    """A constant offset has an exactly known normalized AULC effect."""
    control = synthetic_curve(GRID, offset=0.0)
    treatment = synthetic_curve(GRID, offset=DELTA_TRUE)
    est = transfer_effect(GRID, control, treatment, normalize=True)
    assert est == pytest.approx(DELTA_TRUE, abs=1e-12)


def test_estimator_is_unbiased_under_noise():
    """E[delta_hat] ~ delta_true across simulated experiments."""
    rng = np.random.default_rng(20260816)
    estimates = [
        _simulate_experiment(rng, DELTA_TRUE, N_PAIRS).mean for _ in range(N_TRIALS)
    ]
    assert abs(float(np.mean(estimates)) - DELTA_TRUE) < BIAS_TOL


def test_null_experiments_center_on_zero():
    """The identity-null case must be centered at zero (Gate C's estimator half)."""
    rng = np.random.default_rng(11235)
    estimates = [
        _simulate_experiment(rng, 0.0, N_PAIRS).mean for _ in range(N_TRIALS)
    ]
    assert abs(float(np.mean(estimates))) < BIAS_TOL


# --- Interval coverage ----------------------------------------------------


@pytest.mark.parametrize("delta_true", [0.0, DELTA_TRUE])
def test_confidence_intervals_achieve_nominal_coverage(delta_true):
    """Nominal 95% intervals must cover the truth about 95% of the time."""
    rng = np.random.default_rng(31415 + int(delta_true * 1000))
    covered = 0
    for _ in range(N_TRIALS):
        est = _simulate_experiment(rng, delta_true, N_PAIRS)
        if est.ci_low <= delta_true <= est.ci_high:
            covered += 1
    coverage = covered / N_TRIALS
    assert COVERAGE_LO <= coverage <= COVERAGE_HI, f"coverage {coverage:.4f}"


def test_pairing_cancels_the_shared_component():
    """Pairing must remove seed-family variation, not merely average it.

    The simulator gives each pair a large shared shift. If the paired
    estimator were computed from unpaired arm means, that shift would
    dominate the standard error. This test would catch a pairing bug that
    the coverage test alone might not.
    """
    rng = np.random.default_rng(2718)
    shared_sd = 0.5  # far larger than DELTA_TRUE
    est = _simulate_experiment(rng, DELTA_TRUE, N_PAIRS, shared_sd=shared_sd)
    assert est.sd < shared_sd / 10.0


# --- Baseline correction / the t=0 requirement ----------------------------


def test_baseline_correction_removes_a_pure_head_start():
    """A constant offset is entirely head start and no rate difference."""
    control = synthetic_curve(GRID, offset=0.0)
    treatment = synthetic_curve(GRID, offset=DELTA_TRUE)

    raw = transfer_effect(GRID, control, treatment, normalize=True)
    corrected = transfer_effect(
        GRID, control, treatment, normalize=True, baseline_correct=True
    )
    assert raw == pytest.approx(DELTA_TRUE)
    assert corrected == pytest.approx(0.0, abs=1e-12)


def test_baseline_correction_retains_a_pure_rate_difference():
    """Curves that start together but separate are all rate, no head start."""
    control = synthetic_curve(GRID, rate=3.0)
    treatment = synthetic_curve(GRID, rate=6.0)  # same t=0 value, learns faster

    assert control[0] == pytest.approx(treatment[0])
    corrected = transfer_effect(
        GRID, control, treatment, normalize=True, baseline_correct=True
    )
    assert corrected > 0.0
    assert corrected == pytest.approx(
        transfer_effect(GRID, control, treatment, normalize=True)
    )


def test_baseline_correction_refuses_a_curve_without_t0():
    """The t=0 evaluation is mandatory for the decomposition, not optional."""
    grid = np.linspace(0.1, 1.0, 10)  # starts after target tokens were seen
    control = synthetic_curve(grid, offset=0.0)
    treatment = synthetic_curve(grid, offset=DELTA_TRUE)

    # The conflated quantity is still computable...
    assert transfer_effect(grid, control, treatment) > 0
    # ...but the decomposition refuses rather than silently returning it.
    with pytest.raises(ValueError, match="t=0 evaluation"):
        transfer_effect(grid, control, treatment, baseline_correct=True)


# --- Threshold censoring --------------------------------------------------


def test_threshold_crossing_is_interpolated_between_checkpoints():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.8, 0.4])
    crossing = tokens_to_threshold(t, y, threshold=0.6, direction="below")
    assert not crossing.censored
    assert crossing.time == pytest.approx(1.5)


def test_threshold_never_reached_is_censored_not_dropped():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.9, 0.85])
    crossing = tokens_to_threshold(t, y, threshold=0.1, direction="below")
    assert crossing.censored
    assert crossing.time is None
    assert threshold_effect(t, y, y, threshold=0.1) is None


def test_threshold_direction_is_explicit():
    t = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])  # a rising score
    assert tokens_to_threshold(t, y, 0.5, direction="above").time == pytest.approx(0.5)
    assert tokens_to_threshold(t, y, 0.5, direction="below").time == pytest.approx(0.0)
    with pytest.raises(ValueError, match="direction"):
        tokens_to_threshold(t, y, 0.5, direction="sideways")


def test_censored_summary_reports_probability_and_times_separately():
    crossings = [
        ThresholdCrossing(1.0, False),
        ThresholdCrossing(3.0, False),
        ThresholdCrossing(None, True),
        ThresholdCrossing(None, True),
    ]
    summary = summarize_crossings(crossings)
    assert summary.n == 4
    assert summary.n_reached == 2
    assert summary.p_reached == pytest.approx(0.5)
    assert summary.times.tolist() == [1.0, 3.0]


def test_threshold_crossing_rejects_inconsistent_censoring():
    with pytest.raises(ValueError, match="censored"):
        ThresholdCrossing(time=1.0, censored=True)
    with pytest.raises(ValueError, match="censored"):
        ThresholdCrossing(time=None, censored=False)


# --- Equivalence, the Claim 1 falsifier -----------------------------------


def test_equivalence_verdict_distinguishes_null_from_uninformative():
    """A tight interval around zero is a finding; a wide one is not."""
    tight = PairedEstimate(0.0, 0.01, 0.005, -0.01, 0.01, 8, ALPHA)
    wide = PairedEstimate(0.0, 0.5, 0.2, -0.40, 0.40, 8, ALPHA)
    real = PairedEstimate(0.2, 0.05, 0.02, 0.16, 0.24, 8, ALPHA)

    assert equivalence_verdict(tight, delta_min=0.05) == "equivalent"
    assert equivalence_verdict(wide, delta_min=0.05) == "inconclusive"
    assert equivalence_verdict(real, delta_min=0.05) == "different"


def test_paired_summary_rejects_underpowered_input():
    with pytest.raises(ValueError, match="at least two pairs"):
        paired_summary([0.1])
    with pytest.raises(ValueError, match="non-finite"):
        paired_summary([0.1, np.nan, 0.2])


# --- Simulator ------------------------------------------------------------


def _simulate_experiment(
    rng: np.random.Generator,
    delta_true: float,
    n_pairs: int,
    *,
    shared_sd: float = 0.5,
    arm_sd: float = 0.02,
) -> PairedEstimate:
    """Simulate one experiment of ``n_pairs`` matched pairs.

    Each pair gets a large shared shift, standing in for seed-family and
    checkpoint variation that a matched design cancels, plus independent
    per-arm measurement noise that it does not.
    """
    deltas = []
    for _ in range(n_pairs):
        shared = rng.normal(0.0, shared_sd)
        control = synthetic_curve(
            GRID, offset=-shared, noise_sd=arm_sd, rng=rng
        )
        treatment = synthetic_curve(
            GRID, offset=delta_true - shared, noise_sd=arm_sd, rng=rng
        )
        deltas.append(transfer_effect(GRID, control, treatment, normalize=True))
    return paired_summary(deltas, alpha=ALPHA)
