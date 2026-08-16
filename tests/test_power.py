"""The power planner. Seed count must be derived, not assumed."""

from __future__ import annotations

import pytest

from dsi.power import achieved_power, detectable_effect, plan_power, required_pairs


def test_power_rises_with_pairs():
    powers = [achieved_power(n, 0.04, 0.05, 0.05) for n in range(2, 40)]
    assert powers == sorted(powers)
    assert powers[0] < 0.5 < powers[-1]


def test_power_at_zero_effect_equals_alpha():
    """With no effect, rejection happens at the nominal false-positive rate."""
    assert achieved_power(20, 0.04, 1e-12, 0.05) == pytest.approx(0.05, abs=1e-3)


def test_required_pairs_meets_the_target_and_is_minimal():
    n = required_pairs(0.04, 0.05, alpha=0.05, power=0.90)
    assert achieved_power(n, 0.04, 0.05, 0.05) >= 0.90
    assert achieved_power(n - 1, 0.04, 0.05, 0.05) < 0.90


def test_noisier_measurement_needs_more_pairs():
    assert required_pairs(0.08, 0.05) > required_pairs(0.04, 0.05)


def test_smaller_effects_need_more_pairs():
    assert required_pairs(0.04, 0.01) > required_pairs(0.04, 0.10)


def test_uses_noncentral_t_not_a_normal_approximation():
    """At small n the normal approximation overstates power.

    Believing it would mean running an underpowered confirmatory experiment
    while thinking it adequate, so the difference is asserted rather than
    assumed away.
    """
    from scipy import stats

    n, sigma, delta, alpha = 6, 0.04, 0.05, 0.05
    exact = achieved_power(n, sigma, delta, alpha)
    z = delta / sigma * n**0.5
    normal = stats.norm.sf(stats.norm.isf(alpha / 2) - z)
    assert normal > exact + 0.02


def test_detectable_effect_inverts_the_planner():
    n = 8
    delta = detectable_effect(n, 0.04, power=0.90)
    assert achieved_power(n, 0.04, delta, 0.05) == pytest.approx(0.90, abs=1e-3)
    assert required_pairs(0.04, delta, power=0.90) <= n


def test_plan_reports_runs_tokens_and_cost():
    plan = plan_power(
        0.04, 0.05, tokens_per_run=1_000_000, seconds_per_run=60.0,
        usd_per_accelerator_hour=0.80,
    )
    assert plan.runs == plan.required_pairs * 2
    assert plan.tokens == plan.runs * 1_000_000
    assert plan.estimated_usd == pytest.approx(plan.runs * 60.0 / 3600.0 * 0.80)
    assert plan.achieved_power >= 0.90
    assert "required_pairs" in plan.render()


def test_zero_noise_is_an_error_not_a_free_design():
    """A zero noise floor means null calibration failed."""
    with pytest.raises(ValueError, match="sigma_pair must be positive"):
        required_pairs(0.0, 0.05)


def test_hopeless_designs_are_refused():
    with pytest.raises(ValueError, match="more than"):
        required_pairs(sigma_pair=10.0, delta_min=1e-4)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"power": 1.0}])
def test_invalid_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        required_pairs(0.04, 0.05, **kwargs)
