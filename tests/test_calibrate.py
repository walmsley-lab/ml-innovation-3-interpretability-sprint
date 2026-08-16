"""Gate B's selection rule, and the neutrality property it must have.

The selection rule is a pure function over measured records, so it is tested
without training anything. The point of the tests is less that it picks the
smallest passing candidate — that is easy — than that it cannot be steered by
the phenomenon under study.
"""

from __future__ import annotations

import dataclasses

import pytest

from dsi.calibrate import (
    LearningWindow,
    RegimeCandidate,
    RegimeCriteria,
    learning_window,
    select_regime,
)

CRITERIA = RegimeCriteria(
    tau_w=0.90, tau_p=0.90, tau_generalization=0.80, tau_retention=0.80, min_window=10.0
)

GOOD_WINDOW = LearningWindow(t10=10.0, t90=40.0, width=30.0, floor=0.25, ceiling=1.0, censored=False)
NARROW_WINDOW = LearningWindow(t10=10.0, t90=12.0, width=2.0, floor=0.25, ceiling=1.0, censored=False)
CENSORED_WINDOW = LearningWindow(None, None, None, floor=0.25, ceiling=0.26, censored=True)


def _candidate(label="base", params=100_000, **overrides) -> RegimeCandidate:
    kwargs = dict(
        label=label, params=params, tokens=1_000_000, n_digits=3, d_model=64,
        n_layers=2, learning_rate=1e-3, steps_per_phase=400, n_cues=256,
        seeds=(1000,),
        acc_w=0.99, acc_p=0.99, generalization_worst=0.95,
        retention_pre_washout=0.95, retention_post_washout=0.95,
        window_w=GOOD_WINDOW, window_p=GOOD_WINDOW,
    )
    kwargs.update(overrides)
    return RegimeCandidate(**kwargs)


# --- Learning window ------------------------------------------------------


def test_window_locates_ten_and_ninety_percent_of_acquisition():
    t = [0, 10, 20, 30, 40]
    y = [0.0, 0.25, 0.50, 0.75, 1.0]  # linear acquisition
    window = learning_window(t, y)
    assert window.t10 == pytest.approx(4.0)
    assert window.t90 == pytest.approx(36.0)
    assert window.width == pytest.approx(32.0)


def test_window_thresholds_are_relative_to_observed_range():
    """A capability that tops out at 0.6 still has a measurable window.

    Whether it got good enough is competence, a separate criterion. Mixing
    the two would make a slow, mediocre learner indistinguishable from a fast
    one.
    """
    t = [0, 1, 2, 3, 4]
    window = learning_window(t, [0.2, 0.3, 0.4, 0.5, 0.6])
    assert not window.censored
    assert window.ceiling == pytest.approx(0.6)


def test_flat_curve_is_censored_not_zero_width():
    """Never learning and learning instantly are opposite failures.

    Both would produce width 0 if the floor were not checked, and the
    selection rule would then reject the good regime and accept the dead one.
    """
    t = [0, 1, 2, 3]
    assert learning_window(t, [0.25, 0.25, 0.25, 0.25]).censored
    assert learning_window(t, [0.25, 0.26, 0.25, 0.27]).censored


def test_instant_acquisition_has_a_narrow_but_uncensored_window():
    t = [0, 1, 2, 3]
    window = learning_window(t, [0.0, 1.0, 1.0, 1.0])
    assert not window.censored
    assert window.width < 1.0


def test_window_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least two"):
        learning_window([0], [0.5])


# --- Adequacy -------------------------------------------------------------


def test_adequate_candidate_has_no_failures():
    assert _candidate().is_adequate(CRITERIA)
    assert _candidate().failures(CRITERIA) == ()


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"acc_w": 0.5}, "A_W"),
        ({"acc_p": 0.5}, "A_P"),
        ({"generalization_worst": 0.4}, "gen"),
        ({"retention_pre_washout": 0.3}, "retention_pre"),
        ({"window_w": NARROW_WINDOW}, "R_W"),
        ({"window_p": NARROW_WINDOW}, "R_P"),
        ({"window_w": CENSORED_WINDOW}, "R_W=censored"),
    ],
)
def test_each_criterion_can_reject_independently(overrides, expected):
    candidate = _candidate(**overrides)
    assert not candidate.is_adequate(CRITERIA)
    assert any(expected in reason for reason in candidate.failures(CRITERIA))


def test_memorizing_regime_is_rejected_by_generalization():
    """High training accuracy with low held-out accuracy must not pass."""
    memorizer = _candidate(acc_w=1.0, acc_p=1.0, generalization_worst=0.30)
    assert not memorizer.is_adequate(CRITERIA)


def test_forgetting_regime_is_rejected_by_retention():
    """The smoke regime's failure mode, now a first-class rejection."""
    forgetter = _candidate(acc_w=1.0, acc_p=1.0, retention_pre_washout=0.25)
    assert not forgetter.is_adequate(CRITERIA)


# --- Selection ------------------------------------------------------------


def test_selects_the_smallest_adequate_candidate():
    small_bad = _candidate("small", params=1_000, acc_w=0.2)
    medium_ok = _candidate("medium", params=50_000)
    large_ok = _candidate("large", params=900_000)
    chosen = select_regime([large_ok, small_bad, medium_ok], CRITERIA)
    assert chosen.label == "medium"


def test_selection_is_deterministic_under_reordering():
    a = _candidate("a", params=50_000)
    b = _candidate("b", params=50_000)
    assert select_regime([a, b], CRITERIA).label == select_regime([b, a], CRITERIA).label


def test_ties_break_on_tokens_before_label():
    cheap = _candidate("z_cheap", params=50_000, tokens=100)
    dear = _candidate("a_dear", params=50_000, tokens=999)
    assert select_regime([dear, cheap], CRITERIA).label == "z_cheap"


def test_no_adequate_candidate_is_an_error_not_a_fallback():
    """Gate B failing must not silently return the least-bad regime.

    Returning a best-effort choice would let an inadequate regime carry the
    confirmatory experiment while looking like a passed gate.
    """
    with pytest.raises(ValueError, match="no candidate meets"):
        select_regime([_candidate(acc_w=0.1), _candidate(acc_p=0.1)], CRITERIA)
    with pytest.raises(ValueError, match="no candidates"):
        select_regime([], CRITERIA)


# --- Neutrality -----------------------------------------------------------


def test_candidates_carry_no_conflict_or_order_measurement():
    """The record cannot express the thing selection must not use.

    Neutrality is enforced by the shape of the data, not by reviewer
    discipline: there is no field here for conflict behaviour or for an order
    effect, so no selection rule written against this type can consult one.
    """
    field_names = {f.name for f in dataclasses.fields(RegimeCandidate)}
    for forbidden in ("conflict", "follows_w", "follows_p", "order", "delta"):
        assert not any(forbidden in name for name in field_names), forbidden


def test_retention_is_symmetric_across_sources_and_orders():
    """One worst-case number cannot encode a preference between orders.

    If retention were stored per order, a regime could be chosen because it
    made one ordering look better, which is exactly the contamination the
    neutrality requirement forbids.
    """
    field_names = {f.name for f in dataclasses.fields(RegimeCandidate)}
    assert {"retention_pre_washout", "retention_post_washout"} <= field_names
    assert not {"retention_w_then_p", "retention_p_then_w"} & field_names


def test_post_washout_retention_is_recorded_but_not_used_for_adequacy():
    """Adequacy is a property of the learner, not of what MIX restored."""
    masked = _candidate(retention_pre_washout=0.2, retention_post_washout=0.99)
    assert not masked.is_adequate(CRITERIA)
    assert any("retention_pre" in f for f in masked.failures(CRITERIA))

    held = _candidate(retention_pre_washout=0.95, retention_post_washout=0.2)
    assert held.is_adequate(CRITERIA)


def test_worst_case_over_seeds_takes_the_least_favourable_value():
    """A regime adequate on one seed and not another is not adequate."""
    from dsi.calibrate import worst_case_over_seeds

    lucky = _candidate(acc_w=0.99, retention_pre_washout=0.95, seeds=(1000,))
    unlucky = _candidate(acc_w=0.55, retention_pre_washout=0.30, seeds=(1001,))
    worst = worst_case_over_seeds([lucky, unlucky])

    assert worst.acc_w == 0.55
    assert worst.retention_pre_washout == 0.30
    assert worst.seeds == (1000, 1001)
    assert not worst.is_adequate(CRITERIA)


def test_worst_case_propagates_a_censored_window():
    from dsi.calibrate import worst_case_over_seeds

    ok = _candidate(window_p=GOOD_WINDOW, seeds=(1000,))
    censored = _candidate(window_p=CENSORED_WINDOW, seeds=(1001,))
    assert worst_case_over_seeds([ok, censored]).window_p.censored


def test_worst_case_refuses_to_mix_different_regimes():
    from dsi.calibrate import worst_case_over_seeds

    with pytest.raises(ValueError, match="share a label"):
        worst_case_over_seeds([_candidate("a"), _candidate("b")])
