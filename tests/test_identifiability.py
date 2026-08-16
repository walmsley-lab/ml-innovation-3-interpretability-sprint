"""Well-posedness of the corrected W/P task.

The first construction was unidentifiable. The two families presented
byte-identical model-visible inputs and differed only in the answer, so no
deterministic predictor could be competent at both: with K balanced,
independent answer classes,

    A_W + A_P <= 1 + 1/K     and     min(A_W, A_P) <= (1 + 1/K)/2,

which at K=4 is 0.625, below the prespecified retention threshold of 0.80.
Gate B was measuring something no model could satisfy.

These tests pin the repair: that the bound really did apply to the old
construction, that an oracle clears both gates under the new one, and that
the mode token is the only thing distinguishing the two explicit families.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dsi.data import (
    CONDITIONS,
    GATE_B_CONDITIONS,
    NEUTRAL,
    USE_P,
    USE_W,
    TaskConfig,
    cue_table,
    sample_batch,
)

TASK = TaskConfig(n_digits=3)
N = 20_000


def _oracle_predictions(batch, config: TaskConfig) -> np.ndarray:
    """An oracle that reads MODE and applies the requested strategy.

    Deliberately deterministic and a pure function of the visible input,
    because that is the class of predictor the impossibility bound covers.
    """
    tokens = np.asarray(batch["tokens"])
    mode = tokens[:, 1]
    digits = tokens[:, 2 + 1 : 2 + 1 + config.n_digits] - config.digit_base
    rule = digits.sum(axis=1) % config.n_classes

    cue_token = tokens[:, 2] - config.cue_base
    lookup = np.zeros(config.n_cues, dtype=np.int64)
    for klass, row in enumerate(np.asarray(cue_table(config))):
        lookup[row] = klass
    cue_class = lookup[cue_token]

    return np.where(mode == USE_P, cue_class, rule)


# --- The bound that invalidated the original construction -----------------


def test_impossibility_bound_holds_for_a_mode_blind_predictor():
    """Any predictor ignoring MODE is capped at (1 + 1/K)/2 on min(A_W, A_P).

    Constructed directly: a predictor that always answers with the rule is
    perfect on W and correct on P only where the two coincide.
    """
    w = sample_batch(jr.key(0), "W_COMPETENCE", TASK, N)
    p = sample_batch(jr.key(1), "P_COMPETENCE", TASK, N)

    # Mode-blind predictor: always apply the rule.
    acc_w = float(np.mean(np.asarray(w["w_answer"]) == np.asarray(w["w_answer"])))
    acc_p = float(np.mean(np.asarray(p["w_answer"]) == np.asarray(p["p_answer"])))

    ceiling = (1.0 + 1.0 / TASK.n_classes) / 2.0
    assert acc_w == pytest.approx(1.0)
    assert acc_p == pytest.approx(1.0 / TASK.n_classes, abs=0.02)
    assert min(acc_w, acc_p) <= ceiling
    assert ceiling == pytest.approx(0.625)


def test_the_bound_was_below_the_prespecified_threshold():
    """0.625 < 0.80, so no capacity could have satisfied the old Gate B.

    Recorded as a test so the reason the threshold was *not* lowered stays
    attached to the code.
    """
    from dsi.calibrate import RegimeCriteria

    ceiling = (1.0 + 1.0 / TASK.n_classes) / 2.0
    assert ceiling < RegimeCriteria().tau_retention


def test_rule_and_cue_agree_exactly_one_time_in_k():
    batch = sample_batch(jr.key(2), "W_COMPETENCE", TASK, N)
    agree = float(jnp.mean(batch["w_answer"] == batch["p_answer"]))
    assert agree == pytest.approx(1.0 / TASK.n_classes, abs=0.02)


# --- Well-posedness of the corrected task ---------------------------------


def test_oracle_achieves_both_competences_simultaneously():
    """The repair, stated as the property that matters.

    One deterministic predictor, one evaluation population per mode, both
    accuracies at 1.0. The 0.625 ceiling is gone.
    """
    accuracies = {}
    for condition, key in (("W_COMPETENCE", "w_answer"), ("P_COMPETENCE", "p_answer")):
        batch = sample_batch(jr.key(3), condition, TASK, N)
        predictions = _oracle_predictions(batch, TASK)
        accuracies[condition] = float(np.mean(predictions == np.asarray(batch[key])))

    assert accuracies["W_COMPETENCE"] == pytest.approx(1.0)
    assert accuracies["P_COMPETENCE"] == pytest.approx(1.0)
    assert min(accuracies.values()) > (1.0 + 1.0 / TASK.n_classes) / 2.0


def test_oracle_beats_the_old_ceiling_on_a_shared_population():
    """Both gates cleared by the same predictor, at tau_retention=0.80."""
    from dsi.calibrate import RegimeCriteria

    tau = RegimeCriteria().tau_retention
    worst = min(
        float(np.mean(_oracle_predictions(b, TASK) == np.asarray(b[k])))
        for b, k in (
            (sample_batch(jr.key(4), "W_COMPETENCE", TASK, N), "w_answer"),
            (sample_batch(jr.key(5), "P_COMPETENCE", TASK, N), "p_answer"),
        )
    )
    assert worst >= tau


# --- Distribution matching -------------------------------------------------


def test_content_tokens_are_identical_between_the_explicit_families():
    """Same key, same content. Only MODE and the answer may differ.

    This is the property whose absence made the original task
    unidentifiable, so it is asserted byte-for-byte rather than in
    distribution.
    """
    w = sample_batch(jr.key(6), "W_EXPLICIT", TASK, N)
    p = sample_batch(jr.key(6), "P_EXPLICIT", TASK, N)

    # Columns 2..seq_len-2 are the cue and the digits; column 1 is MODE.
    content = slice(2, TASK.seq_len - 1)
    assert jnp.array_equal(w["tokens"][:, content], p["tokens"][:, content])
    assert not jnp.array_equal(w["tokens"][:, 1], p["tokens"][:, 1])


def test_mode_is_the_only_explicit_task_identifier():
    w = sample_batch(jr.key(7), "W_EXPLICIT", TASK, N)
    p = sample_batch(jr.key(8), "P_EXPLICIT", TASK, N)

    assert set(np.unique(np.asarray(w["tokens"][:, 1]))) == {USE_W}
    assert set(np.unique(np.asarray(p["tokens"][:, 1]))) == {USE_P}

    # Every other visible position is drawn from the same marginal.
    for column in range(2, TASK.seq_len - 1):
        hist_w = np.bincount(np.asarray(w["tokens"][:, column]), minlength=TASK.vocab_size) / N
        hist_p = np.bincount(np.asarray(p["tokens"][:, column]), minlength=TASK.vocab_size) / N
        assert np.abs(hist_w - hist_p).max() < 0.01, column


def test_cue_carries_no_rule_information_in_either_explicit_family():
    """The controlled dependency is absent in the explicit families."""
    for family in ("W_EXPLICIT", "P_EXPLICIT"):
        batch = sample_batch(jr.key(9), family, TASK, N)
        w, p = np.asarray(batch["w_answer"]), np.asarray(batch["p_answer"])
        joint = np.histogram2d(w, p, bins=[TASK.n_classes, TASK.n_classes])[0]
        expected = joint.sum(0)[None, :] * joint.sum(1)[:, None] / joint.sum()
        assert float(((joint - expected) ** 2 / expected).sum()) < 30.0, family


def test_neutral_aligned_has_the_deliberate_dependency():
    """The aligned family is where cue and rule are meant to agree."""
    batch = sample_batch(jr.key(10), "NEUTRAL_ALIGNED", TASK, N)
    assert bool(jnp.all(batch["w_answer"] == batch["p_answer"]))
    assert set(np.unique(np.asarray(batch["tokens"][:, 1]))) == {NEUTRAL}


def test_neutral_conflict_never_agrees_and_is_neutral_moded():
    batch = sample_batch(jr.key(11), "NEUTRAL_CONFLICT", TASK, N)
    assert not bool(jnp.any(batch["w_answer"] == batch["p_answer"]))
    assert set(np.unique(np.asarray(batch["tokens"][:, 1]))) == {NEUTRAL}


# --- Gate B cannot see the phenomenon --------------------------------------


def test_gate_b_conditions_exclude_the_preference_measurement():
    """Structural, not conventional: the gate has no access to it."""
    assert set(GATE_B_CONDITIONS) == {"W_COMPETENCE", "P_COMPETENCE"}
    assert "NEUTRAL_CONFLICT" in CONDITIONS
    assert "NEUTRAL_CONFLICT" not in GATE_B_CONDITIONS


def test_evaluate_defaults_to_gate_b_conditions_only():
    """Measuring preference must always be a deliberate act."""
    import inspect

    from dsi.eval import evaluate

    default = inspect.signature(evaluate).parameters["conditions"].default
    assert default == GATE_B_CONDITIONS
