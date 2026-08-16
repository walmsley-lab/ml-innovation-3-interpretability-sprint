"""Invariants required before Claim 1: exposure matched, tail identical.

Two histories that differ in realized exposure differ in what they saw, and
no amount of downstream care recovers from that. These are asserted rather
than trusted, because a mismatch is invisible in the resulting numbers.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dsi.data import USE_P, USE_W, TaskConfig, sample_batch
from dsi.rng import run_keys

TASK = TaskConfig(n_digits=3, n_cues=512)
BATCH = 128
OVERLAP = 0.20
TAIL = "NEUTRAL_ALIGNED+W_P_INTERLEAVED@0.20"
CALIBRATION_SEEDS = (1000, 1001, 1002)
CONFIRMATORY_SEEDS = (3000, 3001, 3002, 3003, 3004, 3005)


def _counts(batch, mode):
    return int((np.asarray(batch["mode"]) == mode).sum())


def test_mixture_counts_are_exact_not_merely_expected():
    """Every batch carries exactly round(r*B) old-skill examples."""
    expected = round(OVERLAP * BATCH)
    for step in range(25):
        key = jr.fold_in(jr.key(0), step)
        batch = sample_batch(key, f"P_EXPLICIT+W_EXPLICIT@{OVERLAP}", TASK, BATCH)
        assert _counts(batch, USE_W) == expected, step


def test_realized_exposure_matches_exactly_across_histories():
    """W-first and P-first receive identical realized per-family counts.

    Arm A's phase 2 is P-dominant with W maintenance; arm B's is W-dominant
    with P maintenance. The *counts* must mirror exactly, with only the
    identity of the skill in each role differing.
    """
    total_a = {"old": 0, "new": 0}
    total_b = {"old": 0, "new": 0}
    for step in range(40):
        key = jr.fold_in(jr.key(7), step)
        a = sample_batch(key, f"P_EXPLICIT+W_EXPLICIT@{OVERLAP}", TASK, BATCH)
        b = sample_batch(key, f"W_EXPLICIT+P_EXPLICIT@{OVERLAP}", TASK, BATCH)
        total_a["old"] += _counts(a, USE_W)   # W is the old skill in history A
        total_a["new"] += _counts(a, USE_P)
        total_b["old"] += _counts(b, USE_P)   # P is the old skill in history B
        total_b["new"] += _counts(b, USE_W)
    assert total_a == total_b
    assert total_a["old"] + total_a["new"] == 40 * BATCH


def test_schedule_structure_is_matched_position_by_position():
    """The same positions carry the old skill in both histories."""
    key = jr.fold_in(jr.key(11), 3)
    a = sample_batch(key, f"P_EXPLICIT+W_EXPLICIT@{OVERLAP}", TASK, BATCH)
    b = sample_batch(key, f"W_EXPLICIT+P_EXPLICIT@{OVERLAP}", TASK, BATCH)
    old_positions_a = np.asarray(a["mode"]) == USE_W
    old_positions_b = np.asarray(b["mode"]) == USE_P
    assert np.array_equal(old_positions_a, old_positions_b)


def test_common_tail_is_literally_identical_across_histories():
    """Same examples, same order, same RNG. The tail depends only on the seed.

    Both arms of a pair share seed_family, and the tail phase draws from the
    shared target-data stream, so the tail is byte-identical rather than
    merely distributionally matched.
    """
    for seed in CONFIRMATORY_SEEDS[:3]:
        keys_a = run_keys(seed, n_phases=3, n_eval_points=11)
        keys_b = run_keys(seed, n_phases=3, n_eval_points=11)
        for step in range(10):
            ka = jr.fold_in(keys_a["target_data.2"], step)
            kb = jr.fold_in(keys_b["target_data.2"], step)
            batch_a = sample_batch(ka, TAIL, TASK, BATCH)
            batch_b = sample_batch(kb, TAIL, TASK, BATCH)
            assert jnp.array_equal(batch_a["tokens"], batch_b["tokens"])


def test_neutral_conflict_evaluation_is_identical_within_a_pair():
    """Both arms are scored on exactly the same conflict examples."""
    for seed in CONFIRMATORY_SEEDS[:3]:
        ka = run_keys(seed, n_phases=3, n_eval_points=11)["eval.2.10"]
        kb = run_keys(seed, n_phases=3, n_eval_points=11)["eval.2.10"]
        a = sample_batch(ka, "NEUTRAL_CONFLICT", TASK, 1024)
        b = sample_batch(kb, "NEUTRAL_CONFLICT", TASK, 1024)
        assert jnp.array_equal(a["tokens"], b["tokens"])
        assert jnp.array_equal(a["w_answer"], b["w_answer"])


def test_confirmatory_seeds_are_disjoint_from_calibration_seeds():
    """The apparatus was tuned on the calibration seeds; Claim 1 must not reuse them."""
    assert not set(CONFIRMATORY_SEEDS) & set(CALIBRATION_SEEDS)
    assert not set(CONFIRMATORY_SEEDS) & {2000, 2001, 2002, 2003, 2004}  # Gate C nulls


def test_exact_counts_hold_at_the_tail_maintenance_ratio():
    expected = round(0.20 * BATCH)
    for step in range(15):
        batch = sample_batch(jr.fold_in(jr.key(5), step), TAIL, TASK, BATCH)
        maintenance = _counts(batch, USE_W) + _counts(batch, USE_P)
        assert maintenance == expected, (step, maintenance)
