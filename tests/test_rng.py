"""The sharing and divergence contracts that define the paired unit.

A pairing bug here is silent: the runs complete, the numbers look
plausible, and every developmental edge downstream is wrong. These tests
are the only thing standing between that failure mode and the results, so
they assert the contracts directly rather than testing the helpers that
implement them.
"""

from __future__ import annotations

import pytest

from dsi.rng import Role, derive, key_fingerprint, root_key, run_keys

SEED_FAMILY = 7
N_PHASES = 3
N_EVAL_POINTS = 4


def _fingerprints(keys):
    return {path: key_fingerprint(k) for path, k in keys.items()}


# --- No key is ever reused ------------------------------------------------


def test_no_key_is_reused_within_a_run():
    keys = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS)
    prints = _fingerprints(keys)
    assert len(set(prints.values())) == len(prints), "duplicate key in one run"


def test_no_key_is_reused_across_seed_families():
    a = _fingerprints(run_keys(1, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS))
    b = _fingerprints(run_keys(2, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS))
    assert set(a.values()).isdisjoint(set(b.values()))


def test_roles_do_not_collide_with_phase_indices():
    """Folding a role then an index must not equal folding the index alone.

    A flat namespace where Role.SOURCE_DATA == some phase index would make
    two conceptually different streams identical.
    """
    root = root_key(SEED_FAMILY)
    for role in Role:
        assert key_fingerprint(derive(root, role, 0)) != key_fingerprint(
            derive(root, 0, role)
        )


# --- Determinism ----------------------------------------------------------


def test_derivation_is_deterministic():
    a = _fingerprints(run_keys(SEED_FAMILY, n_phases=N_PHASES))
    b = _fingerprints(run_keys(SEED_FAMILY, n_phases=N_PHASES))
    assert a == b


# --- The transfer pair contract -------------------------------------------


def test_transfer_pair_shares_everything_except_the_source_corpus():
    """D_i -> D_j against N -> D_j.

    The two arms train on different corpora, so their source *content*
    differs. Every stream, including the source sampling stream, is
    identical: corpus identity is the only difference between the arms, and
    forcing an RNG difference on top of it would confound the two.
    """
    treatment = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS)
    control = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS)

    assert _fingerprints(treatment) == _fingerprints(control)


def test_transfer_pair_shares_init_target_and_eval():
    """Stated explicitly, so the contract survives a refactor of the above."""
    treatment = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS)
    control = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS)

    shared = [p for p in treatment if p.startswith(("init", "target_data", "eval", "dropout"))]
    assert shared, "expected shared streams to exist"
    for path in shared:
        assert key_fingerprint(treatment[path]) == key_fingerprint(control[path]), path


# --- The identity-null pair contract --------------------------------------


def test_null_pair_diverges_on_the_source_draw_alone():
    """N_1 -> D_j against N_2 -> D_j.

    Structurally identical to the transfer pair, differing in exactly the
    same place, so that sigma_pair estimated from nulls is the right
    yardstick for treatment pairs.
    """
    arm0 = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS, arm=0)
    arm1 = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=N_EVAL_POINTS, arm=1)

    assert set(arm0) == set(arm1)
    for path in arm0:
        same = key_fingerprint(arm0[path]) == key_fingerprint(arm1[path])
        if path.startswith("source_data"):
            assert not same, f"{path} must diverge between null arms"
        else:
            assert same, f"{path} must be shared between null arms"


def test_null_pair_shares_the_base_checkpoint_with_its_treatment_pair():
    """The null must not differ from the treatment pair in its init stream.

    If nulls used independent initializations they would measure
    between-init variance, which the matched design cancels, and would
    inflate the required seed count.
    """
    transfer = run_keys(SEED_FAMILY, n_phases=N_PHASES)
    null = run_keys(SEED_FAMILY, n_phases=N_PHASES, arm=0)
    assert key_fingerprint(transfer["init"]) == key_fingerprint(null["init"])


def test_distinct_null_arms_are_pairwise_independent():
    arms = [run_keys(SEED_FAMILY, n_phases=N_PHASES, arm=a) for a in range(4)]
    prints = [key_fingerprint(a["source_data.0"]) for a in arms]
    assert len(set(prints)) == len(prints)


def test_arm_none_is_not_the_same_as_arm_zero():
    """Opting out of divergence must be distinguishable from arm 0.

    Otherwise a transfer arm and a null arm would silently share a source
    stream, and the null would stop being an independent draw.
    """
    shared = run_keys(SEED_FAMILY, n_phases=N_PHASES, arm=None)
    arm0 = run_keys(SEED_FAMILY, n_phases=N_PHASES, arm=0)
    assert key_fingerprint(shared["source_data.0"]) != key_fingerprint(
        arm0["source_data.0"]
    )


# --- Phase and eval addressing --------------------------------------------


def test_phases_get_distinct_streams():
    keys = run_keys(SEED_FAMILY, n_phases=N_PHASES)
    for role in ("source_data", "target_data", "dropout"):
        prints = [key_fingerprint(keys[f"{role}.{p}"]) for p in range(N_PHASES)]
        assert len(set(prints)) == N_PHASES, role


def test_every_phase_carries_a_zero_token_evaluation_stream():
    """The t=0 evaluation is addressable for every phase, including the target.

    Its key must exist before the phase trains, because the evaluation
    happens before any of the phase's tokens are seen.
    """
    keys = run_keys(SEED_FAMILY, n_phases=N_PHASES, n_eval_points=2)
    for phase in range(N_PHASES):
        assert f"eval.{phase}.0" in keys


# --- Input validation -----------------------------------------------------


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_phases"):
        run_keys(SEED_FAMILY, n_phases=0)
    with pytest.raises(ValueError, match="zero tokens"):
        run_keys(SEED_FAMILY, n_phases=1, n_eval_points=0)
    with pytest.raises(ValueError, match="non-negative"):
        run_keys(SEED_FAMILY, n_phases=1, arm=-1)
    with pytest.raises(ValueError, match="non-negative"):
        root_key(-1)
    with pytest.raises(TypeError, match="int"):
        root_key(True)
    with pytest.raises(ValueError, match="at least one path component"):
        derive(root_key(0))
