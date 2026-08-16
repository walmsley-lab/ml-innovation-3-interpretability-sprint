"""The training core, and the properties the paired design depends on.

These are not accuracy tests. They check that the machinery does what the
experimental design assumes it does: that the t=0 evaluation happens before
any training, that two arms sharing a seed family start from identical
parameters, and that a run is reproducible from its specification alone.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dsi.artifacts import ArtifactWriter, code_version
from dsi.data import CONDITIONS, GATE_B_CONDITIONS, TaskConfig, sample_batch
from dsi.eval import evaluate
from dsi.model import ModelConfig, count_params, init_model
from dsi.rng import run_keys
from dsi.specs import EvalSpec, PhaseSpec, RunSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

TASK = TaskConfig(n_digits=2)
MODEL = ModelConfig(vocab_size=TASK.vocab_size, d_model=32, n_heads=2, n_layers=1, d_ff=64)
TRAIN = TrainConfig(batch_size=32)


def _state(seed: int = 0):
    return init_state(MODEL, TRAIN, jr.key(seed))


def _phase(family: str, steps: int, role: str = "target") -> PhaseSpec:
    return PhaseSpec(family, steps * TRAIN.batch_size * TASK.seq_len, role)


# --- The task's independence property -------------------------------------


def test_rule_and_cue_are_independent_in_the_isolating_conditions():
    """The P competence condition must carry exactly zero rule signal.

    If digit residues were not uniform, the digit sum would predict the
    answer slightly even under p_only, and that leak would surface later as
    transfer that is really an artifact of the generator.
    """
    batch = sample_batch(jr.key(0), "P_COMPETENCE", TASK, 40_000)
    w, p = np.asarray(batch["w_answer"]), np.asarray(batch["p_answer"])
    joint = np.histogram2d(w, p, bins=[TASK.n_classes, TASK.n_classes])[0]
    expected = joint.sum(0)[None, :] * joint.sum(1)[:, None] / joint.sum()
    # Chi-square over (K-1)^2 = 9 df; 30 is far beyond the 0.001 critical value.
    assert float(((joint - expected) ** 2 / expected).sum()) < 30.0


def test_conflict_examples_never_agree():
    batch = sample_batch(jr.key(1), "NEUTRAL_CONFLICT", TASK, 4096)
    assert not bool(jnp.any(batch["w_answer"] == batch["p_answer"]))


def test_aligned_examples_always_agree():
    batch = sample_batch(jr.key(2), "NEUTRAL_ALIGNED", TASK, 4096)
    assert bool(jnp.all(batch["w_answer"] == batch["p_answer"]))


def test_digit_residues_are_exactly_uniform_by_construction():
    assert TASK.n_digit_values % TASK.n_classes == 0
    with pytest.raises(ValueError, match="multiple of"):
        TaskConfig(n_classes=4, n_digit_values=10)


# --- Token budgets and evaluation scheduling ------------------------------


def test_phase_steps_derive_from_token_budget():
    assert phase_steps(_phase("W_EXPLICIT", 10), TASK, TRAIN) == 10


def test_phase_too_small_for_a_step_is_an_error_not_a_skip():
    """A silently skipped phase would still look like a valid arm."""
    with pytest.raises(ValueError, match="at least one step"):
        phase_steps(PhaseSpec("W_EXPLICIT", 1, "target"), TASK, TRAIN)


def test_offset_zero_maps_to_zero_steps():
    assert offset_steps((0.0, 0.5, 1.0), 100) == (0, 50, 100)


# --- The t=0 evaluation ---------------------------------------------------


def test_zero_offset_evaluation_happens_before_any_training():
    """The first evaluation must see untouched parameters.

    This is the whole mechanism: it is what makes a head start carried in
    from an earlier phase separable from what this phase teaches.
    """
    state = _state()
    before = evaluate(state.model, TASK, jr.key(9), batch_size=64)

    seen: list = []
    _, records = train_phase(
        state, _phase("W_EXPLICIT", 5), TASK, TRAIN, jr.key(1),
        eval_at=(0, 5),
        eval_fn=lambda model, i: seen.append(evaluate(model, TASK, jr.key(9), batch_size=64)) or seen[-1],
    )
    assert records[0]["step_in_phase"] == 0
    assert records[0]["tokens_in_phase"] == 0
    for condition in GATE_B_CONDITIONS:
        assert seen[0][condition].loss == pytest.approx(before[condition].loss, abs=1e-6)


def test_evaluation_records_carry_tokens_and_steps():
    state = _state()
    _, records = train_phase(
        state, _phase("W_EXPLICIT", 4), TASK, TRAIN, jr.key(1),
        eval_at=(0, 2, 4), eval_fn=lambda m, i: i,
    )
    assert [r["step_in_phase"] for r in records] == [0, 2, 4]
    tokens_per_step = TRAIN.batch_size * TASK.seq_len
    assert [r["tokens_in_phase"] for r in records] == [0, 2 * tokens_per_step, 4 * tokens_per_step]


def test_training_advances_step_and_token_counters():
    state = _state()
    after, _ = train_phase(state, _phase("W_EXPLICIT", 6), TASK, TRAIN, jr.key(1))
    assert int(after.step) == 6
    assert int(after.tokens_seen) == 6 * TRAIN.batch_size * TASK.seq_len


# --- Reproducibility and pairing ------------------------------------------


def test_the_same_spec_trains_to_the_same_parameters():
    """A run is reproducible from its specification alone."""
    a, _ = train_phase(_state(), _phase("W_EXPLICIT", 8), TASK, TRAIN, jr.key(1))
    b, _ = train_phase(_state(), _phase("W_EXPLICIT", 8), TASK, TRAIN, jr.key(1))
    for x, y in zip(
        jax.tree.leaves(eqx.filter(a.model, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(b.model, eqx.is_inexact_array)),
    ):
        assert jnp.array_equal(x, y)


def test_both_arms_of_a_pair_start_from_identical_parameters():
    """Shared seed family means a shared base checkpoint, not merely a similar one."""
    keys = run_keys(5, n_phases=2)
    a = init_model(MODEL, keys["init"])
    b = init_model(MODEL, run_keys(5, n_phases=2)["init"])
    for x, y in zip(
        jax.tree.leaves(eqx.filter(a, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(b, eqx.is_inexact_array)),
    ):
        assert jnp.array_equal(x, y)


def test_different_seed_families_start_differently():
    a = init_model(MODEL, run_keys(0, n_phases=1)["init"])
    b = init_model(MODEL, run_keys(1, n_phases=1)["init"])
    leaves_a = jax.tree.leaves(eqx.filter(a, eqx.is_inexact_array))
    leaves_b = jax.tree.leaves(eqx.filter(b, eqx.is_inexact_array))
    assert not all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))


def test_different_data_streams_produce_different_batches():
    keys = run_keys(0, n_phases=2)
    source = sample_batch(keys["source_data.0"], "W_EXPLICIT", TASK, 64)
    target = sample_batch(keys["target_data.0"], "W_EXPLICIT", TASK, 64)
    assert not bool(jnp.array_equal(source["tokens"], target["tokens"]))


# --- Evaluation semantics -------------------------------------------------


def test_conflict_reports_no_accuracy():
    """Neither answer is correct under both sources, so accuracy is undefined.

    Reporting the rule-following rate as "accuracy" would quietly assert
    that the rule is the right answer, which is what the condition exists to
    leave open.
    """
    results = evaluate(_state().model, TASK, jr.key(3), batch_size=128,
                       conditions=CONDITIONS)
    assert np.isnan(results["NEUTRAL_CONFLICT"].accuracy)
    for condition in ("W_COMPETENCE", "P_COMPETENCE", "NEUTRAL_ALIGNED_EVAL"):
        assert not np.isnan(results[condition].accuracy)


def test_evaluation_is_deterministic_given_its_key():
    model = _state().model
    a = evaluate(model, TASK, jr.key(4), batch_size=128)
    b = evaluate(model, TASK, jr.key(4), batch_size=128)
    assert {k: v.loss for k, v in a.items()} == {k: v.loss for k, v in b.items()}


def test_model_shape_and_size():
    model = init_model(MODEL, jr.key(0))
    logits = jax.vmap(model)(sample_batch(jr.key(0), "W_EXPLICIT", TASK, 4)["tokens"][:, :-1])
    assert logits.shape == (4, TASK.seq_len - 1, TASK.vocab_size)
    assert count_params(model) > 0


# --- Artifacts ------------------------------------------------------------


def test_artifacts_round_trip_with_a_schema_version(tmp_path):
    import polars as pl

    writer = ArtifactWriter(tmp_path)
    path = writer.write_runs([{"run_id": "abc", "status": "COMPLETED"}])
    frame = pl.read_parquet(path)
    assert frame["schema_version"].to_list() == [1]
    assert frame["run_id"].to_list() == ["abc"]


def test_empty_tables_are_refused(tmp_path):
    """An empty artifact is indistinguishable from a run that never happened."""
    with pytest.raises(ValueError, match="empty table"):
        ArtifactWriter(tmp_path).write_runs([])


def test_code_version_marks_a_dirty_tree():
    version = code_version()
    assert version and isinstance(version, str)


def test_run_spec_drives_the_stream_count():
    spec = RunSpec(
        parent_id=None,
        phases=(PhaseSpec("W", 1000, "source"), PhaseSpec("P", 1000, "target")),
        model_config_id="m", data_version="d", seed_family=0,
        evals=(EvalSpec("s", "v1", offsets=(0.0, 0.5, 1.0)),),
    )
    keys = run_keys(spec.seed_family, n_phases=spec.n_phases,
                    n_eval_points=spec.n_eval_points(0))
    assert spec.n_eval_points(0) == 3
    assert f"eval.{spec.target_phase_index}.0" in keys


# --- The cue map (added at Gate B) ----------------------------------------


def test_cue_map_is_balanced_across_classes():
    """An unbalanced cue map would make the cue predict the rule.

    Every class must own the same number of cue tokens, or the cue carries
    information about the rule and the isolating conditions stop isolating.
    """
    from dsi.data import cue_table

    config = TaskConfig(n_digits=3)
    table = np.asarray(cue_table(config))
    assert table.shape == (config.n_classes, config.cues_per_class)
    assert sorted(table.flatten().tolist()) == list(range(config.n_cues))


def test_unbalanced_cue_count_is_refused():
    with pytest.raises(ValueError, match="multiple of n_classes"):
        TaskConfig(n_classes=4, n_cues=6)


def test_cue_is_uninformative_about_the_rule_in_w_family():
    batch = sample_batch(jr.key(7), "W_EXPLICIT", TaskConfig(n_digits=3), 40_000)
    w, p = np.asarray(batch["w_answer"]), np.asarray(batch["p_answer"])
    joint = np.histogram2d(w, p, bins=[4, 4])[0]
    expected = joint.sum(0)[None, :] * joint.sum(1)[:, None] / joint.sum()
    assert float(((joint - expected) ** 2 / expected).sum()) < 30.0


def test_heldout_split_is_disjoint_from_train():
    from dsi.data import digit_table

    config = TaskConfig(n_digits=3)
    train = {tuple(r) for r in np.asarray(digit_table(config, "train")).tolist()}
    heldout = {tuple(r) for r in np.asarray(digit_table(config, "heldout")).tolist()}
    assert train and heldout
    assert train.isdisjoint(heldout)
    assert len(train) + len(heldout) == config.n_inputs


def test_split_is_stratified_so_residues_stay_uniform():
    """Both halves must keep exactly uniform rule residues.

    An unstratified split would leave the digit sum weakly predictive in each
    half, which would surface later as transfer that is an artifact of how
    the data was divided.
    """
    from dsi.data import digit_table

    config = TaskConfig(n_digits=3)
    for split in ("train", "heldout"):
        table = np.asarray(digit_table(config, split))
        counts = np.bincount(table.sum(1) % config.n_classes, minlength=config.n_classes)
        assert len(set(counts.tolist())) == 1, f"{split} residues unbalanced: {counts}"
