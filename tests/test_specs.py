"""Canonicalization must be stable, and the t=0 evaluation must be mandatory.

Content addressing is only useful if the same experiment hashes identically
everywhere. These tests cover the three ways that fails in practice:
construction order, float representation, and machine-local information
leaking into a scientific identity.
"""

from __future__ import annotations

import socket

import pytest

from dsi.specs import (
    PHASE_ROLES,
    SPEC_SCHEMA_VERSION,
    EvalSpec,
    PhaseSpec,
    RunSpec,
    canonical_dict,
    canonical_json,
    run_id,
)

CODE_VERSION = "0f1e2d3c"


def _spec(**overrides) -> RunSpec:
    kwargs = dict(
        parent_id=None,
        phases=(PhaseSpec("W", 1000, "source"), PhaseSpec("P", 2000, "target")),
        model_config_id="tiny-v1",
        data_version="wp-v1",
        seed_family=3,
        evals=(EvalSpec("wp_diag", "v1"),),
    )
    kwargs.update(overrides)
    return RunSpec(**kwargs)


# --- Hash stability -------------------------------------------------------


def test_identical_specs_hash_identically():
    assert run_id(_spec(), CODE_VERSION) == run_id(_spec(), CODE_VERSION)


def test_hash_is_independent_of_keyword_order():
    """Construction order is not part of the experiment."""
    a = RunSpec(
        parent_id=None,
        phases=(PhaseSpec("W", 1000, "source"), PhaseSpec("P", 2000, "target")),
        model_config_id="tiny-v1",
        data_version="wp-v1",
        seed_family=3,
        evals=(EvalSpec("wp_diag", "v1"),),
    )
    b = RunSpec(
        evals=(EvalSpec(version="v1", suite_id="wp_diag"),),
        seed_family=3,
        data_version="wp-v1",
        model_config_id="tiny-v1",
        phases=(PhaseSpec(role="source", tokens=1000, family="W"),
                PhaseSpec(role="target", tokens=2000, family="P")),
        parent_id=None,
    )
    assert run_id(a, CODE_VERSION) == run_id(b, CODE_VERSION)


def test_canonical_keys_are_sorted():
    d = canonical_dict(_spec())
    assert list(d) == sorted(d)
    assert list(d["phases"][0]) == sorted(d["phases"][0])


def test_float_representation_does_not_change_the_hash():
    """0.5 written three ways is one experiment."""
    a = _spec(evals=(EvalSpec("s", "v1", offsets=(0.0, 0.5, 1.0)),))
    b = _spec(evals=(EvalSpec("s", "v1", offsets=(0.0, 1.0 / 2.0, 1.0)),))
    c = _spec(evals=(EvalSpec("s", "v1", offsets=(0.0, 0.50000000000000000, 1.0)),))
    assert run_id(a, CODE_VERSION) == run_id(b, CODE_VERSION) == run_id(c, CODE_VERSION)


def test_negative_zero_offset_is_normalized():
    """-0.0 and 0.0 are the same moment in a phase."""
    a = _spec(evals=(EvalSpec("s", "v1", offsets=(0.0, 1.0)),))
    b = _spec(evals=(EvalSpec("s", "v1", offsets=(-0.0, 1.0)),))
    assert run_id(a, CODE_VERSION) == run_id(b, CODE_VERSION)


def test_schema_version_is_part_of_the_identity():
    assert _spec().schema_version == SPEC_SCHEMA_VERSION
    assert run_id(_spec(), CODE_VERSION) != run_id(
        _spec(schema_version=SPEC_SCHEMA_VERSION + 1), CODE_VERSION
    )


def test_code_version_changes_the_run_but_not_the_spec():
    """The same specification under two revisions is two runs, one spec."""
    spec = _spec()
    assert run_id(spec, "aaa") != run_id(spec, "bbb")
    assert canonical_json(spec) == canonical_json(spec)
    assert "aaa" not in canonical_json(spec)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed_family": 4},
        {"arm": 0},
        {"data_version": "wp-v2"},
        {"model_config_id": "tiny-v2"},
        {"parent_id": "deadbeef"},
        {"phases": (PhaseSpec("W", 1001, "source"), PhaseSpec("P", 2000, "target"))},
        {"phases": (PhaseSpec("P", 1000, "source"), PhaseSpec("W", 2000, "target"))},
        {"evals": (EvalSpec("wp_diag", "v2"),)},
    ],
)
def test_every_scientific_field_changes_the_hash(overrides):
    """No field is decorative. Changing any of them is a different run."""
    assert run_id(_spec(), CODE_VERSION) != run_id(_spec(**overrides), CODE_VERSION)


def test_arm_none_and_arm_zero_are_different_runs():
    """Mirrors the RNG contract: opting out of divergence is not arm 0."""
    assert run_id(_spec(arm=None), CODE_VERSION) != run_id(_spec(arm=0), CODE_VERSION)


# --- No machine-local information -----------------------------------------


def test_absolute_paths_are_rejected():
    for bad in ("/Users/someone/data", "~/data", "C:\\data", "\\\\host\\share"):
        with pytest.raises(ValueError, match="absolute path"):
            _spec(data_version=bad)


def test_hostname_is_rejected():
    """Hardware metadata belongs in the artifact, not the run identity."""
    host = socket.gethostname()
    with pytest.raises(ValueError, match="hostname"):
        _spec(model_config_id=host)
    with pytest.raises(ValueError, match="hostname"):
        _spec(model_config_id=host.split(".")[0].upper())


def test_canonical_json_is_pure_ascii():
    """Encoding differences across platforms must not reach the hash."""
    payload = canonical_json(_spec(data_version="wp-v1-naive"))
    assert payload.encode("ascii")


def test_non_finite_floats_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        EvalSpec("s", "v1", offsets=(0.0, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        EvalSpec("s", "v1", offsets=(0.0, float("inf")))


# --- The t=0 evaluation requirement ---------------------------------------


def test_eval_spec_defaults_include_zero_tokens():
    assert EvalSpec("s", "v1").offsets[0] == 0.0


def test_eval_spec_without_zero_offset_is_refused():
    """The requirement is enforced by the type, not by convention."""
    with pytest.raises(ValueError, match="must begin at 0.0"):
        EvalSpec("s", "v1", offsets=(0.5, 1.0))
    with pytest.raises(ValueError, match="must begin at 0.0"):
        EvalSpec("s", "v1", offsets=(1.0,))


def test_offsets_must_be_ascending_distinct_and_in_range():
    with pytest.raises(ValueError, match="ascending"):
        EvalSpec("s", "v1", offsets=(0.0, 1.0, 0.5))
    with pytest.raises(ValueError, match="distinct"):
        EvalSpec("s", "v1", offsets=(0.0, 0.5, 0.5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EvalSpec("s", "v1", offsets=(0.0, 1.5))


def test_target_phase_index_locates_the_measured_phase():
    spec = _spec(
        phases=(
            PhaseSpec("W", 1000, "source"),
            PhaseSpec("P", 2000, "target"),
            PhaseSpec("MIX", 500, "washout"),
        )
    )
    assert spec.target_phase_index == 1
    assert spec.n_phases == 3
    assert spec.total_tokens == 3500


def test_mixture_run_has_no_target_phase():
    spec = _spec(phases=(PhaseSpec("ALL", 1000, "mixture"),))
    assert spec.target_phase_index is None


# --- Phase ordering invariants --------------------------------------------


def test_valid_experimental_shapes_are_accepted():
    shapes = {
        "W-only": (PhaseSpec("W", 1000, "target"),),
        "W->P": (PhaseSpec("W", 1000, "source"), PhaseSpec("P", 1000, "target")),
        "W->P->M": (
            PhaseSpec("W", 1000, "source"),
            PhaseSpec("P", 1000, "target"),
            PhaseSpec("MIX", 1000, "washout"),
        ),
        "A+C->B": (
            PhaseSpec("A", 500, "source"),
            PhaseSpec("C", 500, "source"),
            PhaseSpec("B", 1000, "target"),
        ),
        "mixture": (PhaseSpec("ALL", 1000, "mixture"),),
    }
    for name, phases in shapes.items():
        assert _spec(phases=phases).n_phases == len(phases), name


def test_two_target_phases_are_refused():
    with pytest.raises(ValueError, match="at most one target"):
        _spec(phases=(PhaseSpec("A", 10, "target"), PhaseSpec("B", 10, "target")))


def test_source_after_target_is_refused():
    """Ordering is the object of study, so an ambiguous ordering is refused."""
    with pytest.raises(ValueError, match="only 'washout'"):
        _spec(phases=(PhaseSpec("A", 10, "target"), PhaseSpec("B", 10, "source")))


def test_washout_before_target_is_refused():
    with pytest.raises(ValueError, match="only 'source'"):
        _spec(phases=(PhaseSpec("M", 10, "washout"), PhaseSpec("B", 10, "target")))


def test_mixture_cannot_be_combined_with_a_controlled_phase():
    with pytest.raises(ValueError, match="only 'source'"):
        _spec(phases=(PhaseSpec("M", 10, "mixture"), PhaseSpec("B", 10, "target")))


def test_unknown_role_is_refused():
    assert "pretrain" not in PHASE_ROLES
    with pytest.raises(ValueError, match="role must be one of"):
        PhaseSpec("A", 10, "pretrain")


# --- Immutability and general validation ----------------------------------


def test_specs_are_frozen():
    spec = _spec()
    with pytest.raises(Exception):
        spec.seed_family = 99
    with pytest.raises(Exception):
        spec.phases[0].tokens = 99


def test_lists_are_refused_where_tuples_are_required():
    """Mutable containers cannot be part of an immutable identity."""
    with pytest.raises(TypeError, match="tuple"):
        _spec(phases=[PhaseSpec("W", 10, "target")])
    with pytest.raises(TypeError, match="tuple"):
        EvalSpec("s", "v1", offsets=[0.0, 1.0])


def test_empty_and_degenerate_inputs_are_refused():
    with pytest.raises(ValueError, match="non-empty"):
        _spec(phases=())
    with pytest.raises(ValueError, match="non-empty"):
        _spec(evals=())
    with pytest.raises(ValueError, match="positive"):
        PhaseSpec("W", 0, "target")
    with pytest.raises(ValueError, match="non-negative"):
        _spec(seed_family=-1)
    with pytest.raises(ValueError, match="duplicate eval suite_id"):
        _spec(evals=(EvalSpec("s", "v1"), EvalSpec("s", "v2")))


def test_run_id_requires_a_code_version():
    with pytest.raises(ValueError, match="code_version"):
        run_id(_spec(), "")
    with pytest.raises(TypeError, match="RunSpec"):
        run_id(object(), CODE_VERSION)
