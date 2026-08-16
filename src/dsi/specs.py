"""Immutable run specifications and their canonical content addresses.

The scientific layer generates ``RunSpec`` objects. The executor runs them
and decides nothing about what experiment should exist.

Every run has a stable content address, ``run_id = H(RunSpec,
code_version)``, computed from a canonical form with sorted keys, an
explicit schema version, normalized floats, and no machine-local
information. Two different machines specifying the same experiment must
produce the same identifier, or the artifact store cannot be joined across
them.

Note on seeds, relative to technical.md
---------------------------------------
Section 4 sketches ``RunSpec`` with four independent seeds (``init_seed``,
``data_seed``, ``branch_seed``, ``eval_seed``), while section 6 states that
every ``RunSpec`` receives *one* root key from which all streams are derived
by role. These conflict, and this module implements section 6.

Four independent seeds would provide a second, redundant way to specify the
same streams, and the redundancy is not benign: it makes it possible to vary
``eval_seed`` between two arms of a matched pair while believing them
paired. A single ``seed_family`` with roles folded in centrally makes that
error unrepresentable. Divergence, where an identity-null pair needs it, is
expressed by ``arm`` and applied to the source stream alone.
"""

from __future__ import annotations

import json
import math
import re
import socket
from dataclasses import dataclass, fields, is_dataclass
from hashlib import blake2b
from typing import Any

__all__ = [
    "SPEC_SCHEMA_VERSION",
    "PHASE_ROLES",
    "PhaseSpec",
    "EvalSpec",
    "RunSpec",
    "canonical_dict",
    "canonical_json",
    "run_id",
]

SPEC_SCHEMA_VERSION = 1

PHASE_ROLES = frozenset({"source", "target", "washout", "mixture"})
"""Roles a phase may play in a controlled comparison.

A role states what a phase is *for* in the measurement, not where it sits in
the sequence.

``source``
    Exposure whose effect on later learning is being measured. A source
    phase is the thing an intervention varies.

``target``
    **The phase whose acquisition is being measured.** This is the
    definition, and position is not part of it. The target phase is the one
    the learning curve is drawn from, the one whose zero-token evaluation
    the transfer estimand depends on, and the one whose token budget defines
    the integration window. At most one per run.

``washout``
    Identical post-curriculum training given to both arms, so that final
    diagnostics follow shared experience.

``mixture``
    An uncontrolled phase, for curriculum and baseline runs where no single
    phase is the object of measurement.

A solo control such as W-only or P-only is a single ``target`` phase. That
is the definition applied directly, not a workaround for the ordering rule:
the run exists to measure how that one family is acquired from
initialization, so that family's phase is exactly the phase whose
acquisition is being measured. Labelling it ``source`` would be the error,
because nothing downstream measures the effect of that exposure.
"""

_ABSOLUTE_PATH = re.compile(r"^(/|~|[A-Za-z]:[\\/]|\\\\)")


# --------------------------------------------------------------------------
# Field validation
# --------------------------------------------------------------------------


def _check_identifier(value: str, field_name: str) -> None:
    """Reject empty strings and anything machine-local.

    Absolute paths and hostnames are the two ways local environment leaks
    into a scientific identity. Either would make the same experiment hash
    differently on two machines, which defeats content addressing.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if _ABSOLUTE_PATH.match(value):
        raise ValueError(
            f"{field_name}={value!r} looks like a local absolute path; "
            "run identity must not depend on where the code happens to live"
        )
    host = socket.gethostname()
    candidates = {host.lower(), host.split(".")[0].lower()}
    if value.lower() in candidates:
        raise ValueError(
            f"{field_name}={value!r} is this machine's hostname; "
            "hardware metadata belongs in the artifact, not the run identity"
        )


def _norm_float(x: float, field_name: str) -> str:
    """Normalize a float to a stable decimal string.

    ``repr`` differs in edge cases across platforms and ``-0.0`` and ``0.0``
    are distinct in JSON but identical as experimental quantities. Both
    would produce two hashes for one experiment.
    """
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError(f"{field_name} must be a real number, got {x!r}")
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{field_name} must be finite, got {x!r}")
    if x == 0.0:
        x = 0.0  # collapses -0.0
    return format(x, ".17g")


# --------------------------------------------------------------------------
# Specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSpec:
    """One training phase: a corpus, a token budget, and a role."""

    family: str
    tokens: int
    role: str

    def __post_init__(self) -> None:
        _check_identifier(self.family, "family")
        if not isinstance(self.tokens, int) or isinstance(self.tokens, bool):
            raise TypeError(f"tokens must be an int, got {type(self.tokens).__name__}")
        if self.tokens <= 0:
            raise ValueError(f"tokens must be positive, got {self.tokens}")
        if self.role not in PHASE_ROLES:
            raise ValueError(
                f"role must be one of {sorted(PHASE_ROLES)}, got {self.role!r}"
            )


@dataclass(frozen=True)
class EvalSpec:
    """An evaluation suite and the within-phase points at which it runs.

    ``offsets`` are fractions of the phase's token budget. Offset ``0.0`` is
    mandatory and is the reason this field exists rather than a simple
    end-of-phase evaluation.

    The target-phase evaluation at zero tokens is taken *before* the phase
    has trained on anything. Without it, the area under the learning curve
    conflates a head start the arm carried in from its source phase with a
    genuinely faster acquisition rate. The two cannot be separated
    afterwards at any price short of re-running the matrix, so the
    measurement is required at run time even though nothing in Milestone A
    consumes it analytically yet.
    """

    suite_id: str
    version: str
    offsets: tuple[float, ...] = (0.0, 1.0)

    def __post_init__(self) -> None:
        _check_identifier(self.suite_id, "suite_id")
        _check_identifier(self.version, "version")
        if not isinstance(self.offsets, tuple):
            raise TypeError(
                f"offsets must be a tuple, got {type(self.offsets).__name__}; "
                "specs are immutable so that they can be hashed and shared"
            )
        if not self.offsets:
            raise ValueError("offsets must be non-empty")
        for offset in self.offsets:
            _norm_float(offset, "offset")
            if not 0.0 <= offset <= 1.0:
                raise ValueError(f"offsets must lie in [0, 1], got {offset!r}")
        if list(self.offsets) != sorted(self.offsets):
            raise ValueError(f"offsets must be ascending, got {self.offsets!r}")
        if len(set(self.offsets)) != len(self.offsets):
            raise ValueError(f"offsets must be distinct, got {self.offsets!r}")
        if self.offsets[0] != 0.0:
            raise ValueError(
                f"offsets must begin at 0.0, got {self.offsets!r}. The "
                "target-phase evaluation at zero tokens separates a transfer "
                "head start from a faster acquisition rate and is "
                "unrecoverable after the run."
            )


@dataclass(frozen=True)
class RunSpec:
    """A complete, self-contained description of one training run.

    One arm of a matched pair. Both arms of a pair share ``seed_family``;
    that is what makes them paired. ``arm`` forces stochastic divergence and
    is used only by identity-null pairs, where two independent draws of the
    same condition are the measurement.

    Phases carry roles rather than positions. See :data:`PHASE_ROLES`:
    ``target`` means the phase whose acquisition is being measured, so a
    solo control is a single ``target`` phase rather than a lone ``source``.
    """

    parent_id: str | None
    phases: tuple[PhaseSpec, ...]
    model_config_id: str
    data_version: str
    seed_family: int
    evals: tuple[EvalSpec, ...]
    arm: int | None = None
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.parent_id is not None:
            _check_identifier(self.parent_id, "parent_id")
        _check_identifier(self.model_config_id, "model_config_id")
        _check_identifier(self.data_version, "data_version")

        for name in ("phases", "evals"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
            if not value:
                raise ValueError(f"{name} must be non-empty")

        if not all(isinstance(p, PhaseSpec) for p in self.phases):
            raise TypeError("phases must contain PhaseSpec instances")
        if not all(isinstance(e, EvalSpec) for e in self.evals):
            raise TypeError("evals must contain EvalSpec instances")

        suite_ids = [e.suite_id for e in self.evals]
        if len(set(suite_ids)) != len(suite_ids):
            raise ValueError(f"duplicate eval suite_id in {suite_ids}")

        for name in ("seed_family", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if self.arm is not None:
            if not isinstance(self.arm, int) or isinstance(self.arm, bool):
                raise TypeError(f"arm must be an int or None, got {type(self.arm).__name__}")
            if self.arm < 0:
                raise ValueError(f"arm must be non-negative, got {self.arm}")

        self._check_phase_ordering()

    def _check_phase_ordering(self) -> None:
        """Enforce the shape of a controlled comparison.

        A run either measures acquisition of one target phase, in which case
        it reads ``source* target washout*``, or it is an uncontrolled
        mixture run with no target at all. Anything else, such as two target
        phases or a source after the target, is not a comparison this
        project knows how to interpret.

        ``source*`` admits zero sources, which is what makes a solo control
        a well-formed run: a single ``target`` phase measures acquisition
        from initialization with no prior exposure. A run made only of
        source phases is refused because nothing in it is measured.
        """
        roles = [p.role for p in self.phases]
        n_target = roles.count("target")

        if n_target == 0:
            if set(roles) != {"mixture"}:
                raise ValueError(
                    f"a run with no target phase must be all 'mixture', got {roles}"
                )
            return

        if n_target > 1:
            raise ValueError(f"a run may have at most one target phase, got {roles}")

        target_at = roles.index("target")
        before, after = roles[:target_at], roles[target_at + 1 :]
        if any(r != "source" for r in before):
            raise ValueError(f"only 'source' phases may precede the target, got {roles}")
        if any(r != "washout" for r in after):
            raise ValueError(f"only 'washout' phases may follow the target, got {roles}")

    # -- Derived views -----------------------------------------------------

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    @property
    def target_phase_index(self) -> int | None:
        """Where the measured phase sits, or None for a mixture run.

        This is the phase whose zero-token evaluation the transfer estimand
        depends on.
        """
        for i, phase in enumerate(self.phases):
            if phase.role == "target":
                return i
        return None

    @property
    def total_tokens(self) -> int:
        return sum(p.tokens for p in self.phases)

    def n_eval_points(self, phase_index: int) -> int:
        """Evaluation points for a phase, across all suites.

        Offsets are deduplicated across suites so that the RNG stream count
        matches the number of distinct evaluation moments.
        """
        if not 0 <= phase_index < self.n_phases:
            raise IndexError(f"phase_index {phase_index} out of range")
        return len({o for e in self.evals for o in e.offsets})


# --------------------------------------------------------------------------
# Canonicalization and hashing
# --------------------------------------------------------------------------


def canonical_dict(obj: Any, *, field_name: str = "spec") -> Any:
    """Recursively convert a spec to its canonical, hashable form.

    Floats become normalized decimal strings. This form exists to be
    hashed, not to be a data interchange format; artifacts serialize
    separately.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: canonical_dict(getattr(obj, f.name), field_name=f.name)
            for f in sorted(fields(obj), key=lambda f: f.name)
        }
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _norm_float(obj, field_name)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (tuple, list)):
        return [canonical_dict(v, field_name=field_name) for v in obj]
    if isinstance(obj, dict):
        return {
            str(k): canonical_dict(obj[k], field_name=str(k)) for k in sorted(obj, key=str)
        }
    raise TypeError(f"{field_name} has non-canonicalizable type {type(obj).__name__}")


def canonical_json(spec: Any) -> str:
    """Stable JSON encoding of a spec's canonical form."""
    return json.dumps(
        canonical_dict(spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def run_id(spec: RunSpec, code_version: str, *, digest_size: int = 16) -> str:
    """The content address of a run: ``H(RunSpec, code_version)``.

    ``code_version`` is deliberately not a field of ``RunSpec``: the same
    experiment run against two revisions of the code is the same
    *specification* but not the same *run*, and keeping it outside the spec
    makes that relationship explicit at every call site.
    """
    if not isinstance(spec, RunSpec):
        raise TypeError(f"spec must be a RunSpec, got {type(spec).__name__}")
    _check_identifier(code_version, "code_version")

    payload = json.dumps(
        {"code_version": code_version, "spec": canonical_dict(spec)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()
