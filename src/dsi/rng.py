"""Central RNG role discipline.

Every run derives all of its stochastic streams from one root key by
explicit ``fold_in``. Nothing in this project calls ``jax.random.key``
outside this module.

The reason this module exists, and the reason it is tested before any model
code is written, is that the paired experimental unit is defined by exactly
which streams two arms share:

    A transfer pair holds the base checkpoint, the target-phase example
    ordering, and the evaluation protocol identical, and differs in exactly
    one thing: the identity of the source-phase corpus.

    An identity-null pair holds the same streams identical and differs in
    exactly one thing: an independent draw of the *same* source condition.

The null is what calibrates the noise floor for the treatment pair, so the
two must differ in the same place and nowhere else. A null that shared more
than its treatment pair would understate sigma_pair and under-power the
design; one that shared less would overstate it and waste compute. Neither
failure is visible in the resulting numbers, which is why the contracts are
asserted in tests rather than left to convention.
"""

from __future__ import annotations

from enum import IntEnum
from types import MappingProxyType
from typing import Mapping

import jax
import jax.random as jr

__all__ = ["Role", "root_key", "derive", "run_keys", "key_fingerprint"]

KeyArray = jax.Array


class Role(IntEnum):
    """Stream roles, folded in as the first component of every key path.

    These integers are part of the scientific run identity: changing a value
    changes every stream derived under it and silently invalidates
    comparison with previously recorded runs. Append new roles, never
    renumber existing ones.
    """

    INIT = 1
    """Model initialization. Shared across both arms of a matched pair."""

    SOURCE_DATA = 2
    """Source-phase example sampling. The one stream a transfer pair differs on."""

    TARGET_DATA = 3
    """Target-phase example sampling. Shared, so both arms see one ordering."""

    BRANCH = 4
    """Forces divergence between arms that are otherwise identical.

    Used only by identity-null pairs, where the whole point is two
    stochastically independent draws of the same condition.
    """

    EVAL = 5
    """Evaluation sampling. Shared, so both arms face the same eval draw."""

    DROPOUT = 6
    """Any within-training stochasticity. Shared within a pair.

    Sharing this is what makes the two arms differ in their *history* rather
    than in their training noise.
    """


def root_key(seed_family: int) -> KeyArray:
    """The single root key for one seed family.

    A seed family indexes a matched pair, not a run. Both arms of a pair
    derive from this same root; that is what makes them paired.
    """
    if not isinstance(seed_family, int) or isinstance(seed_family, bool):
        raise TypeError(f"seed_family must be an int, got {type(seed_family).__name__}")
    if seed_family < 0:
        raise ValueError(f"seed_family must be non-negative, got {seed_family}")
    return jr.key(seed_family)


def derive(key: KeyArray, *path: int) -> KeyArray:
    """Fold a path of integers into a key, one component at a time.

    Successive ``fold_in`` rather than a combined hash keeps the derivation
    inspectable: any key in the project can be recomputed from its root and
    its integer path, which is what makes the sharing contracts testable.
    """
    if not path:
        raise ValueError("derive requires at least one path component")
    for component in path:
        if not isinstance(component, (int, Role)) or isinstance(component, bool):
            raise TypeError(f"path components must be ints, got {component!r}")
        key = jr.fold_in(key, int(component))
    return key


def key_fingerprint(key: KeyArray) -> tuple[int, ...]:
    """A hashable, comparable value for a key.

    Typed JAX keys do not compare with ``==`` in a way that is useful for
    set membership, and the raw data is what actually determines the stream.
    """
    return tuple(int(v) for v in jr.key_data(key).reshape(-1))


def run_keys(
    seed_family: int,
    *,
    n_phases: int,
    arm: int | None = None,
    n_eval_points: int = 1,
) -> Mapping[str, KeyArray]:
    """All streams for one arm of one seed family, addressed by path.

    Args:
        seed_family: Indexes the matched pair. Both arms pass the same value.
        n_phases: Number of training phases in the run.
        arm: Arm identifier, folded in under :attr:`Role.BRANCH` on the
            source stream only. ``None`` (the default) means the arm does
            not diverge stochastically, which is correct for a transfer
            pair: its arms already differ by training on different corpora,
            so forcing a further RNG difference would confound corpus
            identity with sampling noise. An identity-null pair passes
            distinct integers here, because an independent draw of the same
            condition is precisely what it is measuring.
        n_eval_points: Evaluation points per phase. Must be at least one,
            and the first is the phase-boundary evaluation at zero tokens.

    Returns:
        An immutable mapping from dotted path to key. Keys are distinct by
        construction and asserted so in tests.
    """
    if n_phases < 1:
        raise ValueError(f"n_phases must be at least 1, got {n_phases}")
    if n_eval_points < 1:
        raise ValueError(
            f"n_eval_points must be at least 1, got {n_eval_points}; "
            "the first evaluation is taken at zero tokens into the phase"
        )
    if arm is not None:
        if not isinstance(arm, int) or isinstance(arm, bool):
            raise TypeError(f"arm must be an int or None, got {type(arm).__name__}")
        if arm < 0:
            raise ValueError(f"arm must be non-negative, got {arm}")

    root = root_key(seed_family)
    keys: dict[str, KeyArray] = {"init": derive(root, Role.INIT)}

    for phase in range(n_phases):
        source = derive(root, Role.SOURCE_DATA, phase)
        if arm is not None:
            # Divergence is applied here and only here. Everything else in
            # this mapping is identical across arms of the same pair.
            source = derive(source, Role.BRANCH, arm)
        keys[f"source_data.{phase}"] = source
        keys[f"target_data.{phase}"] = derive(root, Role.TARGET_DATA, phase)
        keys[f"dropout.{phase}"] = derive(root, Role.DROPOUT, phase)
        for point in range(n_eval_points):
            keys[f"eval.{phase}.{point}"] = derive(root, Role.EVAL, phase, point)

    return MappingProxyType(keys)
