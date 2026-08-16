"""Invariants of the hash-based Layer-2 train/held-out split.

Scalability fix, not a scientific decision: the value space is too large to
enumerate at n_values=64, so membership is defined by a hash of the tuple.
The guarantees enumeration gave for free now have to be asserted.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dsi.layer2 import FAMILIES, Layer2Config, sample_layer2_batch
from dsi.layer2 import _heldout_mask, _sample_values

CONFIG = Layer2Config()


def test_split_is_deterministic_from_tuple_and_seed():
    values = jr.randint(jr.key(0), (2000, CONFIG.n_fields), 0, CONFIG.n_values)
    assert jnp.array_equal(_heldout_mask(values, CONFIG), _heldout_mask(values, CONFIG))


def test_split_depends_on_the_split_seed_and_nothing_else():
    """Not on training seed, arm, curriculum, or any observed outcome.

    The mask is a pure function of the tuple and the frozen split seed, so
    there is no channel through which an outcome could influence membership.
    """
    values = jr.randint(jr.key(1), (2000, CONFIG.n_fields), 0, CONFIG.n_values)
    other = Layer2Config(split_seed=CONFIG.split_seed + 1)
    assert not jnp.array_equal(_heldout_mask(values, CONFIG), _heldout_mask(values, other))

    import inspect
    source = inspect.getsource(_heldout_mask)
    for forbidden in ("seed=", "arm", "family", "curriculum", "step"):
        assert forbidden not in source.replace("split_seed", ""), forbidden


def test_train_and_heldout_are_strictly_disjoint():
    train = {tuple(r) for r in np.asarray(_sample_values(jr.key(2), CONFIG, 4096, "train")).tolist()}
    heldout = {tuple(r) for r in np.asarray(_sample_values(jr.key(3), CONFIG, 4096, "heldout")).tolist()}
    assert train and heldout
    assert train.isdisjoint(heldout)


def test_sampled_batches_land_on_the_requested_side():
    assert float((~_heldout_mask(_sample_values(jr.key(4), CONFIG, 4096, "train"), CONFIG)).mean()) == 1.0
    assert float(_heldout_mask(_sample_values(jr.key(5), CONFIG, 4096, "heldout"), CONFIG).mean()) == 1.0


def test_split_proportion_is_close_to_intended():
    values = jr.randint(jr.key(6), (60_000, CONFIG.n_fields), 0, CONFIG.n_values)
    observed = float(_heldout_mask(values, CONFIG).mean())
    assert abs(observed - CONFIG.heldout_fraction) < 0.02, observed


def test_the_same_split_function_applies_to_every_family():
    """One partition of the value space, shared by all six families.

    Per-family splits would make a held-out tuple for one family a training
    tuple for another, and any transfer measurement between them would be
    contaminated.
    """
    key = jr.key(7)
    bodies = {}
    for family in FAMILIES:
        batch = sample_layer2_batch(key, family, CONFIG, 512, split="heldout")
        bodies[family] = np.asarray(batch["tokens"][:, 2 : CONFIG.seq_len - 2])
    reference = bodies[FAMILIES[0]]
    for family in FAMILIES[1:]:
        assert np.array_equal(reference, bodies[family]), family


def test_no_materialization_of_the_full_space():
    """64^4 is 16.7M tuples; nothing may enumerate it."""
    assert CONFIG.n_values ** CONFIG.n_fields > 10_000_000
    import inspect
    source = inspect.getsource(_sample_values) + inspect.getsource(_heldout_mask)
    assert "arange(total)" not in source and "n_values**" not in source
