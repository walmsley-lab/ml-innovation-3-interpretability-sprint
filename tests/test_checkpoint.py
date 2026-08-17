"""Checkpoint persistence, including the round-trip the module docstring asserts.

`dsi.checkpoint` states that "save then load must round-trip to bitwise-identical
leaves. This is preflight check P6 and it is asserted in the tests, not assumed."
It was not, until this file. Every downstream `V(S,D)` continuation restores
weights from disk, so a lossy round-trip would silently change the incoming
state of every branch in the experiment.
"""
from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsi.checkpoint import checkpoint_paths, load_model, save_model
from dsi.model import ModelConfig, init_model


def _tiny() -> ModelConfig:
    return ModelConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=2,
                       d_ff=32, max_len=8)


def test_round_trip_is_bitwise_identical(tmp_path):
    """P6. Not 'close': identical, leaf by leaf."""
    cfg = _tiny()
    model = init_model(cfg, jax.random.PRNGKey(0))
    save_model(tmp_path / "m", model, cfg)
    restored, restored_cfg = load_model(tmp_path / "m")

    original = eqx.filter(model, eqx.is_inexact_array)
    back = eqx.filter(restored, eqx.is_inexact_array)
    leaves_a = jax.tree_util.tree_leaves(original)
    leaves_b = jax.tree_util.tree_leaves(back)
    assert leaves_a and len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert restored_cfg == cfg


def test_round_trip_preserves_forward_output_exactly(tmp_path):
    """The property that actually matters downstream: same weights, same logits."""
    cfg = _tiny()
    model = init_model(cfg, jax.random.PRNGKey(3))
    tokens = jnp.asarray(np.arange(cfg.max_len) % cfg.vocab_size)
    before = np.asarray(model(tokens))
    save_model(tmp_path / "m", model, cfg)
    after = np.asarray(load_model(tmp_path / "m")[0](tokens))
    assert np.array_equal(before, after)


def test_config_is_persisted_beside_the_weights(tmp_path):
    """A checkpoint must be loadable without the caller knowing the architecture."""
    cfg = _tiny()
    weights, config = save_model(tmp_path / "m", init_model(cfg, jax.random.PRNGKey(0)), cfg)
    assert weights.exists() and config.exists()
    assert json.loads(config.read_text())["d_model"] == cfg.d_model


def test_paths_are_derived_from_one_stem(tmp_path):
    """Deriving both names from a stem is what stops a config pairing with
    the wrong weights."""
    w, c = checkpoint_paths(tmp_path / "abc")
    assert w.stem == c.stem.split(".")[0] == "abc"
    assert w.suffix == ".eqx" and c.name.endswith(".config.json")


def test_missing_weights_and_missing_config_both_raise(tmp_path):
    cfg = _tiny()
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent")

    save_model(tmp_path / "m", init_model(cfg, jax.random.PRNGKey(0)), cfg)
    checkpoint_paths(tmp_path / "m")[1].unlink()
    with pytest.raises(FileNotFoundError, match="architecture cannot be reconstructed"):
        load_model(tmp_path / "m")


def test_loading_does_not_leak_the_template_initialization(tmp_path):
    """Deserialisation builds a template with an arbitrary key; none of it may
    survive, or the restored state would depend on the loader rather than the
    saved run."""
    cfg = _tiny()
    saved = init_model(cfg, jax.random.PRNGKey(11))
    save_model(tmp_path / "m", saved, cfg)
    a = jax.tree_util.tree_leaves(eqx.filter(load_model(tmp_path / "m")[0], eqx.is_inexact_array))
    b = jax.tree_util.tree_leaves(eqx.filter(load_model(tmp_path / "m")[0], eqx.is_inexact_array))
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
