"""Layer 2: a six-family synthetic corpus over one shared world.

Implements DESIGN_LAYER2.md. Six capability families composed from five
overlapping latent primitives, so that developmental relationships between
families can exist at all. The W/P pair of Layer 1 shares no mechanism, which
is why its transfer structure was degenerate.

    SELECT     read the value at a named position     F1 F2 F4 F5
    MAP        apply the learned lookup L             F1 F3 F6
    AGGREGATE  sum modulo n_classes                   F2 F3
    COMPARE    order relation between two values      F4
    CHAIN      use one value as the index of another  F5 F6

Every record shares one format and one vocabulary:

    [BOS, MODE, k1, v1, k2, v2, ... , SEP, ANSWER]

MODE is the only explicit task identifier, generalizing the Layer-1
identifiability fix. Record bodies are drawn from one generator, so nuisance
statistics are controlled without requiring byte identity, which would
flatten the chained and comparison families.

The primitive-sharing graph is known latent mechanistic structure, not
developmental ground truth: sharing a primitive implies neither positive
transfer nor a developmental edge in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

__all__ = ["Layer2Config", "FAMILIES", "PRIMITIVES", "primitive_graph",
           "sample_layer2_batch", "answer_for"]

PAD, BOS, SEP = 0, 1, 2
_N_SPECIAL = 3

FAMILIES = ("F1_SELECT_MAP", "F2_SELECT_AGG", "F3_MAP_AGG",
            "F4_SELECT_CMP", "F5_CHAIN_SELECT", "F6_CHAIN_MAP")

PRIMITIVES = {
    "F1_SELECT_MAP": ("SELECT", "MAP"),
    "F2_SELECT_AGG": ("SELECT", "AGGREGATE"),
    "F3_MAP_AGG": ("MAP", "AGGREGATE"),
    "F4_SELECT_CMP": ("SELECT", "COMPARE"),
    "F5_CHAIN_SELECT": ("CHAIN", "SELECT"),
    "F6_CHAIN_MAP": ("CHAIN", "MAP"),
}


def primitive_graph() -> dict:
    """Shared-primitive counts between families. Quarantined from discovery.

    A comparison object for evaluating inferred developmental relationships,
    never an input to them, and never itself a claim about transfer.
    """
    return {
        (a, b): len(set(PRIMITIVES[a]) & set(PRIMITIVES[b]))
        for a in FAMILIES for b in FAMILIES if a != b
    }


@dataclass(frozen=True)
class Layer2Config:
    n_classes: int = 8
    n_fields: int = 4
    n_values: int = 64          # multiple of n_classes, so residues are uniform
    n_keys: int = 4
    heldout_fraction: float = 0.25
    split_seed: int = 0
    map_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_values % self.n_classes:
            raise ValueError(
                f"n_values ({self.n_values}) must be a multiple of n_classes "
                f"({self.n_classes}) so aggregate residues are exactly uniform")
        if self.n_fields < 2:
            raise ValueError("need at least two fields for COMPARE and AGGREGATE")

    @property
    def mode_base(self) -> int:
        return _N_SPECIAL

    @property
    def key_base(self) -> int:
        return self.mode_base + len(FAMILIES)

    @property
    def value_base(self) -> int:
        return self.key_base + self.n_keys

    @property
    def answer_base(self) -> int:
        return self.value_base + self.n_values

    @property
    def vocab_size(self) -> int:
        return self.answer_base + self.n_classes

    @property
    def seq_len(self) -> int:
        """[BOS, MODE, (key, value) x n_fields, SEP, ANSWER]"""
        return 4 + 2 * self.n_fields


@lru_cache(maxsize=None)
def _lookup(config: Layer2Config) -> jnp.ndarray:
    """The MAP primitive: a fixed balanced value -> class table."""
    rng = np.random.default_rng(config.map_seed)
    per = config.n_values // config.n_classes
    table = np.repeat(np.arange(config.n_classes), per)
    return jnp.asarray(rng.permutation(table), dtype=jnp.int32)


def answer_for(family: str, values: jnp.ndarray, config: Layer2Config) -> jnp.ndarray:
    """The answer each family computes from the same record body."""
    lookup = _lookup(config)
    v0, v1 = values[:, 0], values[:, 1]
    if family == "F1_SELECT_MAP":
        return lookup[v0]
    if family == "F2_SELECT_AGG":
        return (v0 + v1) % config.n_classes
    if family == "F3_MAP_AGG":
        return (lookup[v0] + lookup[v1]) % config.n_classes
    if family == "F4_SELECT_CMP":
        # Select the larger of two fields, then bucket it. A bare order
        # relation would emit only three classes and leave this family with a
        # visibly different answer marginal from the others, which is a
        # nuisance statistic the design requires to be matched.
        return jnp.where(v0 > v1, v0, v1) % config.n_classes
    index = v0 % config.n_fields
    chained = jnp.take_along_axis(values, index[:, None], axis=1).squeeze(1)
    if family == "F5_CHAIN_SELECT":
        return chained % config.n_classes
    if family == "F6_CHAIN_MAP":
        return lookup[chained]
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def _heldout_mask(values: jnp.ndarray, config: Layer2Config) -> jnp.ndarray:
    """Deterministic hash split over value tuples.

    The value space is too large to enumerate once n_values grows, so the
    train/held-out partition is defined by a hash of the tuple rather than by
    materializing it. Same guarantee as enumeration — a given tuple is always
    on the same side — without the memory.
    """
    weights = jnp.asarray(
        [(2654435761 * (k + 1) + 40503) % (2**32) for k in range(config.n_fields)],
        dtype=jnp.uint32)
    mixed = jnp.sum(values.astype(jnp.uint32) * weights, axis=1)
    mixed = mixed ^ (mixed >> 15)
    mixed = mixed * jnp.uint32(2246822519) + jnp.uint32(config.split_seed)
    mixed = mixed ^ (mixed >> 13)
    return (mixed % jnp.uint32(1000)) < jnp.uint32(round(config.heldout_fraction * 1000))


def _sample_values(key, config: Layer2Config, batch_size: int, split: str) -> jnp.ndarray:
    """Uniform value tuples restricted to one side of the hash split.

    Oversamples and reorders so the accepted tuples come first. With a 25%
    held-out fraction an 8x pool leaves the chance of underfilling
    negligible, and the operation stays vectorized and deterministic.
    """
    if split not in ("train", "heldout"):
        raise ValueError(f"split must be train or heldout, got {split!r}")
    pool = batch_size * 8
    candidates = jr.randint(key, (pool, config.n_fields), 0, config.n_values)
    heldout = _heldout_mask(candidates, config)
    wanted = heldout if split == "heldout" else ~heldout
    order = jnp.argsort(~wanted)          # accepted first, stable
    return candidates[order[:batch_size]].astype(jnp.int32)


def sample_layer2_batch(key, family: str, config: Layer2Config, batch_size: int,
                        *, split: str = "train"):
    """One batch. Record bodies come from a single shared generator."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    values = _sample_values(key, config, batch_size, split)
    answer = answer_for(family, values, config)

    keys_row = jnp.arange(config.n_fields, dtype=jnp.int32)
    body = jnp.stack(
        [jnp.broadcast_to(config.key_base + keys_row, (batch_size, config.n_fields)),
         config.value_base + values], axis=2).reshape(batch_size, -1)
    mode = config.mode_base + FAMILIES.index(family)
    tokens = jnp.concatenate([
        jnp.full((batch_size, 1), BOS, dtype=jnp.int32),
        jnp.full((batch_size, 1), mode, dtype=jnp.int32),
        body.astype(jnp.int32),
        jnp.full((batch_size, 1), SEP, dtype=jnp.int32),
        (config.answer_base + answer)[:, None].astype(jnp.int32),
    ], axis=1)
    return {"tokens": tokens, "answer": answer.astype(jnp.int32),
            "family": family, "mode": mode}
