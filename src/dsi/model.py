"""A deliberately conventional decoder-only transformer.

The project studies curriculum, not architecture. Nothing here is novel and
nothing here should become novel: token embedding, pre-norm blocks with
RMSNorm, causal attention with RoPE, a GELU feed-forward, and an output
projection.

Model configuration is frozen after capacity calibration (Milestone B). The
defaults below are a placeholder large enough to learn the Milestone A task,
not a calibrated choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = ["ModelConfig", "Transformer", "init_model", "count_params"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_len: int = 32

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must divide evenly into n_heads ({self.n_heads})"
            )
        for name in ("vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def _rope(x: jax.Array, positions: jax.Array) -> jax.Array:
    """Rotary position embedding over a (seq, heads, d_head) array."""
    d_head = x.shape[-1]
    half = d_head // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    cos = jnp.cos(angles)[:, None, :]
    sin = jnp.sin(angles)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Block(eqx.Module):
    """Pre-norm attention + feed-forward, operating on one sequence."""

    norm_attn: eqx.nn.RMSNorm
    norm_ff: eqx.nn.RMSNorm
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, config: ModelConfig, key: jax.Array):
        k_qkv, k_proj, k_in, k_out = jr.split(key, 4)
        self.norm_attn = eqx.nn.RMSNorm(config.d_model)
        self.norm_ff = eqx.nn.RMSNorm(config.d_model)
        self.qkv = eqx.nn.Linear(config.d_model, 3 * config.d_model, use_bias=False, key=k_qkv)
        self.proj = eqx.nn.Linear(config.d_model, config.d_model, use_bias=False, key=k_proj)
        self.ff_in = eqx.nn.Linear(config.d_model, config.d_ff, use_bias=False, key=k_in)
        self.ff_out = eqx.nn.Linear(config.d_ff, config.d_model, use_bias=False, key=k_out)
        self.n_heads = config.n_heads
        self.d_head = config.d_head

    def __call__(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[0]
        positions = jnp.arange(seq_len)

        h = jax.vmap(self.norm_attn)(x)
        qkv = jax.vmap(self.qkv)(h).reshape(seq_len, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = _rope(q, positions), _rope(k, positions)

        scores = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(self.d_head)
        causal = positions[None, :] > positions[:, None]
        scores = jnp.where(causal[None, :, :], -jnp.inf, scores)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hqk,khd->qhd", attn, v).reshape(seq_len, -1)
        x = x + jax.vmap(self.proj)(out)

        h = jax.vmap(self.norm_ff)(x)
        h = jax.vmap(self.ff_out)(jax.nn.gelu(jax.vmap(self.ff_in)(h)))
        return x + h


class Transformer(eqx.Module):
    embed: eqx.nn.Embedding
    blocks: tuple[Block, ...]
    norm: eqx.nn.RMSNorm
    out: eqx.nn.Linear

    def __init__(self, config: ModelConfig, key: jax.Array):
        k_embed, k_out, *k_blocks = jr.split(key, 2 + config.n_layers)
        self.embed = eqx.nn.Embedding(config.vocab_size, config.d_model, key=k_embed)
        self.blocks = tuple(Block(config, k) for k in k_blocks)
        self.norm = eqx.nn.RMSNorm(config.d_model)
        self.out = eqx.nn.Linear(config.d_model, config.vocab_size, use_bias=False, key=k_out)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """One sequence of token ids to logits. vmap over the batch."""
        x = jax.vmap(self.embed)(tokens)
        for block in self.blocks:
            x = block(x)
        return jax.vmap(self.out)(jax.vmap(self.norm)(x))


def init_model(config: ModelConfig, key: jax.Array) -> Transformer:
    """Initialize a model. The only entry point; there is no global state."""
    return Transformer(config, key)


def count_params(model: Transformer) -> int:
    return sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))
