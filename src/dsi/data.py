"""The synthetic W/P task: two independently learnable information sources.

Research plan Stage 3 needs a task in which a model can learn either of two
strategies, so that which one it settles on is a consequence of its training
history rather than of what was learnable.

    W   an underlying rule over the input
    P   a predictive preference cue

Each example is a fixed-length sequence

    [BOS, CUE, d1 ... dn, SEP, ANSWER]

``W`` is the sum of the digits modulo ``n_classes``. ``P`` is the identity of
the cue token. Both determine the answer perfectly on their own, and neither
requires the other, so a model that has learned only one of them is still at
ceiling on the examples that source explains.

Digit values are ``0 .. n_digit_values-1`` with ``n_digit_values`` an exact
multiple of ``n_classes``. That is not cosmetic: it makes each digit's
residue exactly uniform, so the digit sum carries exactly zero information
about the answer in the P-only family. Base-10 digits under a modulus of 4
would leak a small amount of rule signal into the condition that is supposed
to isolate the cue, and the leak would show up as spurious transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = ["TaskConfig", "Family", "Condition", "Batch", "sample_batch", "FAMILIES", "CONDITIONS"]

PAD, BOS, SEP = 0, 1, 2
_N_SPECIAL = 3


@dataclass(frozen=True)
class TaskConfig:
    """Shape of the W/P task. Frozen, and part of the data version."""

    n_classes: int = 4
    n_digits: int = 4
    n_digit_values: int = 8

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {self.n_classes}")
        if self.n_digits < 1:
            raise ValueError(f"n_digits must be at least 1, got {self.n_digits}")
        if self.n_digit_values % self.n_classes != 0:
            raise ValueError(
                f"n_digit_values ({self.n_digit_values}) must be a multiple of "
                f"n_classes ({self.n_classes}) so that digit residues are exactly "
                "uniform; otherwise the rule leaks into the cue-isolation condition"
            )

    # Token layout: specials, cues, digits, answers.
    @property
    def cue_base(self) -> int:
        return _N_SPECIAL

    @property
    def digit_base(self) -> int:
        return self.cue_base + self.n_classes

    @property
    def answer_base(self) -> int:
        return self.digit_base + self.n_digit_values

    @property
    def vocab_size(self) -> int:
        return self.answer_base + self.n_classes

    @property
    def seq_len(self) -> int:
        """[BOS, CUE, digits..., SEP, ANSWER]"""
        return 4 + self.n_digits

    @property
    def answer_position(self) -> int:
        """Index of the answer token in the full sequence."""
        return self.seq_len - 1

    @property
    def answer_target_index(self) -> int:
        """Index of the answer within the *shifted* target sequence.

        Training predicts ``seq[1:]`` from ``seq[:-1]``, so the answer sits
        one place earlier in the target array than in the sequence.
        """
        return self.answer_position - 1


# Training families: which source is informative in the data itself.
FAMILIES = ("W", "P", "MIX")

# Evaluation conditions: the diagnostic suite of research plan Stage 3.
CONDITIONS = ("aligned", "w_only", "p_only", "conflict")


Family = str
Condition = str


class Batch(dict):
    """A batch of examples plus the answers each source implies.

    Carrying both ``w_answer`` and ``p_answer`` on every batch is what makes
    the conflict condition measurable: the question is not whether the model
    is correct, but which of two correct-by-some-source answers it gives.
    """

    tokens: jax.Array
    w_answer: jax.Array
    p_answer: jax.Array


def _assemble(config: TaskConfig, cue: jax.Array, digits: jax.Array, answer: jax.Array) -> jax.Array:
    batch = cue.shape[0]
    return jnp.concatenate(
        [
            jnp.full((batch, 1), BOS, dtype=jnp.int32),
            (config.cue_base + cue)[:, None].astype(jnp.int32),
            (config.digit_base + digits).astype(jnp.int32),
            jnp.full((batch, 1), SEP, dtype=jnp.int32),
            (config.answer_base + answer)[:, None].astype(jnp.int32),
        ],
        axis=1,
    )


def sample_batch(key: jax.Array, kind: str, config: TaskConfig, batch_size: int) -> Batch:
    """Sample a batch from a training family or an evaluation condition.

    Args:
        kind: One of :data:`FAMILIES` or :data:`CONDITIONS`.

            ``W``         rule informative, cue uniform random
            ``P``         cue informative, rule uniform random
            ``MIX``       both informative and agreeing
            ``aligned``   as ``MIX``; the ordinary case W(x) = P(x)
            ``w_only``    clean-rule diagnostic: cue uninformative
            ``p_only``    cue-isolation diagnostic: rule uninformative
            ``conflict``  W(x) != P(x); neither answer is privileged
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    k_digits, k_cue = jr.split(key)
    digits = jr.randint(k_digits, (batch_size, config.n_digits), 0, config.n_digit_values)
    w = jnp.sum(digits, axis=1) % config.n_classes

    if kind in ("W", "w_only"):
        cue = jr.randint(k_cue, (batch_size,), 0, config.n_classes)
        answer = w
    elif kind in ("P", "p_only"):
        cue = jr.randint(k_cue, (batch_size,), 0, config.n_classes)
        answer = cue
    elif kind in ("MIX", "aligned"):
        cue = w
        answer = w
    elif kind == "conflict":
        # Offset by 1..n_classes-1 so the cue never coincides with the rule.
        offset = 1 + jr.randint(k_cue, (batch_size,), 0, config.n_classes - 1)
        cue = (w + offset) % config.n_classes
        # The answer token written into the sequence is never scored in this
        # condition; both w_answer and p_answer are read out instead.
        answer = w
    else:
        raise ValueError(
            f"unknown kind {kind!r}; expected one of {FAMILIES + CONDITIONS}"
        )

    batch = Batch()
    batch["tokens"] = _assemble(config, cue, digits, answer)
    batch["w_answer"] = w.astype(jnp.int32)
    batch["p_answer"] = cue.astype(jnp.int32)
    return batch
