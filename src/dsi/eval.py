"""Diagnostic evaluation over the W/P conditions.

The competence gates of research plan Stage 3 are enforced here in the sense
that they are *measured* here: ``w_only`` and ``p_only`` accuracies are what
later decide whether a conflict result may be interpreted at all. A model
that follows the cue because it never learned the rule is not an instance of
path dependence, and the only way to tell the difference is to have measured
both competences on the same checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from dsi.data import CONDITIONS, GATE_B_CONDITIONS, TaskConfig, sample_batch
from dsi.model import Transformer

__all__ = ["ConditionResult", "evaluate_condition", "evaluate"]


@dataclass(frozen=True)
class ConditionResult:
    """Answer-position metrics for one diagnostic condition.

    ``follows_w`` and ``follows_p`` are only meaningful under
    ``NEUTRAL_CONFLICT``, where the two sources disagree and the question is
    which one the model produces. Elsewhere they are recorded but coincide.
    """

    condition: str
    split: str
    loss: float
    accuracy: float
    follows_w: float
    follows_p: float
    n_examples: int


@eqx.filter_jit
def _answer_logprobs(model: Transformer, tokens: jax.Array, answer_index: int) -> jax.Array:
    """Log-probabilities over the answer position for a batch."""
    logits = jax.vmap(model)(tokens[:, :-1])
    return jax.nn.log_softmax(logits[:, answer_index, :], axis=-1)


def evaluate_condition(
    model: Transformer,
    condition: str,
    config: TaskConfig,
    key: jax.Array,
    batch_size: int,
    *,
    split: str = "train",
) -> ConditionResult:
    batch = sample_batch(key, condition, config, batch_size, split=split)
    logprobs = _answer_logprobs(model, batch["tokens"], config.answer_target_index)

    w_token = config.answer_base + batch["w_answer"]
    p_token = config.answer_base + batch["p_answer"]
    lp_w = jnp.take_along_axis(logprobs, w_token[:, None], axis=1).squeeze(1)
    lp_p = jnp.take_along_axis(logprobs, p_token[:, None], axis=1).squeeze(1)

    predicted = jnp.argmax(logprobs, axis=-1)
    follows_w = jnp.mean(predicted == w_token)
    follows_p = jnp.mean(predicted == p_token)

    if condition == "NEUTRAL_CONFLICT":
        # No answer is correct under both sources, so "accuracy" is not
        # defined. Reporting the W-following rate here would quietly assert
        # that the rule is the right answer, which is the very thing the
        # condition is meant to leave open.
        loss = -jnp.mean(jnp.logaddexp(lp_w, lp_p))
        accuracy = jnp.array(jnp.nan)
    else:
        # Under USE_P the cue map determines the answer; otherwise the rule
        # does. Both are read from the batch rather than recomputed.
        target = config.answer_base + jnp.where(
            jnp.asarray(condition in ("P_COMPETENCE", "P_EXPLICIT")),
            batch["p_answer"], batch["w_answer"],
        )
        loss = -jnp.mean(jnp.take_along_axis(logprobs, target[:, None], axis=1))
        accuracy = jnp.mean(predicted == target)

    return ConditionResult(
        condition=condition,
        split=split,
        loss=float(loss),
        accuracy=float(accuracy),
        follows_w=float(follows_w),
        follows_p=float(follows_p),
        n_examples=batch_size,
    )


def evaluate(
    model: Transformer,
    config: TaskConfig,
    key: jax.Array,
    *,
    batch_size: int = 512,
    conditions: tuple[str, ...] = GATE_B_CONDITIONS,
    split: str = "train",
) -> dict[str, ConditionResult]:
    """Run the diagnostic suite.

    Each condition draws from its own split of the evaluation key, so adding
    or reordering conditions does not change the examples the others see.

    ``conditions`` defaults to :data:`GATE_B_CONDITIONS`, the two explicit
    modes. ``NEUTRAL_CONFLICT`` must be asked for by name, so measuring the
    preference effect is always a deliberate act and can never happen by
    default inside a calibration routine.
    """
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {sorted(unknown)}")
    keys = jr.split(key, len(conditions))
    return {
        condition: evaluate_condition(model, condition, config, k, batch_size, split=split)
        for condition, k in zip(conditions, keys)
    }
