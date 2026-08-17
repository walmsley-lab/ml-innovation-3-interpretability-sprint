"""Functional training. No hidden global state.

Signatures follow technical.md §8: a state goes in, a state comes out, and
every stochastic stream arrives as an explicit key derived by :mod:`dsi.rng`.
Because ``TrainState`` is an immutable PyTree, branching a developmental
history is a pure function call and costs nothing locally, which is why
Milestone A needs no checkpointing to run a paired experiment.

Learning rate is constant by default. Research plan §7 requires it: a decay
schedule makes position in the schedule a confound for phase order, so that
``W -> P`` and ``P -> W`` would differ in how much learning rate each family
received as well as in their order. Warmup is available for later stages
where it is needed, but it is not the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from dsi.data import TaskConfig, sample_batch
from dsi.model import ModelConfig, Transformer, init_model
from dsi.specs import PhaseSpec

__all__ = ["TrainConfig", "TrainState", "init_state", "train_phase", "phase_steps", "offset_steps"]


@dataclass(frozen=True)
class TrainConfig:
    """Nuisance parameters. Frozen after Milestone B calibration, not tuned per curriculum."""

    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    batch_size: int = 128
    warmup_steps: int = 0
    loss_positions: str = "all"
    """``"all"`` for ordinary next-token loss, ``"answer"`` to score only the answer."""

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.loss_positions not in ("all", "answer"):
            raise ValueError(f"loss_positions must be 'all' or 'answer', got {self.loss_positions!r}")


class TrainState(eqx.Module):
    """Parameters, optimizer state, and the counters that define run lineage."""

    model: Transformer
    opt_state: optax.OptState
    step: jax.Array
    tokens_seen: jax.Array


def _optimizer(config: TrainConfig) -> optax.GradientTransformation:
    if config.warmup_steps:
        schedule = optax.linear_schedule(0.0, config.learning_rate, config.warmup_steps)
    else:
        schedule = optax.constant_schedule(config.learning_rate)
    return optax.adamw(schedule, weight_decay=config.weight_decay)


def init_state(
    model_config: ModelConfig, train_config: TrainConfig, key: jax.Array
) -> TrainState:
    model = init_model(model_config, key)
    opt_state = _optimizer(train_config).init(eqx.filter(model, eqx.is_inexact_array))
    return TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_seen=jnp.asarray(0, dtype=jnp.int32),
    )


def phase_steps(phase: PhaseSpec, task: TaskConfig, train_config: TrainConfig) -> int:
    """Optimizer steps in a phase, from its token budget.

    Token budgets rather than step counts are the experimental currency, so
    that two phases with equal budgets are comparable regardless of batch
    size. A budget too small for a single step is an error, not a no-op
    phase, because a silently skipped phase would still look like a valid
    arm of a comparison.
    """
    tokens_per_step = train_config.batch_size * task.seq_len
    steps = phase.tokens // tokens_per_step
    if steps < 1:
        raise ValueError(
            f"phase {phase.family!r} has {phase.tokens} tokens but a step costs "
            f"{tokens_per_step}; a phase must run at least one step"
        )
    return steps


def offset_steps(offsets: tuple[float, ...], n_steps: int) -> tuple[int, ...]:
    """Map within-phase offsets to step counts.

    Offset ``0.0`` maps to zero steps, which is the point of the whole
    mechanism: the evaluation happens before the phase has trained on
    anything, so a head start carried in from an earlier phase is measured
    separately from what this phase teaches.
    """
    return tuple(int(round(o * n_steps)) for o in offsets)


def _loss(model: Transformer, tokens: jax.Array, answer_index: int, positions: str) -> jax.Array:
    logits = jax.vmap(model)(tokens[:, :-1])
    targets = tokens[:, 1:]
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
    if positions == "answer":
        picked = picked[:, answer_index]
    return -jnp.mean(picked)


@eqx.filter_jit
def _step(
    state: TrainState,
    tokens: jax.Array,
    optimizer: optax.GradientTransformation,
    answer_index: int,
    positions: str,
) -> tuple[TrainState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(_loss)(state.model, tokens, answer_index, positions)
    updates, opt_state = optimizer.update(
        grads, state.opt_state, eqx.filter(state.model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(state.model, updates)
    return (
        TrainState(
            model=model,
            opt_state=opt_state,
            step=state.step + 1,
            tokens_seen=state.tokens_seen + tokens.size,
        ),
        loss,
    )


def train_phase(
    state: TrainState,
    phase: PhaseSpec,
    task: TaskConfig,
    train_config: TrainConfig,
    data_key: jax.Array,
    *,
    eval_at: tuple[int, ...] = (),
    eval_fn=None,
    sampler=None,
) -> tuple[TrainState, list[dict]]:
    """Train one phase, evaluating at the requested step counts.

    Args:
        eval_at: Step counts within this phase at which to call ``eval_fn``.
            ``0`` means before the first optimizer step.
        eval_fn: Called as ``eval_fn(model, index)`` where ``index`` is the
            position within ``eval_at``. The index selects the evaluation
            key, so evaluation draws are fixed by the spec rather than by
            when training happens to reach them.
        sampler: Called as ``sampler(key, family, task, batch_size)`` and
            returning a mapping with a ``"tokens"`` entry. Defaults to the W/P
            :func:`dsi.data.sample_batch`. Injecting it is what lets the V2
            micro-world reuse this loop unchanged — ``task`` need only supply
            ``seq_len`` and ``answer_target_index``, which both
            ``TaskConfig`` and ``MicroConfig`` do.

    Returns:
        The updated state, and one record per evaluation point carrying the
        step, tokens into the phase, and cumulative tokens seen.
    """
    draw = sample_batch if sampler is None else sampler
    optimizer = _optimizer(train_config)
    n_steps = phase_steps(phase, task, train_config)
    tokens_per_step = train_config.batch_size * task.seq_len
    schedule = {s: i for i, s in enumerate(eval_at)}
    records: list[dict] = []

    def record(step_in_phase: int) -> None:
        index = schedule[step_in_phase]
        records.append(
            {
                "step_in_phase": step_in_phase,
                "tokens_in_phase": step_in_phase * tokens_per_step,
                "step": int(state.step),
                "tokens_seen": int(state.tokens_seen),
                "result": None if eval_fn is None else eval_fn(state.model, index),
            }
        )

    if 0 in schedule:
        record(0)

    for step_in_phase in range(1, n_steps + 1):
        batch_key = jax.random.fold_in(data_key, step_in_phase)
        batch = draw(batch_key, phase.family, task, train_config.batch_size)
        state, _ = _step(
            state, batch["tokens"], optimizer, task.answer_target_index, train_config.loss_positions
        )
        if step_in_phase in schedule:
            record(step_in_phase)

    return state, records
