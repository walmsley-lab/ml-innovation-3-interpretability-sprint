"""Layer-2 transfer pilot: all 12 directed pairs over the frozen four families.

One unit is a *paired* comparison, both arms in one process from one base
checkpoint:

    treatment   source D_i  ->  target D_j
    control     neutral N   ->  target D_j

so 12 pairs x 3 seeds = 36 units = 72 trajectories.

The t=0 target evaluation is taken before any target token, which is what
keeps immediate transfer separable from a faster acquisition rate. Skipping
it is unrecoverable after the run.

The neutral control N is a balanced mixture of the families that are neither
source nor target, at the same token budget, so the arms differ in the
identity of the source phase and nothing else.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jr, optax

from dsi.artifacts import code_version, utc_now
from dsi.layer2 import Layer2Config, sample_layer2_batch
from dsi.model import ModelConfig, init_model
from dsi.stats import transfer_effect

FROZEN_FAMILIES = ("F1_SELECT_MAP", "F4_SELECT_CMP", "F5_CHAIN_SELECT", "F6_CHAIN_MAP")
D_MODEL, N_LAYERS, LR, BATCH = 64, 4, 3e-3, 128
SOURCE_STEPS, TARGET_STEPS = 600, 600
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))   # includes t=0
EVAL_BATCH = 1024


def _loss(model, tokens):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(lp, tokens[:, 1:][..., None], axis=-1).squeeze(-1))


@eqx.filter_jit
def _step(model, opt_state, tokens, optimizer):
    loss, grads = eqx.filter_value_and_grad(_loss)(model, tokens, )
    updates, opt_state = optimizer.update(grads, opt_state,
                                          eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def _target_loss(model, tokens, answer, answer_index, answer_base):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits[:, answer_index, :], -1)
    return -jnp.mean(jnp.take_along_axis(lp, (answer_base + answer)[:, None], 1))


def _phase(model, opt_state, optimizer, families, config, key, steps, weights=None):
    for step in range(1, steps + 1):
        k = jr.fold_in(key, step)
        family = families[0] if len(families) == 1 else \
            families[int(jr.randint(jr.fold_in(k, 7), (), 0, len(families)))]
        batch = sample_layer2_batch(k, family, config, BATCH, split="train")
        model, opt_state, _ = _step(model, opt_state, batch["tokens"], optimizer)
    return model, opt_state


def run_unit(source: str, target: str, seed: int, out: Path) -> None:
    path = out / "units" / f"{source}__to__{target}__seed{seed}.json"
    if path.exists():
        return
    config = Layer2Config()
    model_config = ModelConfig(vocab_size=config.vocab_size, d_model=D_MODEL,
                               n_layers=N_LAYERS, n_heads=4, d_ff=4 * D_MODEL)
    answer_index = config.seq_len - 2
    optimizer = optax.adamw(optax.constant_schedule(LR), weight_decay=0.01)
    root = jr.key(seed)
    started = time.time()

    # Neutral control: the families that are neither source nor target.
    neutral = tuple(f for f in FROZEN_FAMILIES if f not in (source, target))
    eval_batch = sample_layer2_batch(jr.fold_in(root, 9), target, config,
                                     EVAL_BATCH, split="heldout")

    curves = {}
    for arm, prefix in (("treatment", (source,)), ("control", neutral)):
        model = init_model(model_config, jr.fold_in(root, 1))       # shared init
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        model, opt_state = _phase(model, opt_state, optimizer, prefix, config,
                                  jr.fold_in(root, 2 if arm == "treatment" else 3),
                                  SOURCE_STEPS)
        marks = {int(round(o * TARGET_STEPS)) for o in OFFSETS}
        curve = []
        def measure(step):
            curve.append({"step": step, "offset": step / TARGET_STEPS,
                          "loss": float(_target_loss(model, eval_batch["tokens"],
                                                     eval_batch["answer"], answer_index,
                                                     config.answer_base))})
        measure(0)                                    # t=0, before any target token
        for step in range(1, TARGET_STEPS + 1):
            batch = sample_layer2_batch(jr.fold_in(jr.fold_in(root, 4), step),
                                        target, config, BATCH, split="train")
            model, opt_state, _ = _step(model, opt_state, batch["tokens"], optimizer)
            if step in marks:
                measure(step)
        curves[arm] = curve

    t = [r["offset"] for r in curves["control"]]
    ctrl = [r["loss"] for r in curves["control"]]
    treat = [r["loss"] for r in curves["treatment"]]
    payload = {
        "source": source, "target": target, "seed": seed,
        "neutral_control": list(neutral),
        "T_aulc": transfer_effect(t, ctrl, treat, normalize=True),
        "T_aulc_rate_only": transfer_effect(t, ctrl, treat, normalize=True,
                                            baseline_correct=True),
        "head_start": ctrl[0] - treat[0],           # the t=0 gap
        "endpoint": ctrl[-1] - treat[-1],
        "curve_control": curves["control"], "curve_treatment": curves["treatment"],
        "seconds": time.time() - started,
        "code_version": code_version(), "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2) + "\n"); tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/layer2_transfer"))
    a = ap.parse_args(); run_unit(a.source, a.target, a.seed, a.out)
