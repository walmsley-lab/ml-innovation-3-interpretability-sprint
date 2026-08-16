"""Layer-2 solo calibration: acquisition timescale per family.

Difficulty matching is a gate, not a nicety. Unmatched difficulty makes T_ij
partly a measurement of how hard the target family is, which is exactly the
additive beta_j term the developmental claim has to beat. Layer 1 cost two
calibration cycles to this, with a trivially fast cue against a slow rule.

Adequacy target: max t90 / min t90 <= 2.0 across families, with every family
clearing competence and held-out generalization solo.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jr, optax

from dsi.artifacts import code_version, utc_now
from dsi.layer2 import FAMILIES, Layer2Config, sample_layer2_batch
from dsi.model import ModelConfig, init_model
from dsi.stats import tokens_to_threshold

D_MODEL, N_LAYERS, LR, STEPS, BATCH = 64, 4, 3e-3, 1200, 128
OFFSETS = tuple(round(0.05 * i, 2) for i in range(21))
EVAL_BATCH, TAU_COMP, TAU_GEN = 1024, 0.90, 0.80


def _loss(model, tokens, answer_index):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(lp, tokens[:, 1:][..., None], axis=-1).squeeze(-1)
    return -jnp.mean(picked)


@eqx.filter_jit
def _step(model, opt_state, tokens, optimizer, answer_index):
    loss, grads = eqx.filter_value_and_grad(_loss)(model, tokens, answer_index)
    updates, opt_state = optimizer.update(grads, opt_state,
                                          eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def _accuracy(model, tokens, answer, answer_index, answer_base):
    logits = jax.vmap(model)(tokens[:, :-1])
    return jnp.mean(jnp.argmax(logits[:, answer_index, :], -1) == answer_base + answer)


def run_unit(family: str, seed: int, out: Path) -> None:
    path = out / "units" / f"{family}__seed{seed}.json"
    if path.exists():
        return
    config = Layer2Config()
    model_config = ModelConfig(vocab_size=config.vocab_size, d_model=D_MODEL,
                               n_layers=N_LAYERS, n_heads=4, d_ff=4 * D_MODEL)
    root = jr.key(seed)
    model = init_model(model_config, jr.fold_in(root, 1))
    optimizer = optax.adamw(optax.constant_schedule(LR), weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    answer_index = config.seq_len - 2
    started = time.time()

    eval_key = jr.fold_in(root, 5)
    marks = {int(round(o * STEPS)) for o in OFFSETS}
    curve = []

    def measure(step):
        row = {"step": step, "offset": step / STEPS}
        for split in ("train", "heldout"):
            b = sample_layer2_batch(jr.fold_in(eval_key, 0 if split == "train" else 1),
                                    family, config, EVAL_BATCH, split=split)
            row[split] = float(_accuracy(model, b["tokens"], b["answer"],
                                         answer_index, config.answer_base))
        curve.append(row)

    measure(0)
    for step in range(1, STEPS + 1):
        batch = sample_layer2_batch(jr.fold_in(jr.fold_in(root, 2), step),
                                    family, config, BATCH, split="train")
        model, opt_state, _ = _step(model, opt_state, batch["tokens"], optimizer, answer_index)
        if step in marks:
            measure(step)

    steps = [r["step"] for r in curve]
    train = [r["train"] for r in curve]
    floor, ceiling = train[0], max(train)
    gain = ceiling - floor
    t90 = None
    if gain >= 0.05:
        c = tokens_to_threshold(steps, train, floor + 0.90 * gain, direction="above")
        t90 = None if c.censored else c.time
    payload = {
        "family": family, "seed": seed, "t90": t90,
        "final_train": train[-1], "final_heldout": curve[-1]["heldout"],
        "competent": train[-1] >= TAU_COMP,
        "generalizes": curve[-1]["heldout"] >= TAU_GEN,
        "curve": curve, "seconds": time.time() - started,
        "code_version": code_version(), "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2) + "\n"); tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/layer2_calib"))
    a = ap.parse_args(); run_unit(a.family, a.seed, a.out)
