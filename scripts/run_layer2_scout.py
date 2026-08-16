"""Execute the frozen Layer-2 ceiling-scout manifest.

Reads the manifest and runs one unit per (arm, seed). Outputs are
**idempotent and per-unit**: a unit already on disk is skipped, and writes are
atomic, so an idle worker can steal any shard without coordination and
without risk of a torn or duplicated result.

Every arm sees the identical aggregate family allocation and the identical
total training budget. The arms differ only in presentation order, which is
the whole point: an allocation difference would confound order with dose.

Evaluation is on held-out examples of **every** family after training, plus a
learning curve per family, so a curriculum that wins on one family while
destroying another is visible rather than hidden in an average.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from dsi.artifacts import code_version, utc_now
from dsi.layer2 import Layer2Config, sample_layer2_batch
from dsi.model import ModelConfig, init_model

MANIFEST = Path("artifacts/layer2_scout/scout_manifest.json")
D_MODEL, N_LAYERS, N_HEADS, LR, BATCH = 64, 4, 4, 3e-3, 128
EVAL_BATCH = 1024
EVAL_EVERY = 100


def _loss(model, tokens):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(lp, tokens[:, 1:][..., None], axis=-1).squeeze(-1))


@eqx.filter_jit
def _step(model, opt_state, tokens, optimizer):
    loss, grads = eqx.filter_value_and_grad(_loss)(model, tokens)
    updates, opt_state = optimizer.update(grads, opt_state,
                                          eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def _answer_loss(model, tokens, answer, answer_index, answer_base):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits[:, answer_index, :], -1)
    return -jnp.mean(jnp.take_along_axis(lp, (answer_base + answer)[:, None], 1))


@eqx.filter_jit
def _answer_acc(model, tokens, answer, answer_index, answer_base):
    logits = jax.vmap(model)(tokens[:, :-1])
    pred = jnp.argmax(logits[:, answer_index, :], -1)
    return jnp.mean(pred == (answer_base + answer))


def run_unit(arm: str, seed: int, manifest: dict, out: Path) -> dict:
    path = out / "units" / f"{arm}__seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())

    started = time.time()
    families = manifest["families"]
    spec = manifest["arms"][arm]
    steps_per_phase = manifest["steps_per_phase"]
    total_steps = manifest["total_steps"]

    config = Layer2Config()
    model_config = ModelConfig(vocab_size=config.vocab_size, d_model=D_MODEL,
                               n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=4 * D_MODEL)
    answer_index = config.seq_len - 2
    optimizer = optax.adamw(optax.constant_schedule(LR), weight_decay=0.01)
    root = jr.key(seed)

    model = init_model(model_config, jr.fold_in(root, 1))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    held_out = {f: sample_layer2_batch(jr.fold_in(root, 900 + i), f, config,
                                       EVAL_BATCH, split="heldout")
                for i, f in enumerate(families)}

    # Presentation schedule. Both arm types spend exactly steps_per_phase
    # worth of training on each family; only the order differs.
    if spec["type"] == "sequential":
        schedule = [f for f in spec["order"] for _ in range(steps_per_phase)]
    else:
        schedule = [families[i % len(families)] for i in range(total_steps)]
        rng = np.random.default_rng(seed)
        schedule = list(np.array(schedule)[rng.permutation(total_steps)])
    counts: dict = {}
    for f in schedule:
        counts[f] = counts.get(f, 0) + 1
    if set(counts.values()) != {steps_per_phase}:
        raise AssertionError(f"allocation differs across families: {counts}")

    curve = []

    def measure(step):
        row = {"step": step}
        for f in families:
            batch = held_out[f]
            row[f"{f}_loss"] = float(_answer_loss(model, batch["tokens"], batch["answer"],
                                                  answer_index, config.answer_base))
            row[f"{f}_acc"] = float(_answer_acc(model, batch["tokens"], batch["answer"],
                                                answer_index, config.answer_base))
        curve.append(row)

    measure(0)
    for step in range(1, total_steps + 1):
        family = schedule[step - 1]
        batch = sample_layer2_batch(jr.fold_in(jr.fold_in(root, 4), step), family,
                                    config, BATCH, split="train")
        model, opt_state, _ = _step(model, opt_state, batch["tokens"], optimizer)
        if step % EVAL_EVERY == 0 or step == total_steps:
            measure(step)

    final = curve[-1]
    payload = {
        "arm": arm, "seed": seed, "order": spec["order"], "type": spec["type"],
        "allocation_per_family": counts,
        "final_mean_acc": float(np.mean([final[f"{f}_acc"] for f in families])),
        "final_min_acc": float(min(final[f"{f}_acc"] for f in families)),
        "final_mean_loss": float(np.mean([final[f"{f}_loss"] for f in families])),
        "final_per_family": {f: {"acc": final[f"{f}_acc"], "loss": final[f"{f}_loss"]}
                             for f in families},
        "curve": curve, "seconds": time.time() - started,
        "manifest_sha256": manifest["sha256"],
        "code_version": code_version(), "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/layer2_scout"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    units = manifest["units"]
    for unit in units[args.shard::args.n_shards]:
        row = run_unit(unit["arm"], unit["seed"], manifest, args.out)
        print(f"{unit['arm']:>18s} seed {unit['seed']}  "
              f"mean_acc {row['final_mean_acc']:.4f}  "
              f"min_acc {row['final_min_acc']:.4f}  {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
