"""Execute the frozen r=0.20 overlap diagnostic.

Per-unit outputs are idempotent and atomically written, so idle workers can
steal shards without coordination.

Each unit records the frozen efficiency metrics as well as final accuracy,
because the question is no longer only "is there an order effect" but
"does any ordering buy capability at equal compute". Steps-to-threshold is
recorded per family and jointly; min-across-families is the primary metric,
since the ceiling scout's sequential arms looked tolerable on the mean
(0.383) and catastrophic on the min (0.106).
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

MANIFEST = Path("artifacts/layer2_overlap/overlap_manifest.json")
D_MODEL, N_LAYERS, N_HEADS, LR, BATCH = 64, 4, 4, 3e-3, 128
EVAL_BATCH, EVAL_EVERY = 1024, 50


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
def _acc(model, tokens, answer, answer_index, answer_base):
    logits = jax.vmap(model)(tokens[:, :-1])
    return jnp.mean(jnp.argmax(logits[:, answer_index, :], -1) == (answer_base + answer))


@eqx.filter_jit
def _ans_loss(model, tokens, answer, answer_index, answer_base):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits[:, answer_index, :], -1)
    return -jnp.mean(jnp.take_along_axis(lp, (answer_base + answer)[:, None], 1))


def build_schedule(arm: dict, families, total_steps: int, seed: int):
    """Expand the frozen plan into an explicit per-step family sequence."""
    if arm["type"] == "interleaved":
        seq = [families[i % len(families)] for i in range(total_steps)]
        return list(np.array(seq)[np.random.default_rng(seed).permutation(total_steps)])
    seq = []
    for phase in arm["plan"]:
        block = [phase["family"]] * phase["own_steps"]
        for prev in phase["previous"]:
            block += [prev] * phase["overlap_steps_each"]
        # Interleave the overlap within the phase rather than appending it,
        # so the phase is a mixture throughout instead of a mini-curriculum.
        rng = np.random.default_rng(seed * 100 + phase["phase"])
        seq += list(np.array(block)[rng.permutation(len(block))])
    return seq


def run_unit(arm_name: str, seed: int, manifest: dict, out: Path) -> dict:
    path = out / "units" / f"{arm_name}__seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    families = manifest["families"]
    arm = manifest["arms"][arm_name]
    total_steps = manifest["total_steps"]
    threshold = manifest["threshold"]

    config = Layer2Config()
    model_config = ModelConfig(vocab_size=config.vocab_size, d_model=D_MODEL,
                               n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=4 * D_MODEL)
    answer_index = config.seq_len - 2
    optimizer = optax.adamw(optax.constant_schedule(LR), weight_decay=0.01)
    root = jr.key(seed)
    model = init_model(model_config, jr.fold_in(root, 1))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    held = {f: sample_layer2_batch(jr.fold_in(root, 900 + i), f, config,
                                   EVAL_BATCH, split="heldout")
            for i, f in enumerate(families)}

    schedule = build_schedule(arm, families, total_steps, seed)
    counts: dict = {}
    for f in schedule:
        counts[f] = counts.get(f, 0) + 1
    expected = arm["per_family_total_steps"]
    if counts != expected:
        raise AssertionError(f"allocation {counts} != frozen {expected}")

    tokens_per_step = BATCH * config.seq_len
    curve, reached = [], {f: None for f in families}

    def measure(step):
        row = {"step": step, "tokens": step * tokens_per_step}
        for f in families:
            b = held[f]
            a = float(_acc(model, b["tokens"], b["answer"], answer_index, config.answer_base))
            row[f"{f}_acc"] = a
            row[f"{f}_loss"] = float(_ans_loss(model, b["tokens"], b["answer"],
                                               answer_index, config.answer_base))
            if reached[f] is None and a >= threshold:
                reached[f] = step
        curve.append(row)

    measure(0)
    for step in range(1, total_steps + 1):
        batch = sample_layer2_batch(jr.fold_in(jr.fold_in(root, 4), step),
                                    schedule[step - 1], config, BATCH, split="train")
        model, opt_state, _ = _step(model, opt_state, batch["tokens"], optimizer)
        if step % EVAL_EVERY == 0 or step == total_steps:
            measure(step)

    final = curve[-1]
    steps = np.array([r["step"] for r in curve])
    auc = {f: float(np.trapezoid([r[f"{f}_acc"] for r in curve], steps) / steps[-1])
           for f in families}
    # All-families threshold: first step where EVERY family is at or above it.
    joint = next((r["step"] for r in curve
                  if all(r[f"{f}_acc"] >= threshold for f in families)), None)

    payload = {
        "arm": arm_name, "seed": seed, "order": arm["order"], "type": arm["type"],
        "allocation": counts,
        "fixed_budget_mean_acc": float(np.mean([final[f"{f}_acc"] for f in families])),
        "fixed_budget_min_acc": float(min(final[f"{f}_acc"] for f in families)),
        "fixed_budget_mean_loss": float(np.mean([final[f"{f}_loss"] for f in families])),
        "steps_to_threshold": reached,
        "tokens_to_threshold": {f: (None if v is None else v * tokens_per_step)
                                for f, v in reached.items()},
        "joint_steps_to_threshold": joint,
        "joint_tokens_to_threshold": None if joint is None else joint * tokens_per_step,
        "familywise_auc": auc,
        "mean_auc": float(np.mean(list(auc.values()))),
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
    ap.add_argument("--out", type=Path, default=Path("artifacts/layer2_overlap"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for unit in manifest["units"][args.shard::args.n_shards]:
        row = run_unit(unit["arm"], unit["seed"], manifest, args.out)
        print(f"{unit['arm']:>18s} seed {unit['seed']}  "
              f"min_acc {row['fixed_budget_min_acc']:.4f}  "
              f"mean_acc {row['fixed_budget_mean_acc']:.4f}  "
              f"joint_steps {row['joint_steps_to_threshold']}  "
              f"{row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
