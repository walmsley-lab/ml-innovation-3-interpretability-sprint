"""Why the two sources do not coexist, at the fixed Gate B regime.

Gate B failed on retention at every model size tried: training on the second
source drove the first to chance. Before spending a broader sweep on capacity
and learning rate, this asks whether the training objective is responsible.

The full-token loss spends most of its gradient on predicting the random
digit and cue tokens, which is irreducible noise, and only a small share on
the answer position where W and P actually live. If that dilution is what
prevents coexistence, then answer-only loss fixes it and no capacity change
is needed.

Holds every other variable fixed and varies one thing: loss_positions.

    A   W -> P, loss over all positions          (current)
    B   P -> W, loss over all positions          (current)
    C   W -> P, loss on the answer position only
    D   P -> W, loss on the answer position only

Measurements:

    competence          w_only and p_only accuracy throughout phase 2, so
                        forgetting is observed as it happens rather than
                        inferred from its endpoint;
    gradient cosine     global and per module, between the W objective and
                        the P objective at the end of phase 1. A strongly
                        negative cosine means the two objectives actively
                        oppose each other and no amount of capacity will make
                        them coexist under sequential training; a cosine near
                        zero means they are merely independent, and
                        interference is about capacity or plasticity;
    displacement        normalized per-module parameter movement during
                        phase 2, showing where the second source overwrites.

The conflict condition is not evaluated and no threshold is changed. This is
a diagnostic of the training objective, not a measurement of the phenomenon.

    python scripts/diagnose_coexistence.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.data import TaskConfig, sample_batch
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.rng import run_keys
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, _loss, init_state, offset_steps, phase_steps, train_phase

# The regime under test, unchanged from the Gate B sweep.
REGIME = {"n_digits": 3, "n_cues": 256, "d_model": 64, "n_layers": 2,
          "steps_per_phase": 600, "learning_rate": 3e-3}
SEED = 1000
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 256
GRAD_BATCH = 512
CONDITIONS = ("w_only", "p_only")  # note: no "conflict"


def _flat(tree) -> jnp.ndarray:
    return jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(tree)])


def _cosine(a, b) -> float:
    a, b = _flat(a), _flat(b)
    denom = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return float(jnp.dot(a, b) / denom) if denom > 0 else float("nan")


def _module_paths(tree) -> dict:
    """Group leaves by their top two path components, e.g. blocks.0."""
    groups: dict[str, list] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        name = ".".join(str(getattr(p, "name", getattr(p, "idx", p))) for p in path[:2])
        groups.setdefault(name, []).append(leaf)
    return groups


def gradient_alignment(model, task, train_config, key) -> dict:
    """Cosine between the W and P gradients at the same parameters."""
    kw, kp = jax.random.split(key)
    w_batch = sample_batch(kw, "W", task, GRAD_BATCH)["tokens"]
    p_batch = sample_batch(kp, "P", task, GRAD_BATCH)["tokens"]

    grad = eqx.filter_grad(_loss)
    gw = grad(model, w_batch, task.answer_target_index, train_config.loss_positions)
    gp = grad(model, p_batch, task.answer_target_index, train_config.loss_positions)

    per_module = {}
    groups_w, groups_p = _module_paths(gw), _module_paths(gp)
    for name in sorted(groups_w):
        per_module[name] = _cosine(groups_w[name], groups_p[name])
    return {"global": _cosine(gw, gp), "per_module": per_module}


def displacement(before, after) -> dict:
    """Normalized per-module parameter movement, ||delta|| / ||before||."""
    groups_before, groups_after = _module_paths(before), _module_paths(after)
    out = {}
    for name in sorted(groups_before):
        b, a = _flat(groups_before[name]), _flat(groups_after[name])
        norm = float(jnp.linalg.norm(b))
        out[name] = float(jnp.linalg.norm(a - b) / norm) if norm > 0 else float("nan")
    return out


def run_arm(order, loss_positions: str) -> dict:
    first, second = order
    task = TaskConfig(n_digits=REGIME["n_digits"], n_cues=REGIME["n_cues"])
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=REGIME["d_model"],
        n_layers=REGIME["n_layers"], n_heads=max(1, REGIME["d_model"] // 16),
        d_ff=4 * REGIME["d_model"],
    )
    train_config = TrainConfig(learning_rate=REGIME["learning_rate"],
                               loss_positions=loss_positions)
    tokens = REGIME["steps_per_phase"] * train_config.batch_size * task.seq_len
    phases = (PhaseSpec(first, tokens, "source"), PhaseSpec(second, tokens, "target"))

    keys = run_keys(SEED, n_phases=2, n_eval_points=len(OFFSETS))
    state = init_state(model_config, train_config, keys["init"])
    trace, started = [], time.time()

    for index, phase in enumerate(phases):
        stream = "source_data" if phase.role == "source" else "target_data"

        def eval_fn(model, point, _i=index):
            return evaluate(model, task, keys[f"eval.{_i}.{point}"],
                            batch_size=EVAL_BATCH, conditions=CONDITIONS, split="train")

        state, records = train_phase(
            state, phase, task, train_config, keys[f"{stream}.{index}"],
            eval_at=offset_steps(OFFSETS, phase_steps(phase, task, train_config)),
            eval_fn=eval_fn,
        )
        for offset, record in zip(OFFSETS, records):
            trace.append({
                "phase": index, "phase_family": phase.family, "offset": offset,
                "acc_w": record["result"]["w_only"].accuracy,
                "acc_p": record["result"]["p_only"].accuracy,
            })
        if index == 0:
            end_of_phase1 = state.model
            alignment = gradient_alignment(state.model, task, train_config, keys["eval.0.0"])

    return {
        "order": f"{first}->{second}", "loss_positions": loss_positions,
        "trace": trace,
        "gradient_alignment": alignment,
        "displacement_phase2": displacement(end_of_phase1, state.model),
        "final_acc_w": trace[-1]["acc_w"], "final_acc_p": trace[-1]["acc_p"],
        "coexistence": min(trace[-1]["acc_w"], trace[-1]["acc_p"]),
        "seconds": time.time() - started,
    }


def main() -> None:
    out = Path("artifacts/diagnostic")
    print(f"regime {REGIME}\nseed   {SEED}\n")
    results = []

    for loss_positions in ("all", "answer"):
        for order in (("W", "P"), ("P", "W")):
            result = run_arm(order, loss_positions)
            results.append(result)
            print(f"{result['order']:6s} loss={loss_positions:6s} "
                  f"final A_W={result['final_acc_w']:.3f} A_P={result['final_acc_p']:.3f} "
                  f"coexistence={result['coexistence']:.3f} "
                  f"grad_cos={result['gradient_alignment']['global']:+.3f} "
                  f"{result['seconds']:.0f}s")

    print("\n--- phase 2 competence trace (accuracy of the source NOT being trained) ---")
    for result in results:
        forgotten = "acc_w" if result["order"].startswith("W") else "acc_p"
        phase2 = [r for r in result["trace"] if r["phase"] == 1]
        values = " ".join(f"{r[forgotten]:.2f}" for r in phase2)
        print(f"  {result['order']:6s} loss={result['loss_positions']:6s} {values}")

    print("\n--- W-vs-P gradient cosine at end of phase 1 ---")
    for result in results:
        align = result["gradient_alignment"]
        modules = " ".join(f"{k}={v:+.2f}" for k, v in align["per_module"].items())
        print(f"  {result['order']:6s} loss={result['loss_positions']:6s} "
              f"global={align['global']:+.3f}  {modules}")

    ArtifactWriter(out).write("coexistence", [
        {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
         for k, v in r.items()} | {"code_version": code_version(), "recorded_at": utc_now()}
        for r in results
    ])

    baseline = max(r["coexistence"] for r in results if r["loss_positions"] == "all")
    answer_only = max(r["coexistence"] for r in results if r["loss_positions"] == "answer")
    print(f"\nbest coexistence: full-token {baseline:.3f}   answer-only {answer_only:.3f}")
    print("Coexistence is min(A_W, A_P) after sequential exposure, pre-washout, "
          f"against the unchanged tau_retention of 0.80.")
    if answer_only >= 0.80 > baseline:
        print("\nANSWER-ONLY LOSS RESOLVES IT. Capacity and learning rate need no change.")
    elif answer_only > baseline + 0.10:
        print("\nAnswer-only loss materially improves coexistence but does not reach "
              "the threshold on its own.")
    else:
        print("\nAnswer-only loss does not materially improve coexistence; "
              "proceed to a narrow neutral capacity/LR sweep.")


if __name__ == "__main__":
    main()
