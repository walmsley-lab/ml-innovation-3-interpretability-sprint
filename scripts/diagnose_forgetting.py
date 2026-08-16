"""Blocker classification: why does coexistence fail?

The corrected task is well-posed (an oracle reaches A_W = A_P = 1.0), and
yet no regime in the eight-regime sweep retained both skills: coexistence sat
at 0.195-0.250 across a 6.4x parameter range, two depths and two learning
rates. Something other than capacity is responsible, and these diagnostics
separate the candidates instead of guessing.

Diagnostic A — joint-training upper bound
    Train from initialization on a balanced within-batch interleaving of
    W_EXPLICIT and P_EXPLICIT, on the same total token budget as the
    sequential condition. Asks whether the architecture can represent both
    skills at once when optimization is not sequential.

Diagnostic B — recovery under common integration
    Take the ordinary sequential histories and append an identical
    NEUTRAL_ALIGNED phase to both arms, tracking explicit competence
    throughout. Asks whether the forgotten skill is erased or merely
    inaccessible.

Reading the outcomes:

    joint fails                      -> capacity or task still inadequate
    joint succeeds, sequential fails -> genuine catastrophic interference
    integration restores both        -> immediate retention may be too strong
                                        a gate, and a developmental-recovery
                                        formulation deserves consideration

NEUTRAL_CONFLICT is not evaluated. Nothing here changes tau_retention or the
recorded Gate B result, and no continual-learning method is added to the
primary design.

    python scripts/diagnose_forgetting.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.calibrate import RegimeCriteria
from dsi.data import GATE_B_CONDITIONS, TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig, count_params
from dsi.rng import run_keys
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

STEPS = 600
SEED = 1000
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 512
TAU = RegimeCriteria().tau_retention

# d64/l2 is the specified diagnostic regime. d64/l4 is included because
# d64/l2 reaches only A_W = 0.56 solo, so a joint-training failure there
# would be ambiguous between "cannot hold both" and "cannot learn the rule".
ARCHITECTURES = {"d64l2": (64, 2), "d64l4": (64, 4)}


def _setup(d_model, n_layers, lr=3e-3):
    task = TaskConfig(n_digits=3, n_cues=256)
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=max(1, d_model // 16), d_ff=4 * d_model,
    )
    return task, model_config, TrainConfig(learning_rate=lr, loss_positions="all")


def _train(phases, task, model_config, train_config, n_phases_keys):
    keys = run_keys(SEED, n_phases=n_phases_keys, n_eval_points=len(OFFSETS))
    state = init_state(model_config, train_config, keys["init"])
    trace = []
    for index, phase in enumerate(phases):
        stream = "source_data" if phase.role == "source" else "target_data"

        def eval_fn(model, point, _i=index):
            return evaluate(model, task, keys[f"eval.{_i}.{point}"],
                            batch_size=EVAL_BATCH, conditions=GATE_B_CONDITIONS,
                            split="train")

        state, records = train_phase(
            state, phase, task, train_config, keys[f"{stream}.{index}"],
            eval_at=offset_steps(OFFSETS, phase_steps(phase, task, train_config)),
            eval_fn=eval_fn,
        )
        for offset, record in zip(OFFSETS, records):
            trace.append({
                "phase": index, "phase_family": phase.family, "offset": offset,
                "acc_w": record["result"]["W_COMPETENCE"].accuracy,
                "acc_p": record["result"]["P_COMPETENCE"].accuracy,
            })
    return state, trace


def diagnostic_a() -> list[dict]:
    """Joint training on the same total budget as the sequential condition."""
    print("=== Diagnostic A: joint-training upper bound ===")
    print(f"balanced W/P interleaving, {2 * STEPS} steps "
          f"(= the sequential total), from initialization\n")
    results = []
    for name, (d_model, n_layers) in ARCHITECTURES.items():
        task, model_config, train_config = _setup(d_model, n_layers)
        tokens = 2 * STEPS * train_config.batch_size * task.seq_len
        started = time.time()
        state, trace = _train(
            [PhaseSpec("W_P_INTERLEAVED", tokens, "target")],
            task, model_config, train_config, 1)
        final = trace[-1]
        coexistence = min(final["acc_w"], final["acc_p"])
        results.append({
            "diagnostic": "A_joint", "arch": name,
            "params": count_params(state.model),
            "acc_w": final["acc_w"], "acc_p": final["acc_p"],
            "coexistence": coexistence, "trace": trace,
            "seconds": time.time() - started,
        })
        print(f"  {name:6s} A_W={final['acc_w']:.3f} A_P={final['acc_p']:.3f} "
              f"coexistence={coexistence:.3f} "
              f"{'>= tau' if coexistence >= TAU else '< tau'} "
              f"({time.time()-started:.0f}s)")
    return results


def diagnostic_b() -> list[dict]:
    """Sequential history followed by identical shared integration."""
    print("\n=== Diagnostic B: recovery under common integration ===")
    print("W->P->NEUTRAL_ALIGNED and P->W->NEUTRAL_ALIGNED, identical tail\n")
    results = []
    for name, (d_model, n_layers) in ARCHITECTURES.items():
        task, model_config, train_config = _setup(d_model, n_layers)
        tokens = STEPS * train_config.batch_size * task.seq_len
        for first, second in (("W_EXPLICIT", "P_EXPLICIT"), ("P_EXPLICIT", "W_EXPLICIT")):
            started = time.time()
            state, trace = _train(
                [PhaseSpec(first, tokens, "source"),
                 PhaseSpec(second, tokens, "target"),
                 PhaseSpec("NEUTRAL_ALIGNED", tokens, "washout")],
                task, model_config, train_config, 3)
            pre = [t for t in trace if t["phase"] == 1][-1]
            post = trace[-1]
            results.append({
                "diagnostic": "B_integration", "arch": name,
                "order": f"{first}->{second}",
                "pre_integration": min(pre["acc_w"], pre["acc_p"]),
                "post_integration": min(post["acc_w"], post["acc_p"]),
                "post_acc_w": post["acc_w"], "post_acc_p": post["acc_p"],
                "trace": trace, "seconds": time.time() - started,
            })
            print(f"  {name:6s} {first[0]}->{second[0]}  pre={min(pre['acc_w'], pre['acc_p']):.3f} "
                  f"post={min(post['acc_w'], post['acc_p']):.3f} "
                  f"(A_W={post['acc_w']:.3f} A_P={post['acc_p']:.3f}) "
                  f"({time.time()-started:.0f}s)")
    return results


def main() -> None:
    results = diagnostic_a() + diagnostic_b()

    print("\n--- integration-phase traces (phase 2, the shared tail) ---")
    for r in results:
        if r["diagnostic"] != "B_integration":
            continue
        tail = [t for t in r["trace"] if t["phase"] == 2]
        print(f"  {r['arch']:6s} {r['order']:26s} A_W " +
              " ".join(f"{t['acc_w']:5.2f}" for t in tail))
        print(f"  {'':6s} {'':26s} A_P " +
              " ".join(f"{t['acc_p']:5.2f}" for t in tail))

    joint = max(r["coexistence"] for r in results if r["diagnostic"] == "A_joint")
    integrated = max(r["post_integration"] for r in results if r["diagnostic"] == "B_integration")
    print(f"\nbest joint coexistence      {joint:.3f}  (tau = {TAU})")
    print(f"best post-integration       {integrated:.3f}")
    if joint < TAU:
        print("\nVERDICT: joint training also fails. Capacity or task is still "
              "inadequate; this is not specific to sequential optimization.")
    elif integrated >= TAU:
        print("\nVERDICT: joint succeeds and shared integration restores both. "
              "The skill is not erased. Whether immediate retention is the right "
              "gate is a question about the hypothesis, to be decided explicitly.")
    else:
        print("\nVERDICT: joint succeeds, sequential fails, integration does not "
              "restore. Genuine catastrophic interference; Diagnostic C is "
              "licensed to test whether destructive updates are causal.")

    ArtifactWriter(Path("artifacts/forgetting")).write("diagnostics", [
        {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
        | {"seed": SEED, "code_version": code_version(), "recorded_at": utc_now()}
        for r in results
    ])


if __name__ == "__main__":
    main()
