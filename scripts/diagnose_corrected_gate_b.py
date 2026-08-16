"""First corrected Gate B diagnostic: does one checkpoint retain both skills?

The original task was unidentifiable (research.md 8b): the two families
presented byte-identical visible inputs, capping min(A_W, A_P) at 0.625 for
any deterministic predictor, below the prespecified tau_retention of 0.80.
A MODE token now makes the request part of the input, and an oracle clears
both gates simultaneously (tests/test_identifiability.py).

This asks the capability question and nothing else:

    W_EXPLICIT -> P_EXPLICIT      and      P_EXPLICIT -> W_EXPLICIT

At every checkpoint, and at the final one in particular, W competence is
measured under MODE=USE_W and P competence under MODE=USE_P, on the same
checkpoint. The question is whether both survive in both orders.

NEUTRAL_ALIGNED and NEUTRAL_CONFLICT are deliberately absent. The aligned
tail enters only after this succeeds, and the conflict condition is the
preference measurement, which Gate B must never see.

Model and training are fixed at the values already in hand. Nothing here
sweeps capacity or learning rate.

    python scripts/diagnose_corrected_gate_b.py
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

REGIME = {"d_model": 64, "n_layers": 2, "n_digits": 3, "n_cues": 256,
          "steps_per_phase": 600, "learning_rate": 3e-3, "loss_positions": "all"}
SEED = 1000
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 512
CRITERIA = RegimeCriteria()  # tau_retention unchanged at 0.80

ORDERS = (("W_EXPLICIT", "P_EXPLICIT"), ("P_EXPLICIT", "W_EXPLICIT"))


def run_order(order) -> dict:
    first, second = order
    task = TaskConfig(n_digits=REGIME["n_digits"], n_cues=REGIME["n_cues"])
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=REGIME["d_model"],
        n_layers=REGIME["n_layers"], n_heads=max(1, REGIME["d_model"] // 16),
        d_ff=4 * REGIME["d_model"],
    )
    train_config = TrainConfig(learning_rate=REGIME["learning_rate"],
                               loss_positions=REGIME["loss_positions"])
    tokens = REGIME["steps_per_phase"] * train_config.batch_size * task.seq_len
    phases = (PhaseSpec(first, tokens, "source"), PhaseSpec(second, tokens, "target"))

    keys = run_keys(SEED, n_phases=2, n_eval_points=len(OFFSETS))
    state = init_state(model_config, train_config, keys["init"])
    trace, started = [], time.time()

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

    # Generalization on held-out compositional structures, same checkpoint.
    heldout = evaluate(state.model, task, keys[f"eval.1.{len(OFFSETS)-1}"],
                       batch_size=EVAL_BATCH, conditions=GATE_B_CONDITIONS,
                       split="heldout")
    final = trace[-1]
    return {
        "order": f"{first}->{second}",
        "params": count_params(state.model),
        "trace": trace,
        "final_acc_w": final["acc_w"], "final_acc_p": final["acc_p"],
        "coexistence": min(final["acc_w"], final["acc_p"]),
        "heldout_acc_w": heldout["W_COMPETENCE"].accuracy,
        "heldout_acc_p": heldout["P_COMPETENCE"].accuracy,
        "seconds": time.time() - started,
    }


def main() -> None:
    print("=== corrected Gate B: capability coexistence ===")
    print(f"regime     {REGIME}")
    print(f"seed       {SEED}")
    print(f"conditions {GATE_B_CONDITIONS}  (NEUTRAL_CONFLICT not generated)")
    print(f"threshold  tau_retention={CRITERIA.tau_retention} (unchanged)\n")

    results = [run_order(order) for order in ORDERS]
    for result in results:
        print(f"{result['order']:26s} final A_W={result['final_acc_w']:.3f} "
              f"A_P={result['final_acc_p']:.3f} coexistence={result['coexistence']:.3f} "
              f"heldout {result['heldout_acc_w']:.3f}/{result['heldout_acc_p']:.3f} "
              f"{result['seconds']:.0f}s")

    print("\n--- phase 2: is the first capability still accessible while the second is learned? ---")
    print(f"{'order':26s} {'src':4s} " + " ".join(f"{o:5.1f}" for o in OFFSETS))
    for result in results:
        for key, name in (("acc_w", "W"), ("acc_p", "P")):
            phase2 = [t for t in result["trace"] if t["phase"] == 1]
            print(f"{result['order']:26s} {name:4s} " + " ".join(f"{t[key]:5.2f}" for t in phase2))

    worst = min(r["coexistence"] for r in results)
    print(f"\nworst-order coexistence {worst:.3f} vs tau_retention {CRITERIA.tau_retention}")
    print("PASS: both capabilities retained in both orders" if worst >= CRITERIA.tau_retention
          else "FAIL: retention still not achieved on the corrected task")

    out = Path("artifacts/corrected_gate_b")
    ArtifactWriter(out).write("coexistence", [
        {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
        | {"seed": SEED, "regime": json.dumps(REGIME),
           "code_version": code_version(), "recorded_at": utc_now()}
        for r in results
    ])
    print(f"wrote {out}/coexistence.parquet")


if __name__ == "__main__":
    main()
