"""Continuity-threshold diagnostic: is the abrupt block boundary the cause?

Joint training reaches coexistence 0.98 on the same architecture that reaches
0.23 sequentially, so capacity is not the constraint and the order of updates
is. This asks the next question: how much continuity does a curriculum need
before the first skill survives?

Phase 2 becomes a mixture rather than a pure block:

    W_EXPLICIT -> (1-r) P_EXPLICIT + r W_EXPLICIT
    P_EXPLICIT -> (1-r) W_EXPLICIT + r P_EXPLICIT

for r in {0, .01, .05, .10, .25, .50}, with the phase-2 token budget held
fixed. r=0 is the current Gate B condition and r=0.5 approaches joint
training, so the sweep interpolates between the two results already in hand.

Every r branches from the *same* phase-1 checkpoint. That is both cheaper and
a stricter comparison than retraining phase 1 each time: the arms differ in
the continuity ratio and in nothing else, not even in the parameters they
started phase 2 from.

This is a diagnostic of developmental continuity. It does not change Gate B,
it does not move tau_retention, and it does not evaluate NEUTRAL_CONFLICT.

    python scripts/diagnose_continuity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.calibrate import RegimeCriteria
from dsi.data import GATE_B_CONDITIONS, TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.rng import run_keys
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

# d64/l4: B1-capable, reaching A_W = A_P = 1.00 solo.
D_MODEL, N_LAYERS, LR = 64, 4, 3e-3
STEPS = 600
SEED = 1000
RATIOS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 512
TAU = RegimeCriteria().tau_retention

DIRECTIONS = (("W_EXPLICIT", "P_EXPLICIT"), ("P_EXPLICIT", "W_EXPLICIT"))


def setup():
    task = TaskConfig(n_digits=3, n_cues=256)
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=max(1, D_MODEL // 16), d_ff=4 * D_MODEL,
    )
    return task, model_config, TrainConfig(learning_rate=LR, loss_positions="all")


def main() -> None:
    task, model_config, train_config = setup()
    tokens = STEPS * train_config.batch_size * task.seq_len
    keys = run_keys(SEED, n_phases=2, n_eval_points=len(OFFSETS))
    results = []

    print("=== continuity-threshold diagnostic ===")
    print(f"regime d{D_MODEL}/l{N_LAYERS} lr={LR} steps/phase={STEPS} seed={SEED}")
    print(f"ratios {RATIOS}   phase-2 token budget held fixed")
    print(f"conditions {GATE_B_CONDITIONS}  (NEUTRAL_CONFLICT not evaluated)\n")

    for first, second in DIRECTIONS:
        # Phase 1 once; every ratio branches from this exact checkpoint.
        state = init_state(model_config, train_config, keys["init"])
        state, records = train_phase(
            state, PhaseSpec(first, tokens, "source"), task, train_config,
            keys["source_data.0"],
            eval_at=(phase_steps(PhaseSpec(first, tokens, "source"), task, train_config),),
            eval_fn=lambda m, i: evaluate(m, task, keys["eval.0.0"],
                                          batch_size=EVAL_BATCH,
                                          conditions=GATE_B_CONDITIONS, split="train"),
        )
        after_phase1 = records[-1]["result"]
        print(f"--- {first} -> {second} "
              f"(after phase 1: A_W={after_phase1['W_COMPETENCE'].accuracy:.3f} "
              f"A_P={after_phase1['P_COMPETENCE'].accuracy:.3f}) ---")

        for ratio in RATIOS:
            family = f"{second}+{first}@{ratio}"
            phase = PhaseSpec(family, tokens, "target")
            started = time.time()

            def eval_fn(model, point):
                return evaluate(model, task, keys[f"eval.1.{point}"],
                                batch_size=EVAL_BATCH,
                                conditions=GATE_B_CONDITIONS, split="train")

            # Branching an immutable PyTree costs nothing, so every ratio
            # starts from identical parameters.
            _, phase2 = train_phase(
                state, phase, task, train_config, keys["target_data.1"],
                eval_at=offset_steps(OFFSETS, phase_steps(phase, task, train_config)),
                eval_fn=eval_fn,
            )
            trace = [{"offset": o,
                      "acc_w": r["result"]["W_COMPETENCE"].accuracy,
                      "acc_p": r["result"]["P_COMPETENCE"].accuracy}
                     for o, r in zip(OFFSETS, phase2)]
            final = trace[-1]
            coexistence = min(final["acc_w"], final["acc_p"])
            results.append({
                "direction": f"{first}->{second}", "ratio": ratio,
                "acc_w": final["acc_w"], "acc_p": final["acc_p"],
                "coexistence": coexistence, "trace": trace,
                "seconds": time.time() - started,
            })
            print(f"  r={ratio:<5} A_W={final['acc_w']:.3f} A_P={final['acc_p']:.3f} "
                  f"coexistence={coexistence:.3f} "
                  f"{'>= tau' if coexistence >= TAU else ''} "
                  f"({time.time()-started:.0f}s)")

    print("\n--- phase-2 competence traces ---")
    for r in results:
        print(f"  {r['direction']:26s} r={r['ratio']:<5} "
              f"A_W " + " ".join(f"{t['acc_w']:5.2f}" for t in r["trace"]))
        print(f"  {'':26s} {'':7s} "
              f"A_P " + " ".join(f"{t['acc_p']:5.2f}" for t in r["trace"]))

    print("\n--- worst-direction coexistence by ratio ---")
    passing = []
    for ratio in RATIOS:
        worst = min(r["coexistence"] for r in results if r["ratio"] == ratio)
        flag = "PASS" if worst >= TAU else ""
        if worst >= TAU:
            passing.append(ratio)
        print(f"  r={ratio:<5} worst={worst:.3f}  {flag}")

    if passing:
        print(f"\nSmallest continuity reaching tau={TAU}: r={min(passing)}")
        if min(passing) <= 0.05:
            print("Very small continuity rescues coexistence. The abrupt task "
                  "boundary is the likely blocker, and continual-learning "
                  "methods are not yet warranted.")
    else:
        print(f"\nNo tested ratio reaches tau={TAU}, up to r=0.50. "
              "Continuity alone does not rescue coexistence; Diagnostic C "
              "(layer freeze, then EWC-style consolidation) is the next step.")

    print("\nDiagnostic only. Gate B and tau_retention are unchanged.")
    ArtifactWriter(Path("artifacts/continuity")).write("continuity", [
        {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
        | {"seed": SEED, "d_model": D_MODEL, "n_layers": N_LAYERS,
           "learning_rate": LR, "code_version": code_version(),
           "recorded_at": utc_now()}
        for r in results
    ])


if __name__ == "__main__":
    main()
