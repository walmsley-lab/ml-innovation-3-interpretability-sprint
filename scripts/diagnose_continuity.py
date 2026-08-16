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

import argparse
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
RATIOS = (0.0, 0.05, 0.10, 0.25, 0.50)
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


def phase2_steps(ratio: float, *, budget_corrected: bool) -> int:
    """Length of phase 2 at continuity ratio `ratio`.

    Budget-corrected: the *new* skill keeps a fixed exposure of STEPS pure
    steps and the old-skill samples are added on top, so the phase runs
    STEPS/(1-r) steps in total. Fixed-total instead holds the phase length
    constant, which silently reduces new-skill exposure as r rises and
    starves the slower incoming skill.
    """
    return STEPS if not budget_corrected else int(round(STEPS / (1.0 - ratio)))


def classify(trace, old_key: str, tau: float) -> str:
    """How the prior capability behaved across phase 2."""
    values = [t[old_key] for t in trace]
    final, lowest = values[-1], min(values[1:]) if len(values) > 1 else values[-1]
    if final < tau:
        return "remains lost"
    if lowest >= tau:
        return "continuously preserved"
    return f"collapses to {lowest:.2f} then recovers"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-total", action="store_true",
                        help="hold phase-2 length constant (the confounded variant)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/continuity_corrected"))
    args = parser.parse_args()
    budget_corrected = not args.fixed_total

    task, model_config, train_config = setup()
    tokens = STEPS * train_config.batch_size * task.seq_len
    keys = run_keys(SEED, n_phases=2, n_eval_points=len(OFFSETS))
    results = []

    print("=== continuity-threshold diagnostic ===")
    print(f"regime d{D_MODEL}/l{N_LAYERS} lr={LR} steps/phase={STEPS} seed={SEED}")
    if budget_corrected:
        print("phase 2: new-skill exposure held at "
              f"{STEPS} pure steps; old-skill samples added on top")
        print("  " + "  ".join(
            f"r={r}: {STEPS} new + {phase2_steps(r, budget_corrected=True)-STEPS} old"
            for r in RATIOS))
    else:
        print(f"phase 2: total length held fixed at {STEPS} steps (confounded)")
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
            steps2 = phase2_steps(ratio, budget_corrected=budget_corrected)
            phase = PhaseSpec(
                family, steps2 * train_config.batch_size * task.seq_len, "target")
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
            old_key = "acc_w" if first.startswith("W") else "acc_p"
            behaviour = classify(trace, old_key, TAU)
            results.append({
                "direction": f"{first}->{second}", "ratio": ratio,
                "phase2_steps": steps2, "new_skill_steps": STEPS,
                "budget_corrected": budget_corrected,
                "acc_w": final["acc_w"], "acc_p": final["acc_p"],
                "coexistence": coexistence, "prior_capability": behaviour,
                "trace": trace, "seconds": time.time() - started,
            })
            print(f"  r={ratio:<5} steps={steps2:<5} A_W={final['acc_w']:.3f} "
                  f"A_P={final['acc_p']:.3f} coexistence={coexistence:.3f} "
                  f"{'>= tau' if coexistence >= TAU else '      '} "
                  f"| prior: {behaviour} ({time.time()-started:.0f}s)")

    print("\n--- phase-2 competence traces ---")
    for r in results:
        print(f"  {r['direction']:26s} r={r['ratio']:<5} "
              f"A_W " + " ".join(f"{t['acc_w']:5.2f}" for t in r["trace"]))
        print(f"  {'':26s} {'':7s} "
              f"A_P " + " ".join(f"{t['acc_p']:5.2f}" for t in r["trace"]))

    print("\n--- smallest continuity clearing tau, per direction ---")
    print("  (directional thresholds are kept separate; an asymmetry here "
          "would be a finding, not noise to average away)")
    thresholds = {}
    for first, second in DIRECTIONS:
        name = f"{first}->{second}"
        clearing = sorted(r["ratio"] for r in results
                          if r["direction"] == name and r["coexistence"] >= TAU)
        thresholds[name] = clearing[0] if clearing else None
        print(f"  {name:26s} " +
              (f"r={clearing[0]}" if clearing else f"none up to r={max(RATIOS)}"))

    both = [t for t in thresholds.values() if t is not None]
    if len(both) == len(DIRECTIONS):
        print(f"\nBoth directions clear tau={TAU}. Observed threshold "
              f"r={max(both)} in THIS architecture, task and seed regime; not a "
              "universal threshold pending replication across seeds and scales.")
        print("Stop before EWC/OGD. The question is now whether controlled "
              "overlapping curricula should become the primary developmental "
              "operationalization, which is a decision about the hypothesis.")
    elif both:
        print("\nDirections differ under matched new-task exposure. This is a "
              "genuine directional interference asymmetry, recorded as such.")
    else:
        print(f"\nNeither direction reaches tau={TAU} with matched new-task "
              "exposure. Diagnostic C is licensed.")

    print("\nDiagnostic only. Gate B and tau_retention are unchanged.")
    ArtifactWriter(args.out).write("continuity", [
        {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
        | {"seed": SEED, "d_model": D_MODEL, "n_layers": N_LAYERS,
           "learning_rate": LR, "code_version": code_version(),
           "recorded_at": utc_now()}
        for r in results
    ])


if __name__ == "__main__":
    main()
