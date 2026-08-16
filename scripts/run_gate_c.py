"""Gate C: null distribution and noise floor for the Claim-1 metric.

The Claim-1 metric is the neutral W-choice rate under NEUTRAL_CONFLICT,
measured on a checkpoint that has passed explicit competence. The order
effect is the difference in that rate between two developmental histories.

Identity-null pairs hold the history **identical** in both arms and differ
only in an independent draw of the same condition, so they estimate the noise
floor of the paired estimator without exposing any order effect: both arms
have the same history, so there is nothing for a history contrast to reveal.

Full frozen Layer-1 apparatus. NEUTRAL_CONFLICT is evaluated here because the
null distribution of the metric is what Gate C exists to measure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dsi.artifacts import code_version, utc_now
from dsi.data import TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.rng import run_keys
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

# --- Frozen Layer-1 apparatus. Not reopened. ---
D_MODEL, N_LAYERS, LR, N_CUES, STEPS = 64, 4, 3e-3, 512, 1200
OVERLAP = 0.20
TAIL = "NEUTRAL_ALIGNED+W_P_INTERLEAVED@0.20"   # T1, smallest adequate
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH, TAU = 1024, 0.80
CONDITIONS = ("W_COMPETENCE", "P_COMPETENCE", "NEUTRAL_CONFLICT")
HISTORIES = {"A_WthenP": ("W_EXPLICIT", "P_EXPLICIT"),
             "B_PthenW": ("P_EXPLICIT", "W_EXPLICIT")}


def run_unit(history: str, seed: int, arm: int, out: Path) -> None:
    path = out / "units" / f"{history}__seed{seed}__arm{arm}.json"
    if path.exists():
        return
    task = TaskConfig(n_digits=3, n_cues=N_CUES)
    model_config = ModelConfig(vocab_size=task.vocab_size, d_model=D_MODEL,
                               n_layers=N_LAYERS, n_heads=4, d_ff=4 * D_MODEL)
    train_config = TrainConfig(learning_rate=LR, loss_positions="all")
    first, second = HISTORIES[history]
    # arm < 0 means "no forced divergence": the treatment configuration, where
    # the two arms differ by history rather than by an independent draw.
    keys = run_keys(seed, n_phases=3, arm=(None if arm < 0 else arm),
                    n_eval_points=len(OFFSETS))
    state = init_state(model_config, train_config, keys["init"])
    started = time.time()
    tok = lambda n: n * train_config.batch_size * task.seq_len

    phases = [
        PhaseSpec(first, tok(STEPS), "source"),
        PhaseSpec(f"{second}+{first}@{OVERLAP}",
                  tok(int(round(STEPS / (1.0 - OVERLAP)))), "target"),
        PhaseSpec(TAIL, tok(STEPS), "washout"),
    ]
    for index, phase in enumerate(phases):
        stream = "source_data" if phase.role == "source" else "target_data"
        state, _ = train_phase(
            state, phase, task, train_config, keys[f"{stream}.{index}"],
            eval_at=(phase_steps(phase, task, train_config),),
            eval_fn=lambda m, pt: None,
        )

    final = evaluate(state.model, task, keys[f"eval.2.{len(OFFSETS)-1}"],
                     batch_size=EVAL_BATCH, conditions=CONDITIONS, split="train")
    acc_w = final["W_COMPETENCE"].accuracy
    acc_p = final["P_COMPETENCE"].accuracy
    payload = {
        "history": history, "seed": seed, "arm": arm,
        "acc_w": acc_w, "acc_p": acc_p,
        "competent": min(acc_w, acc_p) >= TAU,
        # The frozen primary Claim-1 estimand, s(x) = log P(W|x) - log P(P|x).
        "neutral_logodds": final["NEUTRAL_CONFLICT"].logodds_w_minus_p,
        # Secondary / descriptive.
        "neutral_follows_w": final["NEUTRAL_CONFLICT"].follows_w,
        "neutral_follows_p": final["NEUTRAL_CONFLICT"].follows_p,
        "neutral_loss": final["NEUTRAL_CONFLICT"].loss,
        "overlap": OVERLAP, "tail": TAIL, "n_cues": N_CUES,
        "seconds": time.time() - started,
        "code_version": code_version(), "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/gate_c"))
    args = parser.parse_args()
    run_unit(args.history, args.seed, args.arm, args.out)


if __name__ == "__main__":
    main()
