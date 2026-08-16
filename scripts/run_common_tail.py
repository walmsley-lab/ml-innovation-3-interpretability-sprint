"""Common-tail adequacy gate (research.md 8d).

Claim 1 requires: different histories -> identical subsequent experience ->
both explicit capabilities retained -> then test neutral behaviour. The third
step is a precondition. A model that defaults to one strategy because it no
longer has the other is not exhibiting a preference.

Three predeclared tails, all identical across histories:

    T0  pure NEUTRAL_ALIGNED
    T1  NEUTRAL-dominant + modest balanced explicit W/P maintenance (20%)
    T2  NEUTRAL-dominant + stronger balanced maintenance (40%)

Maintenance uses W_P_INTERLEAVED, which is balanced 50/50 by construction, so
no tail leaves one family more recently exposed than the other.

Selection is on explicit competence only. NEUTRAL_CONFLICT is not generated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dsi.artifacts import code_version, utc_now
from dsi.data import GATE_B_CONDITIONS, TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.rng import run_keys
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

# Frozen Layer-1 apparatus.
D_MODEL, N_LAYERS, LR, N_CUES, STEPS = 64, 4, 3e-3, 512, 1200
OVERLAP = 0.20                      # frozen overlap floor
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH, TAU = 512, 0.80

TAILS = {
    "T0_pure_neutral": "NEUTRAL_ALIGNED",
    "T1_modest_maint": "NEUTRAL_ALIGNED+W_P_INTERLEAVED@0.20",
    "T2_stronger_maint": "NEUTRAL_ALIGNED+W_P_INTERLEAVED@0.40",
}
HISTORIES = {"A_WthenP": ("W_EXPLICIT", "P_EXPLICIT"),
             "B_PthenW": ("P_EXPLICIT", "W_EXPLICIT")}


def setup():
    task = TaskConfig(n_digits=3, n_cues=N_CUES)
    model = ModelConfig(vocab_size=task.vocab_size, d_model=D_MODEL,
                        n_layers=N_LAYERS, n_heads=4, d_ff=4 * D_MODEL)
    return task, model, TrainConfig(learning_rate=LR, loss_positions="all")


def run_unit(tail: str, history: str, seed: int, out: Path) -> None:
    path = out / "units" / f"{tail}__{history}__seed{seed}.json"
    if path.exists():
        return
    task, model_config, train_config = setup()
    first, second = HISTORIES[history]
    keys = run_keys(seed, n_phases=3, n_eval_points=len(OFFSETS))
    state = init_state(model_config, train_config, keys["init"])
    started = time.time()
    tok = lambda n: n * train_config.batch_size * task.seq_len

    # Phase 1: first skill. Phase 2: second skill at the frozen overlap floor,
    # with incoming exposure matched. Phase 3: the shared tail.
    p2_steps = int(round(STEPS / (1.0 - OVERLAP)))
    phases = [
        PhaseSpec(first, tok(STEPS), "source"),
        PhaseSpec(f"{second}+{first}@{OVERLAP}", tok(p2_steps), "target"),
        PhaseSpec(TAILS[tail], tok(STEPS), "washout"),
    ]
    trace = []
    for index, phase in enumerate(phases):
        stream = "source_data" if phase.role == "source" else "target_data"
        state, records = train_phase(
            state, phase, task, train_config, keys[f"{stream}.{index}"],
            eval_at=offset_steps(OFFSETS, phase_steps(phase, task, train_config)),
            eval_fn=lambda m, pt, _i=index: evaluate(
                m, task, keys[f"eval.{_i}.{pt}"], batch_size=EVAL_BATCH,
                conditions=GATE_B_CONDITIONS, split="train"),
        )
        for offset, record in zip(OFFSETS, records):
            trace.append({"phase": index, "offset": offset,
                          "acc_w": record["result"]["W_COMPETENCE"].accuracy,
                          "acc_p": record["result"]["P_COMPETENCE"].accuracy})

    pre = [t for t in trace if t["phase"] == 1][-1]
    post = trace[-1]
    payload = {
        "tail": tail, "tail_family": TAILS[tail], "history": history, "seed": seed,
        "overlap": OVERLAP, "n_cues": N_CUES, "steps_per_phase": STEPS,
        "pre_tail_w": pre["acc_w"], "pre_tail_p": pre["acc_p"],
        "post_tail_w": post["acc_w"], "post_tail_p": post["acc_p"],
        "pre_tail_coexistence": min(pre["acc_w"], pre["acc_p"]),
        "post_tail_coexistence": min(post["acc_w"], post["acc_p"]),
        "adequate": min(post["acc_w"], post["acc_p"]) >= TAU,
        "trace": trace, "seconds": time.time() - started,
        "code_version": code_version(), "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/common_tail"))
    args = parser.parse_args()
    run_unit(args.tail, args.history, args.seed, args.out)


if __name__ == "__main__":
    main()
