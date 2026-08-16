"""Pre-sweep probe: how cue-map size sets the cue's learning window.

Gate B requires both sources to have an observable learning window. The cue
is the binding constraint: with few cue tokens it is effectively a copy
operation and saturates inside the first checkpoint interval, so no choice of
model size, phase length or learning rate produces a measurable window.

This probe fixes the one task-difficulty parameter that controls it, before
the main sweep, so the sweep does not have to spend its budget rediscovering
a monotone relationship. Recorded output at 600 steps, d_model=64, lr=3e-3:

    n_cues    16    R_P = 0.080   saturated within one checkpoint
    n_cues    64    R_P = 0.128
    n_cues   256    R_P = 0.232   clears min_window = 0.15
    n_cues  1024    R_P = 0.550

The probe measures only the cue, in isolation, and never touches the conflict
condition, so it cannot bias regime selection toward any order effect.

    python scripts/probe_cue_window.py
"""

from __future__ import annotations

import jax.random as jr

from dsi.calibrate import learning_window
from dsi.data import TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.rng import run_keys
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
STEPS = 600
CALIBRATION_SEED_FAMILY = 1000


def main() -> None:
    for n_cues in (16, 64, 256, 1024):
        task = TaskConfig(n_digits=3, n_cues=n_cues)
        model_config = ModelConfig(
            vocab_size=task.vocab_size, d_model=64, n_layers=2, n_heads=4, d_ff=256
        )
        train_config = TrainConfig(learning_rate=3e-3)
        keys = run_keys(CALIBRATION_SEED_FAMILY, n_phases=1, n_eval_points=len(OFFSETS))
        state = init_state(model_config, train_config, keys["init"])
        phase = PhaseSpec("P", STEPS * train_config.batch_size * task.seq_len, "target")

        state, records = train_phase(
            state, phase, task, train_config, keys["target_data.0"],
            eval_at=offset_steps(OFFSETS, phase_steps(phase, task, train_config)),
            eval_fn=lambda model, i: evaluate(
                model, task, keys[f"eval.0.{i}"], batch_size=256,
                conditions=("p_only",), split="train",
            ),
        )
        curve = [r["result"]["p_only"].accuracy for r in records]
        window = learning_window(OFFSETS, curve)
        width = "censored" if window.censored else f"{window.width:.3f}"
        print(f"n_cues={n_cues:5d} vocab={task.vocab_size:5d} "
              f"A_P={max(curve):.3f} R_P={width}")


if __name__ == "__main__":
    main()
