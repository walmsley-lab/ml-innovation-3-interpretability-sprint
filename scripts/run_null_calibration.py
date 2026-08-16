"""Gate C: measure the noise floor, then derive the seed count from it.

Identity-null pairs are conditions that are scientifically identical but
stochastically independent. Their expected effect is zero, and the spread of
their paired differences is the noise floor that the confirmatory design must
be powered against.

Research plan §6.2 distinguishes two notions, and they are not conflated
here:

    execution null   Everything that can be held identical is held identical.
                     Both arms share every RNG stream. What remains is
                     execution nondeterminism alone. Its spread should be at
                     or near zero, and anything else indicates a leak.

    data null        The arms differ in one independent draw of the same
                     neutral condition, which is exactly where a real
                     transfer pair differs. This is the yardstick the power
                     planner consumes.

The data null must differ from its treatment pair in the same place and
nowhere else. A null sharing more would understate sigma_pair and
under-power the design; one sharing less would overstate it and waste
compute. Neither failure is visible in the resulting numbers.

Reads the frozen regime from Gate B. Refuses to run without one.

    python scripts/run_null_calibration.py [--pairs N] [--regime PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.data import TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.power import plan_power
from dsi.rng import run_keys
from dsi.specs import EvalSpec, PhaseSpec, RunSpec, run_id
from dsi.stats import paired_summary, transfer_effect
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

# Prespecified: the smallest order effect on the conflict diagnostic that
# would be scientifically meaningful. Fixed before sigma_pair is known.
DELTA_MIN = 0.10
ALPHA = 0.05
POWER_TARGET = 0.90

OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 512
NULL_SEED_BASE = 2000  # disjoint from calibration (1000) and confirmatory (0)

# The neutral prefix. "MIX" is the balanced aligned family: both sources
# agree on it, so it privileges neither.
NEUTRAL = "MIX"
TARGET = "W"


def run_arm(seed_family: int, arm, task, model_config, train_config, steps, version):
    """One neutral-prefix arm: MIX -> W, returning the target-phase curve."""
    tokens = steps * train_config.batch_size * task.seq_len
    spec = RunSpec(
        parent_id=None,
        phases=(
            PhaseSpec(NEUTRAL, tokens, "source"),
            PhaseSpec(TARGET, tokens, "target"),
        ),
        model_config_id=f"d{model_config.d_model}l{model_config.n_layers}",
        data_version=f"wp-synth-d{task.n_digits}-c{task.n_cues}",
        seed_family=seed_family,
        arm=arm,
        evals=(EvalSpec("wp_null", "v1", offsets=OFFSETS),),
    )
    keys = run_keys(
        spec.seed_family, n_phases=spec.n_phases, arm=spec.arm,
        n_eval_points=spec.n_eval_points(0),
    )
    state = init_state(model_config, train_config, keys["init"])
    curve: list[float] = []

    for index, phase in enumerate(spec.phases):
        stream = "source_data" if phase.role == "source" else "target_data"
        n_steps = phase_steps(phase, task, train_config)

        def eval_fn(model, point, _i=index):
            return evaluate(
                model, task, keys[f"eval.{_i}.{point}"], batch_size=EVAL_BATCH,
                conditions=("w_only", "p_only"), split="train",
            )

        state, records = train_phase(
            state, phase, task, train_config, keys[f"{stream}.{index}"],
            eval_at=offset_steps(OFFSETS, n_steps), eval_fn=eval_fn,
        )
        if phase.role == "target":
            # Target-phase loss curve, including the t=0 point taken before
            # any target tokens.
            curve = [r["result"]["w_only"].loss for r in records]

    return run_id(spec, version), curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--regime", type=Path, default=Path("artifacts/capacity/regime.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/nulls"))
    args = parser.parse_args()

    if not args.regime.exists():
        raise SystemExit(
            f"no frozen regime at {args.regime}. Gate B must pass before the noise "
            "floor means anything: sigma_pair measured in one regime does not "
            "transfer to another."
        )
    regime = json.loads(args.regime.read_text())
    version = code_version()

    task = TaskConfig(n_digits=regime["n_digits"], n_cues=regime["n_cues"])
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=regime["d_model"], n_layers=regime["n_layers"],
        n_heads=max(1, regime["d_model"] // 16), d_ff=4 * regime["d_model"],
    )
    train_config = TrainConfig(learning_rate=regime["learning_rate"])
    steps = regime["steps_per_phase"]

    print(f"regime       {regime['label']} ({regime['params']:,} params)")
    print(f"null design  {NEUTRAL} -> {TARGET}, {args.pairs} pairs of each kind")
    print(f"prespecified delta_min={DELTA_MIN} alpha={ALPHA} power={POWER_TARGET}\n")

    started = time.time()
    rows: list[dict] = []
    deltas: dict[str, list[float]] = {"execution": [], "data": []}

    for kind in ("execution", "data"):
        for pair in range(args.pairs):
            seed_family = NULL_SEED_BASE + pair
            # execution null: both arms share every stream (arm=None).
            # data null: arms diverge on the source draw alone.
            arms = (None, None) if kind == "execution" else (0, 1)
            args_ = (task, model_config, train_config, steps, version)
            id_a, curve_a = run_arm(seed_family, arms[0], *args_)
            id_b, curve_b = run_arm(seed_family, arms[1], *args_)

            delta = transfer_effect(OFFSETS, curve_a, curve_b, normalize=True)
            deltas[kind].append(delta)
            rows.append({
                "kind": kind, "seed_pair": pair, "run_a": id_a, "run_b": id_b,
                "target": TARGET, "prefix": NEUTRAL, "null_delta": delta,
                "estimator": "aulc_normalized", "hardware": "cpu",
                "code_version": version, "recorded_at": utc_now(),
            })
            print(f"  {kind:9s} pair={pair}  delta={delta:+.5f}")

    ArtifactWriter(args.out).write("nulls", rows)
    print(f"\nran {len(rows)} null pairs in {time.time()-started:.0f}s -> {args.out}/nulls.parquet")

    print("\n--- null distributions ---")
    summaries = {}
    for kind, values in deltas.items():
        est = paired_summary(values, alpha=ALPHA)
        summaries[kind] = est
        centred = est.ci_low <= 0.0 <= est.ci_high
        print(f"  {kind:9s} mean={est.mean:+.5f} sd={est.sd:.5f} "
              f"CI=[{est.ci_low:+.5f}, {est.ci_high:+.5f}] "
              f"{'centred on zero' if centred else 'NOT CENTRED — invalidator'}")

    execution_sd = summaries["execution"].sd
    if execution_sd > summaries["data"].sd:
        print("\n  WARNING: execution null is noisier than the data null. The arms "
              "that share every stream should be the quieter pair; this indicates "
              "a leak in the sharing contract rather than a property of the data.")

    sigma_pair = summaries["data"].sd
    print("\n--- power plan (from the data null) ---")
    plan = plan_power(
        sigma_pair, DELTA_MIN, alpha=ALPHA, power=POWER_TARGET,
        tokens_per_run=3 * steps * train_config.batch_size * task.seq_len,
    )
    print(plan.render())

    path = Path(args.out) / "power_plan.json"
    path.write_text(json.dumps({
        "sigma_pair_data_null": sigma_pair,
        "sigma_pair_execution_null": execution_sd,
        "delta_min": DELTA_MIN, "alpha": ALPHA, "power_target": POWER_TARGET,
        "required_pairs": plan.required_pairs, "achieved_power": plan.achieved_power,
        "null_pairs_run": args.pairs, "regime": regime["label"],
        "code_version": version, "planned_at": utc_now(),
    }, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
