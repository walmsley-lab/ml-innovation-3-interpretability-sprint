"""Milestone A exit criterion: one local W->P / P->W paired experiment.

Runs the order comparison of research plan §11,

    Delta_order = Y(W -> P) - Y(P -> W),

paired on seed family, and writes versioned Parquet artifacts.

This is a smoke test of the machinery, not a scientific result. The model
scale, token budgets, and seed count are all uncalibrated: Gate B fixes the
first, and Gate C determines the third from measured sigma_pair. Nothing
printed here should be read as evidence for or against Claim 1.

    python scripts/run_wp_local.py [--seed-families N] [--steps N] [--out DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax.random as jr

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.data import TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig, count_params
from dsi.rng import run_keys
from dsi.specs import EvalSpec, PhaseSpec, RunSpec, run_id
from dsi.stats import equivalence_verdict, paired_summary
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

# Prespecified before looking at any output.
TAU_COMPETENCE = 0.90  # tau_W and tau_P; conflict is uninterpretable below this
DELTA_MIN = 0.10  # smallest order effect worth calling meaningful
EVAL_OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0)
EVAL_BATCH = 512

CURRICULA = {"W_then_P": ("W", "P"), "P_then_W": ("P", "W")}


def phase_data_key(keys, phase_index: int, role: str):
    """Which RNG stream a phase draws its examples from.

    Source phases draw from the source stream, the one place a matched pair
    is permitted to differ. Target and washout phases draw from the target
    stream, which is shared across arms so that both see the same example
    ordering and differ only in the history they bring to it.
    """
    stream = "source_data" if role == "source" else "target_data"
    return keys[f"{stream}.{phase_index}"]


def run_one(spec: RunSpec, task, model_config, train_config, version: str) -> dict:
    """Execute one arm and return its records."""
    keys = run_keys(
        spec.seed_family,
        n_phases=spec.n_phases,
        arm=spec.arm,
        n_eval_points=spec.n_eval_points(0),
    )
    rid = run_id(spec, version)
    state = init_state(model_config, train_config, keys["init"])

    curve_rows: list[dict] = []
    eval_rows: list[dict] = []
    run_rows: list[dict] = []
    suite = spec.evals[0]

    for index, phase in enumerate(spec.phases):
        n_steps = phase_steps(phase, task, train_config)
        eval_at = offset_steps(suite.offsets, n_steps)

        def eval_fn(model, point, _index=index):
            return evaluate(
                model, task, keys[f"eval.{_index}.{point}"], batch_size=EVAL_BATCH
            )

        state, records = train_phase(
            state, phase, task, train_config,
            phase_data_key(keys, index, phase.role),
            eval_at=eval_at, eval_fn=eval_fn,
        )

        for point, record in enumerate(records):
            for condition, result in record["result"].items():
                row = {
                    "run_id": rid,
                    "seed_family": spec.seed_family,
                    "phase_index": index,
                    "phase_family": phase.family,
                    "phase_role": phase.role,
                    "offset": suite.offsets[point],
                    "step_in_phase": record["step_in_phase"],
                    "tokens_in_phase": record["tokens_in_phase"],
                    "tokens_seen": record["tokens_seen"],
                    "condition": condition,
                    "loss": result.loss,
                    "accuracy": result.accuracy,
                    "follows_w": result.follows_w,
                    "follows_p": result.follows_p,
                }
                curve_rows.append(row)
                if point == len(records) - 1:
                    eval_rows.append(
                        {
                            **{k: row[k] for k in ("run_id", "condition", "loss",
                                                   "accuracy", "follows_w", "follows_p")},
                            "checkpoint_step": record["step"],
                            "suite_id": suite.suite_id,
                            "suite_version": suite.version,
                            "n_examples": result.n_examples,
                        }
                    )

        run_rows.append(
            {
                "run_id": rid,
                "parent_id": spec.parent_id,
                "seed_family": spec.seed_family,
                "arm": spec.arm,
                "curriculum": "->".join(p.family for p in spec.phases),
                "phase": index,
                "phase_family": phase.family,
                "phase_role": phase.role,
                "tokens": phase.tokens,
                "code_version": version,
                "data_version": spec.data_version,
                "model_config_id": spec.model_config_id,
                "accelerator": "cpu",
                "provider": "local",
                "cost_usd": 0.0,
                "status": "COMPLETED",
                "recorded_at": utc_now(),
            }
        )

    final = {c: r for c, r in records[-1]["result"].items()}
    return {
        "run_id": rid, "curves": curve_rows, "evals": eval_rows,
        "runs": run_rows, "final": final,
    }


def build_spec(curriculum: tuple[str, str], seed_family: int, tokens: int) -> RunSpec:
    source, target = curriculum
    return RunSpec(
        parent_id=None,
        phases=(PhaseSpec(source, tokens, "source"), PhaseSpec(target, tokens, "target")),
        model_config_id="wp-tiny-v1",
        data_version="wp-synth-v1",
        seed_family=seed_family,
        evals=(EvalSpec("wp_diag", "v1", offsets=EVAL_OFFSETS),),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-families", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250, help="optimizer steps per phase")
    parser.add_argument("--out", type=Path, default=Path("artifacts/wp_local"))
    args = parser.parse_args()

    task = TaskConfig(n_digits=2)
    model_config = ModelConfig(vocab_size=task.vocab_size)
    train_config = TrainConfig()
    tokens = args.steps * train_config.batch_size * task.seq_len
    version = code_version()

    print(f"code_version   {version}")
    print(f"task           n_classes={task.n_classes} n_digits={task.n_digits} "
          f"vocab={task.vocab_size} seq_len={task.seq_len}")
    print(f"phases         {args.steps} steps = {tokens} tokens each")
    print(f"seed families  {args.seed_families}\n")

    all_curves, all_evals, all_runs = [], [], []
    finals: dict[str, dict[int, dict]] = {name: {} for name in CURRICULA}
    started = time.time()

    for name, curriculum in CURRICULA.items():
        for seed_family in range(args.seed_families):
            spec = build_spec(curriculum, seed_family, tokens)
            out = run_one(spec, task, model_config, train_config, version)
            all_curves += out["curves"]
            all_evals += out["evals"]
            all_runs += out["runs"]
            finals[name][seed_family] = out["final"]
            f = out["final"]
            print(f"{name:9s} s={seed_family}  {out['run_id'][:12]}  "
                  f"w_only={f['w_only'].accuracy:.3f} p_only={f['p_only'].accuracy:.3f} "
                  f"follows_w={f['conflict'].follows_w:.3f}")

    writer = ArtifactWriter(args.out)
    paths = [
        writer.write_runs(all_runs),
        writer.write_learning_curves(all_curves),
        writer.write_evaluations(all_evals),
    ]
    print(f"\nwrote {sum(1 for _ in paths)} tables to {args.out}/ in {time.time()-started:.0f}s")
    for p in paths:
        print(f"  {p.name:22s} {p.stat().st_size:>8,} bytes")

    _report(finals, args.seed_families, model_config, task, train_config)


def _report(finals, n_families, model_config, task, train_config) -> None:
    print(f"\nmodel params   {count_params(init_state(model_config, train_config, jr.key(0)).model):,}")

    print("\n--- competence gates (invalidator, not a result) ---")
    failures = []
    for name in CURRICULA:
        for s in range(n_families):
            f = finals[name][s]
            for source, key in (("W", "w_only"), ("P", "p_only")):
                acc = f[key].accuracy
                if acc < TAU_COMPETENCE:
                    failures.append(f"{name} s={s} A_{source}={acc:.3f}")
    if failures:
        print(f"  FAILED tau={TAU_COMPETENCE} in {len(failures)} arm-conditions:")
        for line in failures[:8]:
            print(f"    {line}")
        print("  Conflict behaviour below is UNINTERPRETABLE. A model that follows one")
        print("  source because it never learned the other is not path dependence.")
    else:
        print(f"  passed: all arms reach A_W and A_P >= {TAU_COMPETENCE}")

    print("\n--- order effect on conflict (research plan §11) ---")
    deltas = [
        finals["W_then_P"][s]["conflict"].follows_w - finals["P_then_W"][s]["conflict"].follows_w
        for s in range(n_families)
    ]
    for s, d in enumerate(deltas):
        print(f"  s={s}  delta={d:+.4f}")
    est = paired_summary(deltas)
    print(f"\n  mean {est.mean:+.4f}   95% CI [{est.ci_low:+.4f}, {est.ci_high:+.4f}]"
          f"   sigma_pair {est.sd:.4f}   n={est.n}")
    print(f"  verdict vs delta_min={DELTA_MIN}: {equivalence_verdict(est, DELTA_MIN)}")
    if failures:
        print("\n  Reported for machinery validation only; the competence gate failed.")


if __name__ == "__main__":
    main()
