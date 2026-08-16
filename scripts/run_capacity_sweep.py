"""Gate B: sweep neutral adequacy variables and select the smallest regime.

Searches only over variables that cannot favour one source over the other:

    model size          d_model x n_layers
    task difficulty     n_digits
    phase duration      steps per phase
    learning rate       a nuisance parameter (technical.md Nuisance HPO)

Learning rate is included because the developmental-resolution criterion is
otherwise unreachable: the cue is trivially copyable and saturates within a
few checkpoints at a high learning rate, and no choice of model size or task
difficulty widens that window. It is neutral in the sense that matters here,
affecting both sources the same way, and it is frozen after selection rather
than tuned per curriculum.

The sweep evaluates ``w_only``, ``p_only`` and ``aligned``. It does not
evaluate ``conflict`` at all, so a regime cannot be selected on the magnitude
of the effect the project exists to measure. That is a structural guarantee,
not a convention.

Calibration uses its own seed families, disjoint from the confirmatory
experiment's.

The sweep runs in two stages, because the criteria have different costs:

    Stage B1  solo phases only. Competence, generalization and the learning
              window are properties of learning a source in isolation, and
              cost one phase per source.

    Stage B2  the full source -> target -> washout sequence, run only for
              configurations that already passed B1. Retention is the only
              criterion that needs it, and it costs three times as much.

Generalization is measured at the end of the solo phase, not after the full
sequence. Measured at the end, a forgotten source scores at chance and the
regime is rejected for failing to generalize when it actually failed to
retain. The two are separate criteria and are measured where each is
identifiable.

    python scripts/run_capacity_sweep.py [--pilot] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import jax.random as jr

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.calibrate import RegimeCandidate, RegimeCriteria, learning_window, select_regime
from dsi.data import TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig, count_params
from dsi.rng import run_keys
from dsi.specs import EvalSpec, PhaseSpec, RunSpec, run_id
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

# --- Prespecified before the sweep runs ------------------------------------
CRITERIA = RegimeCriteria(
    tau_w=0.90,
    tau_p=0.90,
    tau_generalization=0.80,
    tau_retention=0.80,
    min_window=0.15,  # fraction of the solo phase; ~2 checkpoints at 11 per phase
)

# Calibration seed families, disjoint from the confirmatory experiment's.
CALIBRATION_SEED_FAMILY = 1000

OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 256
SWEEP_CONDITIONS = ("w_only", "p_only", "aligned")  # note: no "conflict"

# n_cues is fixed at 256 and the learning rate at 3e-3 by the recorded
# pre-sweep probe in scripts/probe_cue_window.py, which showed the cue's
# learning window rising monotonically with cue-map size and clearing
# min_window only from 256, and by the first pilot, in which lr=1e-3 failed
# the rule-competence threshold at every phase length tried. Both were fixed
# on measurements of the sources in isolation. Neither probe measured the
# conflict condition.
FULL_GRID = {
    "n_digits": (3, 4),
    "n_cues": (256,),
    "model": ((32, 2), (64, 2), (128, 2)),
    "steps_per_phase": (600, 1200),
    "learning_rate": (3e-3,),
}
PILOT_GRID = {
    "n_digits": (3,),
    "n_cues": (256,),
    "model": ((64, 2),),
    "steps_per_phase": (600,),
    "learning_rate": (3e-3,),
}


def _phase_stream(role: str, index: int) -> str:
    return f"{'source_data' if role == 'source' else 'target_data'}.{index}"


def _build(order, task, model_config, steps, solo_only: bool) -> RunSpec:
    tokens = steps * TrainConfig().batch_size * task.seq_len
    first, second = order
    if solo_only:
        # A solo run is the W-only / P-only control arm of research plan §7:
        # a single phase whose acquisition is what is being measured, so its
        # role is "target". RunSpec refuses a run made only of source phases,
        # correctly, because that is not a comparison.
        phases = [PhaseSpec(first, tokens, "target")]
    else:
        phases = [
            PhaseSpec(first, tokens, "source"),
            PhaseSpec(second, tokens, "target"),
            PhaseSpec("MIX", tokens, "washout"),
        ]
    return RunSpec(
        parent_id=None,
        phases=tuple(phases),
        model_config_id=f"d{model_config.d_model}l{model_config.n_layers}",
        data_version=f"wp-synth-d{task.n_digits}-c{task.n_cues}-s{task.split_seed}",
        seed_family=CALIBRATION_SEED_FAMILY,
        evals=(EvalSpec("wp_calib", "v1", offsets=OFFSETS),),
    )


def run_sequence(order, task, model_config, train_config, steps, *, solo_only: bool):
    """Train a sequence and return the solo curve plus end-state diagnostics."""
    spec = _build(order, task, model_config, steps, solo_only)
    keys = run_keys(
        spec.seed_family, n_phases=spec.n_phases, arm=spec.arm,
        n_eval_points=spec.n_eval_points(0),
    )
    state = init_state(model_config, train_config, keys["init"])
    own = "w_only" if order[0] == "W" else "p_only"
    solo_curve: list[float] = []
    solo_heldout = float("nan")

    for index, phase in enumerate(spec.phases):
        n_steps = phase_steps(phase, task, train_config)

        def eval_fn(model, point, _i=index):
            return evaluate(
                model, task, keys[f"eval.{_i}.{point}"],
                batch_size=EVAL_BATCH, conditions=SWEEP_CONDITIONS, split="train",
            )

        state, records = train_phase(
            state, phase, task, train_config, keys[_phase_stream(phase.role, index)],
            eval_at=offset_steps(OFFSETS, n_steps), eval_fn=eval_fn,
        )
        if index == 0:
            solo_curve = [r["result"][own].accuracy for r in records]
            # Generalization is measured here, at peak competence, so that it
            # is identifiable separately from retention.
            solo_heldout = evaluate(
                state.model, task, keys[f"eval.0.{len(OFFSETS)-1}"],
                batch_size=EVAL_BATCH, conditions=SWEEP_CONDITIONS, split="heldout",
            )[own].accuracy

    return solo_curve, solo_heldout, records[-1]["result"]


def _configure(settings):
    n_digits = settings["n_digits"]
    d_model, n_layers = settings["model"]
    task = TaskConfig(n_digits=n_digits, n_cues=settings["n_cues"])
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=max(1, d_model // 16), d_ff=4 * d_model,
    )
    train_config = TrainConfig(learning_rate=settings["learning_rate"])
    label = (f"d{d_model}l{n_layers}_nd{n_digits}_nc{settings['n_cues']}"
             f"_s{settings['steps_per_phase']}_lr{settings['learning_rate']:g}")
    return task, model_config, train_config, label


def sweep(grid, out: Path) -> None:
    version = code_version()
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    print(f"code_version {version}")
    print(f"criteria     tau_W={CRITERIA.tau_w} tau_P={CRITERIA.tau_p} "
          f"tau_gen={CRITERIA.tau_generalization} tau_ret={CRITERIA.tau_retention} "
          f"min_window={CRITERIA.min_window}")
    print(f"conditions   {SWEEP_CONDITIONS}  (conflict deliberately not measured)")
    print(f"configs      {len(combos)}\n")
    started = time.time()

    print("--- Stage B1: solo competence, generalization, learning window ---")
    stage1 = []
    for settings in combos:
        task, model_config, train_config, label = _configure(settings)
        steps = settings["steps_per_phase"]
        curve_w, gen_w, _ = run_sequence(("W", "P"), task, model_config, train_config, steps, solo_only=True)
        curve_p, gen_p, _ = run_sequence(("P", "W"), task, model_config, train_config, steps, solo_only=True)
        params = count_params(init_state(model_config, train_config, jr.key(0)).model)
        record = {
            "settings": settings, "label": label, "params": params, "task": task,
            "model_config": model_config, "train_config": train_config,
            "acc_w": max(curve_w), "acc_p": max(curve_p),
            "generalization_worst": min(gen_w, gen_p),
            "window_w": learning_window(OFFSETS, curve_w),
            "window_p": learning_window(OFFSETS, curve_p),
        }
        provisional = _candidate(record, retention=1.0)
        record["solo_failures"] = tuple(
            f for f in provisional.failures(CRITERIA) if not f.startswith("retention")
        )
        stage1.append(record)
        print(f"{label:30s} p={params:>7,} A_W={record['acc_w']:.2f} A_P={record['acc_p']:.2f} "
              f"gen={record['generalization_worst']:.2f} "
              f"R_W={_fmt(record['window_w'])} R_P={_fmt(record['window_p'])} "
              f"{'ok' if not record['solo_failures'] else 'fail: ' + ', '.join(record['solo_failures'])}")

    survivors = [r for r in stage1 if not r["solo_failures"]]
    print(f"\n--- Stage B2: retention after washout ({len(survivors)} survivors) ---")
    if not survivors:
        print("  none; skipping")

    candidates, rows = [], []
    for record in stage1:
        retention = float("nan")
        if not record["solo_failures"]:
            steps = record["settings"]["steps_per_phase"]
            args = (record["task"], record["model_config"], record["train_config"], steps)
            _, _, final_wp = run_sequence(("W", "P"), *args, solo_only=False)
            _, _, final_pw = run_sequence(("P", "W"), *args, solo_only=False)
            retention = min(
                final_wp["w_only"].accuracy, final_wp["p_only"].accuracy,
                final_pw["w_only"].accuracy, final_pw["p_only"].accuracy,
            )
            print(f"{record['label']:30s} retention={retention:.3f}")

        candidate = _candidate(record, retention=0.0 if retention != retention else retention)
        candidates.append(candidate)
        failures = candidate.failures(CRITERIA)
        rows.append({
            "label": record["label"], "params": record["params"], "tokens": candidate.tokens,
            **{k: (v if not isinstance(v, tuple) else f"{v[0]}x{v[1]}")
               for k, v in record["settings"].items()},
            "acc_w": record["acc_w"], "acc_p": record["acc_p"],
            "generalization_worst": record["generalization_worst"],
            "retention_worst": retention,
            "window_w": record["window_w"].width, "window_p": record["window_p"].width,
            "window_w_censored": record["window_w"].censored,
            "window_p_censored": record["window_p"].censored,
            "reached_stage_b2": not record["solo_failures"],
            "adequate": not failures, "failures": "; ".join(failures),
            "code_version": version, "recorded_at": utc_now(),
        })

    ArtifactWriter(out).write("capacity_sweep", rows)
    print(f"\nswept {len(combos)} configs in {time.time()-started:.0f}s "
          f"-> {out}/capacity_sweep.parquet")

    try:
        chosen = select_regime(candidates, CRITERIA)
    except ValueError as exc:
        print(f"\nGATE B NOT PASSED: {exc}")
        return

    print(f"\nselected regime: {chosen.label}  ({chosen.params:,} params)")
    frozen = {
        "label": chosen.label, "params": chosen.params, "n_digits": chosen.n_digits,
        "d_model": chosen.d_model, "n_layers": chosen.n_layers,
        "learning_rate": chosen.learning_rate, "steps_per_phase": chosen.steps_per_phase,
        "acc_w": chosen.acc_w, "acc_p": chosen.acc_p,
        "generalization_worst": chosen.generalization_worst,
        "retention_worst": chosen.retention_worst,
        "window_w": chosen.window_w.width, "window_p": chosen.window_p.width,
        "criteria": CRITERIA.__dict__, "code_version": version, "frozen_at": utc_now(),
    }
    path = Path(out) / "regime.json"
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"froze regime -> {path}")


def _candidate(record, *, retention: float) -> RegimeCandidate:
    settings = record["settings"]
    task, train_config = record["task"], record["train_config"]
    return RegimeCandidate(
        label=record["label"], params=record["params"],
        tokens=3 * settings["steps_per_phase"] * train_config.batch_size * task.seq_len,
        n_digits=settings["n_digits"], d_model=record["model_config"].d_model,
        n_layers=record["model_config"].n_layers,
        learning_rate=settings["learning_rate"],
        steps_per_phase=settings["steps_per_phase"],
        acc_w=record["acc_w"], acc_p=record["acc_p"],
        generalization_worst=record["generalization_worst"],
        retention_worst=retention,
        window_w=record["window_w"], window_p=record["window_p"],
    )


def _fmt(window) -> str:
    return "cens" if window.censored else f"{window.width:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="run the small pilot grid")
    parser.add_argument("--out", type=Path, default=Path("artifacts/capacity"))
    args = parser.parse_args()
    sweep(PILOT_GRID if args.pilot else FULL_GRID, args.out)


if __name__ == "__main__":
    main()
