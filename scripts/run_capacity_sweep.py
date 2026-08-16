"""Gate B: sweep neutral adequacy variables and select the smallest regime.

Searches only over variables that cannot favour one source over the other:
model size, task difficulty (n_digits, n_cues) and phase duration, with the
learning rate as a nuisance parameter. Every criterion is symmetric in W and
P, which is what makes the selection neutral with respect to the phenomenon
under study.

The sweep evaluates ``w_only``, ``p_only`` and ``aligned``. It does not
evaluate ``conflict`` at all, so a regime cannot be selected on the magnitude
of the effect the project exists to measure. That is structural, not a
convention.

Three stages, cheapest first:

    B1  solo competence, generalization and learning window, at the screening
        seed. One phase per source.
    B2  retention, measured immediately after the second source *and* after
        the washout. Adequacy uses the pre-washout figure: retention after
        MIX confounds whether the learner held both sources with whether the
        shared washout restored one, and adequacy is a property of the
        learner.
    B3  replication of the full measurement over the remaining calibration
        seeds. Eligibility is the worst case across seeds, because a regime
        adequate on one seed and not another would fail under the
        confirmatory design it was selected for.

Execution
---------
Work is decomposed into units of (configuration, seed, stage). Each unit runs
in its own process, writes its own result file atomically, and is skipped on
restart if that file already exists. A killed sweep resumes rather than
restarting, which the first full sweep demonstrated the need for by losing
twenty-five minutes to a kill with nothing preserved.

Parallelism uses the memory-gated, staggered, retrying pool from
``scripts/gcp.py``, the same one used on the VM. Waiting is on process
handles, never on process-name matching.

    python scripts/run_capacity_sweep.py --workers 3
    python scripts/run_capacity_sweep.py --pilot --workers 2
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax.random as jr

from dsi.artifacts import ArtifactWriter, code_version, utc_now
from dsi.calibrate import (
    RegimeCandidate,
    RegimeCriteria,
    learning_window,
    select_regime,
    worst_case_over_seeds,
)
from dsi.data import TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig, count_params
from dsi.rng import run_keys
from dsi.specs import EvalSpec, PhaseSpec, RunSpec
from dsi.train import TrainConfig, init_state, offset_steps, phase_steps, train_phase

sys.stdout.reconfigure(line_buffering=True)

# --- Prespecified before the sweep runs ------------------------------------
CRITERIA = RegimeCriteria(
    tau_w=0.90,
    tau_p=0.90,
    tau_generalization=0.80,
    tau_retention=0.80,
    min_window=0.15,
)

CALIBRATION_SEEDS = (1000, 1001, 1002)
"""Disjoint from the confirmatory experiment's seed families.

Screening runs on the first; eligibility is replicated over all three before
the regime is frozen.
"""
SCREEN_SEED = CALIBRATION_SEEDS[0]

OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))
EVAL_BATCH = 256
SWEEP_CONDITIONS = ("w_only", "p_only", "aligned")  # note: no "conflict"

# Narrow neutral grid, eight regimes. Task difficulty and phase duration are
# fixed at the values the first sweep showed adequate for solo competence,
# generalization and both learning windows; only capacity and learning rate
# vary, because retention is the sole criterion still failing.
#
# loss_positions is fixed at "all". scripts/diagnose_coexistence.py tested
# answer-only loss at this regime and rejected it: best pre-washout
# coexistence 0.273 against 0.301 for full-token, with the second source also
# acquired far more weakly. The objective is not the explanation, so it is not
# an axis of this sweep.
FULL_GRID = {
    "n_digits": (3,),
    "n_cues": (256,),
    "model": ((64, 2), (64, 4), (128, 2), (128, 4)),
    "steps_per_phase": (600,),
    "learning_rate": (3e-3, 1e-3),
}
PILOT_GRID = {
    "n_digits": (3,),
    "n_cues": (256,),
    "model": ((64, 2),),
    "steps_per_phase": (600,),
    "learning_rate": (3e-3,),
}


# --- Configuration ---------------------------------------------------------


def label_of(settings) -> str:
    d_model, n_layers = settings["model"]
    return (f"d{d_model}l{n_layers}_nd{settings['n_digits']}"
            f"_nc{settings['n_cues']}_s{settings['steps_per_phase']}"
            f"_lr{settings['learning_rate']:g}")


def configure(settings):
    d_model, n_layers = settings["model"]
    task = TaskConfig(n_digits=settings["n_digits"], n_cues=settings["n_cues"])
    model_config = ModelConfig(
        vocab_size=task.vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=max(1, d_model // 16), d_ff=4 * d_model,
    )
    # loss_positions is fixed, not swept; see the note on FULL_GRID.
    return task, model_config, TrainConfig(
        learning_rate=settings["learning_rate"], loss_positions="all"
    )


def _spec(order, task, model_config, settings, seed: int, solo: bool) -> RunSpec:
    tokens = settings["steps_per_phase"] * TrainConfig().batch_size * task.seq_len
    first, second = order
    phases = (
        # A solo run is the W-only / P-only control: a single phase whose
        # acquisition is measured, so its role is "target".
        (PhaseSpec(first, tokens, "target"),) if solo else
        (PhaseSpec(first, tokens, "source"),
         PhaseSpec(second, tokens, "target"),
         PhaseSpec("MIX", tokens, "washout"))
    )
    return RunSpec(
        parent_id=None, phases=phases,
        model_config_id=f"d{model_config.d_model}l{model_config.n_layers}",
        data_version=f"wp-synth-d{task.n_digits}-c{task.n_cues}-s{task.split_seed}",
        seed_family=seed,
        evals=(EvalSpec("wp_calib", "v1", offsets=OFFSETS),),
    )


def run_sequence(order, task, model_config, train_config, settings, seed, *, solo):
    """Train a sequence, returning the solo curve and per-phase diagnostics."""
    spec = _spec(order, task, model_config, settings, seed, solo)
    keys = run_keys(spec.seed_family, n_phases=spec.n_phases, arm=spec.arm,
                    n_eval_points=spec.n_eval_points(0))
    state = init_state(model_config, train_config, keys["init"])
    own = "w_only" if order[0] == "W" else "p_only"
    curve, solo_heldout, per_phase = [], float("nan"), {}

    for index, phase in enumerate(spec.phases):
        stream = "source_data" if phase.role == "source" else "target_data"

        def eval_fn(model, point, _i=index):
            return evaluate(model, task, keys[f"eval.{_i}.{point}"],
                            batch_size=EVAL_BATCH, conditions=SWEEP_CONDITIONS,
                            split="train")

        state, records = train_phase(
            state, phase, task, train_config, keys[f"{stream}.{index}"],
            eval_at=offset_steps(OFFSETS, phase_steps(phase, task, train_config)),
            eval_fn=eval_fn,
        )
        per_phase[phase.role] = records[-1]["result"]
        if index == 0 and solo:
            curve = [r["result"][own].accuracy for r in records]
            # Generalization at peak competence, so it is identifiable
            # separately from retention.
            solo_heldout = evaluate(
                state.model, task, keys[f"eval.0.{len(OFFSETS)-1}"],
                batch_size=EVAL_BATCH, conditions=SWEEP_CONDITIONS, split="heldout",
            )[own].accuracy

    return curve, solo_heldout, per_phase


# --- Units -----------------------------------------------------------------


def unit_path(out: Path, label: str, seed: int, stage: str) -> Path:
    return out / "units" / f"{label}__seed{seed}__{stage}.json"


def write_unit(path: Path, payload: dict) -> None:
    """Atomic write, so a kill mid-write cannot leave a half-parsed unit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def execute_unit(settings, seed: int, stage: str, out: Path) -> None:
    """Run one (configuration, seed, stage) and record it."""
    task, model_config, train_config = configure(settings)
    label = label_of(settings)
    started = time.time()
    payload = {"label": label, "seed": seed, "stage": stage,
               "code_version": code_version(), **{
                   k: (f"{v[0]}x{v[1]}" if isinstance(v, tuple) else v)
                   for k, v in settings.items()}}

    if stage == "b1":
        curve_w, gen_w, _ = run_sequence(("W", "P"), task, model_config,
                                         train_config, settings, seed, solo=True)
        curve_p, gen_p, _ = run_sequence(("P", "W"), task, model_config,
                                         train_config, settings, seed, solo=True)
        window_w, window_p = learning_window(OFFSETS, curve_w), learning_window(OFFSETS, curve_p)
        payload.update({
            "params": count_params(init_state(model_config, train_config, jr.key(0)).model),
            "acc_w": max(curve_w), "acc_p": max(curve_p),
            "generalization_worst": min(gen_w, gen_p),
            "window_w": window_w.width, "window_w_censored": window_w.censored,
            "window_p": window_p.width, "window_p_censored": window_p.censored,
        })
    elif stage == "b2":
        _, _, wp = run_sequence(("W", "P"), task, model_config, train_config,
                                settings, seed, solo=False)
        _, _, pw = run_sequence(("P", "W"), task, model_config, train_config,
                                settings, seed, solo=False)
        # Both sides of the washout. Adequacy uses the pre-washout figure.
        payload.update({
            "retention_pre_washout": min(
                wp["target"]["w_only"].accuracy, wp["target"]["p_only"].accuracy,
                pw["target"]["w_only"].accuracy, pw["target"]["p_only"].accuracy),
            "retention_post_washout": min(
                wp["washout"]["w_only"].accuracy, wp["washout"]["p_only"].accuracy,
                pw["washout"]["w_only"].accuracy, pw["washout"]["p_only"].accuracy),
        })
    else:
        raise ValueError(f"unknown stage {stage!r}")

    payload["seconds"] = time.time() - started
    payload["recorded_at"] = utc_now()
    write_unit(unit_path(out, label, seed, stage), payload)


# --- Orchestration ---------------------------------------------------------


def _load_pool():
    spec = importlib.util.spec_from_file_location(
        "dsi_gcp", Path(__file__).parent / "gcp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch(pending, out: Path, workers: int, stagger: float) -> None:
    """Run pending units through the memory-gated, staggered, retrying pool."""
    if not pending:
        return
    pool = _load_pool()
    pool.STAGGER_SEC = stagger  # 90s is tuned for corpus loads, not 30s configs
    pool.POLL_SEC = 5
    commands = []
    for settings, seed, stage in pending:
        d_model, n_layers = settings["model"]
        commands.append((
            [sys.executable, __file__, "--unit",
             "--n-digits", str(settings["n_digits"]), "--n-cues", str(settings["n_cues"]),
             "--d-model", str(d_model), "--n-layers", str(n_layers),
             "--steps", str(settings["steps_per_phase"]),
             "--lr", str(settings["learning_rate"]),
             "--seed", str(seed), "--stage", stage, "--out", str(out)],
            # One XLA thread pool per worker, or N workers each grab every core
            # and contend instead of parallelising.
            {"OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 8) // max(1, workers)))},
        ))
    pool.run_pool(commands, parallel=workers)


def load_units(out: Path, stage: str) -> dict:
    directory = out / "units"
    if not directory.exists():
        return {}
    units = {}
    for path in directory.glob(f"*__{stage}.json"):
        payload = json.loads(path.read_text())
        units[(payload["label"], payload["seed"])] = payload
    return units


def candidate_from(settings, b1: dict, b2: dict, seed: int) -> RegimeCandidate:
    from dsi.calibrate import LearningWindow

    def window(prefix):
        return LearningWindow(
            t10=None, t90=None,
            width=b1[prefix], floor=0.0, ceiling=1.0,
            censored=b1[f"{prefix}_censored"],
        )

    task, model_config, train_config = configure(settings)
    steps = settings["steps_per_phase"]
    return RegimeCandidate(
        label=label_of(settings), params=b1["params"],
        tokens=3 * steps * train_config.batch_size * task.seq_len,
        n_digits=settings["n_digits"], d_model=model_config.d_model,
        n_layers=model_config.n_layers, learning_rate=settings["learning_rate"],
        steps_per_phase=steps, n_cues=settings["n_cues"], seeds=(seed,),
        acc_w=b1["acc_w"], acc_p=b1["acc_p"],
        generalization_worst=b1["generalization_worst"],
        retention_pre_washout=b2["retention_pre_washout"] if b2 else 0.0,
        retention_post_washout=b2["retention_post_washout"] if b2 else 0.0,
        window_w=window("window_w"), window_p=window("window_p"),
    )


def sweep(grid, out: Path, workers: int, stagger: float) -> None:
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    version = code_version()
    print(f"code_version {version}")
    print(f"criteria     {CRITERIA}")
    print(f"conditions   {SWEEP_CONDITIONS}  (conflict deliberately not measured)")
    print(f"seeds        screen={SCREEN_SEED} replicate={CALIBRATION_SEEDS}")
    print(f"configs      {len(combos)}   workers {workers}\n")
    started = time.time()

    def pending_for(stage, subset, seeds):
        done = load_units(out, stage)
        return [(s, seed, stage) for s in subset for seed in seeds
                if (label_of(s), seed) not in done]

    # --- B1: solo, screening seed -----------------------------------------
    todo = pending_for("b1", combos, [SCREEN_SEED])
    print(f"--- B1 solo ({len(todo)} to run, {len(combos) - len(todo)} resumed) ---")
    dispatch(todo, out, workers, stagger)
    b1 = load_units(out, "b1")

    def solo_ok(settings):
        record = b1.get((label_of(settings), SCREEN_SEED))
        if not record:
            return False
        candidate = candidate_from(settings, record, None, SCREEN_SEED)
        return not [f for f in candidate.failures(CRITERIA) if not f.startswith("retention")]

    survivors = [s for s in combos if solo_ok(s)]
    for settings in combos:
        record = b1.get((label_of(settings), SCREEN_SEED))
        if record:
            print(f"{label_of(settings):31s} A_W={record['acc_w']:.2f} "
                  f"A_P={record['acc_p']:.2f} gen={record['generalization_worst']:.2f} "
                  f"R_W={record['window_w']:.2f} R_P={record['window_p']:.2f} "
                  f"{record['seconds']:.0f}s {'ok' if solo_ok(settings) else 'fail'}")

    # --- B2: retention, screening seed ------------------------------------
    todo = pending_for("b2", survivors, [SCREEN_SEED])
    print(f"\n--- B2 retention ({len(survivors)} survivors, {len(todo)} to run) ---")
    dispatch(todo, out, workers, stagger)
    b2 = load_units(out, "b2")
    for settings in survivors:
        record = b2.get((label_of(settings), SCREEN_SEED))
        if record:
            print(f"{label_of(settings):31s} retention pre={record['retention_pre_washout']:.3f} "
                  f"post={record['retention_post_washout']:.3f} {record['seconds']:.0f}s")

    def fully_ok(settings, seed):
        first = b1.get((label_of(settings), seed))
        second = b2.get((label_of(settings), seed))
        return bool(first and second and
                    candidate_from(settings, first, second, seed).is_adequate(CRITERIA))

    replicate = [s for s in survivors if fully_ok(s, SCREEN_SEED)]

    # --- B3: replication over the remaining calibration seeds -------------
    others = [s for s in CALIBRATION_SEEDS if s != SCREEN_SEED]
    todo = pending_for("b1", replicate, others) + pending_for("b2", replicate, others)
    print(f"\n--- B3 replication ({len(replicate)} candidates x {len(others)} seeds, "
          f"{len(todo)} to run) ---")
    dispatch(todo, out, workers, stagger)
    b1, b2 = load_units(out, "b1"), load_units(out, "b2")

    # --- Aggregate, select, freeze ----------------------------------------
    candidates, rows = [], []
    for settings in combos:
        replicates = [
            candidate_from(settings, b1[(label_of(settings), seed)],
                           b2.get((label_of(settings), seed)), seed)
            for seed in CALIBRATION_SEEDS if (label_of(settings), seed) in b1
        ]
        if not replicates:
            continue
        worst = worst_case_over_seeds(replicates)
        candidates.append(worst)
        rows.append({
            "label": worst.label, "params": worst.params, "tokens": worst.tokens,
            "n_digits": worst.n_digits, "n_cues": worst.n_cues,
            "d_model": worst.d_model, "n_layers": worst.n_layers,
            "learning_rate": worst.learning_rate, "steps_per_phase": worst.steps_per_phase,
            "seeds": str(list(worst.seeds)), "n_seeds": len(worst.seeds),
            "acc_w": worst.acc_w, "acc_p": worst.acc_p,
            "generalization_worst": worst.generalization_worst,
            "retention_pre_washout": worst.retention_pre_washout,
            "retention_post_washout": worst.retention_post_washout,
            "window_w": worst.window_w.width, "window_p": worst.window_p.width,
            "adequate": worst.is_adequate(CRITERIA),
            "failures": "; ".join(worst.failures(CRITERIA)),
            "code_version": version, "recorded_at": utc_now(),
        })
    ArtifactWriter(out).write("capacity_sweep", rows)
    print(f"\nswept in {time.time()-started:.0f}s -> {out}/capacity_sweep.parquet")

    try:
        chosen = select_regime(candidates, CRITERIA)
    except ValueError as exc:
        print(f"\nGATE B NOT PASSED: {exc}")
        return
    freeze(chosen, out, version)


def freeze(chosen: RegimeCandidate, out: Path, version: str) -> None:
    """Write the complete frozen configuration, not a summary of it.

    Everything needed to reconstruct the regime exactly belongs here,
    including the task-complexity parameters. A regime recorded partially is
    a regime that quietly changes when a default moves.
    """
    settings = {
        "n_digits": chosen.n_digits, "n_cues": chosen.n_cues,
        "model": (chosen.d_model, chosen.n_layers),
        "steps_per_phase": chosen.steps_per_phase,
        "learning_rate": chosen.learning_rate,
    }
    task, model_config, train_config = configure(settings)
    frozen = {
        "label": chosen.label, "params": chosen.params,
        "task": asdict(task), "model": asdict(model_config),
        "train": asdict(train_config),
        "steps_per_phase": chosen.steps_per_phase,
        "tokens_per_phase": chosen.steps_per_phase * train_config.batch_size * task.seq_len,
        "measured": {
            "acc_w": chosen.acc_w, "acc_p": chosen.acc_p,
            "generalization_worst": chosen.generalization_worst,
            "retention_pre_washout": chosen.retention_pre_washout,
            "retention_post_washout": chosen.retention_post_washout,
            "window_w": chosen.window_w.width, "window_p": chosen.window_p.width,
        },
        "criteria": asdict(CRITERIA),
        "calibration_seeds": list(chosen.seeds),
        "eval_offsets": list(OFFSETS),
        "sweep_conditions": list(SWEEP_CONDITIONS),
        "code_version": version, "frozen_at": utc_now(),
    }
    path = Path(out) / "regime.json"
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"\nselected regime: {chosen.label} ({chosen.params:,} params), "
          f"replicated over {len(chosen.seeds)} seeds")
    print(f"froze complete configuration -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stagger", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/capacity"))
    parser.add_argument("--unit", action="store_true", help=argparse.SUPPRESS)
    for name in ("n-digits", "n-cues", "d-model", "n-layers", "steps", "seed"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--stage")
    args = parser.parse_args()

    if args.unit:
        settings = {
            "n_digits": args.n_digits, "n_cues": args.n_cues,
            "model": (args.d_model, args.n_layers),
            "steps_per_phase": args.steps, "learning_rate": args.lr,
        }
        execute_unit(settings, args.seed, args.stage, args.out)
        return

    sweep(PILOT_GRID if args.pilot else FULL_GRID, args.out, args.workers, args.stagger)


if __name__ == "__main__":
    main()
