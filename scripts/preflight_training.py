"""V2 preflight, training half: checks P3, P4, P5.

No scientific claim is made here and no run is reported as a result. These are
the three preflight checks that need a trained model:

    P3  the off-distribution M probe detects the mechanism
        Training on A must raise M substantially more than training on A'.
        If it does not, either the manipulation does not induce the mechanism
        or the probe cannot see it, and the G1 scout would be uninterpretable.

    P4  B is learnable but does not ceiling
        The background arm must end with measurable headroom on BIND. The
        ceiling scout paid for this lesson: interleaving at 0.987 left nothing
        for any method to improve on.

    P5  the phase boundary alone produces nothing
        A BG -> BG run crosses a phase boundary with no informative source. If
        BIND accuracy jumps across that boundary, the boundary itself — not the
        source — is producing effects, and every arm is contaminated.

    PYTHONPATH=src python scripts/preflight_training.py --steps 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

from dsi.mechanism import MProbeSpec, mediator_score
from dsi.microworld import BatchCache, MicroConfig, evaluate_stream
from dsi.model import ModelConfig
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, phase_steps, train_phase

# Criteria, fixed before any check runs.
MIN_M_RATIO = 2.0        # M(A) must exceed M(A') by this factor over baseline
MIN_M_ABSOLUTE = 0.05    # and must rise at least this far above its own start
B_FLOOR = 0.05           # BIND must beat chance-ish
B_CEILING = 0.90         # and must leave headroom
MAX_BOUNDARY_JUMP = 0.03 # BIND accuracy change across an uninformative boundary


def _phase(stream: str, steps: int, cfg: MicroConfig, tc: TrainConfig, role: str) -> PhaseSpec:
    return PhaseSpec(family=stream, tokens=steps * tc.batch_size * cfg.seq_len, role=role)


def _run(cfg, mc, tc, spec, streams, steps, seed, probe_every):
    """Train through ``streams`` in order, sampling telemetry on the way."""
    state = init_state(mc, tc, jax.random.PRNGKey(seed))
    eval_at = tuple(range(0, steps + 1, probe_every))
    trace = []

    for phase_index, stream in enumerate(streams):
        def eval_fn(model, _index, _p=phase_index):
            return {
                "M": mediator_score(model, spec, cfg.vocab_size),
                "BIND": evaluate_stream(model, cfg, "BIND", 90001, 512)["accuracy"],
                "FACT": evaluate_stream(model, cfg, "FACT", 90002, 512)["accuracy"],
                "phase": _p,
            }

        data_key = jax.random.PRNGKey(1000 + seed * 10 + phase_index)
        phase = _phase(stream, steps, cfg, tc, "source" if phase_index == 0 else "target")
        state, records = train_phase(
            state, phase, cfg, tc, data_key,
            eval_at=eval_at,
            eval_fn=eval_fn,
            sampler=BatchCache(data_key, stream, cfg, tc.batch_size,
                               phase_steps(phase, cfg, tc)),
        )
        for r in records:
            trace.append({"step": int(r["step"]), **r["result"]})
    return trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/preflight"))
    ap.add_argument("--arms", type=str, default="",
                    help="comma-separated subset of A,A_prime,BG,BG_BG; empty means all. "
                         "The arms are independent, so running them as separate "
                         "processes is a pure wall-clock win.")
    ap.add_argument("--combine", action="store_true",
                    help="skip training; evaluate criteria from per-arm traces on disk")
    args = ap.parse_args()

    cfg = MicroConfig()
    mc = ModelConfig(vocab_size=cfg.vocab_size, d_model=args.d_model,
                     n_heads=args.heads, n_layers=args.layers,
                     d_ff=4 * args.d_model, max_len=cfg.seq_len)
    tc = TrainConfig(batch_size=args.batch, loss_positions="all")
    spec = MProbeSpec()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    ARMS = {
        "A": ("IND", "BIND"),
        "A_prime": ("IND_R", "BIND"),
        "BG": ("BG", "BIND"),
        "BG_BG": ("BG", "BG"),
    }
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ARMS)

    if not args.combine:
        print(f"model d{args.d_model} l{args.layers} h{args.heads}, "
              f"batch {args.batch}, {args.steps} steps/phase, arms {wanted}\n", flush=True)
        for label in wanted:
            trace = _run(cfg, mc, tc, spec, ARMS[label], args.steps,
                         args.seed, args.probe_every)
            (args.out / f"trace_{label}.json").write_text(json.dumps(trace) + "\n")
            print(f"  {label:8s} done  ({time.time() - t0:.0f}s)", flush=True)

    traces = {}
    for label in ARMS:
        path = args.out / f"trace_{label}.json"
        if path.exists():
            traces[label] = json.loads(path.read_text())
    missing = [a for a in ARMS if a not in traces]
    if missing:
        print(f"traces present: {sorted(traces)}; still missing {missing}. "
              "Re-run with --combine once every arm has finished.")
        return 0

    def at_end_of_phase(trace, phase):
        pts = [t for t in trace if t["phase"] == phase]
        return pts[-1]

    def at_start(trace):
        return trace[0]

    m_start = at_start(traces["A"])["M"]
    m_a = at_end_of_phase(traces["A"], 0)["M"]
    m_ap = at_end_of_phase(traces["A_prime"], 0)["M"]

    rise_a, rise_ap = m_a - m_start, m_ap - m_start
    p3 = {
        "M_at_init": m_start,
        "M_after_A": m_a,
        "M_after_A_prime": m_ap,
        "rise_A": rise_a,
        "rise_A_prime": rise_ap,
        "ratio": (rise_a / rise_ap) if rise_ap > 1e-6 else float("inf"),
        "pass": bool(rise_a >= MIN_M_ABSOLUTE
                     and (rise_ap <= 1e-6 or rise_a / rise_ap >= MIN_M_RATIO)),
    }

    bg_final = at_end_of_phase(traces["BG"], 1)["BIND"]
    p4 = {
        "BIND_final_background_arm": bg_final,
        "floor": B_FLOOR, "ceiling": B_CEILING,
        "pass": bool(B_FLOOR <= bg_final <= B_CEILING),
    }

    before = at_end_of_phase(traces["BG_BG"], 0)["BIND"]
    after = [t for t in traces["BG_BG"] if t["phase"] == 1][0]["BIND"]
    p5 = {
        "BIND_before_boundary": before,
        "BIND_after_boundary": after,
        "jump": abs(after - before),
        "pass": bool(abs(after - before) <= MAX_BOUNDARY_JUMP),
    }

    checks = {"P3_m_probe": p3, "P4_b_headroom": p4, "P5_phase_boundary": p5}
    for name, r in checks.items():
        print(f"\n[{'PASS' if r['pass'] else 'FAIL'}] {name}")
        for k, v in r.items():
            if k == "pass":
                continue
            print(f"         {k:28s} {v:.4f}" if isinstance(v, float) else f"         {k:28s} {v}")

    (args.out / "preflight_training.json").write_text(
        json.dumps({"checks": checks, "traces": traces}, indent=2) + "\n")

    failed = [n for n, r in checks.items() if not r["pass"]]
    if failed:
        print(f"\nPREFLIGHT FAILED: {', '.join(failed)}. Fix only the failed property.")
        return 1
    print("\nP3/P4/P5 pass. The G1 six-seed scout is licensed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
