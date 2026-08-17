"""Continue saved checkpoints on a candidate corpus: the `V(S, D)` primitive.

One job, used by three lanes:

* **Track 1** — give every discovery checkpoint the identical `B` continuation,
  so each saved state acquires a measured *future learnability*. That converts
  the discovery question from "which feature correlates with the arm label?"
  (confounded with the label itself) into "which internal variable predicts how
  much future benefit **this particular checkpoint** will receive?" — which is
  the estimand that matters and the one that can be tested on held-out seeds.

* **Temporal microscope** — the same call over densely-spaced checkpoints from
  one trajectory, cloned and given identical continuations, estimates
  `S_t -> future learnability` directly rather than plotting an activation
  curve and hoping.

* **State x data matrix** — sweep `--target-family` over candidate pools and
  the checkpoint set over incoming states to fill `V(S_i, D_j)`. That surface
  is the central estimand: does identical data have different marginal value
  depending on incoming state, and can telemetry predict it?

Every continuation from a given checkpoint set uses the **same data key**, so
branches differ in their incoming state and their corpus, never in the draw.

What the estimand actually is
-----------------------------
This script restores **weights only** and initializes a fresh optimizer state.
The quantity it measures is therefore

    V(W_t, D)      weight-conditioned future data value

and **not** the full training-state value ``V(S_t, D)``. Optimizer moments,
learning-rate position and accumulated history are deliberately discarded so
that branches differ in weights and corpus alone rather than in momentum.

That is the right control for "does this *model* respond differently to this
data", and it is the wrong instrument for "does readiness live in the weights
or in the optimizer/history". Answering the second needs a matched
continuation that restores optimizer state too, run against these same
branches. That variant is **not implemented**: `dsi.checkpoint` persists
weights and config only. It is recorded in `BACKLOG.md` rather than assumed
away, because a readiness signal that lived partly in optimizer state would be
invisible here and would change the interpretation.

    PYTHONPATH=src python scripts/continue_from_state.py \\
        --checkpoints artifacts/mediator_discovery/checkpoints \\
        --target-family BIND --target-steps 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

from dsi.checkpoint import checkpoint_paths, load_model
from dsi.microworld import BatchCache, MicroConfig, evaluate_stream
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, TrainState, phase_steps, train_phase


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--target-family", type=str, default="BIND")
    ap.add_argument("--target-steps", type=int, default=2000)
    ap.add_argument("--probe-every", type=int, default=125)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data-key", type=int, default=777,
                    help="shared across every branch, so branches differ only "
                         "in incoming state and corpus")
    ap.add_argument("--tag", type=str, default="",
                    help="label for this (corpus, budget) cell")
    ap.add_argument("--only-states", type=str, default="",
                    help="comma-separated state labels this worker owns. The "
                         "unit of ownership is the (state, corpus) CELL, not "
                         "the corpus: corpus-level sharding makes the slowest "
                         "shard the critical path, and the matrix is only "
                         "analyzable when it is balanced.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = MicroConfig()
    tc = TrainConfig(batch_size=args.batch, loss_positions="all")
    tag = args.tag or args.target_family.replace("#", "_").replace("+", "_")

    units = args.out / "units"
    units.mkdir(parents=True, exist_ok=True)

    stems = sorted({p.with_suffix("") for p in args.checkpoints.glob("*.eqx")})
    if args.only_states:
        owned = {x.strip() for x in args.only_states.split(",") if x.strip()}
        stems = [s for s in stems if s.name in owned]
    if not stems:
        print(f"no checkpoints under {args.checkpoints} matching this assignment")
        return 0

    t0 = time.time()
    for stem in stems:
        name = f"{stem.name}__{tag}.json"
        path = units / name
        if path.exists():
            continue

        model, mc = load_model(stem)
        # A fresh optimizer state: the continuation is a controlled branch from
        # a state, not a resumption of the run that produced it. Carrying
        # optimizer moments would make branches differ in accumulated momentum
        # as well as in weights.
        from dsi.train import init_state
        fresh = init_state(mc, tc, jax.random.PRNGKey(0))
        state = TrainState(model=model, opt_state=fresh.opt_state,
                           step=fresh.step, tokens_seen=fresh.tokens_seen)

        phase = PhaseSpec(family=args.target_family,
                          tokens=args.target_steps * tc.batch_size * cfg.seq_len,
                          role="target")
        key = jax.random.PRNGKey(args.data_key)
        eval_at = tuple(range(0, args.target_steps + 1, args.probe_every))

        def telemetry(m, _i):
            return {
                "BIND": evaluate_stream(m, cfg, "BIND", 90001, 512),
                "FACT": evaluate_stream(m, cfg, "FACT", 90002, 512),
                "BINDT": evaluate_stream(m, cfg, "BINDT", 90004, 512),
            }

        state, records = train_phase(
            state, phase, cfg, tc, key,
            eval_at=eval_at, eval_fn=telemetry,
            sampler=BatchCache(key, args.target_family, cfg, tc.batch_size,
                               phase_steps(phase, cfg, tc)),
        )
        trace = [{"step": int(r["step"]), **r["result"]} for r in records]
        acc = [x["BIND"]["accuracy"] for x in trace]

        payload = {
            "checkpoint": str(stem), "state_label": stem.name,
            "target_family": args.target_family, "target_steps": args.target_steps,
            "data_key": args.data_key, "tag": tag,
            "trace": trace,
            # The three summary functionals, kept separate. t=0 is a head start
            # carried in; AULC mixes head start with rate; final is endpoint.
            # Reporting only their sum has repeatedly hidden opposing effects.
            "t0": acc[0], "aulc": sum(acc) / len(acc), "final": acc[-1],
            "rate_only": (sum(acc) / len(acc)) - acc[0],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload) + "\n")
        tmp.replace(path)
        print(f"  {stem.name:22s} {tag:12s} t0 {acc[0]:.4f}  aulc {payload['aulc']:.4f}"
              f"  final {acc[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"done in {time.time() - t0:.0f}s -> {units}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
