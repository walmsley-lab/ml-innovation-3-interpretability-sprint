"""E4b prospective prototype — streaming runner with a hard confirmatory barrier.

Protocol: `docs/experiments/e4b_prototype_protocol.md` (`018ddd863114f662`).
**Feasibility prototype only.** See §0 of the protocol for what it cannot claim.

Two modes in one file, so a worker subprocess is this same script:

* **driver** (default) — maintains a slot pool and dispatches work in the
  protocol's priority order, launching each unit the moment its dependency is
  durably on disk;
* ``--work SPEC`` — executes exactly one unit and exits.

Why the readout is fused into the source unit
---------------------------------------------
The source runner writes checkpoint, behavioral readouts, internal readouts and
provenance in a single atomic unit at the 4000-step endpoint. A downstream job
therefore never has to wait for process exit, and never has to guess whether a
half-written checkpoint is the protocol state. There is no separate
"measure the state" pass to fall behind.

Priority, per protocol §6
-------------------------
``dev source`` > ``dev continuation`` > ``reserve source`` > ``reserve
continuation``. Development continuations preempt unfinished reserve
generation: a development row completes the fit sooner, and the fit is the
critical path. Nothing that is blocked on a dependency ever occupies a slot.

The barrier
-----------
Reserve continuations are refused unless the freeze artifact exists **and**
``--release-reserve`` is passed. Both conditions, deliberately: the flag alone
would let a mistyped command reveal held-out outcomes before predictions were
frozen, which is the one error this design exists to prevent.

    PYTHONPATH=src python scripts/run_e4b_prototype.py --slots 7
    PYTHONPATH=src python scripts/run_e4b_prototype.py --slots 7 --release-reserve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ARMS = {"A": "IND", "A_prime": "IND_R", "BG": "BG"}
DEV_SEEDS = (800, 801, 802)
RESERVE_SEEDS = (803,)
CORPORA = {"BIND": "BIND", "BINDT": "BINDT", "FACT": "FACT", "BIND_FACT": "BIND+FACT"}
SOURCE_STEPS = 4000
CONT_STEPS = 2000
PROBE_EVERY = 250
CONT_DATA_KEY = 777
OUT = Path("artifacts/e4b_prototype")
FREEZE = OUT / "freeze.json"
PROTOCOL_SHA = "018ddd863114f6628a54e3cda7f01f9672745abb074876ec711586d896bc67f3"


def state_labels(seeds):
    return [f"{arm}__seed{s:03d}" for s in seeds for arm in ARMS]


def state_path(label):
    return OUT / "states" / f"{label}.json"


def cell_path(label, tag):
    return OUT / "cells" / f"{label}__{tag}.json"


def ckpt_stem(label):
    return OUT / "checkpoints" / label


# --------------------------------------------------------------------------
# one unit of work
# --------------------------------------------------------------------------

def run_source(label: str) -> None:
    """Train to the protocol endpoint, then emit checkpoint + readouts atomically."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dsi.checkpoint import save_model
    from dsi.mechanism import (
        MProbeSpec, mediator_score, prefix_matching_scores, retrieval_scores,
    )
    from dsi.microworld import BatchCache, MicroConfig, evaluate_stream, sample_documents
    from dsi.model import ModelConfig
    from dsi.specs import PhaseSpec
    from dsi.state import StateProbeSpec, extract_state_features
    from dsi.train import TrainConfig, init_state, phase_steps, train_phase

    arm, seed_s = label.split("__seed")
    seed = int(seed_s)
    stream = ARMS[arm]

    cfg = MicroConfig()
    mc = ModelConfig(vocab_size=cfg.vocab_size, d_model=128, n_heads=4,
                     n_layers=4, d_ff=512, max_len=cfg.seq_len)
    tc = TrainConfig(batch_size=64, loss_positions="all")
    mspec, sspec = MProbeSpec(), StateProbeSpec(n_examples=128)
    probe = jnp.asarray(np.asarray(
        sample_documents(90777, cfg, "BIND", sspec.n_examples)))

    state = init_state(mc, tc, jax.random.PRNGKey(seed))
    phase = PhaseSpec(family=stream, tokens=SOURCE_STEPS * tc.batch_size * cfg.seq_len,
                      role="source")
    key = jax.random.PRNGKey(10_000 + seed)
    state, _ = train_phase(state, phase, cfg, tc, key,
                           sampler=BatchCache(key, stream, cfg, tc.batch_size,
                                              phase_steps(phase, cfg, tc)))

    # Checkpoint first: it is the artifact downstream work depends on, and a
    # continuation may legitimately start before this process exits.
    save_model(ckpt_stem(label), state.model, mc)

    rs = np.asarray(retrieval_scores(state.model, cfg))
    payload = {
        "protocol_sha256": PROTOCOL_SHA,
        "arm": arm, "source": stream, "seed": seed, "steps": SOURCE_STEPS,
        "split": "development" if seed in DEV_SEEDS else "reserve",
        "state_label": label, "checkpoint": str(ckpt_stem(label)),
        "M_scalar": mediator_score(state.model, mspec, cfg.vocab_size),
        "prefix_matching_per_head":
            np.asarray(prefix_matching_scores(state.model, mspec, cfg.vocab_size)).tolist(),
        "retrieval_per_head": rs.tolist(),
        "retrieval_max": float(rs.max()), "retrieval_mean": float(rs.mean()),
        "state_features": extract_state_features(
            state.model, probe, sspec, cfg.answer_target_index),
        "zero_shot_BIND": evaluate_stream(state.model, cfg, "BIND", 90001, 512),
        "zero_shot_FACT": evaluate_stream(state.model, cfg, "FACT", 90002, 512),
    }
    p = state_path(label)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(p)


def run_cont(label: str, tag: str) -> None:
    """One `V(S,D)` cell: continue this checkpoint on one candidate corpus."""
    import jax

    from dsi.checkpoint import load_model
    from dsi.microworld import BatchCache, MicroConfig, evaluate_stream
    from dsi.specs import PhaseSpec
    from dsi.train import TrainConfig, TrainState, init_state, phase_steps, train_phase

    family = CORPORA[tag]
    cfg = MicroConfig()
    tc = TrainConfig(batch_size=64, loss_positions="all")

    model, mc = load_model(ckpt_stem(label))
    # Weights only, fresh optimizer state: branches differ in incoming state
    # and corpus, never in accumulated momentum. This is `V(W_t, D)`.
    fresh = init_state(mc, tc, jax.random.PRNGKey(0))
    state = TrainState(model=model, opt_state=fresh.opt_state,
                       step=fresh.step, tokens_seen=fresh.tokens_seen)

    phase = PhaseSpec(family=family, tokens=CONT_STEPS * tc.batch_size * cfg.seq_len,
                      role="target")
    key = jax.random.PRNGKey(CONT_DATA_KEY)

    def telemetry(m, _i):
        return {"BIND": evaluate_stream(m, cfg, "BIND", 90001, 512),
                "FACT": evaluate_stream(m, cfg, "FACT", 90002, 512),
                "BINDT": evaluate_stream(m, cfg, "BINDT", 90004, 512)}

    _, records = train_phase(
        state, phase, cfg, tc, key,
        eval_at=tuple(range(0, CONT_STEPS + 1, PROBE_EVERY)), eval_fn=telemetry,
        sampler=BatchCache(key, family, cfg, tc.batch_size, phase_steps(phase, cfg, tc)))

    trace = [{"step": int(r["step"]), **r["result"]} for r in records]
    end = trace[-1]
    payload = {
        "protocol_sha256": PROTOCOL_SHA,
        "state_label": label, "corpus_tag": tag, "corpus_family": family,
        "split": "development" if int(label.split("seed")[1]) in DEV_SEEDS else "reserve",
        "target_steps": CONT_STEPS, "data_key": CONT_DATA_KEY, "trace": trace,
        # The common yardstick: mean final accuracy across the three
        # capabilities, never the capability this cell's own corpus trains.
        "objective_mean": sum(end[c]["accuracy"] for c in ("BIND", "FACT", "BINDT")) / 3.0,
        "final_by_capability": {c: end[c]["accuracy"] for c in ("BIND", "FACT", "BINDT")},
    }
    p = cell_path(label, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(p)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_queue(release_reserve: bool):
    """Work items as (priority, spec, dependency-ready predicate)."""
    q = []
    for label in state_labels(DEV_SEEDS):
        q.append((0, f"source:{label}", lambda: True))
    for label in state_labels(DEV_SEEDS):
        for tag in CORPORA:
            q.append((1, f"cont:{label}:{tag}",
                      (lambda s: (lambda: ckpt_stem(s).with_suffix(".eqx").exists()))(label)))
    for label in state_labels(RESERVE_SEEDS):
        q.append((2, f"source:{label}", lambda: True))
    if release_reserve:
        for label in state_labels(RESERVE_SEEDS):
            for tag in CORPORA:
                q.append((3, f"cont:{label}:{tag}",
                          (lambda s: (lambda: ckpt_stem(s).with_suffix(".eqx").exists()
                                      and FREEZE.exists()))(label)))
    return q


def done(spec: str) -> bool:
    kind, rest = spec.split(":", 1)
    if kind == "source":
        return state_path(rest).exists()
    label, tag = rest.rsplit(":", 1)
    return cell_path(label, tag).exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=str, default="")
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--cores-per-slot", type=int, default=4)
    ap.add_argument("--release-reserve", action="store_true")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated state labels this driver owns. The "
                         "unit of ownership is the STATE, so a state's source "
                         "and its four cells stay on one machine and the "
                         "continuation never waits on a checkpoint transfer.")
    args = ap.parse_args()

    if args.work:
        kind, rest = args.work.split(":", 1)
        if kind == "source":
            run_source(rest)
        else:
            label, tag = rest.rsplit(":", 1)
            run_cont(label, tag)
        return 0

    if args.release_reserve and not FREEZE.exists():
        print("REFUSED: --release-reserve passed but no freeze artifact at "
              f"{FREEZE}. Predictions must be frozen before reserve outcomes "
              "are generated (protocol §6).")
        return 2

    queue = sorted(build_queue(args.release_reserve), key=lambda t: t[0])
    if args.only:
        owned = {x.strip() for x in args.only.split(",") if x.strip()}
        queue = [w for w in queue
                 if w[1].split(":", 1)[1].split(":")[0] in owned]
        print(f"  owning {len(owned)} states -> {len(queue)} work items")
    running: dict[subprocess.Popen, tuple[str, float, int]] = {}
    free_cores = list(range(args.slots))
    t0 = time.time()
    launched = completed = 0

    while queue or running:
        for proc in list(running):
            if proc.poll() is None:
                continue
            spec, started, slot = running.pop(proc)
            free_cores.append(slot)
            completed += 1
            ok = "ok" if proc.returncode == 0 and done(spec) else "FAILED"
            print(f"  [{time.time()-t0:6.0f}s] {ok:6s} {spec:42s} "
                  f"({time.time()-started:.0f}s)  {completed} done, "
                  f"{len(queue)} queued", flush=True)

        progressed = True
        while progressed and free_cores:
            progressed = False
            for i, (prio, spec, ready) in enumerate(queue):
                if done(spec):
                    queue.pop(i)
                    progressed = True
                    break
                if not ready():
                    continue
                slot = free_cores.pop(0)
                lo = slot * args.cores_per_slot
                cmd = ["taskset", "-c", f"{lo}-{lo + args.cores_per_slot - 1}",
                       sys.executable, __file__, "--work", spec]
                env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1",
                           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false "
                                     "intra_op_parallelism_threads=1")
                running[subprocess.Popen(cmd, env=env,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)] = (
                    spec, time.time(), slot)
                queue.pop(i)
                launched += 1
                print(f"  [{time.time()-t0:6.0f}s] start  {spec:42s} "
                      f"prio {prio}  slot {slot}", flush=True)
                progressed = True
                break

        if running:
            time.sleep(args.poll)
        elif queue and not any(r() for _, _, r in queue):
            print("STALLED: remaining work is blocked on dependencies:")
            for _, spec, _ in queue:
                print(f"    {spec}")
            return 1

    print(f"\ndone: {launched} launched, {completed} completed in "
          f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
