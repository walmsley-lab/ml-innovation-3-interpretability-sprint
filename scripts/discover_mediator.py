"""Track 1: find what actually carries the transfer, now that `M` is falsified.

`CLAIMS.md` H1-H4 stand — the advantage is real, selective, identity-general
and scale-persistent — while A2b/A2c are falsified: the off-distribution
prefix-matching score `M` is not A-selective. Something else is doing the work.

This script is the **candidate generator**, and is explicitly exploratory. It
trains the source phase for each arm, saves the checkpoint at the phase
boundary, and records three families of state measurement side by side:

* ``M`` — off-distribution prefix matching. Included so the falsification is
  visible in the same table as its replacement rather than argued from memory.
* ``retrieval`` — on-distribution attention from the prediction site to the
  bound value. Task-specific by construction and therefore circular on its own;
  it is here to *nominate* heads, not to prove anything.
* the 137 basis-invariant state features from :mod:`dsi.state` — norms,
  participation ratios, example-Gram spectra, attention statistics.

Nothing here is causal. The output is a ranked list of candidate head sets and
a per-feature A-vs-A' separation. Causal weight comes afterwards, from ablating
the nominated heads across **all three arms** and reading the arm x ablation
interaction with ``run_v2_units.py --ablation``.

    PYTHONPATH=src python scripts/discover_mediator.py --seeds 500-502
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import jax
import numpy as np

from dsi.checkpoint import save_model
from dsi.mechanism import (
    MProbeSpec, mediator_score, prefix_matching_scores, retrieval_scores,
)
from dsi.microworld import BatchCache, MicroConfig, evaluate_stream
from dsi.model import ModelConfig
from dsi.specs import PhaseSpec
from dsi.state import StateProbeSpec, extract_state_features
from dsi.train import TrainConfig, init_state, phase_steps, train_phase

ARMS = {"A": "IND", "A_prime": "IND_R", "BG": "BG"}


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="500-502")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("artifacts/mediator_discovery"))
    args = ap.parse_args()

    cfg = MicroConfig()
    mc = ModelConfig(vocab_size=cfg.vocab_size, d_model=args.d_model,
                     n_heads=args.heads, n_layers=args.layers,
                     d_ff=4 * args.d_model, max_len=cfg.seq_len)
    tc = TrainConfig(batch_size=args.batch, loss_positions="all")
    mspec = MProbeSpec()
    sspec = StateProbeSpec(n_examples=128)

    units = args.out / "units"
    ckpts = args.out / "checkpoints"
    units.mkdir(parents=True, exist_ok=True)
    ckpts.mkdir(parents=True, exist_ok=True)

    # The state probe reuses micro-world documents, held fixed across every
    # model so a feature difference cannot be a difference in what was shown.
    from dsi.microworld import sample_documents
    probe = np.asarray(sample_documents(90777, cfg, "BIND", sspec.n_examples))
    import jax.numpy as jnp
    probe = jnp.asarray(probe)

    t0 = time.time()
    for arm, stream in ARMS.items():
        for seed in _parse_seeds(args.seeds):
            path = units / f"{arm}__seed{seed:03d}.json"
            if path.exists():
                continue

            state = init_state(mc, tc, jax.random.PRNGKey(seed))
            phase = PhaseSpec(family=stream,
                              tokens=args.steps * tc.batch_size * cfg.seq_len,
                              role="source")
            key = jax.random.PRNGKey(10_000 + seed)
            state, _ = train_phase(
                state, phase, cfg, tc, key,
                sampler=BatchCache(key, stream, cfg, tc.batch_size,
                                   phase_steps(phase, cfg, tc)),
            )

            stem = ckpts / f"{arm}__seed{seed:03d}"
            save_model(stem, state.model, mc)

            pm = np.asarray(prefix_matching_scores(state.model, mspec, cfg.vocab_size))
            rs = np.asarray(retrieval_scores(state.model, cfg))
            payload = {
                "arm": arm, "source": stream, "seed": seed, "steps": args.steps,
                "checkpoint": str(stem),
                "M_scalar": mediator_score(state.model, mspec, cfg.vocab_size),
                "prefix_matching_per_head": pm.tolist(),
                "retrieval_per_head": rs.tolist(),
                "retrieval_max": float(rs.max()),
                "retrieval_mean": float(rs.mean()),
                "state_features": extract_state_features(
                    state.model, probe, sspec, cfg.answer_target_index),
                "zero_shot_BIND": evaluate_stream(state.model, cfg, "BIND", 90001, 512),
                "zero_shot_FACT": evaluate_stream(state.model, cfg, "FACT", 90002, 512),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload) + "\n")
            tmp.replace(path)
            print(f"  {arm:8s} seed {seed}  retrieval_max {rs.max():.4f}  "
                  f"M {payload['M_scalar']:.4f}  zsB {payload['zero_shot_BIND']['accuracy']:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    # Ranked separation report over everything measured.
    rows = [json.loads(p.read_text()) for p in sorted(units.glob("*.json"))]
    if len({r["arm"] for r in rows}) < 2:
        print("not enough arms finished for a separation report")
        return 0

    def by(arm, key):
        return [r[key] for r in rows if r["arm"] == arm]

    print("\n=== candidate separation, A vs A_prime (exploratory) ===")
    for label, key in (("retrieval_max", "retrieval_max"),
                       ("retrieval_mean", "retrieval_mean"),
                       ("M_scalar", "M_scalar"),
                       ("zero_shot_BIND", None)):
        a = by("A", key) if key else [r["zero_shot_BIND"]["accuracy"] for r in rows if r["arm"] == "A"]
        p = by("A_prime", key) if key else [r["zero_shot_BIND"]["accuracy"] for r in rows if r["arm"] == "A_prime"]
        g = by("BG", key) if key else [r["zero_shot_BIND"]["accuracy"] for r in rows if r["arm"] == "BG"]
        if not (a and p):
            continue
        sd = st.pstdev(a + p) or 1e-9
        print(f"  {label:16s} A {st.mean(a):8.4f}  A' {st.mean(p):8.4f}  "
              f"BG {st.mean(g) if g else float('nan'):8.4f}  "
              f"sep {(st.mean(a) - st.mean(p)) / sd:+6.2f} sd")

    names = sorted(rows[0]["state_features"])
    seps = []
    for n in names:
        a = [r["state_features"][n] for r in rows if r["arm"] == "A"]
        p = [r["state_features"][n] for r in rows if r["arm"] == "A_prime"]
        sd = st.pstdev(a + p)
        if sd > 1e-12:
            seps.append((abs(st.mean(a) - st.mean(p)) / sd, n, st.mean(a), st.mean(p)))
    seps.sort(reverse=True)
    print("\n=== top state features separating A from A_prime ===")
    for sep, n, ma, mp in seps[:12]:
        print(f"  {sep:6.2f} sd  {n:34s} A {ma:10.4f}  A' {mp:10.4f}")

    (args.out / "separation.json").write_text(json.dumps(
        {"per_feature": [{"sep_sd": s, "feature": n, "A": a, "A_prime": p}
                         for s, n, a, p in seps]}, indent=2) + "\n")
    print(f"\ncheckpoints in {ckpts} — ablate the nominated heads next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
