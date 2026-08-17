"""Protocol B: dense checkpoints across the frozen 150-450 step window.

Implements `docs/experiments/downstream_protocols.md` Protocol B
(sha256 8fc78c4087e2f87b). The window comes from T1 and is NOT refitted here;
if the learnability break lands outside it, that is the result.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import jax
from dsi.checkpoint import save_model
from dsi.microworld import BatchCache, MicroConfig
from dsi.model import ModelConfig
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, phase_steps, train_phase

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="900,901,902")
    ap.add_argument("--source", type=str, default="IND")
    ap.add_argument("--lo", type=int, default=150)
    ap.add_argument("--hi", type=int, default=450)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("artifacts/temporal_replay"))
    args = ap.parse_args()

    cfg = MicroConfig()
    mc = ModelConfig(vocab_size=cfg.vocab_size, d_model=128, n_heads=4,
                     n_layers=4, d_ff=512, max_len=cfg.seq_len)
    tc = TrainConfig(batch_size=args.batch, loss_positions="all")
    ck = args.out / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for seed in [int(x) for x in args.seeds.split(",")]:
        state = init_state(mc, tc, jax.random.PRNGKey(seed))
        key = jax.random.PRNGKey(10_000 + seed)
        marks = list(range(args.lo, args.hi + 1, args.every))
        prev = 0
        for m in marks:
            stem = ck / f"s{seed:03d}_t{m:04d}"
            if stem.with_suffix(".eqx").exists():
                prev = m; continue
            span = m - prev
            if span > 0:
                # Train the increment. The data key is the run's key so the
                # trajectory is identical to one uninterrupted run of length m.
                phase = PhaseSpec(family=args.source,
                                  tokens=span * tc.batch_size * cfg.seq_len,
                                  role="source")
                state, _ = train_phase(
                    state, phase, cfg, tc, jax.random.fold_in(key, prev),
                    sampler=BatchCache(jax.random.fold_in(key, prev), args.source,
                                       cfg, tc.batch_size, phase_steps(phase, cfg, tc)))
            save_model(stem, state.model, mc)
            prev = m
            print(f"  seed {seed} t={m} saved ({time.time()-t0:.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {ck}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
