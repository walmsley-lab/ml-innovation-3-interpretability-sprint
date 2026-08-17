"""Lane B, stage SP-a: generate the behaviourally-matched population.

Trains ``n_pairs`` initializations through both histories (``W -> P`` and
``P -> W``) to the matching checkpoint ``t*``, saves weights there, records the
behavioural vector, then continues each checkpoint under two frozen future
interventions and records the outcome.

**The freeze contract.** This script writes the behavioural vector, the
checkpoints and the outcomes. It does **not** compute divergence, fit a
predictor, or form matched pairs. Those live in the analysis script, which must
not run until `docs/experiments/sprint_latent_state_protocol.json` is written
and hashed. Separating them is what stops the matching rule from being chosen
after the outcomes are visible — the failure mode that invalidated LOPO model
selection at 20NG.

Paired initialization is load-bearing: both histories for a given pair start
from the *same* weights, so history is decorrelated from initialization and an
internal-state probe cannot succeed by reading the init seed instead. The
shared init is asserted per pair on an embedding fingerprint, not assumed.

    PYTHONPATH=src python scripts/sprint_population.py --pairs 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from dsi.checkpoint import save_model
from dsi.data import CONDITIONS, SPLITS, TaskConfig
from dsi.eval import evaluate
from dsi.model import ModelConfig
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, train_phase

# The frozen Layer-1 regime is **overlapping**, not block-sequential. Pure
# block-sequential phases were abandoned at Gate B because they produce
# catastrophic interference: only the last phase's skill survives. The overlap
# floor r = 0.20 was calibrated to give coexistence (worst case 0.969), and
# coexistence is a precondition here — a preference between two skills is not
# measurable unless the model retains both.
#
# A first run of this script used pure phases and every one of 256 models
# failed the competence gate: W at chance (0.248) in both histories, and only
# the final phase's skill retained. That is recorded in CLAIMS.md as B1.
OVERLAP = 0.20

# The frozen apparatus ends with a **common tail** — research.md 8d, T1
# maintenance — which is what produced worst-case post-tail coexistence 0.947.
# Omitting it was an implementation error, not a design choice: without the
# tail the histories end at different recency and neither retains both skills.
COMMON_TAIL = "NEUTRAL_ALIGNED+W_P_INTERLEAVED@0.20"
HISTORIES = {
    "W_first": ("W_EXPLICIT", f"P_EXPLICIT+W_EXPLICIT@{OVERLAP:.2f}"),
    "P_first": ("P_EXPLICIT", f"W_EXPLICIT+P_EXPLICIT@{OVERLAP:.2f}"),
}

# The two frozen future interventions. Fitting a predictor on one and
# evaluating on the other is a generalization test a single intervention
# cannot provide: it asks whether internal state predicts divergence in
# general or only one specific perturbation.
INTERVENTIONS = {
    "I_conflict": "NEUTRAL_ALIGNED",
    "I_continue": "W_P_INTERLEAVED",
}


def _fingerprint(model) -> str:
    """Hash of the embedding matrix. Two runs sharing an init must agree."""
    w = np.asarray(model.embed.weight)
    return hashlib.sha256(w.tobytes()).hexdigest()[:16]


def _behaviour(model, task: TaskConfig, key) -> dict:
    """The full behavioural vector at ``t*``.

    Deliberately the *complete* observable picture — every condition on every
    split — rather than a handful of scalars. The behaviour-only predictor has
    to be a fair baseline, and a weak one would manufacture the sprint result
    the way a near-saturated additive model did at 20NG.
    """
    out = {}
    results = []
    for split in SPLITS:
        results += list(
            evaluate(model, task, jax.random.fold_in(key, SPLITS.index(split)),
                     conditions=CONDITIONS, split=split).values()
        )
    for r in results:
        prefix = f"{r.condition}.{r.split}"
        out[f"{prefix}.loss"] = r.loss
        out[f"{prefix}.accuracy"] = r.accuracy
        out[f"{prefix}.follows_w"] = r.follows_w
        out[f"{prefix}.logodds"] = r.logodds_w_minus_p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1200,
                    help="steps per phase to t*; the frozen Layer-1 regime is 1,200")
    ap.add_argument("--intervention-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("artifacts/sprint_population"))
    ap.add_argument("--start", type=int, default=0, help="first pair index (for sharding)")
    ap.add_argument("--limit", type=int, default=0, help="max pairs this shard runs")
    args = ap.parse_args()

    task = TaskConfig(n_cues=512)
    tc = TrainConfig(batch_size=args.batch, loss_positions="answer")
    mc = ModelConfig(vocab_size=task.vocab_size, d_model=64, n_heads=4,
                     n_layers=4, d_ff=256, max_len=task.seq_len)
    tokens = args.steps * tc.batch_size * task.seq_len
    itokens = args.intervention_steps * tc.batch_size * task.seq_len

    units = args.out / "units"
    ckpts = args.out / "checkpoints"
    units.mkdir(parents=True, exist_ok=True)
    ckpts.mkdir(parents=True, exist_ok=True)

    end = args.pairs if not args.limit else min(args.pairs, args.start + args.limit)
    t0 = time.time()
    written = 0

    for pair in range(args.start, end):
        out_path = units / f"pair_{pair:04d}.json"
        if out_path.exists():          # idempotent: finished units are never recomputed
            continue

        record = {"pair": pair, "histories": {}}
        init_key = jax.random.PRNGKey(700_000 + pair)
        fingerprints = set()

        for history, (first, second) in HISTORIES.items():
            state = init_state(mc, tc, init_key)
            fingerprints.add(_fingerprint(state.model))

            for phase_index, family in enumerate((first, second, COMMON_TAIL)):
                state, _ = train_phase(
                    state,
                    PhaseSpec(family=family, tokens=tokens,
                              role=("source", "target", "washout")[phase_index]),
                    task, tc,
                    jax.random.fold_in(jax.random.PRNGKey(800_000 + pair), phase_index),
                )

            stem = ckpts / f"pair_{pair:04d}_{history}"
            save_model(stem, state.model, mc)
            entry = {
                "t_star_behaviour": _behaviour(
                    state.model, task, jax.random.PRNGKey(900_000 + pair)),
                "checkpoint": str(stem),
                "outcomes": {},
            }

            # Both interventions start from the same t* state, so they are
            # counterfactuals of one another rather than a sequence.
            for name, family in INTERVENTIONS.items():
                cont, _ = train_phase(
                    state,
                    PhaseSpec(family=family, tokens=itokens, role="mixture"),
                    task, tc,
                    jax.random.PRNGKey(950_000 + pair),
                )
                entry["outcomes"][name] = _behaviour(
                    cont.model, task, jax.random.PRNGKey(960_000 + pair))
            record["histories"][history] = entry

        if len(fingerprints) != 1:
            raise RuntimeError(
                f"pair {pair}: the two histories did not share an initialization "
                f"({len(fingerprints)} distinct embedding fingerprints). Paired "
                "initialization is what decorrelates history from init; without "
                "it the S1 probe is uninterpretable."
            )
        record["init_fingerprint"] = fingerprints.pop()

        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record) + "\n")
        tmp.replace(out_path)          # atomic, so a killed shard leaves no partial unit
        written += 1
        if written % 8 == 0:
            done = pair - args.start + 1
            rate = (time.time() - t0) / done
            print(f"  {done} pairs, {rate:.1f}s/pair, "
                  f"~{rate * (end - pair - 1) / 60:.0f} min left", flush=True)

    print(f"wrote {written} pairs to {units} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
