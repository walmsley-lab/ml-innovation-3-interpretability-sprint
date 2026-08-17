"""V2 execution unit: one arm, one seed, one ablation condition.

Covers both licensed V2 stages with the same code path, because they are the
same experiment with one field changed:

* **G1 scout / expansion** — ``--ablation none`` across arms ``A``, ``A_prime``
  and ``BG``. Answers H2.1 (does A selectively accelerate B, with C unmoved)
  and H2.2 (does M precede the acceleration).
* **H2.3 ablation** — ``--ablation M`` and ``--ablation random`` across the
  **same three arms**. The estimand is the arm x ablation interaction

      D = [B_A - B_A']_intact - [B_A - B_A']_do(M-)

  which is why the control arms must be ablated too. "Ablation hurt B" on the
  A arm alone is uninformative: ablating heads damages a model generally, and
  differencing against A' is what removes that.

Ablation is applied **at the start of phase 2**, so the question is about
learnability going forward rather than about scoring a damaged model. The
matched-random control is layer-matched to the M heads, so treatment and
control differ in which heads are removed, not in where in the network the
damage lands.

Units are idempotent and atomically written: a finished unit is never
recomputed and a killed shard leaves no partial file.

    PYTHONPATH=src python scripts/run_v2_units.py --arms A,A_prime,BG --seeds 0-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

from dsi.mechanism import (
    MProbeSpec, ablate_heads, matched_random_heads, mediator_score,
    retrieval_scores, top_heads, top_retrieval_heads,
)
from dsi.microworld import BatchCache, MicroConfig, evaluate_stream
from dsi.model import ModelConfig
from dsi.specs import PhaseSpec
from dsi.train import TrainConfig, init_state, phase_steps, train_phase

ARMS = {"A": "IND", "A_prime": "IND_R", "BG": "BG"}
ABLATIONS = ("none", "M", "retrieval", "random")


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


def run_unit(cfg, mc, tc, spec, source: str, seed: int, ablation: str,
             steps: int, probe_every: int, target_steps: int = 0,
             target_family: str = "BIND+FACT") -> dict:
    """Phase 1 on ``source``, optional ablation, phase 2 on the target."""
    state = init_state(mc, tc, jax.random.PRNGKey(seed))
    target_steps = target_steps or steps
    eval_at = tuple(range(0, steps + 1, probe_every))
    target_eval_at = tuple(range(0, target_steps + 1, probe_every))
    trace: list[dict] = []

    # The target stream is evaluated by name, not assumed to be BIND. An
    # earlier version hardcoded BIND, so runs whose target was BINDT trained on
    # one capability and measured another — the wrong estimand entirely.
    target_eval = target_family.split("+")[0]

    def telemetry(model, phase_index):
        out = {
            "phase": phase_index,
            "M": mediator_score(model, spec, cfg.vocab_size),
            "BIND": evaluate_stream(model, cfg, "BIND", 90001, 512),
            "FACT": evaluate_stream(model, cfg, "FACT", 90002, 512),
            "source_stream": evaluate_stream(model, cfg, source, 90003, 512),
        }
        out["target"] = (out["BIND"] if target_eval.startswith("BIND#")
                         or target_eval == "BIND"
                         else evaluate_stream(model, cfg, target_eval, 90005, 512))
        out["target_stream_name"] = target_eval
        return out

    # Phase 1: the source.
    p1 = PhaseSpec(family=source, tokens=steps * tc.batch_size * cfg.seq_len, role="source")
    k1 = jax.random.PRNGKey(10_000 + seed)
    state, records = train_phase(
        state, p1, cfg, tc, k1,
        eval_at=eval_at,
        eval_fn=lambda m, _i: telemetry(m, 0),
        sampler=BatchCache(k1, source, cfg, tc.batch_size, phase_steps(p1, cfg, tc)),
    )
    trace += [{"step": int(r["step"]), **r["result"]} for r in records]

    # Ablation at the phase boundary. The head set is selected on the model as
    # it stands at the end of phase 1, so it is the mechanism this run actually
    # built rather than a set fixed in advance.
    m_heads = top_heads(state.model, spec, cfg.vocab_size)
    r_heads = top_retrieval_heads(state.model, cfg, k=spec.top_k)
    ablation_record = {
        "selected_M_heads": [list(h) for h in m_heads],
        "selected_retrieval_heads": [list(h) for h in r_heads],
        "retrieval_per_head": np.asarray(retrieval_scores(state.model, cfg)).tolist(),
    }
    # The head set that is *measured* and the set that is *removed* must be the
    # same object, or the interaction tests something other than the candidate.
    if ablation == "M":
        selected, heads = m_heads, m_heads
    elif ablation == "retrieval":
        selected, heads = r_heads, r_heads
    elif ablation == "random":
        selected = r_heads
        heads = matched_random_heads(state.model, spec, cfg.vocab_size,
                                     seed=50_000 + seed, exclude=r_heads)
    else:
        selected, heads = r_heads, ()
    ablation_record["ablated_heads"] = [list(h) for h in heads]
    # Efficacy is measured as a **causal contribution**, not as an attention
    # statistic. `ablate_heads` zeroes a head's output projection, which leaves
    # its attention pattern untouched — so `retrieval_scores` is structurally
    # unable to detect its own ablation and is useless as an efficacy check.
    # Zero-shot BIND accuracy before/after is what actually moves if the
    # ablated heads were carrying the capability.
    ablation_record["M_before"] = mediator_score(state.model, spec, cfg.vocab_size)
    ablation_record["retrieval_attn_before"] = float(
        np.asarray(retrieval_scores(state.model, cfg)).max())
    ablation_record["zeroshot_BIND_before"] = evaluate_stream(
        state.model, cfg, "BIND", 90001, 512)["accuracy"]

    if heads:
        state = type(state)(
            model=ablate_heads(state.model, heads),
            opt_state=state.opt_state,
            step=state.step,
            tokens_seen=state.tokens_seen,
        )
    ablation_record["M_after"] = mediator_score(state.model, spec, cfg.vocab_size)
    ablation_record["retrieval_attn_after"] = float(
        np.asarray(retrieval_scores(state.model, cfg)).max())
    ablation_record["zeroshot_BIND_after"] = evaluate_stream(
        state.model, cfg, "BIND", 90001, 512)["accuracy"]

    # Phase 2: the target. The t=0 evaluation is mandatory and separates a head
    # start carried in from phase 1 from a genuinely faster acquisition rate.
    p2 = PhaseSpec(family=target_family,
                   tokens=target_steps * tc.batch_size * cfg.seq_len, role="target")
    k2 = jax.random.PRNGKey(20_000 + seed)
    state, records = train_phase(
        state, p2, cfg, tc, k2,
        eval_at=target_eval_at,
        eval_fn=lambda m, _i: telemetry(m, 1),
        sampler=BatchCache(k2, target_family, cfg, tc.batch_size, phase_steps(p2, cfg, tc)),
    )
    trace += [{"step": int(r["step"]), **r["result"]} for r in records]

    return {"trace": trace, "ablation": ablation_record}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="A,A_prime,BG")
    ap.add_argument("--seeds", type=str, default="0-5")
    ap.add_argument("--ablation", type=str, default="none", choices=ABLATIONS)
    ap.add_argument("--steps", type=int, default=4000, help="source-phase steps")
    ap.add_argument("--target-steps", type=int, default=0,
                    help="target-phase steps; 0 means same as --steps")
    ap.add_argument("--target-family", type=str, default="BIND+FACT",
                    help="target-phase stream. BIND+FACT puts BOTH the target "
                         "capability and the negative control in the training "
                         "data, which is what makes the specificity test "
                         "meaningful: FACT was previously never trained, so "
                         "'C did not move' was vacuous.")
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("artifacts/v2"))
    args = ap.parse_args()

    cfg = MicroConfig()
    mc = ModelConfig(vocab_size=cfg.vocab_size, d_model=args.d_model,
                     n_heads=args.heads, n_layers=args.layers,
                     d_ff=4 * args.d_model, max_len=cfg.seq_len)
    tc = TrainConfig(batch_size=args.batch, loss_positions="all")
    spec = MProbeSpec()

    units = args.out / "units"
    units.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = _parse_seeds(args.seeds)
    t0 = time.time()

    # An arm may be a preset name or an explicit "label=STREAM" pair. The
    # explicit form is what the C1 discrimination scout needs: it runs sources
    # over a disjoint entity sub-pool ("IND#h1") against a target on the other
    # half ("BIND#h2"), so source and target share no entity token and any
    # transfer must be structural rather than content-level.
    resolved = []
    for arm in arms:
        if "=" in arm:
            label, stream = arm.split("=", 1)
            resolved.append((label, stream))
        elif arm in ARMS:
            resolved.append((arm, ARMS[arm]))
        else:
            raise SystemExit(f"unknown arm {arm!r}; use a preset {sorted(ARMS)} or label=STREAM")

    for arm, source_stream in resolved:
        for seed in seeds:
            name = f"{arm}__abl_{args.ablation}__seed{seed:02d}.json"
            path = units / name
            if path.exists():
                continue
            result = run_unit(cfg, mc, tc, spec, source_stream, seed, args.ablation,
                              args.steps, args.probe_every,
                              args.target_steps, args.target_family)
            payload = {
                "arm": arm, "source": source_stream, "seed": seed,
                "ablation_kind": args.ablation, "steps": args.steps,
                "target_steps": args.target_steps or args.steps,
                "target_family": args.target_family,
                "model": {"d_model": args.d_model, "n_layers": args.layers,
                          "n_heads": args.heads, "batch": args.batch},
                **result,
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload) + "\n")
            tmp.replace(path)
            print(f"  {name}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"done in {time.time() - t0:.0f}s -> {units}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
