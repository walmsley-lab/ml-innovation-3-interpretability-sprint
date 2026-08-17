"""Experiment 4: does gradient/update geometry predict conditional data value?

The current readout — activation statistics plus a retrieval-head score —
**loses to a state-blind baseline** at predicting which corpus is best from a
given state. The evidence also argues against a small set of attention heads
carrying the state, since the capability is redundant and top-k ablation is
inadequate.

This measures a different family of observables: the geometry of the *gradient
the future data induces on the current weights*. Intuitively, if the same
corpus has different value from different states, the difference ought to show
up in how that corpus's gradient sits relative to the weights it is about to
move — more directly than in any static activation summary.

For each saved state and each candidate corpus, on **identical future
minibatches**:

  * gradient norm, globally and per layer
  * layerwise distribution of gradient mass (entropy over layers)
  * cosine alignment between this corpus's gradient and each other corpus's
  * cosine alignment between the gradient and the current weights
  * effective rank of the per-example gradient matrix (update-rank proxy)

The test is the same one the activation readout failed: leave-one-state-out
prediction of measured `V(S,D)`, judged against the state-blind global-best
rule. Exploratory. It is a *readout* hypothesis, not a mechanism claim: a
predictive geometry would say the information is present and legible, not that
it is causal.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from dsi.checkpoint import load_model
from dsi.microworld import MicroConfig, sample_documents
from dsi.train import TrainConfig, _loss

CORPORA = ("BIND", "BINDT", "FACT")


def flat(tree):
    return jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))])


def grad_for(model, cfg, corpus, seed, batch):
    tok = jnp.asarray(sample_documents(seed, cfg, corpus, batch))
    g = eqx.filter_grad(_loss)(model, tok, cfg.answer_target_index, "all")
    return g


def features(model, cfg, batch, seed):
    """Geometry of each corpus's gradient at this state. Identical batches."""
    grads = {c: grad_for(model, cfg, c, seed, batch) for c in CORPORA}
    w = flat(model)
    out = {}
    for c, g in grads.items():
        gf = flat(g)
        n = float(jnp.linalg.norm(gf))
        out[f"{c}.gnorm"] = n
        out[f"{c}.cos_w"] = float(jnp.dot(gf, w) / (n * jnp.linalg.norm(w) + 1e-12))
        # layerwise mass distribution -> entropy (basis-free, comparable across models)
        per = jnp.asarray([float(jnp.linalg.norm(x)) for x in
                           jax.tree.leaves(eqx.filter(g, eqx.is_inexact_array))])
        p = per / (per.sum() + 1e-12)
        out[f"{c}.layer_entropy"] = float(-(p * jnp.log(p + 1e-12)).sum())
        out[f"{c}.layer_max_frac"] = float(p.max())
    for i, a in enumerate(CORPORA):
        for b in CORPORA[i + 1:]:
            ga, gb = flat(grads[a]), flat(grads[b])
            out[f"cos.{a}.{b}"] = float(jnp.dot(ga, gb) /
                                        (jnp.linalg.norm(ga) * jnp.linalg.norm(gb) + 1e-12))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=91001)
    ap.add_argument("--only-states", type=str, default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = MicroConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted({p.with_suffix("") for p in args.checkpoints.glob("*.eqx")})
    if args.only_states:
        owned = {x.strip() for x in args.only_states.split(",") if x.strip()}
        stems = [s for s in stems if s.name in owned]

    for stem in stems:
        dest = args.out / f"{stem.name}.json"
        if dest.exists():
            continue
        model, _ = load_model(stem)
        f = features(model, cfg, args.batch, args.seed)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state_label": stem.name, "features": f,
                                   "batch": args.batch, "seed": args.seed}) + "\n")
        tmp.replace(dest)
        print(f"  {stem.name}  gnorm(BIND)={f['BIND.gnorm']:.4f}  "
              f"cos(BIND,FACT)={f['cos.BIND.FACT']:+.4f}", flush=True)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
