"""Stage-5 natural transfer: paired source -> target units on 20 Newsgroups.

One unit is a paired comparison, both arms in one process from one shared
initialization:

    treatment   source D_i  ->  target D_j
    control     neutral N   ->  target D_j

The arms differ in the identity of the source phase and in nothing else. The
target chunk stream is materialized once and both arms consume it in the same
order, the initialization is shared, and the two source phases are matched
**exactly** in LM training tokens, which is asserted per unit rather than
assumed.

The neutral control is a balanced mixture of the two families that are
neither source nor target, and its chunks are **interleaved rather than
blocked**. A blocked mixture would be a two-phase curriculum in its own
right, and Layer 1 established that abrupt isolated phases produce
catastrophic interference; the control must be a neutral prefix, not a
curriculum.

The target ``t=0`` evaluation is taken before any target token. It is what
separates a head start carried in from the source phase from a genuinely
faster acquisition rate, and it is unrecoverable after the run.

Held-out discipline: training streams are drawn from the frozen ``train``
split only, evaluation from the frozen ``val`` split only, and ``test``
is never touched. Both are asserted per unit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from dsi.artifacts import code_version, utc_now
from dsi.corpus import Corpus, Document, deduplicate, split_documents
from dsi.model import ModelConfig, init_model
from dsi.natural import Vocabulary, chunk_documents
from dsi.stats import transfer_effect

FROZEN = Path("artifacts/natural_pilot")
USABLE_FAMILIES = (1, 2, 3, 4)

# Natural-path regime. Chunk budgets, not step counts, are the currency:
# family 3 supplies 1,601 train chunks, so 1,536 is the largest round budget
# every usable family can serve in a single pass with no repetition.
SEQ_LEN, CHUNKS_PER_PHASE, BATCH = 128, 1536, 16
D_MODEL, N_LAYERS, N_HEADS, LR = 64, 4, 4, 3e-3
EVAL_CHUNKS = 256
OFFSETS = tuple(round(0.1 * i, 1) for i in range(11))   # includes t=0


def _loss(model, tokens):
    logits = jax.vmap(model)(tokens[:, :-1])
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(lp, tokens[:, 1:][..., None], axis=-1).squeeze(-1))


@eqx.filter_jit
def _step(model, opt_state, tokens, optimizer):
    loss, grads = eqx.filter_value_and_grad(_loss)(model, tokens)
    updates, opt_state = optimizer.update(grads, opt_state,
                                          eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def _eval_loss(model, chunks):
    return _loss(model, chunks)


def load_frozen() -> dict:
    """Frozen assignment, vocabulary and pair split. Nothing is refitted here."""
    frozen = json.loads((FROZEN / "assignments.json").read_text())
    pairs = json.loads((FROZEN / "frozen_pairs.json").read_text())
    vocabulary = Vocabulary.from_json(FROZEN / "vocab.json")
    raw_pairs = [tuple(p) for p in pairs["observed_pairs"]]
    heldout = [tuple(p) for p in pairs["heldout_pairs"]]

    from sklearn.datasets import fetch_20newsgroups
    raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    dedup, _ = deduplicate(Corpus("20newsgroups", "v1",
                                  tuple(Document.of(t) for t in raw.data)))
    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))
    if {d.doc_id for d in splits["train"].documents} != set(frozen["train"]):
        raise AssertionError(
            "ingestion no longer reproduces the frozen train split; the pilot "
            "provenance is broken and no transfer result from it is valid")

    documents = {}
    for split in ("train", "val"):
        for family in USABLE_FAMILIES:
            documents[(split, family)] = [
                d for d in splits[split].documents
                if frozen[split].get(d.doc_id) == family]
    return {"documents": documents, "vocabulary": vocabulary,
            "observed_pairs": raw_pairs, "heldout_pairs": heldout,
            "test_ids": set(frozen["test"]), "val_ids": set(frozen["val"])}


def family_chunks(state: dict, family: int, n_chunks: int, seed: int) -> np.ndarray:
    """Exactly ``n_chunks`` single-pass chunks from a family's train documents."""
    docs = state["documents"][("train", family)]
    chunks = chunk_documents(docs, state["vocabulary"], SEQ_LEN, seed=seed)
    if len(chunks) < n_chunks:
        raise ValueError(
            f"family {family} supplies {len(chunks)} chunks, short of {n_chunks}; "
            "the arms cannot be exposure-matched from this family without repetition")
    return chunks[:n_chunks]


def neutral_chunks(state: dict, source: int, target: int, n_chunks: int,
                   seed: int) -> tuple:
    """Balanced, interleaved mixture of the families that are neither arm's."""
    others = [f for f in USABLE_FAMILIES if f not in (source, target)]
    per = n_chunks // len(others)
    parts = [family_chunks(state, f, per, seed + 100 + f) for f in others]
    mixed = np.concatenate(parts, axis=0)
    if len(mixed) != n_chunks:
        raise ValueError(f"neutral mixture is {len(mixed)} chunks, not {n_chunks}")
    order = np.random.default_rng(seed + 7).permutation(len(mixed))
    return mixed[order], others


def _train(model, opt_state, optimizer, chunks):
    """Run one source phase: every chunk once, in the order given."""
    for step in range(1, len(chunks) // BATCH + 1):
        batch = jnp.asarray(chunks[(step - 1) * BATCH: step * BATCH])
        model, opt_state, _ = _step(model, opt_state, batch, optimizer)
    return model, opt_state


def run_unit(source: int, target: int, seed: int, out: Path, state: dict) -> dict:
    path = out / "units" / f"f{source}__to__f{target}__seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    vocabulary = state["vocabulary"]
    model_config = ModelConfig(vocab_size=vocabulary.size, d_model=D_MODEL,
                               n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=4 * D_MODEL,
                               max_len=SEQ_LEN)
    optimizer = optax.adamw(optax.constant_schedule(LR), weight_decay=0.01)
    root = jr.key(seed)

    # Evaluation set: target family, val split, fixed across seeds and arms so
    # the same yardstick measures every unit with this target.
    val_docs = state["documents"][("val", target)]
    eval_chunks = jnp.asarray(
        chunk_documents(val_docs, vocabulary, SEQ_LEN, seed=0)[:EVAL_CHUNKS])

    source_stream = family_chunks(state, source, CHUNKS_PER_PHASE, seed)
    control_stream, neutral_families = neutral_chunks(state, source, target,
                                                      CHUNKS_PER_PHASE, seed)
    target_stream = family_chunks(state, target, CHUNKS_PER_PHASE, seed + 1000)

    # Exposure matching is asserted in LM-training-token units, not assumed.
    treat_tokens, control_tokens = source_stream.size, control_stream.size
    if treat_tokens != control_tokens:
        raise AssertionError(
            f"source exposure {treat_tokens} != control exposure {control_tokens} "
            "LM tokens; the arms are not budget-matched and the unit is invalid")

    n_steps = CHUNKS_PER_PHASE // BATCH
    marks = sorted({int(round(o * n_steps)) for o in OFFSETS})
    curves, fingerprints = {}, {}
    for arm, prefix in (("treatment", source_stream), ("control", control_stream)):
        model = init_model(model_config, jr.fold_in(root, 1))       # shared init
        fingerprints[arm] = float(jnp.sum(model.embed.weight))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        model, opt_state = _train(model, opt_state, optimizer, prefix)

        curve, holder = [], {"model": model}

        def measure(step):
            curve.append({"step": step, "offset": step / n_steps,
                          "loss": float(_eval_loss(holder["model"], eval_chunks))})

        measure(0)                              # t=0, before any target token
        for step in range(1, n_steps + 1):
            batch = jnp.asarray(target_stream[(step - 1) * BATCH: step * BATCH])
            model, opt_state, _ = _step(model, opt_state, batch, optimizer)
            holder["model"] = model
            if step in marks:
                measure(step)
        curves[arm] = curve

    if fingerprints["treatment"] != fingerprints["control"]:
        raise AssertionError("paired arms did not share the initialization")

    t = [r["offset"] for r in curves["control"]]
    ctrl = [r["loss"] for r in curves["control"]]
    treat = [r["loss"] for r in curves["treatment"]]
    payload = {
        "source": source, "target": target, "seed": seed,
        "neutral_families": neutral_families,
        "lm_tokens_source": int(treat_tokens), "lm_tokens_control": int(control_tokens),
        "lm_tokens_target": int(target_stream.size),
        "T_aulc": transfer_effect(t, ctrl, treat, normalize=True),
        "T_aulc_rate_only": transfer_effect(t, ctrl, treat, normalize=True,
                                            baseline_correct=True),
        "head_start": ctrl[0] - treat[0],
        "endpoint": ctrl[-1] - treat[-1],
        "target_learning": ctrl[0] - ctrl[-1],      # does the target phase teach anything
        "curve_control": curves["control"], "curve_treatment": curves["treatment"],
        "vocab_size": vocabulary.size, "seq_len": SEQ_LEN, "batch": BATCH,
        "chunks_per_phase": CHUNKS_PER_PHASE, "steps_per_phase": n_steps,
        "seconds": time.time() - started,
        "code_version": code_version(), "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one paired unit with invariants")
    ap.add_argument("--source", type=int)
    ap.add_argument("--target", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002])
    ap.add_argument("--out", type=Path, default=Path("artifacts/natural_transfer"))
    ap.add_argument("--pairs", choices=("observed", "heldout"), default="observed",
                    help="'heldout' runs the 3 pairs frozen as the untouched pool; "
                         "everything else about the unit is identical by construction")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    state = load_frozen()
    train_ids = {d.doc_id for f in USABLE_FAMILIES
                 for d in state["documents"][("train", f)]}
    if train_ids & (state["val_ids"] | state["test_ids"]):
        raise AssertionError("held-out documents present in the training pool")

    if args.smoke:
        source = args.source or 2
        target = args.target or 1
        result = run_unit(source, target, args.seeds[0], args.out, state)
        print(f"smoke unit f{source} -> f{target} seed {args.seeds[0]}  "
              f"{result['seconds']:.0f}s")
        print(f"  LM tokens   source {result['lm_tokens_source']}  "
              f"control {result['lm_tokens_control']}  "
              f"{'MATCHED' if result['lm_tokens_source'] == result['lm_tokens_control'] else 'MISMATCH'}")
        print(f"  control target loss  t=0 {result['curve_control'][0]['loss']:.4f} "
              f"-> t=1 {result['curve_control'][-1]['loss']:.4f}  "
              f"(learned {result['target_learning']:.4f} nats)")
        print(f"  head start {result['head_start']:+.4f}   "
              f"AULC {result['T_aulc']:+.4f}   "
              f"rate-only {result['T_aulc_rate_only']:+.4f}   "
              f"endpoint {result['endpoint']:+.4f}")
        return

    pairs = ([(args.source, args.target)] if args.source and args.target
             else state["heldout_pairs"] if args.pairs == "heldout"
             else state["observed_pairs"])
    # One process per shard, so the corpus is ingested once per worker rather
    # than once per unit. Units are content-addressed by filename and skipped
    # if already present, which makes a re-run idempotent.
    units = [(source, target, seed) for source, target in pairs for seed in args.seeds]
    for source, target, seed in units[args.shard::args.n_shards]:
        result = run_unit(source, target, seed, args.out, state)
        print(f"f{source}->f{target} seed {seed}  T_aulc {result['T_aulc']:+.4f}  "
              f"head {result['head_start']:+.4f}  {result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
