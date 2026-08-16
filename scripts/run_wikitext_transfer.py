"""WikiText Stage-5 development wave, against the frozen protocol.

Everything scientific here was frozen before any outcome existed and is read
from disk rather than restated: the seven-family universe (family 6 excluded
as a residual cluster), the equal-family common control, the 1,536-chunk
dose, and the hashed 13/5/3 unordered-pair pools.

    treatment   source D_i  ->  target D_j
    control     common N_j  ->  target D_j       (identical for every i)

`N_j` is an equal mixture of the six non-target used families, 256 chunks
each. It is a function of the target and the seed alone, cached once per
`(j, seed)` and reused by every source.

**Exact sample identity is recorded per intervention.** Each arm stores the
ordered document ids it consumed, the chunk count, and a stream hash. Seed
variance in this design mixes training variance with family-subsample
variance — a treatment is 1,536 distinct chunks drawn from a pool three to
fifteen times larger — and this experiment does **not** attempt to
disentangle them. Recording the identifiers is what makes a later
disentangling possible without re-running anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_natural_transfer import (  # noqa: E402  frozen constants, not restated
    BATCH, CHUNKS_PER_PHASE, D_MODEL, EVAL_CHUNKS, LR, N_HEADS, N_LAYERS, OFFSETS,
    SEQ_LEN, _eval_loss, _step, _train)

from dsi.artifacts import code_version, utc_now  # noqa: E402
from dsi.corpus import Document  # noqa: E402
from dsi.model import ModelConfig, init_model  # noqa: E402
from dsi.natural import Vocabulary, chunk_tokens  # noqa: E402
from dsi.stats import transfer_effect  # noqa: E402

CORPUS = Path("artifacts/corpus_v2")


def load_frozen() -> dict:
    pools = json.loads((CORPUS / "frozen_pools.json").read_text())
    assignments = json.loads((CORPUS / "assignments_k8.json").read_text())
    vocabulary = Vocabulary.from_json(CORPUS / "vocab.json")
    families = tuple(pools["families_used"])
    if pools["phase_chunks"] != CHUNKS_PER_PHASE:
        raise AssertionError(
            f"frozen dose {pools['phase_chunks']} != runner {CHUNKS_PER_PHASE}")

    # Per-family slices are cached by build_corpus_cache.py, which runs the
    # same deterministic derivation once instead of once per worker. The
    # document-id hash is verified so a stale cache cannot be used silently.
    cache = CORPUS / "cache"
    index = json.loads((cache / "index.json").read_text())
    documents = {}
    for split in ("train", "val"):
        for family in families:
            rows = [json.loads(line) for line in
                    (cache / f"{split}_f{family}.jsonl").read_text().splitlines() if line]
            digest = hashlib.blake2b(
                "".join(r["id"] for r in rows).encode(), digest_size=8).hexdigest()
            if digest != index["splits"][f"{split}_f{family}"]["doc_id_hash"]:
                raise AssertionError(f"stale corpus cache for {split} f{family}")
            documents[(split, family)] = [Document(r["id"], r["text"], ()) for r in rows]
    return {"documents": documents, "vocabulary": vocabulary, "families": families,
            "pools": pools}


def sample_family(state: dict, family: int, n_chunks: int, seed: int) -> tuple:
    """Exactly ``n_chunks`` chunks, with the manifest identifying the sample.

    Documents are shuffled by seed and consumed in order until the budget is
    met, so the manifest lists exactly the documents that contributed.
    """
    docs = state["documents"][("train", family)]
    order = np.random.default_rng(seed).permutation(len(docs))
    vocabulary = state["vocabulary"]
    stream, consumed = [], []
    for i in order:
        stream.extend(vocabulary.encode_document(docs[i].text))
        consumed.append(docs[i].doc_id)
        if len(stream) >= n_chunks * SEQ_LEN:
            break
    if len(stream) < n_chunks * SEQ_LEN:
        raise ValueError(
            f"family {family} supplies {len(stream) // SEQ_LEN} chunks, short of "
            f"{n_chunks}; the arms cannot be exposure-matched without repetition")
    chunks = chunk_tokens(stream, SEQ_LEN)[:n_chunks]
    manifest = {"family": family, "seed": seed, "n_chunks": int(n_chunks),
                "documents_consumed": consumed,
                "n_documents": len(consumed),
                "n_documents_available": len(docs),
                "stream_hash": hashlib.blake2b(
                    np.ascontiguousarray(chunks).tobytes(), digest_size=8).hexdigest()}
    return chunks, manifest


def common_control(state: dict, target: int, n_chunks: int, seed: int) -> tuple:
    """Equal-family mixture over the non-target used families, interleaved."""
    others = [f for f in state["families"] if f != target]
    per = n_chunks // len(others)
    parts, manifests = [], []
    for family in others:
        chunks, manifest = sample_family(state, family, per, seed + 100 + family)
        parts.append(chunks); manifests.append(manifest)
    mixed = np.concatenate(parts, axis=0)
    if len(mixed) != per * len(others):
        raise AssertionError("control mixture is the wrong size")
    order = np.random.default_rng(seed + 7).permutation(len(mixed))
    mixed = mixed[order]
    return mixed, {"weighting": "equal_family", "families": others,
                   "chunks_per_family": per, "components": manifests,
                   "stream_hash": hashlib.blake2b(
                       np.ascontiguousarray(mixed).tobytes(), digest_size=8).hexdigest()}


def _setup(state: dict, target: int, seed: int):
    vocabulary = state["vocabulary"]
    config = ModelConfig(vocab_size=vocabulary.size, d_model=D_MODEL, n_layers=N_LAYERS,
                         n_heads=N_HEADS, d_ff=4 * D_MODEL, max_len=SEQ_LEN)
    optimizer = optax.adamw(optax.constant_schedule(LR), weight_decay=0.01)
    val_docs = state["documents"][("val", target)]
    stream = []
    for d in val_docs:
        stream.extend(vocabulary.encode_document(d.text))
        if len(stream) >= EVAL_CHUNKS * SEQ_LEN:
            break
    eval_chunks = jnp.asarray(chunk_tokens(stream, SEQ_LEN)[:EVAL_CHUNKS])
    target_stream, target_manifest = sample_family(state, target, CHUNKS_PER_PHASE,
                                                   seed + 1000)
    return config, optimizer, eval_chunks, target_stream, target_manifest


def _trajectory(prefix, config, optimizer, eval_chunks, target_stream, seed):
    n_steps = CHUNKS_PER_PHASE // BATCH
    marks = {int(round(o * n_steps)) for o in OFFSETS}
    model = init_model(config, jr.fold_in(jr.key(seed), 1))          # shared init
    fingerprint = float(jnp.sum(model.embed.weight))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    model, opt_state = _train(model, opt_state, optimizer, prefix)
    curve = [{"step": 0, "offset": 0.0, "loss": float(_eval_loss(model, eval_chunks))}]
    for step in range(1, n_steps + 1):
        batch = jnp.asarray(target_stream[(step - 1) * BATCH: step * BATCH])
        model, opt_state, _ = _step(model, opt_state, batch, optimizer)
        if step in marks:
            curve.append({"step": step, "offset": step / n_steps,
                          "loss": float(_eval_loss(model, eval_chunks))})
    return curve, fingerprint


def control_arm(target: int, seed: int, out: Path, state: dict) -> dict:
    path = out / "controls" / f"f{target}__seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    config, optimizer, eval_chunks, target_stream, target_manifest = _setup(state, target, seed)
    stream, manifest = common_control(state, target, CHUNKS_PER_PHASE, seed)
    curve, fingerprint = _trajectory(stream, config, optimizer, eval_chunks,
                                     target_stream, seed)
    payload = {"target": target, "seed": seed, "control_manifest": manifest,
               "target_manifest": target_manifest, "lm_tokens": int(stream.size),
               "init_fingerprint": fingerprint, "curve": curve,
               "seconds": time.time() - started, "code_version": code_version(),
               "recorded_at": utc_now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(path)
    return payload


def unit(source: int, target: int, seed: int, out: Path, state: dict) -> dict:
    path = out / "units" / f"f{source}__to__f{target}__seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    control = control_arm(target, seed, out, state)
    config, optimizer, eval_chunks, target_stream, target_manifest = _setup(state, target, seed)
    stream, manifest = sample_family(state, source, CHUNKS_PER_PHASE, seed)
    curve, fingerprint = _trajectory(stream, config, optimizer, eval_chunks,
                                     target_stream, seed)

    if int(stream.size) != control["lm_tokens"]:
        raise AssertionError(
            f"treatment {stream.size} != control {control['lm_tokens']} LM tokens")
    if fingerprint != control["init_fingerprint"]:
        raise AssertionError("arms did not share the initialization")

    t = [r["offset"] for r in control["curve"]]
    ctrl = [r["loss"] for r in control["curve"]]
    treat = [r["loss"] for r in curve]
    payload = {
        "source": source, "target": target, "seed": seed, "corpus": "wikitext103",
        "control": "common_N_j_equal_family",
        "source_manifest": manifest, "control_manifest": control["control_manifest"],
        "target_manifest": target_manifest,
        "lm_tokens_source": int(stream.size), "lm_tokens_control": control["lm_tokens"],
        "head_start": ctrl[0] - treat[0],
        "T_aulc_rate_only": transfer_effect(t, ctrl, treat, normalize=True,
                                            baseline_correct=True),
        "T_aulc": transfer_effect(t, ctrl, treat, normalize=True),
        "endpoint": ctrl[-1] - treat[-1],
        "curve_control": control["curve"], "curve_treatment": curve,
        "seconds": time.time() - started, "code_version": code_version(),
        "recorded_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="development",
                    choices=("development", "confirmatory", "adaptive"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002])
    ap.add_argument("--out", type=Path, default=Path("artifacts/wikitext_transfer"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()

    state = load_frozen()
    pairs = [tuple(p) for p in state["pools"]["directed"][args.pool]]

    if args.controls_only:
        targets = sorted({t for _, t in pairs})
        work = [(t, s) for t in targets for s in args.seeds]
        for target, seed in work[args.shard::args.n_shards]:
            row = control_arm(target, seed, args.out, state)
            print(f"control N_{target} seed {seed}  "
                  f"{row['control_manifest']['stream_hash']}  {row['seconds']:.0f}s",
                  flush=True)
        return

    work = [(i, j, s) for i, j in pairs for s in args.seeds]
    for source, target, seed in work[args.shard::args.n_shards]:
        row = unit(source, target, seed, args.out, state)
        print(f"f{source}->f{target} seed {seed}  head {row['head_start']:+.4f}  "
              f"rate {row['T_aulc_rate_only']:+.4f}  aulc {row['T_aulc']:+.4f}",
              flush=True)


if __name__ == "__main__":
    main()
