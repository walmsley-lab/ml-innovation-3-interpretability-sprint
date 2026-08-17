"""The mediator ``M``: per-head prefix-matching score, and head ablation.

``M`` is measured on **random repeated token sequences**, never on micro-world
records. That independence is the single most important guard in the V2
design: a mediator statistic computed on the same corpus whose transfer it is
asked to explain would be measuring the design back at itself. Here the probe
shares only the vocabulary.

The probe sequence is ``[BOS, r, r]`` where ``r`` is a block of tokens drawn
uniformly at random. In the second copy, position ``i`` is preceded by the
same token that preceded position ``i`` in the first copy, so a head that
implements prefix matching attends from each token in the second copy back to
the token *following* its earlier occurrence. The prefix-matching score of a
head is the average attention mass it places on exactly those positions.

This is a measurement, not a definition of what the head "is". A high score
means the head moves information the way an induction head would; it does not
establish that the head participates in any particular downstream computation.
That is what the ablation is for.

Everything in ``MProbeSpec`` is a researcher degree of freedom that could
manufacture H2.2 or H2.3, so the spec is frozen and hashed before phase 2 of
the first scored run, alongside the other frozen objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsi.model import Transformer, _rope

__all__ = [
    "MProbeSpec", "probe_sequences", "head_attention", "prefix_matching_scores",
    "mediator_score", "top_heads", "matched_random_heads", "ablate_heads",
    "retrieval_scores", "top_retrieval_heads", "transplant_heads",
]


@dataclass(frozen=True)
class MProbeSpec:
    """Frozen definition of how ``M`` is measured.

    ``top_k`` is the number of heads treated as "the mechanism" for ablation.
    ``emergence_threshold`` is the value of the aggregate score at which ``M``
    is called present; emergence time is the first telemetry checkpoint at
    which the score crosses it *and stays above it*, which is stricter than a
    first crossing and much less sensitive to a single noisy checkpoint.
    """

    block_len: int = 24
    n_sequences: int = 64
    vocab_low: int = 5
    seed: int = 0
    top_k: int = 4
    emergence_threshold: float = 0.20

    def __post_init__(self) -> None:
        if self.block_len < 4:
            raise ValueError(f"block_len must be at least 4, got {self.block_len}")
        if self.n_sequences < 1:
            raise ValueError(f"n_sequences must be positive, got {self.n_sequences}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not 0.0 < self.emergence_threshold < 1.0:
            raise ValueError(
                f"emergence_threshold must lie in (0, 1), got {self.emergence_threshold}"
            )

    @property
    def seq_len(self) -> int:
        """``[BOS] + two copies of the random block``."""
        return 1 + 2 * self.block_len


def probe_sequences(spec: MProbeSpec, vocab_size: int) -> jnp.ndarray:
    """``(n_sequences, seq_len)`` random-repeat sequences.

    Tokens are drawn from ``[vocab_low, vocab_size)``, excluding the structural
    specials so the probe cannot be answered by template regularities. The
    draw is seeded from the spec, so the probe set is identical across every
    model, checkpoint and architecture it is applied to — without that, a
    difference in ``M`` between two runs could be a difference in the probe.
    """
    if vocab_size <= spec.vocab_low:
        raise ValueError(
            f"vocab_size ({vocab_size}) must exceed vocab_low ({spec.vocab_low})"
        )
    rng = np.random.default_rng(spec.seed)
    block = rng.integers(spec.vocab_low, vocab_size, size=(spec.n_sequences, spec.block_len))
    bos = np.zeros((spec.n_sequences, 1), dtype=block.dtype)
    return jnp.asarray(np.concatenate([bos, block, block], axis=1), dtype=jnp.int32)


def _block_attention(block: eqx.Module, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one block, returning ``(attention, output)``.

    Duplicates the attention arithmetic of ``model.Block.__call__`` because
    the model returns only logits and the probe needs the patterns themselves.
    Kept adjacent to the original so the two are diffable by eye; if the model
    changes, this must change with it, which the tests assert by comparing the
    reconstructed block output against the model's own.
    """
    seq_len = x.shape[0]
    positions = jnp.arange(seq_len)

    h = jax.vmap(block.norm_attn)(x)
    qkv = jax.vmap(block.qkv)(h).reshape(seq_len, 3, block.n_heads, block.d_head)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q, k = _rope(q, positions), _rope(k, positions)

    scores = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(block.d_head)
    causal = positions[None, :] > positions[:, None]
    scores = jnp.where(causal[None, :, :], -jnp.inf, scores)
    attn = jax.nn.softmax(scores, axis=-1)

    out = jnp.einsum("hqk,khd->qhd", attn, v).reshape(seq_len, -1)
    x = x + jax.vmap(block.proj)(out)
    h = jax.vmap(block.norm_ff)(x)
    h = jax.vmap(block.ff_out)(jax.nn.gelu(jax.vmap(block.ff_in)(h)))
    return attn, x + h


@eqx.filter_jit
def head_attention(model: Transformer, tokens: jnp.ndarray) -> jnp.ndarray:
    """Attention patterns for one sequence: ``(layer, head, query, key)``."""
    x = jax.vmap(model.embed)(tokens)
    patterns = []
    for block in model.blocks:
        attn, x = _block_attention(block, x)
        patterns.append(attn)
    return jnp.stack(patterns)


def prefix_matching_scores(
    model: Transformer, spec: MProbeSpec, vocab_size: int
) -> jnp.ndarray:
    """Per-head prefix-matching score, shape ``(n_layers, n_heads)``.

    For each query position ``1 + block_len + i`` in the second copy, the
    induction target is position ``1 + i + 1`` — the token *following* the
    earlier occurrence of the current token. The score is the mean attention
    placed on that target, averaged over probe sequences and over queries.

    The last query of the second copy is excluded: its induction target would
    be the first position of the second copy, which is a repeat boundary
    rather than a continuation.
    """
    tokens = probe_sequences(spec, vocab_size)
    queries = jnp.arange(spec.block_len - 1) + 1 + spec.block_len
    targets = jnp.arange(spec.block_len - 1) + 2

    def one(seq: jnp.ndarray) -> jnp.ndarray:
        attn = head_attention(model, seq)              # (layer, head, q, k)
        picked = attn[:, :, queries, targets]          # (layer, head, n_query)
        return picked.mean(axis=-1)

    return jax.vmap(one)(tokens).mean(axis=0)


def mediator_score(model: Transformer, spec: MProbeSpec, vocab_size: int) -> float:
    """The scalar ``M``: mean of the ``top_k`` per-head prefix-matching scores.

    Aggregating over the top ``k`` rather than over all heads reflects that the
    mechanism is expected to be carried by a few heads, and matches the set
    that ablation removes — the measured quantity and the intervened quantity
    are the same object, which they would not be under a mean over all heads.
    """
    scores = prefix_matching_scores(model, spec, vocab_size)
    flat = jnp.sort(scores.reshape(-1))[::-1]
    return float(flat[: spec.top_k].mean())


def top_heads(
    model: Transformer, spec: MProbeSpec, vocab_size: int
) -> tuple[tuple[int, int], ...]:
    """The ``top_k`` ``(layer, head)`` pairs by prefix-matching score."""
    scores = np.asarray(prefix_matching_scores(model, spec, vocab_size))
    order = np.argsort(scores.reshape(-1))[::-1][: spec.top_k]
    return tuple((int(i // scores.shape[1]), int(i % scores.shape[1])) for i in order)


def matched_random_heads(
    model: Transformer,
    spec: MProbeSpec,
    vocab_size: int,
    seed: int,
    *,
    exclude: tuple[tuple[int, int], ...] = (),
) -> tuple[tuple[int, int], ...]:
    """A control set of ``top_k`` heads, matched on layer depth.

    Matching on layer matters. Induction-style heads sit in later layers, and
    an unmatched random draw would tend to pick early-layer heads whose
    ablation damages the model differently. Matching the *layer profile* of the
    ablated set means the control differs from the treatment in which heads,
    not in where in the network the damage lands.
    """
    rng = np.random.default_rng(seed)
    n_heads = model.blocks[0].n_heads
    excluded = set(exclude)
    picked: list[tuple[int, int]] = []
    for layer, _ in top_heads(model, spec, vocab_size):
        options = [
            (layer, h) for h in range(n_heads)
            if (layer, h) not in excluded and (layer, h) not in picked
        ]
        if not options:
            raise ValueError(
                f"layer {layer} has no unused head available for the matched-random "
                "control; reduce top_k or widen the model"
            )
        picked.append(options[rng.integers(len(options))])
    return tuple(picked)


def ablate_heads(
    model: Transformer, heads: tuple[tuple[int, int], ...]
) -> Transformer:
    """Zero the output projection columns belonging to ``heads``.

    Ablation is applied to ``proj``, the per-block output projection, whose
    input is the concatenation of head outputs. Zeroing the columns for head
    ``h`` removes that head's contribution to the residual stream while
    leaving every other head untouched, and leaving the attention computation
    itself intact.

    The model is returned as a new tree; nothing is mutated in place. Ablating
    an empty set returns an equivalent model, which the tests assert, because a
    silently non-empty no-op would invalidate the control arm.
    """
    n_heads = model.blocks[0].n_heads
    d_head = model.blocks[0].d_head
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        if not 0 <= layer < len(model.blocks):
            raise ValueError(f"layer {layer} out of range for {len(model.blocks)} blocks")
        if not 0 <= head < n_heads:
            raise ValueError(f"head {head} out of range for {n_heads} heads")
        by_layer.setdefault(layer, []).append(head)

    updated = list(model.blocks)
    for layer, layer_heads in by_layer.items():
        block = updated[layer]
        mask = np.ones(block.proj.weight.shape[1], dtype=np.float32)
        for head in layer_heads:
            mask[head * d_head : (head + 1) * d_head] = 0.0
        weight = block.proj.weight * jnp.asarray(mask)[None, :]
        updated[layer] = eqx.tree_at(lambda b: b.proj.weight, block, weight)

    return eqx.tree_at(lambda m: m.blocks, model, tuple(updated))


# ---------------------------------------------------------------------------
# Track 1: on-distribution retrieval scores.
#
# The off-distribution prefix-matching score above is deliberately measured on
# random repeated tokens, so that the mediator statistic cannot be a function
# of the corpus whose transfer it explains. That guard is why `M` is
# trustworthy — and it is also the most likely reason `M` was falsified while
# the behavioural effect stood: a circuit specialised to the micro-world's
# clause template need not generalise to random-token sequences at all.
#
# The score below is the complementary measurement: attention, in real
# micro-world documents, from the position that predicts the answer back to the
# position holding the bound value. It is task-specific by construction, so on
# its own it is *correlational and circular-ish*. It earns its keep only as a
# candidate generator — the head set it nominates is then ablated, and the
# arm x ablation interaction is what carries causal weight.


def retrieval_scores(model: Transformer, config, seed: int = 90101,
                     batch: int = 256) -> jnp.ndarray:
    """Per-head attention from the prediction site to the bound value.

    In a ``BIND`` document the queried entity appears once earlier, and the
    value bound to it sits two positions after that earlier mention. The model
    predicts the answer from the final ``REL`` position. A head implementing
    in-context retrieval must therefore move information from the earlier value
    position to that final position, and this is the mass it places there.

    Returns ``(n_layers, n_heads)``, averaged over documents.
    """
    from dsi.microworld import CLAUSE_LEN, sample_documents

    docs = np.asarray(sample_documents(seed, config, "BIND", batch))
    ent_cols = [1 + j * CLAUSE_LEN + 1 for j in range(config.n_clauses - 1)]
    query_col = config.seq_len - 3
    predict_from = config.seq_len - 2

    # Locate, per document, the earlier mention of the queried entity and the
    # column holding its value. BIND binds in clause 0 by construction, but the
    # search is explicit so the statistic does not silently depend on that.
    value_cols = []
    keep = []
    for i, doc in enumerate(docs):
        target = doc[query_col]
        hit = [c for c in ent_cols if doc[c] == target]
        if hit:
            value_cols.append(hit[0] + 2)
            keep.append(i)
    if not keep:
        raise RuntimeError("no document contained an earlier mention of its query")

    tokens = jnp.asarray(docs[keep])
    cols = jnp.asarray(np.array(value_cols))

    def one(seq, col):
        attn = head_attention(model, seq)          # (layer, head, q, k)
        return attn[:, :, predict_from, col]

    return jax.vmap(one)(tokens, cols).mean(axis=0)


def top_retrieval_heads(model: Transformer, config, k: int = 4,
                        seed: int = 90101, batch: int = 256):
    """The ``k`` highest-scoring ``(layer, head)`` pairs by retrieval score."""
    scores = np.asarray(retrieval_scores(model, config, seed, batch))
    order = np.argsort(scores.reshape(-1))[::-1][:k]
    return tuple((int(i // scores.shape[1]), int(i % scores.shape[1])) for i in order)


def transplant_heads(recipient: Transformer, donor: Transformer,
                     heads: tuple[tuple[int, int], ...]) -> Transformer:
    """Copy specific attention heads from ``donor`` into ``recipient``.

    The positive-direction counterpart to :func:`ablate_heads`. Ablation tests
    **necessity** — remove the candidate and see whether the advantage goes.
    This tests **partial sufficiency** — install the candidate into a model that
    never received the source history and see whether readiness follows.

    A head's parameters are its slice of the fused ``qkv`` projection (rows
    ``c * d_model + h * d_head`` for each of q, k, v) and its slice of the
    output projection (columns ``h * d_head``). Both move together; moving one
    without the other would install a head that reads from one basis and writes
    into another.

    **The confound this cannot escape on its own.** A donor's heads arrive
    expecting the donor's residual basis. Transplanting therefore imports more
    than the mechanism, and a naive gain could be basis compatibility rather
    than readiness. Two controls are required and are the reason this function
    takes an arbitrary head set rather than choosing one itself:

    * transplant *matched-random* heads from the same donor — controls for "any
      donor weights help";
    * transplant the nominated heads from a **BG** donor — controls for "any
      transplant from a trained model helps".

    Using a donor and recipient that share an initialization seed reduces, but
    does not eliminate, the basis mismatch. Interpret accordingly.
    """
    n_heads = recipient.blocks[0].n_heads
    d_head = recipient.blocks[0].d_head
    d_model = n_heads * d_head

    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        if not 0 <= layer < len(recipient.blocks):
            raise ValueError(f"layer {layer} out of range")
        if not 0 <= head < n_heads:
            raise ValueError(f"head {head} out of range")
        by_layer.setdefault(layer, []).append(head)

    updated = list(recipient.blocks)
    for layer, layer_heads in by_layer.items():
        rblock, dblock = updated[layer], donor.blocks[layer]

        qkv = np.asarray(rblock.qkv.weight).copy()
        dqkv = np.asarray(dblock.qkv.weight)
        proj = np.asarray(rblock.proj.weight).copy()
        dproj = np.asarray(dblock.proj.weight)

        for head in layer_heads:
            for component in range(3):          # q, k, v
                lo = component * d_model + head * d_head
                qkv[lo : lo + d_head, :] = dqkv[lo : lo + d_head, :]
            lo = head * d_head
            proj[:, lo : lo + d_head] = dproj[:, lo : lo + d_head]

        rblock = eqx.tree_at(lambda b: b.qkv.weight, rblock, jnp.asarray(qkv))
        rblock = eqx.tree_at(lambda b: b.proj.weight, rblock, jnp.asarray(proj))
        updated[layer] = rblock

    return eqx.tree_at(lambda m: m.blocks, recipient, tuple(updated))
