"""The V2 language micro-world: five streams over one vocabulary.

Ordinary next-token prediction, no MODE tokens, no task heads. Every stream
uses the same clause template and the same surface form, so a model cannot
identify the stream from surface statistics alone.

    clause    [DET, entity, REL, value, DOT]
    document  [BOS, clause, clause, ..., query-clause]

The five streams, and what each is for:

``IND``   source ``A``. Entities recur within a document and each entity keeps
          a consistent value, so a repeated ``DET entity REL`` prefix predicts
          its continuation. Prefix matching pays.

``IND_R`` control ``A'``. The same template, the same clause count, the same
          per-document entity set and the *same entity recurrence* — but a
          recurring entity carries a fresh value each time, so copying from its
          earlier occurrence predicts nothing. Prefix matching pays nothing.

``BIND``  target capability ``B``. A nonce entity is bound to a value early in
          the document and queried at the end. Nonce identities are resampled
          per example, so the binding cannot be stored in the weights and the
          answer is recoverable only from context.

``FACT``  negative control ``C``. The queried entity is a *known* entity whose
          value is fixed globally across the whole corpus, and — critically —
          its binding does **not** appear in the document. The answer is
          recoverable only from the weights.

``BG``    background filler at matched surface statistics, used for the
          no-informative-source arm and for the phase-boundary control.

How ``A``/``A'`` reached this construction, and what the audit rejected
----------------------------------------------------------------------
Three constructions were considered. The record matters because the entire V2
causal claim rests on these two streams differing in exactly one property.

1. **Bigram resampler** (the original design). Generate ``A'`` by resampling
   each document from its own bigram statistics. Rejected as hard to verify
   and matching the marginals only asymptotically.

2. **Fresh-entity sampling.** Keep the template and slot positions identical;
   draw every entity slot i.i.d. from the pool so nothing recurs.
   **Rejected by ``scripts/audit_microworld_shortcuts.py``**, which found that
   short-range predictors could tell the streams apart: unigram cross-entropy
   gap 0.0060 against a null limit of 0.0030, bigram 0.0117 against 0.0056.
   The cause was visible in the audit's own diagnostic — 2.88 distinct
   entities per document in ``A`` against 7.94 in ``A'``. Recurrence
   necessarily concentrates entity mass, so *any* construction that removes
   recurrence also changes entity diversity, and a model could exploit the
   diversity difference instead of the repeat structure.

3. **Value-rebinding** (what is implemented). Both streams draw the same
   per-document entity set with the same slot picks, so entity recurrence is
   identically distributed. Only the *binding* differs: in ``A`` a recurring
   entity keeps its value, in ``A'`` it draws a fresh one. Entity diversity,
   unigram and bigram marginals now match by construction rather than by
   luck, and the sole remaining difference is whether copying from an earlier
   occurrence predicts anything.

The residual asymmetry, recorded
--------------------------------
``A`` contains learnable long-range structure and ``A'`` does not, so a model
trained on ``A`` reaches a lower loss on its own stream. That is intrinsic to
the manipulation — it is what "prefix matching pays" *means* — and it cannot
be designed away. It is a live confound (C12 in the design doc): the arms end
phase 1 at different loss and therefore in different optimization states. The
guards are matched token budget, achieved loss recorded as telemetry rather
than assumed equal, and the ``BG`` arm as a third reference point.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "MicroConfig", "STREAMS", "sample_documents", "fact_table",
    "known_entities", "nonce_entities", "micro_sampler", "evaluate_stream",
    "value_permutation",
    "BatchCache",
]

PAD, BOS, DET, REL, DOT = 0, 1, 2, 3, 4
_N_SPECIAL = 5

STREAMS = ("IND", "IND_R", "BIND", "BINDT", "FACT", "BG")

CLAUSE_LEN = 5  # [DET, entity, REL, value, DOT]
QUERY_LEN = 4   # [DET, entity, REL, answer] — no DOT, answer is the last token


@dataclass(frozen=True)
class MicroConfig:
    """Shape of the micro-world. Frozen, and part of the data version.

    ``n_known`` entities carry a globally fixed value and are the substrate for
    ``FACT``; the remaining ``n_entities - n_known`` are nonce entities, used
    for ``BIND`` and for the ``IND``/``IND_R`` streams. The pools are disjoint
    so that a ``FACT`` answer can never be obtained from context and a ``BIND``
    answer can never be obtained from the weights.
    """

    n_entities: int = 512
    n_known: int = 64
    n_values: int = 64
    n_clauses: int = 8
    entities_per_doc: int = 3
    fact_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_known < 2:
            raise ValueError(f"n_known must be at least 2, got {self.n_known}")
        if self.n_known >= self.n_entities:
            raise ValueError(
                f"n_known ({self.n_known}) must be smaller than n_entities "
                f"({self.n_entities}); the nonce pool would otherwise be empty"
            )
        if self.n_clauses < 3:
            raise ValueError(
                f"n_clauses must be at least 3, got {self.n_clauses}; a document "
                "needs a binding clause, at least one filler, and a query"
            )
        if not 1 <= self.entities_per_doc <= self.n_clauses - 1:
            raise ValueError(
                f"entities_per_doc ({self.entities_per_doc}) must lie in "
                f"[1, n_clauses - 1] ({self.n_clauses - 1})"
            )
        if self.n_known % 2 or self.n_values % 2:
            raise ValueError("n_known and n_values must be even for balanced tables")
        if self.n_known > self.n_values * (self.n_known // self.n_values + 1):
            raise ValueError("n_known/n_values ratio leaves the fact table unbalanced")

    # Token layout: specials, entities, values.
    @property
    def entity_base(self) -> int:
        return _N_SPECIAL

    @property
    def value_base(self) -> int:
        return self.entity_base + self.n_entities

    @property
    def vocab_size(self) -> int:
        return self.value_base + self.n_values

    @property
    def seq_len(self) -> int:
        """``[BOS, (n_clauses - 1) full clauses, query clause]``."""
        return 1 + (self.n_clauses - 1) * CLAUSE_LEN + QUERY_LEN

    @property
    def answer_position(self) -> int:
        return self.seq_len - 1

    @property
    def answer_target_index(self) -> int:
        """Index of the answer within the *shifted* target sequence."""
        return self.answer_position - 1

    @property
    def n_nonce(self) -> int:
        return self.n_entities - self.n_known


@lru_cache(maxsize=None)
def known_entities(config: MicroConfig) -> np.ndarray:
    """Entity ids reserved for ``FACT``. The first ``n_known`` of the pool."""
    return np.arange(config.n_known, dtype=np.int64)


@lru_cache(maxsize=None)
def nonce_entities(config: MicroConfig, half: str = "") -> np.ndarray:
    """Entity ids usable as nonce. Disjoint from :func:`known_entities`.

    ``half`` selects a disjoint sub-pool: ``"h1"`` the first half, ``"h2"`` the
    second, ``""`` the whole pool. The split exists for the C1 discrimination
    scout, which asks whether the ``A -> B`` advantage is ordinary content
    transfer or a mechanism that generalizes across token identity. Training the
    source on ``h1`` and the target on ``h2`` removes every shared entity token
    while leaving the copy *structure* identical.
    """
    full = np.arange(config.n_known, config.n_entities, dtype=np.int64)
    if not half:
        return full
    mid = len(full) // 2
    if half == "h1":
        return full[:mid]
    if half == "h2":
        return full[mid:]
    raise ValueError(f"half must be '', 'h1' or 'h2', got {half!r}")


@lru_cache(maxsize=None)
def value_permutation(config: MicroConfig) -> np.ndarray:
    """A fixed derangement of the value vocabulary, behind ``BINDT``.

    ``BINDT`` is capability ``B2``: retrieve the in-context binding **and then**
    map it through this permutation. It exists to separate *direct task
    transfer* from *developmental readiness*.

    A model carrying the copy circuit that source ``A`` induces will retrieve
    the bound value and emit it unchanged. Because the permutation has **no
    fixed points**, that answer is always wrong, so zero-shot accuracy is
    driven to zero rather than merely to chance. Any advantage the ``A`` arm
    shows on ``B2`` therefore cannot be a head start on the task itself; it has
    to appear as a faster acquisition *rate*.

    A systematic **below-chance** ``t = 0`` on the ``A`` arm is itself
    informative: it is the signature of a transferred retrieval mechanism
    feeding the wrong output map.
    """
    rng = np.random.default_rng(config.fact_seed + 977)
    n = config.n_values
    perm = rng.permutation(n)
    # Repair fixed points by rotating them among themselves, so the result is a
    # derangement and "copy the retrieved value" is never accidentally correct.
    fixed = np.flatnonzero(perm == np.arange(n))
    if fixed.size == 1:
        other = int((fixed[0] + 1) % n)
        perm[[fixed[0], other]] = perm[[other, fixed[0]]]
    elif fixed.size > 1:
        perm[fixed] = perm[np.roll(fixed, 1)]
    return perm


@lru_cache(maxsize=None)
def fact_table(config: MicroConfig) -> np.ndarray:
    """The globally fixed known-entity to value map behind ``FACT``.

    Balanced across values by construction: each value is assigned to the same
    number of known entities, so the marginal answer distribution of ``FACT``
    is uniform and matches ``BIND``'s. An unbalanced table would make the
    answer weakly predictable without reading the entity, which would let the
    negative control be solved by a frequency heuristic.
    """
    rng = np.random.default_rng(config.fact_seed)
    reps = int(np.ceil(config.n_known / config.n_values))
    values = np.tile(np.arange(config.n_values, dtype=np.int64), reps)[: config.n_known]
    return rng.permutation(values)


def _assemble(config: MicroConfig, entities: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Lay out ``(batch, n_clauses)`` entity/value pairs as token sequences.

    The final clause is the query: its value token is the answer and carries no
    trailing DOT, so the answer is always the last position of the sequence and
    the loss index is a constant.
    """
    batch, n_clauses = entities.shape
    ent_tok = config.entity_base + entities
    val_tok = config.value_base + values

    body = np.empty((batch, n_clauses - 1, CLAUSE_LEN), dtype=np.int64)
    body[:, :, 0] = DET
    body[:, :, 1] = ent_tok[:, :-1]
    body[:, :, 2] = REL
    body[:, :, 3] = val_tok[:, :-1]
    body[:, :, 4] = DOT

    query = np.empty((batch, QUERY_LEN), dtype=np.int64)
    query[:, 0] = DET
    query[:, 1] = ent_tok[:, -1]
    query[:, 2] = REL
    query[:, 3] = val_tok[:, -1]

    bos = np.full((batch, 1), BOS, dtype=np.int64)
    return np.concatenate([bos, body.reshape(batch, -1), query], axis=1)


def _sample_ind(rng, config: MicroConfig, batch: int, *, recur: bool, half: str = "") -> np.ndarray:
    """``IND`` (``recur=True``) and ``IND_R`` (``recur=False``).

    Both draw ``n_clauses`` entity slots per document and both draw entities
    uniformly from the nonce pool, so the corpus-level entity marginal is
    uniform in each. They differ in exactly one respect: ``IND`` draws the
    slots from a small per-document set, with each drawn entity keeping a
    fixed value for the whole document, so a repeated prefix is predictive;
    ``IND_R`` draws every slot independently, so nothing recurs.
    """
    pool = nonce_entities(config, half)

    # Both streams draw the SAME per-document entity set and the SAME slot
    # picks, so the entity token sequence is identically distributed and
    # entities recur just as often in each. See the audit note below for why
    # this is not the obvious construction.
    doc_sets = np.stack(
        [rng.choice(pool, size=config.entities_per_doc, replace=False) for _ in range(batch)]
    )
    picks = rng.integers(0, config.entities_per_doc, size=(batch, config.n_clauses))
    entities = np.take_along_axis(doc_sets, picks, axis=1)

    if recur:
        # A recurring entity keeps its value, so the repeated prefix
        # "DET entity REL" predicts the value that follows it.
        doc_values = rng.integers(0, config.n_values, size=(batch, config.entities_per_doc))
        values = np.take_along_axis(doc_values, picks, axis=1)
    else:
        # The entity recurs just as often, but carries a fresh value each time,
        # so copying from its earlier occurrence predicts nothing.
        values = rng.integers(0, config.n_values, size=(batch, config.n_clauses))

    return _assemble(config, entities, values)


def _sample_bind(rng, config: MicroConfig, batch: int, half: str = "") -> np.ndarray:
    """``BIND`` — capability ``B``. In-context retrieval of a nonce binding.

    The queried entity is bound in the first clause and queried in the last;
    the intervening clauses are distractors over *other* nonce entities. The
    binding is only ever available from context.
    """
    pool = nonce_entities(config, half)
    n_slots = config.n_clauses
    entities = rng.choice(pool, size=(batch, n_slots), replace=True)
    values = rng.integers(0, config.n_values, size=(batch, n_slots))

    # Clause 0 binds the queried entity; the query repeats it with its value.
    entities[:, -1] = entities[:, 0]
    values[:, -1] = values[:, 0]

    # A distractor that happens to reuse the queried entity would create a
    # second, possibly contradictory binding. Resample those slots away from
    # the queried identity so the retrieval target is unambiguous.
    for j in range(1, n_slots - 1):
        clash = entities[:, j] == entities[:, 0]
        while clash.any():
            entities[clash, j] = rng.choice(pool, size=int(clash.sum()), replace=True)
            clash = entities[:, j] == entities[:, 0]
    return _assemble(config, entities, values)


def _sample_bindt(rng, config: MicroConfig, batch: int, half: str = "") -> np.ndarray:
    """``BINDT`` — capability ``B2``. In-context retrieval **then** a transform.

    Identical to ``BIND`` in template, clause count, query position and answer
    marginal; the only difference is that the queried answer is the permuted
    value rather than the bound value.
    """
    docs = _sample_bind(rng, config, batch, half)
    perm = value_permutation(config)
    bound = docs[:, -1] - config.value_base
    docs[:, -1] = config.value_base + perm[bound]
    return docs


def _sample_fact(rng, config: MicroConfig, batch: int, half: str = "") -> np.ndarray:
    """``FACT`` — capability ``C``. Parametric recall of a global binding.

    Identical template, identical clause count, identical query position. The
    single difference from ``BIND`` is that the queried entity does **not**
    appear earlier in the document, so context cannot supply the answer and
    only the globally fixed table can. That is the minimal possible
    contextual-versus-parametric contrast; anything smaller would not separate
    the two, and anything larger would let a surface cue distinguish them.
    """
    table = fact_table(config)
    known = known_entities(config)
    pool = nonce_entities(config, half)

    entities = rng.choice(pool, size=(batch, config.n_clauses), replace=True)
    values = rng.integers(0, config.n_values, size=(batch, config.n_clauses))

    queried = rng.choice(known, size=batch, replace=True)
    entities[:, -1] = queried
    values[:, -1] = table[queried]
    return _assemble(config, entities, values)


def _sample_bg(rng, config: MicroConfig, batch: int, half: str = "") -> np.ndarray:
    """``BG`` — filler at matched surface statistics and no learnable binding.

    Every slot is independent, and the query answer is independent of
    everything in the document, so ``BG`` is at the entropy floor: it teaches
    the template and nothing else. That is what makes it usable both as the
    no-informative-source arm and as the phase-boundary control.
    """
    pool = nonce_entities(config, half)
    entities = rng.choice(pool, size=(batch, config.n_clauses), replace=True)
    values = rng.integers(0, config.n_values, size=(batch, config.n_clauses))
    return _assemble(config, entities, values)


_SAMPLERS = {
    "IND": lambda rng, c, b, h: _sample_ind(rng, c, b, recur=True, half=h),
    "IND_R": lambda rng, c, b, h: _sample_ind(rng, c, b, recur=False, half=h),
    "BIND": lambda rng, c, b, h: _sample_bind(rng, c, b, h),
    "BINDT": lambda rng, c, b, h: _sample_bindt(rng, c, b, h),
    "FACT": lambda rng, c, b, h: _sample_fact(rng, c, b, h),
    "BG": lambda rng, c, b, h: _sample_bg(rng, c, b, h),
}


def micro_sampler(key, family: str, config: MicroConfig, batch_size: int) -> dict:
    """Adapter matching :func:`dsi.train.train_phase`'s ``sampler`` contract.

    ``train_phase`` hands out JAX keys; generation here is numpy with integer
    seeds. The key is reduced to an integer seed, which costs one device sync
    per step. That is negligible against a training step and keeps the
    determinism contract intact: the same key yields the same documents.

    A mixture family may be written ``"IND+BG"``, splitting the batch evenly.
    That is how a background stream is held at a fixed weight in every phase,
    which is what makes a curriculum change emphasis rather than swapping the
    visible distribution wholesale.
    """
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    return {"tokens": jnp.asarray(_sample_family(seed, config, family, batch_size))}


@eqx.filter_jit
def _answer_metrics(model, tokens: jnp.ndarray, answer_index: int):
    """Accuracy and loss at the answer position, compiled once.

    Without ``filter_jit`` this retraces on every call and costs ~186 ms per
    evaluation; compiled it is roughly two orders of magnitude cheaper. Pure
    throughput — the arithmetic is unchanged.
    """
    logits = jax.vmap(model)(tokens[:, :-1])
    logprobs = jax.nn.log_softmax(logits[:, answer_index], axis=-1)
    target = tokens[:, -1]
    picked = jnp.take_along_axis(logprobs, target[:, None], axis=-1).squeeze(-1)
    return (logprobs.argmax(axis=-1) == target).mean(), -picked.mean()


def evaluate_stream(model, config: MicroConfig, stream: str, seed: int, batch: int) -> dict:
    """Answer-position accuracy and loss for one stream.

    Scored at the final position only. Capability ``B`` and capability ``C``
    are both "produce the right value at the query", so a whole-sequence loss
    would be dominated by the template both streams share and would not
    separate them.
    """
    tokens = jnp.asarray(sample_documents(seed, config, stream, batch))
    accuracy, loss = _answer_metrics(model, tokens, config.answer_target_index)
    return {"accuracy": float(accuracy), "loss": float(loss)}


class BatchCache:
    """Pre-generated batches for one phase, matching the online sampler exactly.

    The online path derives a fresh key per step with ``fold_in`` and reduces it
    to an integer seed, which costs a host-device sync every step (~1.7 ms, a
    fifth of step time on a small model). This precomputes **every** seed in one
    vectorized device call, generates the documents up front, and then serves
    them by step index with no sync at all.

    It is a throughput change only. The seeds, the generator, the stream, the
    batch size and the ordering are identical to the online path, which
    ``scripts/verify_batch_cache.py`` asserts elementwise before any cached run
    is allowed to count.
    """

    def __init__(self, data_key, family: str, config: MicroConfig,
                 batch_size: int, n_steps: int):
        seeds = jax.vmap(
            lambda i: jax.random.randint(
                jax.random.fold_in(data_key, i), (), 0, 2**31 - 1)
        )(jnp.arange(1, n_steps + 1))
        self._seeds = [int(s) for s in np.asarray(seeds)]   # one sync, not n_steps
        self._tokens = [
            jnp.asarray(_sample_family(seed, config, family, batch_size))
            for seed in self._seeds
        ]
        self._index = 0
        self.n_steps = n_steps

    def __call__(self, key, family: str, task, batch_size: int) -> dict:
        if self._index >= len(self._tokens):
            raise RuntimeError(
                f"BatchCache exhausted after {len(self._tokens)} steps; the "
                "training loop asked for more batches than the cache was built "
                "for, so cached and online paths would diverge"
            )
        tokens = self._tokens[self._index]
        self._index += 1
        return {"tokens": tokens}


def _sample_family(seed: int, config: MicroConfig, family: str, batch_size: int):
    """Generation shared by the online sampler and the cache, mixtures included."""
    parts = family.split("+")
    if len(parts) == 1:
        return sample_documents(seed, config, family, batch_size)
    per = batch_size // len(parts)
    if per < 1:
        raise ValueError(
            f"batch_size {batch_size} cannot be split across {len(parts)} streams")
    chunks = [
        sample_documents(seed + i, config, part,
                         per if i else batch_size - per * (len(parts) - 1))
        for i, part in enumerate(parts)
    ]
    return np.concatenate(chunks, axis=0)


def sample_documents(seed: int, config: MicroConfig, stream: str, batch: int) -> np.ndarray:
    """Draw ``batch`` documents from ``stream``.

    Generation is numpy with an explicit integer seed rather than a JAX key.
    The structured index manipulation here is awkward under ``jit`` and the
    cost is off the critical path, but the determinism contract is unchanged:
    the same ``(seed, config, stream, batch)`` yields the same tokens, which
    preflight check P6 asserts.
    """
    base, _, half = stream.partition("#")
    if base not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}, got {base!r}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    rng = np.random.default_rng(seed)
    return _SAMPLERS[base](rng, config, batch, half)
