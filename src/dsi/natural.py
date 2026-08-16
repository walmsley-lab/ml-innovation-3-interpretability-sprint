"""Natural-corpus token path: a frozen vocabulary and a chunker.

The synthetic layers generate examples on demand, so exposure is exact by
construction. Natural text is a fixed, finite, unequally distributed supply,
and three things have to be made explicit before it can carry a transfer
measurement.

**The vocabulary is fitted on train only.** It is the same discipline as the
family proposer in :mod:`dsi.corpus`, for the same reason: a vocabulary
fitted on all documents has seen the held-out split, and the held-out loss it
produces is contaminated. ``fit_vocabulary`` records what it was fitted on
and :class:`Vocabulary` cannot be refitted.

**One token means one whitespace token, everywhere.** Ids are assigned over
``str.split()`` pieces, so a document's token count is identical before and
after the vocabulary exists. The support audit frozen in
``artifacts/natural_pilot`` counted tokens that way, which is what lets the
200,000-token floor recorded there remain exactly the currency the exposure
budget is denominated in. Rare pieces map to ``UNK`` rather than
disappearing, so vocabulary size changes what the model can express and never
changes how much text an arm was exposed to.

**Packing is contiguous and single-epoch.** Documents are concatenated in a
seeded order with ``EOS`` between them and cut into fixed-length chunks. A
family is exposed once, not cycled, because the four usable 20NG families
differ in supply by a factor of four (788,953 tokens against 204,445) and a
repeated small family would make the transfer estimate partly a measurement
of how many times its documents were seen.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["UNK", "EOS", "N_SPECIAL", "whitespace_tokens", "Vocabulary",
           "fit_vocabulary", "chunk_tokens", "chunk_documents", "coverage"]

UNK, EOS = 0, 1
N_SPECIAL = 2


def whitespace_tokens(text: str) -> list[str]:
    """The frozen audit tokenizer: exactly ``str.split()``.

    Committed as a named function because the support audit in
    ``DESIGN_LAYER2.md`` 14 was computed with it, and an audit whose
    tokenizer lives only in a notebook cannot be rechecked.
    """
    return text.split()


@dataclass(frozen=True)
class Vocabulary:
    """A frozen train-only word vocabulary. Ids are stable across runs."""

    itos: tuple
    fitted_on: str
    min_count: int
    max_size: int

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def stoi(self) -> dict:
        return {token: i for i, token in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        """Ids for one document's whitespace tokens. Length-preserving."""
        table = self._table()
        return [table.get(piece, UNK) for piece in whitespace_tokens(text)]

    def encode_document(self, text: str) -> list[int]:
        """``encode`` plus the terminating ``EOS``.

        This is the tokenizer handed to :func:`dsi.corpus.token_budget_sample`,
        so the boundary token is inside the budget rather than added to it
        afterwards; otherwise two arms matched on budget would differ in
        total exposure by the number of documents each happened to draw.
        """
        return self.encode(text) + [EOS]

    def _table(self) -> dict:
        cached = getattr(self, "_stoi_cache", None)
        if cached is None:
            cached = {token: i for i, token in enumerate(self.itos)}
            object.__setattr__(self, "_stoi_cache", cached)
        return cached

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(
            {"itos": list(self.itos), "fitted_on": self.fitted_on,
             "min_count": self.min_count, "max_size": self.max_size}) + "\n")

    @staticmethod
    def from_json(path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text())
        return Vocabulary(tuple(payload["itos"]), payload["fitted_on"],
                          payload["min_count"], payload["max_size"])


def fit_vocabulary(corpus, *, max_size: int = 8192, min_count: int = 2,
                   fitted_on: str = "train") -> Vocabulary:
    """Fit a vocabulary on one split. Call it on train, never on all documents.

    The vocabulary is pooled across families rather than fitted per family.
    A per-family vocabulary would make the source phase's ids unreadable in
    the target phase, and the resulting transfer effect would be a
    measurement of vocabulary mismatch wearing the name of a developmental
    effect.
    """
    if fitted_on != "train":
        raise ValueError(
            f"vocabulary must be fitted on the train split, got {fitted_on!r}; "
            "fitting on val or test leaks the evaluation set into the token ids")
    counts = Counter()
    for document in corpus.documents:
        counts.update(whitespace_tokens(document.text))
    ranked = sorted((t for t, c in counts.items() if c >= min_count),
                    key=lambda t: (-counts[t], t))
    itos = ("<unk>", "<eos>") + tuple(ranked[: max(0, max_size - N_SPECIAL)])
    return Vocabulary(itos, fitted_on, min_count, max_size)


def coverage(vocabulary: Vocabulary, corpus) -> dict:
    """Fraction of tokens that are in-vocabulary, per corpus.

    Reported rather than optimized. The four 20NG families differ in how
    much rare jargon they carry, so they will differ in ``UNK`` rate, and a
    family whose text is largely ``UNK`` has an easy next-token task for a
    reason that has nothing to do with development.
    """
    total = known = 0
    for document in corpus.documents:
        ids = vocabulary.encode(document.text)
        total += len(ids)
        known += sum(1 for i in ids if i != UNK)
    return {"tokens": total, "in_vocabulary": known,
            "unk_rate": 0.0 if not total else 1.0 - known / total}


def chunk_tokens(tokens, seq_len: int) -> np.ndarray:
    """Cut a flat token stream into ``(n_chunks, seq_len)``.

    The trailing partial chunk is dropped, so every chunk carries the same
    number of loss positions and a step is the same amount of learning
    regardless of which family it came from.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    ids = np.asarray(tokens, dtype=np.int32)
    n_chunks = len(ids) // seq_len
    if n_chunks < 1:
        raise ValueError(
            f"{len(ids)} tokens is short of a single {seq_len}-token chunk")
    return ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)


def chunk_documents(documents, vocabulary: Vocabulary, seq_len: int, *,
                    seed: int = 0) -> np.ndarray:
    """Shuffle documents, concatenate with ``EOS``, and chunk contiguously.

    The shuffle is seeded and applied to documents rather than to chunks, so
    a chunk never mixes two distant parts of the corpus, and the same
    documents in the same order produce the same chunks in every run.
    """
    order = np.random.default_rng(seed).permutation(len(documents))
    stream: list[int] = []
    for i in order:
        stream.extend(vocabulary.encode_document(documents[i].text))
    return chunk_tokens(stream, seq_len)
