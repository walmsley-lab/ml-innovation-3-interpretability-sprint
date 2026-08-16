"""Invariants of the natural token path.

These guard the failure modes that are silent: a vocabulary that saw the
evaluation split, a tokenizer that changes how much text an arm was exposed
to, and a chunker whose output depends on something other than its seed.
"""

from __future__ import annotations

import pytest

from dsi.corpus import Corpus, Document
from dsi.natural import (EOS, UNK, Vocabulary, chunk_documents, chunk_tokens,
                         coverage, fit_vocabulary, whitespace_tokens)


def corpus_of(*texts: str) -> Corpus:
    return Corpus("t", "v1", tuple(Document.of(t) for t in texts))


@pytest.fixture
def train() -> Corpus:
    return corpus_of("the cat sat on the mat", "the dog sat on the log",
                     "a cat and a dog", "the mat and the log")


def test_audit_tokenizer_is_exactly_split():
    # The frozen support audit was computed with str.split(); if this drifts,
    # the 200,000-token gate silently means something else.
    text = "  spaced   out\ttabbed\nlines  "
    assert whitespace_tokens(text) == text.split()


def test_vocabulary_refuses_non_train_splits(train):
    for split in ("val", "test", "all"):
        with pytest.raises(ValueError, match="train split"):
            fit_vocabulary(train, fitted_on=split)


def test_encode_preserves_token_count(train):
    vocabulary = fit_vocabulary(train, min_count=1)
    for document in train.documents:
        assert len(vocabulary.encode(document.text)) == len(whitespace_tokens(document.text))


def test_encode_document_adds_exactly_one_eos(train):
    vocabulary = fit_vocabulary(train, min_count=1)
    ids = vocabulary.encode_document("the cat sat")
    assert ids[-1] == EOS
    assert ids.count(EOS) == 1
    assert len(ids) == 4


def test_out_of_vocabulary_maps_to_unk_not_dropped(train):
    vocabulary = fit_vocabulary(train, min_count=1)
    ids = vocabulary.encode("the quetzalcoatl sat")
    assert len(ids) == 3, "an unknown word must occupy a position, not vanish"
    assert ids[1] == UNK


def test_min_count_excludes_rare_tokens():
    corpus = corpus_of("cat cat", "a solitary word")
    frequent = fit_vocabulary(corpus, min_count=2)
    assert "cat" in frequent.itos          # appears twice
    assert "solitary" not in frequent.itos  # once


def test_max_size_is_respected(train):
    assert fit_vocabulary(train, min_count=1, max_size=5).size == 5


def test_vocabulary_round_trips(tmp_path, train):
    vocabulary = fit_vocabulary(train, min_count=1)
    path = tmp_path / "vocab.json"
    vocabulary.to_json(path)
    restored = Vocabulary.from_json(path)
    assert restored.itos == vocabulary.itos
    assert restored.encode("the cat") == vocabulary.encode("the cat")


def test_train_only_fitting_keeps_heldout_words_out(train):
    # The leakage this discipline exists to prevent: a val-only word must not
    # have an id, because a vocabulary that knows it has read the val split.
    held_out = corpus_of("the platypus sat on the mat")
    vocabulary = fit_vocabulary(train, min_count=1)
    assert "platypus" not in vocabulary.itos
    assert vocabulary.encode("platypus")[0] == UNK
    assert 0.0 < coverage(vocabulary, held_out)["unk_rate"] < 1.0


def test_chunking_drops_only_the_partial_tail():
    chunks = chunk_tokens(list(range(70)), 32)
    assert chunks.shape == (2, 32)
    assert chunks[0].tolist() == list(range(32))


def test_chunking_refuses_a_stream_shorter_than_one_chunk():
    with pytest.raises(ValueError, match="short of a single"):
        chunk_tokens([1, 2, 3], 32)


def test_document_chunking_is_seed_deterministic(train):
    vocabulary = fit_vocabulary(train, min_count=1)
    docs = train.documents
    a = chunk_documents(docs, vocabulary, 4, seed=0)
    assert (a == chunk_documents(docs, vocabulary, 4, seed=0)).all()
    b = chunk_documents(docs, vocabulary, 4, seed=1)
    assert a.shape == b.shape and not (a == b).all()


def test_equal_chunk_budgets_are_equal_token_budgets(train):
    # Exposure matching in the transfer runner rests on this: two arms cut to
    # the same number of chunks have received exactly the same token count,
    # whatever the underlying documents looked like.
    vocabulary = fit_vocabulary(train, min_count=1)
    other = corpus_of("a dog and a cat sat", "the log and the mat")
    a = chunk_documents(train.documents, vocabulary, 4, seed=0)[:2]
    b = chunk_documents(other.documents, vocabulary, 4, seed=0)[:2]
    assert a.size == b.size
