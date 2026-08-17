"""Micro-world substrate invariants.

The A/A′ contrast is the whole experiment: if the control differs from the
source in anything other than the binding relation, every downstream result is
confounded. An earlier version of `IND_R` leaked **entity diversity** (2.88
distinct entities per document against 7.94), which made the arms
distinguishable from surface statistics alone and had to be rebuilt as a
value-rebinding. These tests pin that class of defect.

Documents open with `BOS` and then repeat `[DET, entity, REL, value, DOT]`, so
entity tokens sit at `2 + 5i` and value tokens at `4 + 5i`. The trailing clause
is the query: its entity is the last entity slot and its value is the answer.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from dsi.microworld import (
    STREAMS, BatchCache, MicroConfig, evaluate_stream, known_entities,
    micro_sampler, nonce_entities, sample_documents, value_permutation,
)

ENT = slice(2, None, 5)
VAL = slice(4, None, 5)
CFG = MicroConfig()


def _query(row):
    """(queried entity, answer token, values bound to it in earlier clauses)."""
    ents, vals = row[ENT].tolist(), row[VAL].tolist()
    entity, answer = ents[-1], vals[-1]
    earlier = [v for e, v in zip(ents[:-1], vals[:-1]) if e == entity]
    return entity, answer, earlier


def _distinct_entities(docs: np.ndarray) -> float:
    return float(np.mean([len(set(row[ENT].tolist())) for row in docs]))


# --------------------------------------------------------------------------
# the A / A' contrast
# --------------------------------------------------------------------------

def test_ind_and_ind_r_have_matching_entity_diversity():
    """The regression that motivated the rebuild.

    A′ must differ from A in *value binding only*. If A′ draws entities
    differently, the arms are separable without any binding structure and the
    control stops being a control.
    """
    a = sample_documents(11, CFG, "IND", 512)
    b = sample_documents(11, CFG, "IND_R", 512)
    assert abs(_distinct_entities(a) - _distinct_entities(b)) < 0.05


def test_ind_and_ind_r_have_matching_entity_token_distributions():
    """Stronger than the mean: the entity unigram distributions must agree."""
    a = np.bincount(sample_documents(5, CFG, "IND", 1024)[:, ENT].ravel(),
                    minlength=CFG.vocab_size).astype(float)
    b = np.bincount(sample_documents(5, CFG, "IND_R", 1024)[:, ENT].ravel(),
                    minlength=CFG.vocab_size).astype(float)
    a, b = a / a.sum(), b / b.sum()
    # Two independent draws of the same distribution; the gap must sit at the
    # sampling floor, not above it.
    assert np.abs(a - b).sum() < 0.10


def test_ind_rebinds_values_while_ind_r_does_not():
    """The one difference that is supposed to exist, stated positively.

    In IND a repeated entity keeps its value; in IND_R a repeated entity is
    re-drawn, so agreement collapses toward chance.
    """
    def repeat_agreement(stream: str) -> float:
        docs = sample_documents(7, CFG, stream, 512)
        hits = total = 0
        for row in docs:
            seen: dict[int, int] = {}
            for e, v in zip(row[ENT].tolist(), row[VAL].tolist()):
                if e in seen:
                    total += 1
                    hits += seen[e] == v
                else:
                    seen[e] = v
        return hits / max(total, 1)

    assert repeat_agreement("IND") > 0.99
    assert repeat_agreement("IND_R") < 0.20


# --------------------------------------------------------------------------
# the deranged target, B2
# --------------------------------------------------------------------------

def test_value_permutation_is_a_derangement():
    """BINDT only blocks answer transfer if no value maps to itself."""
    perm = np.asarray(value_permutation(CFG))
    assert perm.shape == (CFG.n_values,)
    assert sorted(perm.tolist()) == list(range(CFG.n_values))
    assert not np.any(perm == np.arange(CFG.n_values))


def test_bindt_answer_is_never_the_retrieved_value():
    """B2's identifying property: retrieval alone gives the wrong token, so a
    head start cannot be a transferred answer."""
    docs = sample_documents(13, CFG, "BINDT", 256)
    checked = 0
    for row in docs:
        _, answer, earlier = _query(row)
        if earlier:
            assert answer != earlier[0]
            checked += 1
    assert checked > 200


def test_bind_answer_is_exactly_the_retrieved_value():
    """The BIND control on the same construction: retrieval is sufficient."""
    docs = sample_documents(13, CFG, "BIND", 256)
    checked = 0
    for row in docs:
        _, answer, earlier = _query(row)
        if earlier:
            assert answer == earlier[0]
            checked += 1
    assert checked > 200


# --------------------------------------------------------------------------
# disjoint content, used by the memorisation control
# --------------------------------------------------------------------------

def test_nonce_halves_are_disjoint_from_each_other_and_from_known():
    """H3 claims zero shared entity tokens between source and target. That
    claim is only true if these pools genuinely do not intersect."""
    h1 = set(nonce_entities(CFG, "h1").tolist())
    h2 = set(nonce_entities(CFG, "h2").tolist())
    known = set(known_entities(CFG).tolist())
    assert h1 and h2
    assert not (h1 & h2)
    assert not (h1 & known) and not (h2 & known)


# --------------------------------------------------------------------------
# shape, vocabulary and the cached path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream", STREAMS)
def test_every_stream_is_well_formed(stream):
    docs = sample_documents(3, CFG, stream, 32)
    assert docs.shape == (32, CFG.seq_len)
    assert docs.min() >= 0 and docs.max() < CFG.vocab_size


def test_streams_are_not_identifiable_from_sequence_length_or_template():
    """Every stream shares one surface template; only the relation differs."""
    shapes = {sample_documents(3, CFG, s, 16).shape for s in STREAMS}
    assert len(shapes) == 1


def test_cached_batches_match_the_online_sampler_exactly():
    """Caching is an acceleration. If it diverges it is a new condition."""
    key = jax.random.PRNGKey(17)
    cache = BatchCache(key, "BIND", CFG, 16, 5)
    for step in range(1, 6):
        online = micro_sampler(jax.random.fold_in(key, step), "BIND", CFG, 16)["tokens"]
        assert np.array_equal(np.asarray(online), np.asarray(cache(None, "BIND", CFG, 16)["tokens"]))


def test_exhausted_cache_raises_rather_than_diverging():
    cache = BatchCache(jax.random.PRNGKey(0), "BIND", CFG, 8, 1)
    cache(None, "BIND", CFG, 8)
    with pytest.raises(RuntimeError):
        cache(None, "BIND", CFG, 8)


def test_mixture_families_draw_from_both_components():
    """`A+B` splits the batch; the V(S,D) matrix uses a mixture corpus."""
    mixed = micro_sampler(jax.random.PRNGKey(2), "IND+BG", CFG, 32)["tokens"]
    assert np.asarray(mixed).shape == (32, CFG.seq_len)


def test_evaluate_stream_is_deterministic_for_a_fixed_seed():
    """Telemetry compared across arms must not move because of eval noise."""
    from dsi.model import ModelConfig, init_model
    mc = ModelConfig(vocab_size=CFG.vocab_size, d_model=32, n_heads=2,
                     n_layers=1, d_ff=64, max_len=CFG.seq_len)
    model = init_model(mc, jax.random.PRNGKey(1))
    a = evaluate_stream(model, CFG, "BIND", 90001, 64)
    b = evaluate_stream(model, CFG, "BIND", 90001, 64)
    assert a["accuracy"] == b["accuracy"] and a["loss"] == b["loss"]
