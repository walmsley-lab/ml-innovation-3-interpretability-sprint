"""V2 preflight: substrate and apparatus checks P1, P2, P6, P7.

No scientific claim is made from this script and no run it performs is
reported as a result. Its only job is to answer the most dangerous question
in the V2 programme before any compute is spent on the scout:

    Could the substrate itself manufacture the effect?

Checks implemented here are the ones that need no trained model:

    P1  A / A' match on low-order statistics, and differ in repeat structure
    P2  B / C match except for contextual vs parametric retrieval
    P6  generation and checkpointing are deterministic and round-trip
    P7  head ablation does what it claims and is a no-op when empty

P3 (the M probe detects the mechanism), P4 (B is learnable but does not
ceiling) and P5 (the phase boundary alone produces nothing) each need
training and are separate.

Each check prints its measured quantities and a PASS/FAIL against a criterion
fixed here, before any of them ran.

    PYTHONPATH=src python scripts/preflight_microworld.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsi.checkpoint import load_model, save_model
from dsi.mechanism import MProbeSpec, ablate_heads, head_attention, probe_sequences
from dsi.microworld import (
    CLAUSE_LEN, DET, MicroConfig, fact_table, known_entities, nonce_entities,
    sample_documents,
)
from dsi.model import ModelConfig, init_model

# Criteria, fixed before any check runs.
#
# REVISION, recorded: the first version of this file used absolute tolerances
# (unigram 0.02, bigram 0.03, answer 0.02). Those were mis-specified — they sat
# *below the sampling noise floor*, so they could never pass no matter how well
# the streams matched. Total-variation distance between two independent draws
# over k categories with n samples is ~sqrt(k / 2*pi*n) by construction: ~0.05
# for the 448-entity marginal and ~0.05 for the 64-way answer marginal at these
# batch sizes.
#
# The replacement is null-calibrated rather than absolute, which is the
# project's existing discipline: the cross-stream distance must be no larger
# than the *same-stream* distance between two different seeds. That criterion
# can still fail — a real structural mismatch would exceed the null — and it
# does not move when batch size changes.
#
# This is a criterion correction, not a threshold relaxed against an outcome.
# The structural checks, which are what actually test the design, passed
# unchanged in both versions and are absolute.
NULL_SIGMA = 3.0        # cross-stream TV must be within mean + 3sd of the null
N_NULL_PAIRS = 8        # seed pairs used to estimate the same-stream null
MIN_BINDING_GAP = 0.50  # min difference in binding-consistency rate; a huge effect, no noise issue
MAX_REPEAT_MISMATCH = 0.01  # entity recurrence is a matched control and must agree
MIN_ENTROPY_RATIO = 0.99  # B and C answer entropy, each against log(n_values)


def _tv(a: Counter, b: Counter) -> float:
    """Total-variation distance between two empirical distributions."""
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def _unigram(docs: np.ndarray) -> Counter:
    return Counter(docs.reshape(-1).tolist())


def _bigram(docs: np.ndarray) -> Counter:
    pairs = np.stack([docs[:, :-1], docs[:, 1:]], axis=-1).reshape(-1, 2)
    return Counter(map(tuple, pairs.tolist()))


def _entity_cols(config: MicroConfig) -> list[int]:
    cols = [1 + j * CLAUSE_LEN + 1 for j in range(config.n_clauses - 1)]
    cols.append(config.seq_len - 3)
    return cols


def _repeat_rate(docs: np.ndarray, config: MicroConfig) -> float:
    """Fraction of entity slots whose entity already appeared in the document.

    Under the value-rebinding construction this is a **matched control**, not
    the manipulation: entities recur equally often in A and A'. It is measured
    so that the match is asserted rather than assumed.
    """
    ents = docs[:, _entity_cols(config)]
    seen_before = np.zeros_like(ents, dtype=bool)
    for j in range(1, ents.shape[1]):
        seen_before[:, j] = (ents[:, :j] == ents[:, j : j + 1]).any(axis=1)
    return float(seen_before[:, 1:].mean())


def _binding_rate(docs: np.ndarray, config: MicroConfig) -> float:
    """Fraction of recurring-entity value slots that repeat the earlier value.

    This is the property A and A' actually differ in. ~1.0 in A by
    construction, ~1/n_values in A'. See
    ``scripts/audit_microworld_shortcuts.py`` for why entity recurrence itself
    could not serve as the discriminator: removing recurrence also changes
    entity diversity, which short-range predictors could exploit.
    """
    ent_cols = _entity_cols(config)
    val_cols = [c + 2 for c in ent_cols]
    hits = total = 0
    for doc in docs:
        first: dict[int, int] = {}
        for ec, vc in zip(ent_cols, val_cols):
            ent = int(doc[ec])
            if ent in first:
                total += 1
                hits += first[ent] == int(doc[vc])
            else:
                first[ent] = int(doc[vc])
    return hits / total if total else 0.0


def _null_band(config: MicroConfig, stream: str, batch: int, featurizer) -> tuple[float, float]:
    """Mean and sd of same-stream TV distance across independent seeds.

    This is the noise floor a cross-stream comparison has to be judged against.
    Estimating it from the *same* generator that produces the comparison means
    the criterion adapts to batch size and category count instead of encoding a
    guess about them.
    """
    distances = [
        _tv(
            featurizer(sample_documents(1000 + 2 * i, config, stream, batch)),
            featurizer(sample_documents(1001 + 2 * i, config, stream, batch)),
        )
        for i in range(N_NULL_PAIRS)
    ]
    return float(np.mean(distances)), float(np.std(distances, ddof=1))


def check_p1(config: MicroConfig, batch: int) -> dict:
    """A / A' matched on low-order statistics, differing in repeat structure."""
    a = sample_documents(11, config, "IND", batch)
    a_r = sample_documents(11, config, "IND_R", batch)

    result: dict = {
        "unigram_tv": _tv(_unigram(a), _unigram(a_r)),
        "bigram_tv": _tv(_bigram(a), _bigram(a_r)),
        "length_a": int(a.shape[1]),
        "length_a_r": int(a_r.shape[1]),
        "repeat_rate_a": _repeat_rate(a, config),
        "repeat_rate_a_r": _repeat_rate(a_r, config),
        "binding_rate_a": _binding_rate(a, config),
        "binding_rate_a_r": _binding_rate(a_r, config),
    }
    # Entity recurrence is a matched control; the binding is the manipulation.
    result["repeat_rate_matched"] = bool(
        abs(result["repeat_rate_a"] - result["repeat_rate_a_r"]) <= MAX_REPEAT_MISMATCH
    )
    result["binding_gap"] = result["binding_rate_a"] - result["binding_rate_a_r"]

    # Null: how far apart do two draws of the *same* stream land? The
    # cross-stream distance is compared against the wider of the two nulls, so
    # the criterion is not gamed by whichever stream happens to be quieter.
    for name, featurizer in (("unigram", _unigram), ("bigram", _bigram)):
        bands = [_null_band(config, s, batch, featurizer) for s in ("IND", "IND_R")]
        mean, sd = max(bands, key=lambda b: b[0] + NULL_SIGMA * b[1])
        limit = mean + NULL_SIGMA * sd
        result[f"{name}_null_mean"] = mean
        result[f"{name}_null_limit"] = limit
        result[f"{name}_within_null"] = bool(result[f"{name}_tv"] <= limit)

    result["pass"] = bool(
        result["unigram_within_null"]
        and result["bigram_within_null"]
        and result["length_a"] == result["length_a_r"]
        and result["repeat_rate_matched"]
        and result["binding_gap"] >= MIN_BINDING_GAP
    )
    return result


def check_p2(config: MicroConfig, batch: int) -> dict:
    """B / C matched except contextual vs parametric retrieval."""
    b = sample_documents(23, config, "BIND", batch)
    c = sample_documents(23, config, "FACT", batch)

    ans_b = Counter(b[:, -1].tolist())
    ans_c = Counter(c[:, -1].tolist())

    ent_cols = [1 + j * CLAUSE_LEN + 1 for j in range(config.n_clauses - 1)]
    q_col = config.seq_len - 3

    def queried_in_context(docs: np.ndarray) -> float:
        return float((docs[:, ent_cols] == docs[:, q_col : q_col + 1]).any(axis=1).mean())

    # Structural identity: everything except the entity and value tokens must
    # be the same template in both streams.
    template_cols = [i for i in range(config.seq_len) if i not in set(ent_cols + [q_col])]
    template_cols = [i for i in template_cols if b[:, i].std() == 0 and c[:, i].std() == 0]

    known = set(known_entities(config).tolist())
    table = fact_table(config)
    q_ent_c = c[:, q_col] - config.entity_base
    fact_consistent = float((c[:, -1] - config.value_base == table[q_ent_c]).mean())

    def answers(docs: np.ndarray) -> Counter:
        return Counter(docs[:, -1].tolist())

    uniform = float(np.log(config.n_values))
    bands = [_null_band(config, s, batch, answers) for s in ("BIND", "FACT")]
    mean, sd = max(bands, key=lambda x: x[0] + NULL_SIGMA * x[1])
    limit = mean + NULL_SIGMA * sd

    result = {
        "answer_marginal_tv": _tv(ans_b, ans_c),
        "answer_null_mean": mean,
        "answer_null_limit": limit,
        "answer_entropy_b": float(-sum(
            (v / batch) * np.log(v / batch) for v in ans_b.values())),
        "answer_entropy_c": float(-sum(
            (v / batch) * np.log(v / batch) for v in ans_c.values())),
        "uniform_entropy": uniform,
        "template_positions_identical": bool(
            (b[:, template_cols][0] == c[:, template_cols][0]).all()),
        "queried_in_context_b": queried_in_context(b),
        "queried_in_context_c": queried_in_context(c),
        "c_queries_known_only": bool(set(q_ent_c.tolist()) <= known),
        "c_answers_match_table": fact_consistent,
    }
    result["answer_within_null"] = bool(result["answer_marginal_tv"] <= limit)
    result["pass"] = bool(
        result["answer_within_null"]
        and result["answer_entropy_b"] / uniform >= MIN_ENTROPY_RATIO
        and result["answer_entropy_c"] / uniform >= MIN_ENTROPY_RATIO
        and result["template_positions_identical"]
        and result["queried_in_context_b"] == 1.0
        and result["queried_in_context_c"] == 0.0
        and result["c_queries_known_only"]
        and result["c_answers_match_table"] == 1.0
    )
    return result


def check_p6(config: MicroConfig, batch: int) -> dict:
    """Determinism of generation, and checkpoint round-trip."""
    same = all(
        np.array_equal(
            sample_documents(7, config, s, batch), sample_documents(7, config, s, batch)
        )
        for s in ("IND", "IND_R", "BIND", "FACT", "BG")
    )
    differs = not np.array_equal(
        sample_documents(7, config, "IND", batch),
        sample_documents(8, config, "IND", batch),
    )

    mc = ModelConfig(vocab_size=config.vocab_size, d_model=32, n_heads=4,
                     n_layers=2, d_ff=64, max_len=config.seq_len)
    model = init_model(mc, jax.random.PRNGKey(3))
    with tempfile.TemporaryDirectory() as tmp:
        save_model(Path(tmp) / "ckpt", model, mc)
        loaded, loaded_cfg = load_model(Path(tmp) / "ckpt")

    leaves_a = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    leaves_b = jax.tree.leaves(eqx.filter(loaded, eqx.is_inexact_array))
    bitwise = all(np.array_equal(np.asarray(x), np.asarray(y))
                  for x, y in zip(leaves_a, leaves_b))

    result = {
        "generation_deterministic": bool(same),
        "seed_changes_output": bool(differs),
        "checkpoint_bitwise_identical": bool(bitwise),
        "config_round_trip": bool(loaded_cfg == mc),
    }
    result["pass"] = all(result.values())
    return result


def check_p7(config: MicroConfig) -> dict:
    """Head ablation is targeted, and empty ablation is a no-op."""
    mc = ModelConfig(vocab_size=config.vocab_size, d_model=32, n_heads=4,
                     n_layers=2, d_ff=64, max_len=config.seq_len)
    model = init_model(mc, jax.random.PRNGKey(5))
    spec = MProbeSpec(block_len=8, n_sequences=4)
    tokens = probe_sequences(spec, config.vocab_size)[0]

    noop = ablate_heads(model, ())
    noop_same = bool(jnp.allclose(model(tokens), noop(tokens)))

    ablated = ablate_heads(model, ((1, 2),))
    d_head = mc.d_model // mc.n_heads
    cols = slice(2 * d_head, 3 * d_head)
    target_zero = bool(jnp.all(ablated.blocks[1].proj.weight[:, cols] == 0.0))
    others_intact = bool(
        jnp.allclose(
            jnp.delete(ablated.blocks[1].proj.weight, jnp.arange(2 * d_head, 3 * d_head), axis=1),
            jnp.delete(model.blocks[1].proj.weight, jnp.arange(2 * d_head, 3 * d_head), axis=1),
        )
    )
    layer0_intact = bool(jnp.allclose(ablated.blocks[0].proj.weight,
                                      model.blocks[0].proj.weight))
    changes_output = not bool(jnp.allclose(model(tokens), ablated(tokens)))

    attn = head_attention(model, tokens)
    rows_sum_to_one = bool(jnp.allclose(attn.sum(axis=-1), 1.0, atol=1e-5))

    result = {
        "empty_ablation_is_noop": noop_same,
        "target_columns_zeroed": target_zero,
        "other_columns_intact": others_intact,
        "other_layers_intact": layer0_intact,
        "ablation_changes_output": changes_output,
        "attention_rows_normalised": rows_sum_to_one,
    }
    result["pass"] = all(result.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path("artifacts/preflight"))
    args = parser.parse_args()

    config = MicroConfig()
    checks = {
        "P1_a_aprime_matching": check_p1(config, args.batch),
        "P2_b_c_matching": check_p2(config, args.batch),
        "P6_determinism_checkpoint": check_p6(config, 256),
        "P7_ablation": check_p7(config),
    }

    print(f"micro-world: vocab {config.vocab_size}, seq_len {config.seq_len}, "
          f"{config.n_clauses} clauses, {config.n_known} known / "
          f"{config.n_nonce} nonce entities\n")
    for name, result in checks.items():
        status = "PASS" if result["pass"] else "FAIL"
        print(f"[{status}] {name}")
        for key, value in result.items():
            if key == "pass":
                continue
            shown = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"         {key:32s} {shown}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preflight.json").write_text(json.dumps(checks, indent=2) + "\n")

    failed = [n for n, r in checks.items() if not r["pass"]]
    if failed:
        print(f"PREFLIGHT FAILED: {', '.join(failed)}")
        print("Fix the apparatus. No scout runs are licensed.")
        return 1
    print("All implemented preflight checks pass. P3/P4/P5 still require training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
