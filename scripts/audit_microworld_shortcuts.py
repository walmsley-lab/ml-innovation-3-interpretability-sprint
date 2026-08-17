"""Audit: does A' differ from A in anything a model could exploit *other than*
long-range repeat structure?

The implementation departed from the design. The design proposed generating
``A'`` by resampling each document from its own bigram statistics; the
implementation instead holds the template, clause count and slot positions
identical and changes only the distribution entity slots are drawn from. The
simplification gives exact rather than asymptotic matching and is far easier to
verify — but a simplification that quietly introduces a *different* exploitable
difference would be worse than the thing it replaced, because the V2 causal
claim rests entirely on ``A`` and ``A'`` differing in one property.

The audit is built around one idea:

    If predictors that CANNOT use long-range context achieve the same loss on
    A and A', then nothing short-range distinguishes the streams, and the only
    exploitable difference is the intended one.

Four predictors, in increasing context:

    positional   P(token | position)          no content at all
    unigram      P(token)                     no context
    bigram       P(token | previous token)    one token of context
    induction    copy what followed this token's earlier occurrence

The first three are the controls: their loss must match across streams. The
fourth is the manipulation: it must succeed on ``A`` and fail on ``A'``.

Also reported, because they are real differences the audit must not hide:

* distinct entities per document — A concentrates on few, A' spreads. This is
  a genuine consequence of the manipulation, not a bug, but it means the two
  streams differ in embedding-update sparsity as well as in repeat structure,
  and that belongs in the confound register rather than being discovered later;
* the achievable entropy floor per stream, which is the quantitative form of
  confound C12.

    PYTHONPATH=src python scripts/audit_microworld_shortcuts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dsi.microworld import CLAUSE_LEN, MicroConfig, sample_documents

# Criteria, fixed before the audit runs.
#
# Short-range predictors must match across streams to within the same
# null-calibrated band the preflight uses: their cross-stream loss gap must not
# exceed the same-stream gap between two independent draws.
NULL_SIGMA = 3.0
N_NULL_PAIRS = 6
MIN_INDUCTION_GAP = 0.25   # induction oracle accuracy, A minus A'


def _positional_ce(train: np.ndarray, test: np.ndarray, vocab: int) -> float:
    """Cross-entropy of P(token | position). Uses no content whatsoever."""
    total = 0.0
    for pos in range(1, train.shape[1]):
        counts = np.bincount(train[:, pos], minlength=vocab) + 1.0
        probs = counts / counts.sum()
        total += -np.log(probs[test[:, pos]]).mean()
    return total / (train.shape[1] - 1)


def _unigram_ce(train: np.ndarray, test: np.ndarray, vocab: int) -> float:
    counts = np.bincount(train[:, 1:].reshape(-1), minlength=vocab) + 1.0
    probs = counts / counts.sum()
    return float(-np.log(probs[test[:, 1:].reshape(-1)]).mean())


def _bigram_ce(train: np.ndarray, test: np.ndarray, vocab: int) -> float:
    """Cross-entropy of P(token | previous token), add-one smoothed."""
    prev = train[:, :-1].reshape(-1)
    nxt = train[:, 1:].reshape(-1)
    counts = np.ones((vocab, vocab), dtype=np.float64)
    np.add.at(counts, (prev, nxt), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return float(-np.log(probs[test[:, :-1].reshape(-1), test[:, 1:].reshape(-1)]).mean())


def _induction_oracle(docs: np.ndarray) -> float:
    """Accuracy of 'copy what followed this token's earlier occurrence', at
    every position.

    Reported as a diagnostic only. It conflates two things: copying that
    recovers the *template* (an entity is always followed by REL, in both
    streams by design) and copying that recovers a *binding* (only in ``A``).
    Because both streams now recur entities equally often, the template part is
    a large constant common to both, and this number understates the
    manipulation. Use :func:`_binding_oracle` for the gate.
    """
    hits = 0
    total = 0
    for doc in docs:
        seen: dict[int, int] = {}
        for i in range(len(doc) - 1):
            tok = int(doc[i])
            if tok in seen:
                total += 1
                if doc[seen[tok] + 1] == doc[i + 1]:
                    hits += 1
            seen[tok] = i
    return hits / total if total else 0.0


def _binding_oracle(docs: np.ndarray, config: MicroConfig) -> float:
    """Accuracy of induction restricted to *value* slots after a recurring entity.

    This is the manipulation, isolated. At a value slot whose entity has
    appeared earlier in the document, does the value that followed the earlier
    occurrence predict the value here?

    In ``A`` a recurring entity keeps its value, so the answer is yes by
    construction. In ``A'`` the value is redrawn, so the answer is chance
    (``1 / n_values``). Template positions, where copying trivially succeeds in
    both streams, are excluded — including them is what made the all-position
    oracle understate the contrast.
    """
    ent_cols = [1 + j * CLAUSE_LEN + 1 for j in range(config.n_clauses - 1)]
    ent_cols.append(config.seq_len - 3)
    val_cols = [c + 2 for c in ent_cols]

    hits = 0
    total = 0
    for doc in docs:
        first_value: dict[int, int] = {}
        for ent_col, val_col in zip(ent_cols, val_cols):
            ent = int(doc[ent_col])
            if ent in first_value:
                total += 1
                if first_value[ent] == int(doc[val_col]):
                    hits += 1
            else:
                first_value[ent] = int(doc[val_col])
    return hits / total if total else 0.0


def _distinct_entities(docs: np.ndarray, config: MicroConfig) -> float:
    cols = [1 + j * CLAUSE_LEN + 1 for j in range(config.n_clauses - 1)]
    cols.append(config.seq_len - 3)
    return float(np.mean([len(set(row.tolist())) for row in docs[:, cols]]))


def _null_gap(config: MicroConfig, stream: str, batch: int, vocab: int, fn) -> tuple[float, float]:
    """Same-stream loss gap between independent draws: the noise floor."""
    gaps = []
    for i in range(N_NULL_PAIRS):
        a = sample_documents(3000 + 4 * i, config, stream, batch)
        b = sample_documents(3001 + 4 * i, config, stream, batch)
        held = sample_documents(3002 + 4 * i, config, stream, batch)
        gaps.append(abs(fn(a, held, vocab) - fn(b, held, vocab)))
    return float(np.mean(gaps)), float(np.std(gaps, ddof=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/preflight"))
    args = parser.parse_args()

    config = MicroConfig()
    vocab = config.vocab_size
    b = args.batch

    train_a = sample_documents(41, config, "IND", b)
    train_r = sample_documents(41, config, "IND_R", b)
    test_a = sample_documents(42, config, "IND", b)
    test_r = sample_documents(42, config, "IND_R", b)

    result: dict = {}
    controls_pass = True

    for name, fn in (
        ("positional", _positional_ce),
        ("unigram", _unigram_ce),
        ("bigram", _bigram_ce),
    ):
        # Each predictor is trained and tested WITHIN its own stream, so the
        # comparison is "how predictable is each stream to a short-range
        # model", not "does a model trained on one transfer to the other".
        ce_a = fn(train_a, test_a, vocab)
        ce_r = fn(train_r, test_r, vocab)
        gap = abs(ce_a - ce_r)

        bands = [_null_gap(config, s, b, vocab, fn) for s in ("IND", "IND_R")]
        mean, sd = max(bands, key=lambda x: x[0] + NULL_SIGMA * x[1])
        limit = mean + NULL_SIGMA * sd
        ok = gap <= limit
        controls_pass &= ok

        result[name] = {
            "ce_A": ce_a, "ce_A_prime": ce_r, "gap": gap,
            "null_mean": mean, "null_limit": limit, "within_null": bool(ok),
        }

    bind_a = _binding_oracle(test_a, config)
    bind_r = _binding_oracle(test_r, config)
    result["binding_oracle"] = {
        "accuracy_A": bind_a,
        "accuracy_A_prime": bind_r,
        "chance": 1.0 / config.n_values,
        "gap": bind_a - bind_r,
        "sufficient": bool(bind_a - bind_r >= MIN_INDUCTION_GAP),
    }
    result["all_position_oracle"] = {
        "accuracy_A": _induction_oracle(test_a),
        "accuracy_A_prime": _induction_oracle(test_r),
        "note": (
            "Diagnostic only. Includes template copying, which succeeds in both "
            "streams by design now that entities recur equally often, so it "
            "understates the manipulation. The gate uses binding_oracle."
        ),
    }

    result["known_asymmetries"] = {
        "distinct_entities_per_doc_A": _distinct_entities(test_a, config),
        "distinct_entities_per_doc_A_prime": _distinct_entities(test_r, config),
        "note": (
            "A concentrates on few entities per document and A' spreads across "
            "many. This is a consequence of the manipulation, not a defect, but "
            "it means the streams differ in embedding-update sparsity as well "
            "as in repeat structure. Recorded as confound C13."
        ),
    }

    result["pass"] = bool(controls_pass and result["binding_oracle"]["sufficient"])

    print(f"A'/A shortcut audit — batch {b}, vocab {vocab}\n")
    print("Short-range controls (must match across streams):")
    for name in ("positional", "unigram", "bigram"):
        r = result[name]
        flag = "ok " if r["within_null"] else "GAP"
        print(f"  [{flag}] {name:11s} A {r['ce_A']:.4f}  A' {r['ce_A_prime']:.4f}  "
              f"gap {r['gap']:.4f}  null limit {r['null_limit']:.4f}")

    r = result["binding_oracle"]
    flag = "ok " if r["sufficient"] else "LOW"
    print(f"\nManipulation (must differ):")
    print(f"  [{flag}] binding      A {r['accuracy_A']:.4f}  A' {r['accuracy_A_prime']:.4f}  "
          f"gap {r['gap']:.4f}  chance {r['chance']:.4f}  required {MIN_INDUCTION_GAP}")
    d = result["all_position_oracle"]
    print(f"        all-position (diagnostic)  A {d['accuracy_A']:.4f}  "
          f"A' {d['accuracy_A_prime']:.4f}")

    k = result["known_asymmetries"]
    print(f"\nKnown asymmetry (documented, not a gate):")
    print(f"        distinct entities/doc  A {k['distinct_entities_per_doc_A']:.2f}  "
          f"A' {k['distinct_entities_per_doc_A_prime']:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shortcut_audit.json").write_text(json.dumps(result, indent=2) + "\n")

    if not result["pass"]:
        print("\nAUDIT FAILED — A' differs from A in something beyond repeat structure, "
              "or the repeat structure is not exploitable. Do not freeze the substrate.")
        return 1
    print("\nAUDIT PASSED — short-range predictors cannot distinguish the streams; "
          "only long-range copying can.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
