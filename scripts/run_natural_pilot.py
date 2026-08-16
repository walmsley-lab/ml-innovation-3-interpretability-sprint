"""Stage-5 natural-corpus ingestion, frozen and re-checkable.

The 20 Newsgroups pilot recorded in ``DESIGN_LAYER2.md`` 14 was run before
this script existed, so its artifacts had no committed producer. This script
is that producer. It reproduces the frozen pipeline exactly and **asserts**
the reproduction rather than overwriting it: the document split, the family
assignment and the 9-observed / 3-held-out pair split are frozen scientific
objects, and a script that silently regenerates them would let a later
re-run quietly move the evaluation set.

Recovered ingestion path, verified byte-for-byte against the frozen
assignment:

    fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
      -- sklearn's own defaults, shuffle=True and random_state=42, which fix
         the document order the split permutation then consumes
      -> Document.of (content-addressed ids)
      -> deduplicate: 18,846 -> 18,287, 559 removed
      -> split_documents(seed=0, fractions=(0.7, 0.15, 0.15))

Two token units are kept distinct and are both reported:

* **support-audit tokens** — plain ``str.split()``. The unit the frozen
  200,000-token source-support gate was computed in. Not redefined here.
* **LM training tokens** — ids from the train-only vocabulary, one per
  whitespace piece plus one ``EOS`` per document. The unit the transfer
  intervention matches exposure in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

from dsi.corpus import Corpus, Document, deduplicate, split_documents
from dsi.natural import coverage, fit_vocabulary, whitespace_tokens

FROZEN = Path("artifacts/natural_pilot")
USABLE_FAMILIES = (1, 2, 3, 4)
SUPPORT_GATE_TOKENS = 200_000


def ingest() -> dict:
    """The recovered ingestion path. Deterministic, no arguments to tune."""
    raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    corpus = Corpus("20newsgroups", "v1", tuple(Document.of(t) for t in raw.data))
    dedup, stats = deduplicate(corpus)
    return {"corpus": dedup, "dedup_stats": stats,
            "splits": split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))}


def verify_against_frozen(ingested: dict, frozen: dict) -> list[str]:
    """Regression assertions. Every failure here invalidates the pilot."""
    checks, splits = [], ingested["splits"]

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            raise AssertionError(f"frozen-provenance check failed: {name} {detail}")

    check("dedup statistics", ingested["dedup_stats"] == frozen["dedup_stats"],
          f"{ingested['dedup_stats']['unique_documents']} unique, "
          f"{ingested['dedup_stats']['duplicates_removed']} removed")

    current = {d.doc_id for d in ingested["corpus"].documents}
    frozen_ids = set(frozen["train"]) | set(frozen["val"]) | set(frozen["test"])
    check("document id set equals frozen id set", current == frozen_ids,
          f"{len(current)} ids")
    check("no content duplicates remain",
          len(current) == len(ingested["corpus"].documents), f"{len(current)} unique")

    seen: dict = {}
    for name in ("train", "val", "test"):
        for doc_id in frozen[name]:
            if doc_id in seen:
                raise AssertionError(f"{doc_id} appears in {seen[doc_id]} and {name}")
            seen[doc_id] = name
    check("every document maps to exactly one frozen split", len(seen) == len(frozen_ids))

    for name in ("train", "val", "test"):
        got = {d.doc_id for d in splits[name].documents}
        check(f"{name} membership preserved exactly", got == set(frozen[name]),
              f"{len(got)} documents")

    # The support audit is re-derived, never recomputed with a new tokenizer.
    text = {d.doc_id: d.text for d in ingested["corpus"].documents}
    for family, row in sorted(frozen["audit"].items()):
        ids = [k for k, v in frozen["train"].items() if v == int(family)]
        tokens = sum(len(whitespace_tokens(text[i])) for i in ids)
        check(f"family {family} support-audit tokens", tokens == row["train_tokens"],
              f"{tokens} whitespace tokens")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=FROZEN)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    frozen = json.loads((args.out / "assignments.json").read_text())
    ingested = ingest()
    for line in verify_against_frozen(ingested, frozen):
        print(line)

    splits = ingested["splits"]
    vocabulary = fit_vocabulary(splits["train"], max_size=args.vocab_size,
                                min_count=args.min_count, fitted_on="train")
    vocabulary.to_json(args.out / "vocab.json")
    print(f"\nvocabulary: {vocabulary.size} types, fitted on train only, "
          f"min_count={args.min_count} -> {args.out / 'vocab.json'}")

    print(f"\n{'family':>7s} {'audit tok':>10s} {'lm tok':>9s} {'chunks':>7s} "
          f"{'unk':>6s} {'val chunks':>11s}  source-capable")
    supply = {}
    for family in USABLE_FAMILIES:
        rows = {}
        for split in ("train", "val"):
            docs = [d for d in splits[split].documents
                    if frozen[split].get(d.doc_id) == family]
            sub = Corpus("f", "v", tuple(docs))
            audit = sum(len(whitespace_tokens(d.text)) for d in docs)
            cov = coverage(vocabulary, sub)
            rows[split] = {"documents": len(docs), "audit_tokens": audit,
                           "lm_tokens": cov["tokens"] + len(docs),  # + one EOS per document
                           "unk_rate": cov["unk_rate"]}
        chunks = rows["train"]["lm_tokens"] // args.seq_len
        supply[family] = {**rows, "train_chunks": chunks,
                          "val_chunks": rows["val"]["lm_tokens"] // args.seq_len}
        print(f"{family:>7d} {rows['train']['audit_tokens']:>10d} "
              f"{rows['train']['lm_tokens']:>9d} {chunks:>7d} "
              f"{rows['train']['unk_rate']:>6.3f} {supply[family]['val_chunks']:>11d}"
              f"  {'yes' if rows['train']['audit_tokens'] >= SUPPORT_GATE_TOKENS else 'NO'}")

    payload = {"seq_len": args.seq_len, "vocab_size": vocabulary.size,
               "support_gate_tokens": SUPPORT_GATE_TOKENS, "supply": supply,
               "note": "support-audit tokens are str.split(); lm tokens are vocabulary "
                       "ids plus one EOS per document. The support gate is not redefined."}
    (args.out / "token_supply.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out / 'token_supply.json'}")


if __name__ == "__main__":
    main()
