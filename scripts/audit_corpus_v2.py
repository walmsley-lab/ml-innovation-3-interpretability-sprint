"""Outcome-blind audit of the WikiText k=8 family proposal.

Run **before** any transfer outcome exists on this substrate, so that every
design constant it fixes — phase budget, vocabulary size, pool sizes — is
chosen from properties of the corpus rather than from results. Four
dimensions:

* **nuisance** — family size imbalance and document-length distribution, the
  quantities exposure matching has to absorb;
* **content** — top terms per family, descriptive only, so the families can
  be described honestly rather than assumed to be topics;
* **vocabulary** — train-only fit and per-family UNK rate, since a family
  that is largely UNK has an easy next-token task for reasons unrelated to
  development;
* **runtime** — chunk supply, the largest single-pass phase budget every
  family can serve, and the resulting trajectory arithmetic.

Nothing here proposes families or touches the frozen 20NG pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dsi.corpus import Corpus, deduplicate, split_documents
from dsi.natural import chunk_documents, coverage, fit_vocabulary, whitespace_tokens

OUT = Path("artifacts/corpus_v2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--source", type=Path, default=Path("data/wikitext103_train.parquet"))
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_corpus_intake import load_wikitext

    assignments = json.loads((OUT / f"assignments_k{args.k}.json").read_text())
    corpus = load_wikitext(args.source)
    dedup, _ = deduplicate(corpus)
    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))
    families = sorted({int(v) for v in assignments["train"].values()})

    vocabulary = fit_vocabulary(splits["train"], max_size=args.vocab_size,
                                min_count=args.min_count, fitted_on="train")
    vocabulary.to_json(OUT / "vocab.json")
    print(f"vocabulary {vocabulary.size} types, train-only, min_count={args.min_count}\n")

    docs = {}
    for split in ("train", "val"):
        for family in families:
            docs[(split, family)] = [d for d in splits[split].documents
                                     if assignments[split].get(d.doc_id) == family]

    print(f"{'fam':>4s} {'train docs':>10s} {'audit tok':>11s} {'lm tok':>10s} "
          f"{'chunks':>7s} {'val chunks':>10s} {'unk':>6s} {'med len':>8s} {'p90 len':>8s}")
    rows = {}
    for family in families:
        train_docs = docs[("train", family)]
        lengths = np.array([len(whitespace_tokens(d.text)) for d in train_docs])
        cov = coverage(vocabulary, Corpus("f", "v", tuple(train_docs)))
        lm_tokens = cov["tokens"] + len(train_docs)
        val_chunks = len(chunk_documents(docs[("val", family)], vocabulary,
                                         args.seq_len, seed=0))
        rows[family] = {
            "train_documents": len(train_docs),
            "audit_tokens": int(lengths.sum()),
            "lm_tokens": int(lm_tokens),
            "train_chunks": int(lm_tokens // args.seq_len),
            "val_chunks": int(val_chunks),
            "unk_rate": cov["unk_rate"],
            "median_doc_tokens": float(np.median(lengths)),
            "p90_doc_tokens": float(np.percentile(lengths, 90)),
        }
        r = rows[family]
        print(f"{family:>4d} {r['train_documents']:>10d} {r['audit_tokens']:>11d} "
              f"{r['lm_tokens']:>10d} {r['train_chunks']:>7d} {r['val_chunks']:>10d} "
              f"{r['unk_rate']:>6.3f} {r['median_doc_tokens']:>8.0f} "
              f"{r['p90_doc_tokens']:>8.0f}")

    chunks = {f: rows[f]["train_chunks"] for f in families}
    binding = min(chunks, key=chunks.get)
    tokens = {f: rows[f]["audit_tokens"] for f in families}
    unk = [rows[f]["unk_rate"] for f in families]

    # A common control N_j draws from the other k-1 families, so each
    # contributes budget/(k-1); the binding constraint is the source phase.
    budget_chunks = (chunks[binding] // 128) * 128
    n_pairs = len(families) * (len(families) - 1)
    summary = {
        "k": args.k, "families": families, "per_family": rows,
        "size_imbalance_ratio": max(tokens.values()) / min(tokens.values()),
        "unk_rate_range": [min(unk), max(unk)],
        "binding_family": binding, "binding_chunks": chunks[binding],
        "single_pass_phase_chunks": int(budget_chunks),
        "single_pass_phase_lm_tokens": int(budget_chunks * args.seq_len),
        "n_directed_pairs": n_pairs,
        "seq_len": args.seq_len, "vocab_size": vocabulary.size,
    }
    print(f"\nnuisance   size imbalance {summary['size_imbalance_ratio']:.1f}x "
          f"(largest/smallest audit tokens); UNK rate range "
          f"{min(unk):.3f}-{max(unk):.3f}")
    print(f"runtime    binding family {binding} at {chunks[binding]} train chunks")
    print(f"           single-pass phase budget {budget_chunks} chunks "
          f"= {budget_chunks * args.seq_len:,} LM tokens")
    print(f"           {n_pairs} directed pairs; common-control design needs "
          f"{n_pairs} treatment + {len(families)} control arms per seed "
          f"= {n_pairs + len(families)} trajectories")

    (OUT / f"audit_k{args.k}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {OUT / f'audit_k{args.k}.json'}")


if __name__ == "__main__":
    main()
