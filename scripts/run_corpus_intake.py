"""Intake and support audit for a larger natural corpus.

The four-family 20NG universe cannot host disjoint observed, held-out and
adaptive pools at once — `docs/experiments/natural_corpus.md` 15 records that as a structural
limit. A legitimate Stage-5 attempt needs enough automatically proposed
families that the three roles can be separated, which means roughly 8 usable
families and therefore 56 directed pairs.

This runs the same frozen pipeline order on an arbitrary corpus, and the
order is the whole point: documents are deduplicated, then split, and only
then is the proposer fitted on train alone and frozen. A proposer fitted on
all documents has read the evaluation set.

Support is audited in **support-audit tokens** — plain `str.split()` — the
same unit the 20NG 200,000-token gate was computed in. The gate is not
redefined; it is applied to a new corpus.

Nothing here touches the frozen 20NG pilot.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dsi.corpus import (Corpus, Document, TfidfLsaProposer, audit_proposal,
                        deduplicate, split_documents)
from dsi.natural import whitespace_tokens

ARTICLE = re.compile(r"^ = [^=].* = $")


def load_wikitext(path: Path, min_tokens: int = 200) -> Corpus:
    """Reassemble WikiText-103 lines into articles.

    The distribution is line-level; an article begins at a single-level
    heading (` = Title = `) while deeper headings are sections within one.
    Splitting on the wrong level would turn one article into many short
    fragments and inflate the document count without adding text.
    """
    import polars as pl

    lines = pl.read_parquet(path)["text"].to_list()
    articles, current = [], []
    for line in lines:
        if ARTICLE.match(line.rstrip("\n")):
            if current:
                articles.append("".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        articles.append("".join(current))
    kept = [a for a in articles if len(whitespace_tokens(a)) >= min_tokens]
    print(f"{len(lines)} lines -> {len(articles)} articles -> {len(kept)} "
          f"with >= {min_tokens} tokens")
    return Corpus("wikitext103", "raw-v1", tuple(Document.of(a) for a in kept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/wikitext103_train.parquet"))
    ap.add_argument("--k", type=int, nargs="+", default=[8, 10, 12, 14])
    ap.add_argument("--min-train-tokens", type=int, default=200_000)
    ap.add_argument("--target-families", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("artifacts/corpus_v2"))
    args = ap.parse_args()

    corpus = load_wikitext(args.source)
    dedup, stats = deduplicate(corpus)
    print(f"dedup: {stats['input_documents']} -> {stats['unique_documents']} "
          f"({stats['duplicates_removed']} removed, "
          f"{stats['duplicate_fraction']:.3%})")

    total = sum(len(whitespace_tokens(d.text)) for d in dedup.documents)
    print(f"corpus: {len(dedup)} documents, {total:,} support-audit tokens")

    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))
    print(f"split BEFORE proposal: train {len(splits['train'])} / "
          f"val {len(splits['val'])} / test {len(splits['test'])}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"corpus": "wikitext103-raw-v1", "dedup_stats": stats,
               "total_support_tokens": total, "min_train_tokens": args.min_train_tokens,
               "sweep": {}}

    for k in args.k:
        proposer = TfidfLsaProposer(n_families=k, n_components=64, seed=0).fit(splits["train"])
        train_proposal = proposer.assign(splits["train"], "train")
        val_proposal = proposer.assign(splits["val"], "val")
        audit = audit_proposal(splits["train"], splits["val"], train_proposal,
                               val_proposal, tokenize=whitespace_tokens,
                               min_train_tokens=args.min_train_tokens,
                               min_heldout_documents=3)
        usable = len(audit.usable_families)
        pairs = usable * (usable - 1)
        print(f"\nk={k}: {usable}/{k} usable, {pairs} directed pairs"
              f"{'   <- sufficient' if usable >= args.target_families else ''}")
        print(audit.render())
        summary["sweep"][str(k)] = {
            "usable_families": list(audit.usable_families),
            "unusable_families": list(audit.unusable_families),
            "n_usable": usable, "n_directed_pairs": pairs,
            "per_family": audit.per_family,
        }
        if usable >= args.target_families:
            (args.out / f"assignments_k{k}.json").write_text(json.dumps(
                {"train": train_proposal.assignments, "val": val_proposal.assignments,
                 "proposer": train_proposal.proposer}, indent=2) + "\n")

    (args.out / "intake_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out / 'intake_summary.json'}")


if __name__ == "__main__":
    main()
