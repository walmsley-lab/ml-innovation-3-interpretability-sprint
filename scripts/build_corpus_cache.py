"""Pre-materialize per-family document slices so workers skip corpus parsing.

Every worker was independently reading a 150MB parquet, reassembling 14,900
articles, content-hashing 14,491 documents and re-deriving the split — the
same deterministic work, up to eighteen times at once. The result is a pure
function of frozen inputs, so it is computed once and cached.

This changes no science: the cache is written by the same code path the
runner would have executed, and the runner verifies the frozen split hash
before using it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CORPUS = Path("artifacts/corpus_v2")
CACHE = CORPUS / "cache"


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_corpus_intake import load_wikitext

    from dsi.corpus import deduplicate, split_documents

    pools = json.loads((CORPUS / "frozen_pools.json").read_text())
    assignments = json.loads((CORPUS / "assignments_k8.json").read_text())
    families = tuple(pools["families_used"])

    dedup, _ = deduplicate(load_wikitext(Path("data/wikitext103_train.parquet")))
    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))

    CACHE.mkdir(parents=True, exist_ok=True)
    index = {"families": list(families), "splits": {}}
    for split in ("train", "val"):
        for family in families:
            docs = [d for d in splits[split].documents
                    if assignments[split].get(d.doc_id) == family]
            path = CACHE / f"{split}_f{family}.jsonl"
            with open(path, "w") as handle:
                for d in docs:
                    handle.write(json.dumps({"id": d.doc_id, "text": d.text}) + "\n")
            digest = hashlib.blake2b(
                "".join(d.doc_id for d in docs).encode(), digest_size=8).hexdigest()
            index["splits"][f"{split}_f{family}"] = {
                "documents": len(docs), "doc_id_hash": digest}
            print(f"{split} f{family}: {len(docs)} docs  {digest}")
    (CACHE / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {CACHE}/index.json")


if __name__ == "__main__":
    main()
