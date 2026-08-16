"""Freeze the WikiText k=8 pair partition before any transfer outcome exists.

**Unordered pairs are the unit of partition, not directed pairs.** Holding out
`i->j` while `j->i` sits in the development set leaks the pair's identity:
the two share families, features, and the same unordered-pair-level nuisance,
so a model fitted on one has effectively seen the other. Both directions of an
unordered pair therefore move together into exactly one pool.

Three disjoint pools, which the 20NG universe was structurally too small to
provide at once:

* **development** — fitting and model selection, LOPO inside this pool only;
* **confirmatory** — a batch held-out set, never fitted on, scored once;
* **adaptive** — untouched candidates for model-selected execution.

The partition is derived from a seeded permutation and hashed, so the pools
cannot be quietly reshuffled after outcomes appear.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/corpus_v2")
FAMILIES = tuple(range(8))
N_DEV, N_CONF, N_ADAPT = 18, 6, 4       # unordered pairs; 28 total
SEED = 20260816


def main() -> None:
    unordered = [(a, b) for i, a in enumerate(FAMILIES) for b in FAMILIES[i + 1:]]
    assert len(unordered) == 28, len(unordered)

    order = np.random.default_rng(SEED).permutation(len(unordered))
    pools = {}
    cut = [(0, N_DEV, "development"), (N_DEV, N_DEV + N_CONF, "confirmatory"),
           (N_DEV + N_CONF, N_DEV + N_CONF + N_ADAPT, "adaptive")]
    for lo, hi, name in cut:
        pools[name] = [unordered[i] for i in order[lo:hi]]

    directed = {name: [[a, b] for pair in group for a, b in (pair, pair[::-1])]
                for name, group in pools.items()}
    for name, group in directed.items():
        print(f"{name:>13s}: {len(pools[name]):>2d} unordered -> "
              f"{len(group):>2d} directed")

    overlap = set(map(tuple, directed["development"])) & set(map(tuple, directed["confirmatory"]))
    overlap |= set(map(tuple, directed["development"])) & set(map(tuple, directed["adaptive"]))
    overlap |= set(map(tuple, directed["confirmatory"])) & set(map(tuple, directed["adaptive"]))
    if overlap:
        raise AssertionError(f"pools overlap: {overlap}")
    total = sum(len(g) for g in directed.values())
    if total != 56:
        raise AssertionError(f"{total} directed pairs, expected 56")

    payload = {
        "corpus": "wikitext103-raw-v1", "k": 8, "seed": SEED,
        "partition_unit": "unordered pair; both directions move together so a "
                          "reverse-direction observation cannot leak pair identity",
        "unordered": {k: [list(p) for p in v] for k, v in pools.items()},
        "directed": directed,
        "counts": {k: len(v) for k, v in directed.items()},
        "frozen_before_any_wikitext_transfer": True,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()

    path = OUT / "frozen_pools.json"
    if path.exists():
        raise AssertionError(f"{path} exists; a frozen partition is never rewritten")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsha256 {payload['sha256']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
