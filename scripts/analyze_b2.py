"""B2 factorial analysis: readiness versus computation-specific transfer.

Written before the units landed, so the reading is fixed in advance.

`B2` (`BINDT`) is retrieval composed with a fixed **derangement**. A model
carrying the copy circuit that `A` induces will retrieve the bound value and
emit it unchanged, which is wrong every time — so zero-shot competence is
blocked by construction and cannot be the source of any advantage.

The two summaries are therefore reported **separately and never summed**:

* ``t=0`` — head start carried in. On `B2` this should sit at or *below*
  chance for the `A` arm. Below-chance is not noise: it is the signature of a
  transferred retrieval mechanism feeding the wrong output map, and it is only
  measurable because the permutation has no fixed points.
* ``rate_only = AULC - t0`` — acquisition speed after the head start is removed.
  **This is the readiness estimand.** Any `A` advantage on `B2` has to appear
  here.

Crossed with surface overlap (shared vs disjoint entity pools), the design
separates three explanations. The reading table is in
`docs/experiments/campaign_2026_08_17.md` §4b.

    PYTHONPATH=src python scripts/analyze_b2.py --roots artifacts/b2_factorial_h2 ...
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

CHANCE = 1.0 / 64


def summarize(values):
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f} (n=1)"
    return f"{st.mean(values):.4f}+-{st.stdev(values):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for p in sorted((root / "units").glob("*.json")):
            d = json.loads(p.read_text())
            p1 = [x for x in d["trace"] if x["phase"] == 1]
            if len(p1) < 3:
                continue
            key = "BINDT" if "BINDT" in str(d.get("target_family", "")) else "BIND"
            acc = [x[key]["accuracy"] for x in p1]
            rows.append({
                "arm": d["arm"], "seed": d["seed"], "root": root.name,
                "target": d.get("target_family", "?"),
                "t0": acc[0], "aulc": sum(acc) / len(acc), "final": acc[-1],
                "rate_only": sum(acc) / len(acc) - acc[0],
            })

    if not rows:
        print("no B2 units yet")
        return 0

    print(f"B2 FACTORIAL — target BINDT (retrieval o derangement), chance = {CHANCE:.4f}")
    print("Zero-shot is blocked by construction; readiness must appear in rate_only.\n")

    by_root = {}
    for r in rows:
        by_root.setdefault(r["root"], {}).setdefault(r["arm"], []).append(r)

    for root in sorted(by_root):
        surface = ("disjoint surface (source h1 -> target h2)" if "h1" in root
                   else "shared surface (source h2 -> target h2)")
        print(f"=== {root} — {surface} ===")
        print(f"  {'arm':12s} {'n':>3s} {'t=0':>20s} {'rate_only':>20s} {'final':>20s}")
        arms = by_root[root]
        for arm in sorted(arms):
            v = arms[arm]
            print(f"  {arm:12s} {len(v):3d} "
                  f"{summarize([x['t0'] for x in v]):>20s} "
                  f"{summarize([x['rate_only'] for x in v]):>20s} "
                  f"{summarize([x['final'] for x in v]):>20s}")

        a_key = next((k for k in arms if k.startswith("A_") and "p" not in k.split("_")[1][:1]), None)
        a_key = next((k for k in arms if k.startswith("A_h")), a_key)
        p_key = next((k for k in arms if k.startswith("Ap")), None)
        if a_key and p_key and len(arms[a_key]) > 1 and len(arms[p_key]) > 1:
            for label in ("t0", "rate_only", "final"):
                a = [x[label] for x in arms[a_key]]
                p = [x[label] for x in arms[p_key]]
                d = st.mean(a) - st.mean(p)
                sd = st.pstdev(a + p) or 1e-9
                print(f"    {label:10s} A - A' = {d:+.4f}   pooled sd {sd:.4f}"
                      f"   effect/noise {d / sd:+.2f}")
            a_t0 = st.mean([x["t0"] for x in arms[a_key]])
            verdict = ("AT/BELOW chance — zero-shot correctly blocked"
                       if a_t0 <= CHANCE * 1.5 else
                       "ABOVE chance — zero-shot NOT fully blocked, interpret with care")
            print(f"    A arm t=0 = {a_t0:.4f} vs chance {CHANCE:.4f}: {verdict}")
        print()

    print("Reading: an A advantage in rate_only with t=0 at or below chance is")
    print("readiness. An advantage only where surface AND computation overlap is")
    print("ordinary task transfer. A split across the two surface conditions maps")
    print("the boundary and is the most informative outcome.")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
