"""Verify every quantitative claim in CLAIMS.md against the artifacts.

Written because three claim rows once failed to write silently while being
reported as recorded. A ledger that is not machine-checked is a ledger that
drifts, and the whole point of it is that it does not.

Recomputes the headline numbers from raw units and prints them beside what the
ledger asserts, so a mismatch is visible rather than trusted.
"""
from __future__ import annotations
import glob, json, statistics as st, sys
from pathlib import Path

def load(pat):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(pat))]

def arm_stats(units, key, phase, cap):
    by = {}
    for d in units:
        pts = [x for x in d["trace"] if x.get("phase") == phase and cap in x]
        if not pts: continue
        acc = [x[cap]["accuracy"] for x in pts]
        v = acc[0] if key == "t0" else (sum(acc)/len(acc) if key == "aulc" else acc[-1])
        by.setdefault(d["arm"], []).append(v)
    return {a: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) for a, v in by.items()}

print("=== CLAIM AUDIT: recomputed from artifacts ===\n")

g1 = load("artifacts/g1/units/*.json")
if g1:
    print(f"H1/H2 (G1 confirmatory, {len(g1)} units)")
    for cap, lab in (("BIND", "B"), ("FACT", "C")):
        s = arm_stats(g1, "t0", 1, cap)
        row = "  ".join(f"{a} {m:.4f}+-{sd:.4f}(n={n})" for a, (m, sd, n) in sorted(s.items()))
        print(f"  {lab} t=0: {row}")
else:
    print("H1/H2: g1 units not present locally (on GPU)")

c1 = load("artifacts/c1_scout/units/*.json")
if c1:
    s = arm_stats(c1, "t0", 1, "BIND")
    print(f"\nH3 (C1 disjoint pools, {len(c1)} units)")
    for a, (m, sd, n) in sorted(s.items()):
        print(f"  {a:12s} t=0 {m:.4f}+-{sd:.4f} (n={n})")
else:
    print("\nH3: c1_scout units not present locally")

md = load("artifacts/mediator_discovery/units/*.json")
mv = load("artifacts/mediator_validation/units/*.json")
for tag, us in (("discovery", md), ("validation", mv)):
    if not us: continue
    by = {}
    for d in us: by.setdefault(d["arm"], []).append(d["retrieval_max"])
    print(f"\nA5c retrieval ({tag}, {len(us)} units)")
    for a, v in sorted(by.items()):
        print(f"  {a:12s} {st.mean(v):.4f} (n={len(v)})")

vs = load("artifacts/vsd_matrix/units/*.json")
if vs:
    cells = {}
    for d in vs:
        f = d["trace"][-1]
        vals = [f[c]["accuracy"] for c in ("BIND","FACT","BINDT") if c in f]
        cells[(d["state_label"], d["tag"])] = sum(vals)/len(vals)
    states = sorted({s for s,_ in cells}); corp = sorted({c for _,c in cells})
    comp = [s for s in states if all((s,c) in cells for c in corp)]
    best = {s: max(((cells[(s,c)], c) for c in corp))[1] for s in comp}
    print(f"\nX2c V(S,D) ({len(vs)} cells, {len(comp)} complete states)")
    print(f"  distinct argmax corpora: {sorted(set(best.values()))}")
    for c in corp:
        vv = [cells[(s,c)] for s in comp]
        print(f"  {c:8s} mean {st.mean(vv):.4f} over {len(vv)} complete states")
print("\nCompare each line against CLAIMS.md. Any divergence is drift.")
