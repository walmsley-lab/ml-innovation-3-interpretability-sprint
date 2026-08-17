"""Matched-State Counterfactual Fork — the frozen analysis.

Implements `docs/experiments/fork_protocol.md` (facef66229928f4a) and reports in
the order it fixes: aggregate interaction, sign fraction, reversal count,
matching-distance dependence, outlier sensitivity. Illustrative pairs last.

Estimand, per matched pair (S1,S2) and corpora (D1,D2):

    delta = [V(S1,D1) - V(S1,D2)] - [V(S2,D1) - V(S2,D2)]

A genuine ordering reversal is V(S1,D1) > V(S1,D2) while V(S2,D2) > V(S2,D1):
the two behaviourally matched states prefer *different* futures.
"""
from __future__ import annotations
import glob, json, random, statistics as st
from pathlib import Path
import numpy as np

META = json.load(open("artifacts/fork_frozen.json"))
D1, D2 = META["corpora"]
BOOT = 10000


def load():
    """AULC of the trained target, per (state, corpus). AULC not final:
    1 in 19 units shows a late instability dip that moves `final` by ~0.9."""
    v, unstable = {}, set()
    for f in glob.glob("artifacts/fork/units/*.json"):
        d = json.loads(Path(f).read_text())
        tag = d["tag"].replace("FORK_", "")
        key = "BINDT" if tag == "BINDT" else "BIND"
        tr = sorted(d["trace"], key=lambda x: x["step"])
        acc = [x[key]["accuracy"] for x in tr]
        v[(d["state_label"], tag)] = sum(acc) / len(acc)
        if max(acc) - acc[-1] > 0.3:
            unstable.add(d["state_label"])
    return v, unstable


def deltas(v, pairs):
    out = []
    for a, b in pairs:
        if not all((s, c) in v for s in (a, b) for c in (D1, D2)):
            continue
        pa = v[(a, D1)] - v[(a, D2)]
        pb = v[(b, D1)] - v[(b, D2)]
        out.append({"pair": (a, b), "pref_a": pa, "pref_b": pb, "delta": pa - pb,
                    "reversal": pa * pb < 0})
    return out


def boot_ci(x, n=BOOT):
    rng = random.Random(0)
    m = sorted(st.mean([x[rng.randrange(len(x))] for _ in x]) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def report(rows, label):
    if len(rows) < 3:
        print(f"  {label}: only {len(rows)} complete pairs"); return
    d = [r["delta"] for r in rows]
    lo, hi = boot_ci(d)
    same = sum(1 for r in rows if np.sign(r["delta"]) == np.sign(st.mean(d)))
    rev = sum(1 for r in rows if r["reversal"])
    print(f"  {label}: n={len(rows)}")
    print(f"    mean interaction  {st.mean(d):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   {'EXCLUDES zero' if lo > 0 or hi < 0 else 'includes zero'}")
    print(f"    |mean| vs sd      {abs(st.mean(d)):.4f} vs {st.stdev(d):.4f}")
    print(f"    pairs sharing the aggregate sign: {same}/{len(rows)}")
    print(f"    genuine ordering reversals:       {rev}/{len(rows)}")


def main():
    v, unstable = load()
    pairs = [tuple(p) for p in META["pairs"]]
    rows = deltas(v, pairs)
    print(f"FORK — {D1} vs {D2}, {len(rows)} of {len(pairs)} frozen pairs complete\n")
    print("=== 1. AGGREGATE STATE x DATA INTERACTION ===")
    report(rows, "all pairs")
    print("\n=== 2. OUTLIER SENSITIVITY (drop pairs touching an unstable unit) ===")
    report([r for r in rows if not (set(r["pair"]) & unstable)], "stable pairs")
    print("\n=== 3. MATCHING-DISTANCE DEPENDENCE ===")
    if len(rows) > 2:
        dist = [META["matching_distance"][f"{a}|{b}"] for a, b in (r["pair"] for r in rows)]
        mag = [abs(r["delta"]) for r in rows]
        print(f"  corr(matching distance, |interaction|) = {np.corrcoef(dist, mag)[0,1]:+.3f}")
    print("\n=== 4. ILLUSTRATIVE PAIR (chosen last, for legibility only) ===")
    revs = [r for r in rows if r["reversal"]]
    if revs:
        r = max(revs, key=lambda r: abs(r["delta"]))
        a, b = r["pair"]
        print(f"  {a}:  {D1} {v[(a,D1)]:.3f}  {D2} {v[(a,D2)]:.3f}  -> prefers "
              f"{D1 if r['pref_a']>0 else D2}")
        print(f"  {b}:  {D1} {v[(b,D1)]:.3f}  {D2} {v[(b,D2)]:.3f}  -> prefers "
              f"{D1 if r['pref_b']>0 else D2}")
    else:
        print("  no genuine ordering reversal among complete pairs")
    json.dump({"rows": [{**r, "pair": list(r["pair"])} for r in rows]},
              open("artifacts/fork_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
