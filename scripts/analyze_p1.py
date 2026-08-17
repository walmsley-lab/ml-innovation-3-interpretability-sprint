"""P1 hidden futures — the frozen analysis.

Implements `docs/experiments/p1_analysis_plan.md` (a228f36586623fa4) and reports
in the order it fixes: aggregate, then outlier robustness, then the
matching-distance check. Illustrative pairs are chosen only afterwards.
"""
from __future__ import annotations
import glob, json, random, statistics as st
from pathlib import Path
import numpy as np

META = json.load(open("artifacts/p1_frozen_pairs.json"))
BOOT = 10000
PERM = 10000


def load():
    runs = {}
    for f in glob.glob("artifacts/p1_continuations/units/*__P1.json"):
        d = json.loads(Path(f).read_text())
        tr = sorted(d["trace"], key=lambda x: x["step"])
        acc = [x["BIND"]["accuracy"] for x in tr]
        runs[d["state_label"]] = {
            "final": acc[-1], "t0": acc[0],
            "rate_only": sum(acc) / len(acc) - acc[0],
            "aulc": sum(acc) / len(acc), "max": max(acc),
        }
    return runs


def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(vals, n=BOOT):
    rng = random.Random(0)
    m = [st.mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n)]
    m.sort()
    return m[int(0.025 * n)], m[int(0.975 * n)]


def analyse(runs, matched, arm_of, label, drop_unstable=False):
    """matched: list of (a,b). Null: all same-arm pairs that are NOT matched."""
    states = sorted(runs)
    if drop_unstable:
        bad = {s for s, v in runs.items() if v["max"] - v["final"] > 0.3}
        states = [s for s in states if s not in bad]
        matched = [(a, b) for a, b in matched if a in states and b in states]
    sset = set(states)
    mset = {tuple(sorted(p)) for p in matched}
    null = [(a, b) for i, a in enumerate(states) for b in states[i+1:]
            if arm_of(a) == arm_of(b) and tuple(sorted((a, b))) not in mset]
    out = {}
    for metric in ("final", "t0", "rate_only"):
        dm = [abs(runs[a][metric] - runs[b][metric]) for a, b in matched]
        dn = [abs(runs[a][metric] - runs[b][metric]) for a, b in null]
        if len(dm) < 3 or len(dn) < 3:
            continue
        diff = st.mean(dm) - st.mean(dn)
        pool = dm + dn
        rng = random.Random(0)
        cnt = 0
        for _ in range(PERM):
            rng.shuffle(pool)
            if abs(st.mean(pool[:len(dm)]) - st.mean(pool[len(dm):])) >= abs(diff):
                cnt += 1
        lo, hi = boot_ci(dm)
        out[metric] = dict(n_matched=len(dm), n_null=len(dn),
                           mean_matched=st.mean(dm), mean_null=st.mean(dn),
                           diff=diff, p=cnt / PERM,
                           cliffs=cliffs_delta(dm, dn), ci=(lo, hi))
    return out


def main():
    runs = load()
    order = [tuple(p) for p in META["run_order"]]
    matched = [(a, b) for a, b in order if a in runs and b in runs]
    arm_of = lambda s: s.split("__")[0]

    print(f"P1 HIDDEN FUTURES — {len(matched)} of {META['n_pairs']} frozen pairs, "
          f"{len(runs)} states\n")
    print("=== 1. AGGREGATE (matched pairs vs within-arm unmatched null) ===")
    res = analyse(runs, matched, arm_of, "all")
    for m, r in res.items():
        print(f"  {m:10s} matched {r['mean_matched']:.4f} (n={r['n_matched']})   "
              f"null {r['mean_null']:.4f} (n={r['n_null']})")
        print(f"             diff {r['diff']:+.4f}   perm p={r['p']:.4f}   "
              f"Cliff's d={r['cliffs']:+.3f}   matched 95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")

    print("\n=== 2. OUTLIER ROBUSTNESS (drop units >0.3 below own max) ===")
    res2 = analyse(runs, matched, arm_of, "robust", drop_unstable=True)
    for m, r in res2.items():
        print(f"  {m:10s} matched {r['mean_matched']:.4f} (n={r['n_matched']})   "
              f"null {r['mean_null']:.4f}   diff {r['diff']:+.4f}   perm p={r['p']:.4f}")

    print("\n=== 3. MATCHING-DISTANCE CHECK (does divergence track residual mismatch?) ===")
    for metric in ("final", "t0", "rate_only"):
        d = [META["distances"][f"{a}|{b}"] for a, b in matched]
        v = [abs(runs[a][metric] - runs[b][metric]) for a, b in matched]
        r = np.corrcoef(d, v)[0, 1]
        print(f"  {metric:10s} corr(matching distance, divergence) = {r:+.3f}")

    json.dump({"aggregate": res, "robust": res2}, open("artifacts/p1_analysis.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
