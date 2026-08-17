"""Protocol A: do behaviour-matched states have divergent futures?

Implements `docs/experiments/downstream_protocols.md` Protocol A
(sha256 8fc78c4087e2f87b), frozen before any continuation was scored.

Primary design is **within-arm** matching on the complete observable vector,
which has no arm confound. The cross-arm variant is secondary and weaker,
because `A` and `A'` differ visibly on the capability being predicted.

Uses existing V(W, BIND) continuations, so the primary costs no new compute.
"""
from __future__ import annotations
import argparse, glob, itertools, json, os, random, statistics as st
from pathlib import Path
import numpy as np

def behaviour(u: dict) -> dict:
    return {"zsB_acc": u["zero_shot_BIND"]["accuracy"], "zsB_loss": u["zero_shot_BIND"]["loss"],
            "zsC_acc": u["zero_shot_FACT"]["accuracy"], "zsC_loss": u["zero_shot_FACT"]["loss"]}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=Path, nargs="+", required=True)
    ap.add_argument("--vsd", type=Path, required=True)
    ap.add_argument("--corpus", type=str, default="BIND")
    ap.add_argument("--quantile", type=float, default=0.10)
    args = ap.parse_args()

    states = {}
    for root in args.states:
        for f in sorted((root / "units").glob("*.json")):
            u = json.loads(f.read_text())
            states[f"{u['arm']}__seed{u['seed']:03d}"] = {"arm": u["arm"], "beh": behaviour(u)}
    fut = {}
    for f in sorted((args.vsd / "units").glob(f"*__{args.corpus}.json")):
        d = json.loads(f.read_text())
        acc = [x["BIND"]["accuracy"] for x in d["trace"]]
        fut[d["state_label"]] = {"final": acc[-1], "t0": acc[0],
                                 "rate_only": sum(acc)/len(acc) - acc[0]}
    usable = sorted(set(states) & set(fut))
    print(f"=== Protocol A: behaviour-matched hidden futures (corpus {args.corpus}) ===")
    print(f"  {len(usable)} states have both behaviour and a measured future\n")
    if len(usable) < 8:
        print("  too few states; not testable")
        return 0

    keys = sorted(states[usable[0]]["beh"])
    M = np.array([[states[s]["beh"][k] for k in keys] for s in usable], float)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
    dist = {(a, b): float(np.max(np.abs(Z[i] - Z[j])))
            for i, a in enumerate(usable) for j, b in enumerate(usable) if i < j}

    same_arm = [d for (a, b), d in dist.items() if states[a]["arm"] == states[b]["arm"]]
    eps = float(np.quantile(same_arm, args.quantile)) if same_arm else 0.0
    print(f"  epsilon = {eps:.4f} ({args.quantile:.0%} quantile of same-arm distances)")

    for label, pred in (("WITHIN-ARM (primary)", lambda a, b: states[a]["arm"] == states[b]["arm"]),
                        ("CROSS-ARM (secondary, weaker)", lambda a, b: states[a]["arm"] != states[b]["arm"])):
        pairs = [(a, b) for (a, b), d in dist.items() if pred(a, b) and d <= eps]
        allp = [(a, b) for (a, b) in dist if pred(a, b)]
        if len(pairs) < 3:
            print(f"\n  {label}: only {len(pairs)} matched pairs at frozen epsilon "
                  f"— not testable. DO NOT loosen epsilon.")
            continue
        for metric in ("final", "rate_only"):
            div_m = [abs(fut[a][metric] - fut[b][metric]) for a, b in pairs]
            div_r = [abs(fut[a][metric] - fut[b][metric]) for a, b in allp]
            rng = random.Random(0); null = []
            for _ in range(5000):
                samp = rng.sample(div_r, min(len(pairs), len(div_r)))
                null.append(st.mean(samp))
            obs = st.mean(div_m)
            pval = sum(1 for x in null if x >= obs) / len(null)
            print(f"\n  {label} [{metric}]  n_matched={len(pairs)}")
            print(f"    matched-pair divergence {obs:.4f}   "
                  f"all-pair divergence {st.mean(div_r):.4f}   p={pval:.4f}")
            if pval < 0.05 and obs > st.mean(div_r):
                print("    --> matched pairs diverge MORE than typical: behaviour")
                print("        underdetermines future learnability")
            else:
                print("    --> no excess divergence: behaviour is sufficient at this")
                print("        resolution. A real and reportable negative.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
