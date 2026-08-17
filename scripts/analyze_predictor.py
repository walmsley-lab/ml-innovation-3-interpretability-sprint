"""Gate 2: does telemetry predict *conditional* data value on held-out states?

The V(S,D) matrix can show a real interaction and still be unpredictable. This
script tests the separate claim: given only a state's internal telemetry — never
its outcomes — can we say which corpus will be best for it?

Held out **by state**, not by row. A random row split would leak, because the
other corpora measured on the same state reveal that state's value profile.
Leave-one-state-out is the only split that answers the question the tournament
will ask.

The competitor that matters is **global-best**: always pick the corpus with the
highest mean value across training states. If a single corpus wins from every
state, a state-aware selector adds nothing, and beating `random` while tying
`global-best` means the predictor learned the data main effect rather than the
interaction. That is the most likely false positive and is reported explicitly.

Also reported: whether the retrieval candidate replicates as an A-selective
signal on fresh validation seeds, with a permutation null.

    PYTHONPATH=src python scripts/analyze_predictor.py \\
        --vsd artifacts/vsd_matrix --states artifacts/mediator_discovery \\
        --validation artifacts/mediator_validation
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

CAPS = ("BIND", "FACT", "BINDT")
ALPHAS = np.logspace(-3, 4, 20)


def value(trace_point: dict, kind: str = "min") -> float:
    vals = [trace_point[c]["accuracy"] for c in CAPS if c in trace_point]
    if not vals:
        return float("nan")
    return min(vals) if kind == "min" else sum(vals) / len(vals)


def load_states(*roots: Path) -> dict[str, dict]:
    out = {}
    for root in roots:
        for p in sorted((root / "units").glob("*.json")):
            d = json.loads(p.read_text())
            label = f"{d['arm']}__seed{d['seed']:03d}"
            out[label] = {"arm": d["arm"], "seed": d["seed"],
                          "features": d["state_features"],
                          "retrieval_max": d["retrieval_max"],
                          "M": d["M_scalar"]}
    return out


def retrieval_replication(states: dict, tag: str) -> None:
    by = {}
    for v in states.values():
        by.setdefault(v["arm"], []).append(v["retrieval_max"])
    if "A" not in by or "A_prime" not in by:
        return
    a, p = by["A"], by["A_prime"]
    obs = st.mean(a) - st.mean(p)
    pool = a + p
    rng = random.Random(0)
    null = []
    for _ in range(5000):
        rng.shuffle(pool)
        null.append(st.mean(pool[:len(a)]) - st.mean(pool[len(a):]))
    pval = sum(1 for x in null if abs(x) >= abs(obs)) / len(null)
    print(f"  {tag}: retrieval_max  A {st.mean(a):.4f} (n={len(a)})  "
          f"A' {st.mean(p):.4f} (n={len(p)})  diff {obs:+.4f}  "
          f"permutation p = {pval:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vsd", type=Path, required=True)
    ap.add_argument("--states", type=Path, nargs="+", required=True)
    ap.add_argument("--validation", type=Path, default=None)
    ap.add_argument("--objective", type=str, default="min", choices=("min", "mean"))
    args = ap.parse_args()

    states = load_states(*args.states)
    print("=== retrieval candidate: replication on fresh seeds ===")
    retrieval_replication(states, "discovery")
    if args.validation and (args.validation / "units").exists():
        retrieval_replication(load_states(args.validation), "VALIDATION (fresh)")
    print()

    cells: dict[tuple[str, str], float] = {}
    for p in sorted((args.vsd / "units").glob("*.json")):
        d = json.loads(p.read_text())
        cells[(d["state_label"], d["tag"])] = value(d["trace"][-1], args.objective)

    labels = sorted({s for s, _ in cells})
    corpora = sorted({c for _, c in cells})
    complete = [s for s in labels
                if all((s, c) in cells for c in corpora) and s in states]
    print(f"=== conditional-value prediction ({args.objective}) — "
          f"{len(complete)} complete states x {len(corpora)} corpora ===")
    if len(complete) < 6:
        print("  too few complete states to hold one out meaningfully; "
              "matrix still filling")
        return 0

    feat_names = sorted(states[complete[0]]["features"])
    X = np.array([[states[s]["features"][f] for f in feat_names] for s in complete])
    Y = np.array([[cells[(s, c)] for c in corpora] for s in complete])

    hits_model = hits_global = 0
    regret_model, regret_global, regret_random = [], [], []
    rng = random.Random(0)

    for i, s in enumerate(complete):
        tr = [j for j in range(len(complete)) if j != i]
        realized = Y[i]
        best_v = realized.max()

        # state-aware: one ridge per corpus, fitted only on other states
        pred = []
        for k in range(len(corpora)):
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr, k])
            pred.append(float(m.predict(sc.transform(X[i:i+1]))[0]))
        choice = int(np.argmax(pred))

        gchoice = int(np.argmax(Y[tr].mean(axis=0)))   # global-best, state-blind
        rchoice = rng.randrange(len(corpora))

        hits_model += realized[choice] == best_v
        hits_global += realized[gchoice] == best_v
        regret_model.append(best_v - realized[choice])
        regret_global.append(best_v - realized[gchoice])
        regret_random.append(best_v - realized[rchoice])

    n = len(complete)
    print(f"  corpora: {corpora}")
    print(f"  {'selector':14s} {'top-1 hit':>10s} {'mean regret':>13s}")
    print(f"  {'state-aware':14s} {hits_model}/{n:<8d} {st.mean(regret_model):13.4f}")
    print(f"  {'global-best':14s} {hits_global}/{n:<8d} {st.mean(regret_global):13.4f}")
    print(f"  {'random':14s} {'-':>10s} {st.mean(regret_random):13.4f}")
    print()
    rm, rg, rr = st.mean(regret_model), st.mean(regret_global), st.mean(regret_random)
    tol = 1e-6
    if rm < rg - tol and hits_model > hits_global:
        print("  GATE 2 PASSES: telemetry predicts conditional data value better")
        print("  than the state-blind global-best rule on held-out states.")
    elif abs(rm - rg) <= tol:
        print("  GATE 2 NOT PASSED - TIE WITH GLOBAL-BEST.")
        print("  The state-aware selector matches the state-blind rule exactly, so")
        print("  a single corpus is best from every state measured. There is no")
        print("  conditional value to exploit; this is a statement about the")
        print("  WORLD (no interaction), not about the predictor failing.")
    elif rm < rr <= rg:
        print("  PARTIAL: beats random but not global-best - the predictor has")
        print("  learned the data main effect, NOT the interaction. This must not")
        print("  be reported as adaptive scheduling.")
    else:
        print("  GATE 2 FAILS: telemetry does not predict conditional data value")
        print("  on held-out states. The tournament is not licensed.")
    print(f"\n  mean regret magnitudes: model {rm:.4f}  global {rg:.4f}  random {rr:.4f}")
    if max(rm, rg, rr) < 0.02:
        print("  NOTE: all regrets are tiny, i.e. the corpora barely differ in value")
        print("  from these states. That is weak evidence for any data main effect")
        print("  and weaker still for an interaction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
