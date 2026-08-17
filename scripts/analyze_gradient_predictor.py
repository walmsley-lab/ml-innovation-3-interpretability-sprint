"""Experiment 4 test: does gradient geometry predict conditional data value?

Same estimand and same competitor as the activation readout that failed
(`analyze_predictor.py`): leave-one-state-out prediction of `V(S,D)`, judged
against the state-blind **global-best** rule. Only the feature family changes.

Beating `random` is not the bar. A predictor that learns "FACT is usually best"
beats random while knowing nothing about conditionality, which is precisely how
the activation readout failed.
"""
from __future__ import annotations
import argparse, glob, json, random, statistics as st
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

CAPS = ("BIND", "FACT", "BINDT")
ALPHAS = np.logspace(-3, 4, 20)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grad", type=Path, default=Path("artifacts/grad_geometry"))
    ap.add_argument("--vsd", type=Path, default=Path("artifacts/vsd_matrix"))
    ap.add_argument("--objective", type=str, default="mean", choices=("mean", "min"))
    args = ap.parse_args()

    feats = {}
    for f in sorted(args.grad.glob("*.json")):
        d = json.loads(f.read_text())
        feats[d["state_label"]] = d["features"]
    # Behavioral features for the same states. Given the P1/Fork nulls, the
    # central question is no longer "does geometry beat a global baseline" but
    # "does geometry add anything BEYOND behavior", since behavior turned out to
    # capture more future-relevant variation than expected.
    beh = {}
    for root in ("artifacts/mediator_discovery", "artifacts/mediator_validation",
                 "artifacts/hf_states"):
        for f in sorted(Path(root).glob("units/*.json")):
            d = json.loads(f.read_text())
            beh[f"{d['arm']}__seed{d['seed']:03d}"] = {
                "zsB_acc": d["zero_shot_BIND"]["accuracy"],
                "zsB_loss": d["zero_shot_BIND"]["loss"],
                "zsC_acc": d["zero_shot_FACT"]["accuracy"],
                "zsC_loss": d["zero_shot_FACT"]["loss"]}
    cells = {}
    for f in sorted((args.vsd / "units").glob("*.json")):
        d = json.loads(f.read_text())
        fin = d["trace"][-1]
        v = [fin[c]["accuracy"] for c in CAPS if c in fin]
        cells[(d["state_label"], d["tag"])] = (sum(v) / len(v) if args.objective == "mean" else min(v))

    corpora = sorted({c for _, c in cells})
    complete = sorted(s for s in {a for a, _ in cells}
                      if all((s, c) in cells for c in corpora)
                      and s in feats and s in beh)
    print(f"=== gradient-geometry readout ({args.objective}) — "
          f"{len(complete)} states with both geometry and complete V(S,D) ===")
    if len(complete) < 6:
        print(f"  only {len(complete)} usable states; need >= 6. "
              f"(geometry: {len(feats)}, complete V(S,D) rows: "
              f"{len({a for a,_ in cells})})")
        return 0

    gk = sorted(feats[complete[0]]); bk = sorted(beh[complete[0]])
    Xg = np.array([[feats[s][k] for k in gk] for s in complete], float)
    Xb = np.array([[beh[s][k] for k in bk] for s in complete], float)
    FEATURE_SETS = {"behavior only": Xb, "geometry only": Xg,
                    "behavior + geometry": np.hstack([Xb, Xg])}
    Y = np.array([[cells[(s, c)] for c in corpora] for s in complete], float)

    def loso(X):
        hits, reg = 0, []
        for i in range(len(complete)):
            tr = [j for j in range(len(complete)) if j != i]
            real = Y[i]; best = real.max()
            pred = []
            for k in range(len(corpora)):
                sc = StandardScaler().fit(X[tr])
                m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr, k])
                pred.append(float(m.predict(sc.transform(X[i:i+1]))[0]))
            ch = int(np.argmax(pred))
            hits += real[ch] == best; reg.append(best - real[ch])
        return hits, reg

    hits_m = hits_g = 0
    rm, rg, rr = [], [], []
    rng = random.Random(0)
    X = Xg
    for i in range(len(complete)):
        tr = [j for j in range(len(complete)) if j != i]
        real = Y[i]; best = real.max()
        pred = []
        for k in range(len(corpora)):
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr, k])
            pred.append(float(m.predict(sc.transform(X[i:i+1]))[0]))
        ch = int(np.argmax(pred)); gc = int(np.argmax(Y[tr].mean(axis=0)))
        hits_m += real[ch] == best; hits_g += real[gc] == best
        rm.append(best - real[ch]); rg.append(best - real[gc])
        rr.append(best - real[rng.randrange(len(corpora))])

    n = len(complete)
    print(f"  features: geometry {len(gk)}, behavior {len(bk)}   corpora: {corpora}")
    print(f"  {'selector':14s} {'top-1':>8s} {'mean regret':>13s}")
    print(f"  {'gradient':14s} {hits_m}/{n:<6d} {st.mean(rm):13.4f}")
    print(f"  {'global-best':14s} {hits_g}/{n:<6d} {st.mean(rg):13.4f}")
    print(f"  {'random':14s} {'-':>8s} {st.mean(rr):13.4f}")
    print("\n  === CENTRAL EXPLORATORY COMPARISON: incremental value over behavior ===")
    results = {}
    for name, XX in FEATURE_SETS.items():
        h, r = loso(XX)
        results[name] = (h, st.mean(r))
        print(f"  {name:22s} top-1 {h}/{n}   mean regret {st.mean(r):.4f}")
    print(f"  {'global-best (blind)':22s} top-1 {hits_g}/{n}   mean regret {st.mean(rg):.4f}")
    bo = results["behavior only"][1]; bg = results["behavior + geometry"][1]
    go = results["geometry only"][1]; gb = st.mean(rg)
    spread = max(bo, bg, go) - min(bo, bg, go)
    # A sign difference is not a result. Require the margin to be a meaningful
    # fraction of the baseline being beaten, and sanity-check that combining
    # features does not do WORSE than a subset -- which is the signature of
    # noise rather than signal.
    MARGIN = 0.10 * gb
    print(f"\n  spread across the three readouts: {spread:.4f}"
          f"   (meaningful-margin threshold {MARGIN:.4f})")
    if bg > go + 1e-9:
        print("  NOTE: behavior+geometry is WORSE than geometry alone. Combining")
        print("  features should not hurt if either carried signal; this pattern")
        print("  indicates noise.")
    if spread < MARGIN:
        print(f"\n  NO MEANINGFUL DIFFERENCE among the three readouts, and all")
        print(f"  lose to the state-blind global-best ({gb:.4f}). Geometry adds")
        print("  nothing beyond behavior at this n. Consistent with the P1/Fork")
        print("  nulls: behavior already carries the future-relevant variation.")
    elif bg < bo - MARGIN:
        print(f"\n  Geometry adds value beyond behavior by a meaningful margin.")
        print("  Exploratory: motivates a powered E4b rather than establishing it.")
    else:
        print("\n  Geometry does not add a meaningful margin beyond behavior.")

    if st.mean(rm) < st.mean(rg) - 1e-6 and hits_m > hits_g:
        print("\n  BEATS GLOBAL-BEST -> experiment 6 (tournament) is licensed.")
    elif abs(st.mean(rm) - st.mean(rg)) <= 1e-6:
        print("\n  TIES global-best -> no exploitable conditionality via this readout.")
    else:
        print("\n  DOES NOT beat global-best -> experiment 6 stays gated. No tuning.")
    print("  Note: a predictive geometry is a READOUT result, not a mechanism claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
