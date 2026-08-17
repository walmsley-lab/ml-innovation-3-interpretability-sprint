"""Exploratory: is developmental change smooth, abrupt, or multi-stage?

**Off the critical path. Costs no new compute in its current form.**

The question is whether readiness for `B` arrives gradually or in a localized
window. The honest way to ask it is by **held-out model comparison**, not by
looking at a curve and calling it a phase transition. Three descriptions are
frozen in advance and compared on data they were not fitted to:

* ``linear``      — smooth, no distinguished moment
* ``sigmoid``     — smooth but localized; has a characteristic time
* ``changepoint`` — piecewise linear with one break, fitted on training seeds only

Held out **by seed**: fit on all but one seed's trajectory, predict the held-out
seed. A split within a trajectory would let neighbouring points leak.

### What is being modelled, and the caveat that limits it

The quantity that properly defines a developmental transition is
`V(S_t, B)` — the *future learnability* of `B` from the state at source step
`t`, measured by continuing from a checkpoint at `t`. **We do not have that**:
the discovery runs save a single checkpoint at the end of the source phase, so
there are no temporally-ordered states to continue from.

What this script uses instead is **zero-shot `B` accuracy during source
training**, read from phase-0 telemetry. It is a *proxy*: it measures what the
model can already do, not how fast it would learn. The two can dissociate — a
model can gain learnability without gaining zero-shot competence, which is
precisely the distinction `BINDT` was built to test.

So a transition found here is a transition in **acquired competence**, and is
suggestive about readiness rather than evidence for it. The `V(S_t, B)` version
requires dense multi-step checkpointing plus identical continuations, and is
recorded in `BACKLOG.md` behind the stretch-goal gates.

    PYTHONPATH=src python scripts/analyze_transition.py --roots artifacts/g1 ...
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def f_linear(t, a, b):
    return a + b * t


def f_sigmoid(t, lo, hi, k, t0):
    return lo + (hi - lo) / (1.0 + np.exp(-k * (t - t0)))


def f_changepoint(t, a, b1, b2, tc):
    return a + b1 * np.minimum(t, tc) + b2 * np.maximum(t - tc, 0.0)


MODELS = {
    "linear": (f_linear, lambda t, y: [y.mean(), 0.0]),
    "sigmoid": (f_sigmoid, lambda t, y: [y.min(), y.max(), 5.0, t.mean()]),
    "changepoint": (f_changepoint, lambda t, y: [y[0], 0.0, 0.0, t.mean()]),
}


def fit_predict(name, t_tr, y_tr, t_te):
    fn, guess = MODELS[name]
    try:
        popt, _ = curve_fit(fn, t_tr, y_tr, p0=guess(t_tr, y_tr), maxfev=20000)
        return fn(t_te, *popt), popt
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--arm", type=str, default="A")
    ap.add_argument("--capability", type=str, default="BIND")
    args = ap.parse_args()

    traj: dict[int, list[tuple[float, float]]] = {}
    for root in args.roots:
        for p in sorted((root / "units").glob("*.json")):
            d = json.loads(p.read_text())
            if d.get("arm") != args.arm:
                continue
            pts = [(x["step"], x[args.capability]["accuracy"])
                   for x in d["trace"] if x.get("phase") == 0
                   and args.capability in x]
            if len(pts) >= 6:
                traj.setdefault(d["seed"], []).extend(pts)

    if len(traj) < 3:
        print(f"need >=3 seeds with phase-0 traces for arm {args.arm}; got {len(traj)}")
        return 0

    seeds = sorted(traj)
    print(f"=== smooth vs transition, arm {args.arm}, {args.capability} zero-shot "
          f"during source training ===")
    print(f"  {len(seeds)} seeds, held out one at a time\n")

    errors: dict[str, list[float]] = {m: [] for m in MODELS}
    breaks = []
    for held in seeds:
        tr = [(t, y) for s in seeds if s != held for t, y in traj[s]]
        te = sorted(traj[held])
        t_tr = np.array([t for t, _ in tr], float)
        y_tr = np.array([y for _, y in tr], float)
        t_te = np.array([t for t, _ in te], float)
        y_te = np.array([y for _, y in te], float)
        for m in MODELS:
            pred, popt = fit_predict(m, t_tr, y_tr, t_te)
            if pred is None:
                continue
            errors[m].append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
            if m == "changepoint" and popt is not None:
                breaks.append(float(popt[3]))

    print(f"  {'model':14s} {'held-out RMSE':>16s}")
    ranked = []
    for m in MODELS:
        if errors[m]:
            mean = st.mean(errors[m])
            ranked.append((mean, m))
            sd = st.stdev(errors[m]) if len(errors[m]) > 1 else 0.0
            print(f"  {m:14s} {mean:10.4f} +- {sd:.4f}")
    ranked.sort()
    if not ranked:
        print("  no model fitted successfully")
        return 0

    best_err, best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    print(f"\n  best held-out description: {best}")
    if second:
        margin = (second[0] - best_err) / (second[0] or 1e-9)
        print(f"  margin over next ({second[1]}): {margin:+.1%}")
        if margin < 0.05:
            print("  --> margin is small; the descriptions are not meaningfully")
            print("      distinguishable at this resolution. Do not claim a shape.")
    if best == "changepoint" and breaks:
        print(f"  fitted break at step {st.mean(breaks):.0f} "
              f"(sd {st.stdev(breaks) if len(breaks) > 1 else 0:.0f} across folds)")
        print("  --> a localized window is the better held-out description;")
        print("      dense replay around it would be the next test IF compute frees up.")
    elif best == "linear":
        print("  --> development looks smooth at this resolution; no distinguished")
        print("      moment. That is a substantive answer, not a null result.")
    elif best == "sigmoid":
        print("  --> smooth but localized: there is a characteristic time without a")
        print("      discontinuity. Avoid phase-transition language for this.")
    print("\n  REMINDER: this models acquired competence, not V(S_t,B) future")
    print("  learnability. A transition here is suggestive about readiness, not")
    print("  evidence for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
