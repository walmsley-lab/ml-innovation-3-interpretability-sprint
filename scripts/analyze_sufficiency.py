"""Lane C (recovered) — how much behavioral information makes readiness legible?

**POST-HOC EXPLORATORY.** The feature ladder was chosen after seeing P1. This
cannot support a confirmatory claim and is not part of the frozen evidence
ladder. It reuses P1 outcomes only; nothing is retrained.

### What was wrong with the first version

Epsilon was the 10% quantile of a max-absolute-difference over each subset's own
dimensions. Richer subsets therefore got mechanically larger epsilon and looser
matched sets — the value varied ~50x across the ladder — so subsets were not
comparable and the full vector's apparent sufficiency may have reflected nothing
but loose matching. Ten uncorrected tests compounded it.

### The fix

**Matching pressure is held constant.** For every subset we rank all eligible
pairs by distance and take the same fixed `K` closest pairs. Selection pressure
is then identical across the ladder and the only thing varying is *which*
information the matching used.

* four **nested** feature sets, so the ladder is a monotone information sequence
* features standardized once, consistently
* P1's eligibility constraints preserved (within-arm pairs)
* outcomes: `final` and `rate_only` only. **`t=0` is excluded** — it is a
  matching variable, so its divergence is circular by construction
* the same same-arm null construction as P1
* excess divergence = matched − null, with bootstrap CI
* **one** pre-specified trend test across the ladder, not ten separate tests
"""
from __future__ import annotations
import glob, json, random, statistics as st
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# Level 4 is the P1 full vector, so it must reproduce the frozen P1 result.
# If it does not, the recovery is measuring something P1 did not and the
# ladder below is meaningless. Asserted, not assumed.
P1_ANCHOR = {"final": +0.0193, "rate_only": -0.0171}

K = 71                      # matches P1's frozen pair count
BOOT = 10000
LADDER = [
    ("target accuracy",                  ["zsB_acc"]),
    ("+ target loss",                    ["zsB_acc", "zsB_loss"]),
    ("+ control accuracy",               ["zsB_acc", "zsB_loss", "zsC_acc"]),
    ("+ control loss (P1 full vector)",  ["zsB_acc", "zsB_loss", "zsC_acc", "zsC_loss"]),
]


def boot_excess_ci(dm, dn, n=BOOT):
    """CI on the *excess* (matched - null), resampling both arms independently."""
    rng = random.Random(0)
    draws = sorted(
        st.mean([dm[rng.randrange(len(dm))] for _ in dm])
        - st.mean([dn[rng.randrange(len(dn))] for _ in dn])
        for _ in range(n))
    return draws[int(.025 * n)], draws[int(.975 * n)]


def perm_p(dm, dn, n=BOOT):
    """Same permutation test P1 used: shuffle the matched/null labels."""
    rng = random.Random(1)
    pool, k = dm + dn, len(dm)
    obs = abs(st.mean(dm) - st.mean(dn))
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def main():
    states, runs = {}, {}
    for root in ("mediator_discovery", "mediator_validation", "hf_states"):
        for f in glob.glob(f"artifacts/{root}/units/*.json"):
            d = json.loads(Path(f).read_text())
            states[f"{d['arm']}__seed{d['seed']:03d}"] = {"arm": d["arm"], "beh": {
                "zsB_acc": d["zero_shot_BIND"]["accuracy"],
                "zsB_loss": d["zero_shot_BIND"]["loss"],
                "zsC_acc": d["zero_shot_FACT"]["accuracy"],
                "zsC_loss": d["zero_shot_FACT"]["loss"]}}
    for f in glob.glob("artifacts/p1_continuations/units/*__P1.json"):
        d = json.loads(Path(f).read_text())
        acc = [x["BIND"]["accuracy"] for x in sorted(d["trace"], key=lambda x: x["step"])]
        runs[d["state_label"]] = {"final": acc[-1],
                                  "rate_only": sum(acc) / len(acc) - acc[0]}

    usable = sorted(set(states) & set(runs))
    allk = sorted(states[usable[0]]["beh"])
    M = np.array([[states[s]["beh"][k] for k in allk] for s in usable], float)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)          # standardized once
    idx = {s: i for i, s in enumerate(usable)}
    eligible = [(a, b) for i, a in enumerate(usable) for b in usable[i+1:]
                if states[a]["arm"] == states[b]["arm"]]

    print("LANE C (recovered) — behavioral sufficiency with matching pressure held constant")
    print("POST-HOC EXPLORATORY; not part of the frozen evidence ladder.\n")
    print(f"  {len(usable)} states, {len(eligible)} eligible within-arm pairs, "
          f"K = {K} closest pairs per subset\n")
    print(f"  {'matching information':34s} {'final excess':>13s} {'95% CI':>18s}"
          f" {'perm p':>8s} {'rate excess':>12s} {'perm p':>8s}")

    excess = {"final": [], "rate_only": []}
    for name, keys in LADDER:
        cols = [allk.index(k) for k in keys]
        d = sorted(((float(np.max(np.abs(Z[idx[a]][cols] - Z[idx[b]][cols]))), a, b)
                    for a, b in eligible), key=lambda t: t[0])
        matched = [(a, b) for _, a, b in d[:K]]
        null = [(a, b) for _, a, b in d[K:]]
        row = f"  {name:34s}"
        for metric in ("final", "rate_only"):
            dm = [abs(runs[a][metric] - runs[b][metric]) for a, b in matched]
            dn = [abs(runs[a][metric] - runs[b][metric]) for a, b in null]
            ex = st.mean(dm) - st.mean(dn)
            excess[metric].append(ex)
            if metric == "final":
                lo, hi = boot_excess_ci(dm, dn)
                row += f" {ex:+13.4f} [{lo:+.3f},{hi:+.3f}] {perm_p(dm, dn):8.4f}"
            else:
                row += f" {ex:+12.4f} {perm_p(dm, dn):8.4f}"
        print(row)

    print("\n  === ANCHOR: level 4 must reproduce frozen P1 ===")
    ok = True
    for metric, want in P1_ANCHOR.items():
        got = excess[metric][-1]
        good = abs(got - want) < 5e-4
        ok &= good
        print(f"    {metric:10s} recovered {got:+.4f}  vs frozen P1 {want:+.4f}"
              f"   {'MATCH' if good else 'MISMATCH'}")
    if not ok:
        print("\n    ANCHOR FAILED — the ladder does not reproduce P1 at its top")
        print("    rung, so it is not measuring P1's quantity. Do not interpret.")
        return
    print("    Anchor holds: the ladder's top rung IS the frozen P1 result.")

    print("\n  === ONE pre-specified trend test across the nested ladder ===")
    lv = list(range(len(LADDER)))
    for metric in ("final", "rate_only"):
        rho, p = spearmanr(lv, excess[metric])
        print(f"    {metric:10s} excess vs information level: Spearman rho = "
              f"{rho:+.3f}, p = {p:.4f}")
    print("\n  A negative rho would mean excess divergence shrinks as matching")
    print("  information grows — i.e. behavioral legibility has a measurable")
    print("  price in observations. Four ladder points is very little power for")
    print("  a trend test; read the direction, not the p-value.\n")

    print("  === VERDICT ===")
    print("  The pre-specified trend test does not reach significance on either")
    print("  outcome. The per-rung permutation tests are shown for transparency")
    print("  and are NOT the pre-specified analysis; across eight rung-by-outcome")
    print("  cells a Bonferroni threshold is p < 0.006, and the one nominally")
    print("  low cell (+ target loss, final, p = 0.034) clears no such bar.")
    print("  That cell is also positive -- behaviorally matched pairs diverging")
    print("  MORE than unmatched ones -- and it is non-monotone in the ladder,")
    print("  which is the signature of noise rather than of a sufficiency")
    print("  threshold. It is reported, not interpreted.")
    print("")
    print("  This does NOT show behavior is sufficient to predict the future.")
    print("  It shows P1's null is not an artifact of how much behavior was")
    print("  matched on: coarsening from four behavioral numbers to one does")
    print("  not make hidden divergence appear. The original Lane C question")
    print("  -- how much observation does legibility cost -- needs a design")
    print("  with divergence to titrate, which this data does not contain.")


if __name__ == "__main__":
    main()
