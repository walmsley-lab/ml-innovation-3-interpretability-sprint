"""Exploratory: how much behavioral observation makes readiness legible?

**Post-hoc exploratory. The subsets below were chosen after seeing P1, so this
is not a pre-registered analysis and cannot support a confirmatory claim.**

P1 found that matching on a modest behavioral vector erased the history-
conditioned difference in future learning. That is a binary statement. This asks
the more useful question: *which parts of the vector mattered?* If matching on
target competence alone leaves a difference, but adding a second capability
removes it, then behavioral legibility has a measurable price in observations.

Re-runs the P1 matched-vs-null comparison under progressively richer subsets of
the matching vector, holding everything else fixed.
"""
from __future__ import annotations
import glob, json, random, statistics as st
from pathlib import Path
import numpy as np

SUBSETS = {
    "target accuracy only":        ["zsB_acc"],
    "target loss only":            ["zsB_loss"],
    "target acc + loss":           ["zsB_acc", "zsB_loss"],
    "control capability only":     ["zsC_acc", "zsC_loss"],
    "both capabilities (P1 full)": ["zsB_acc", "zsB_loss", "zsC_acc", "zsC_loss"],
}
PERM = 4000


def main():
    states = {}
    for root in ("mediator_discovery", "mediator_validation", "hf_states"):
        for f in glob.glob(f"artifacts/{root}/units/*.json"):
            d = json.loads(Path(f).read_text())
            lab = f"{d['arm']}__seed{d['seed']:03d}"
            states[lab] = {"arm": d["arm"], "beh": {
                "zsB_acc": d["zero_shot_BIND"]["accuracy"],
                "zsB_loss": d["zero_shot_BIND"]["loss"],
                "zsC_acc": d["zero_shot_FACT"]["accuracy"],
                "zsC_loss": d["zero_shot_FACT"]["loss"]}}
    runs = {}
    for f in glob.glob("artifacts/p1_continuations/units/*__P1.json"):
        d = json.loads(Path(f).read_text())
        tr = sorted(d["trace"], key=lambda x: x["step"])
        acc = [x["BIND"]["accuracy"] for x in tr]
        runs[d["state_label"]] = {"final": acc[-1],
                                  "rate_only": sum(acc)/len(acc) - acc[0]}
    usable = sorted(set(states) & set(runs))
    print("EXPLORATORY behavioral-sufficiency map (post-hoc; subsets chosen "
          "after seeing P1)\n")
    print(f"  {len(usable)} states with both behavior and a measured future\n")
    print(f"  {'matching vector':30s} {'eps':>7s} {'n_match':>8s} "
          f"{'final diff':>11s} {'p':>7s}   {'rate diff':>10s} {'p':>7s}")

    for name, keys in SUBSETS.items():
        M = np.array([[states[s]["beh"][k] for k in keys] for s in usable], float)
        Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
        d = {(a, b): float(np.max(np.abs(Z[i]-Z[j])))
             for i, a in enumerate(usable) for j, b in enumerate(usable) if i < j
             and states[a]["arm"] == states[b]["arm"]}
        if not d: continue
        eps = float(np.quantile(list(d.values()), 0.10))
        matched = [p for p, v in d.items() if v <= eps]
        null = [p for p, v in d.items() if v > eps]
        if len(matched) < 5 or len(null) < 5: continue
        row = f"  {name:30s} {eps:7.3f} {len(matched):8d}"
        for metric in ("final", "rate_only"):
            dm = [abs(runs[a][metric]-runs[b][metric]) for a, b in matched]
            dn = [abs(runs[a][metric]-runs[b][metric]) for a, b in null]
            diff = st.mean(dm) - st.mean(dn)
            pool = dm + dn; rng = random.Random(0); cnt = 0
            for _ in range(PERM):
                rng.shuffle(pool)
                if abs(st.mean(pool[:len(dm)]) - st.mean(pool[len(dm):])) >= abs(diff):
                    cnt += 1
            row += f" {diff:+11.4f} {cnt/PERM:7.3f}" if metric == "final" else \
                   f"   {diff:+10.4f} {cnt/PERM:7.3f}"
        print(row)
    print("\n  Positive diff = matched pairs still diverge more than unmatched,")
    print("  i.e. that subset was NOT sufficient to make readiness legible.")
    print("\n  NOT INTERPRETABLE AS CONSTRUCTED. Two defects:")
    print("  1. epsilon is the 10% quantile of a max-abs-difference over the")
    print("     subset's dimensions, so richer vectors mechanically get LARGER")
    print("     epsilon and therefore LOOSER matched sets. The column above")
    print("     varies ~50x, so subsets are not comparable and the full vector's")
    print("     apparent sufficiency may only reflect its looser matching.")
    print("  2. Ten tests without correction; a Bonferroni bar would be 0.005.")
    print("  A valid version must hold matching stringency constant across")
    print("  subsets rather than holding the quantile constant.")


if __name__ == "__main__":
    main()
