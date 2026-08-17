"""Lane D — readiness dose-response, per the frozen plan.

Implements `docs/experiments/dose_response_plan.md` (9096cd66d13b5313), frozen
before any continuation existed.

Primary: Spearman rank correlation between A-exposure level and `rate_only` on
the target. Control: `t=0` must stay at or near chance across levels, otherwise
the derangement is not blocking zero-shot transfer and the primary cannot be
read as readiness.

Changepoints, sigmoids and other shapes are prohibited by the plan. Only
monotonic association is tested.
"""
from __future__ import annotations
import glob, json, statistics as st
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, linregress

CHANCE = 1 / 64


def main():
    rows = {}
    for f in glob.glob("artifacts/dose_response_v/units/*.json"):
        d = json.loads(Path(f).read_text())
        lab = d["state_label"]                       # sNNN_tTTTT
        seed = int(lab[1:4]); step = int(lab.split("_t")[1])
        tr = sorted(d["trace"], key=lambda x: x["step"])
        acc = [x["BINDT"]["accuracy"] for x in tr] if "BINDT" in tr[0] \
              else [x["BIND"]["accuracy"] for x in tr]
        rows.setdefault(seed, []).append(
            {"step": step, "t0": acc[0],
             "rate_only": sum(acc) / len(acc) - acc[0], "final": acc[-1]})
    if not rows:
        print("no dose-response continuations yet"); return
    seeds = sorted(rows)
    print(f"LANE D — readiness dose-response, {len(seeds)} seeds "
          f"(plan 9096cd66d13b5313)\n")

    print("  === CONTROL: t=0 competence must stay near chance ===")
    all_t0 = []
    for s in seeds:
        v = sorted(rows[s], key=lambda r: r["step"])
        all_t0 += [r["t0"] for r in v]
        print(f"    seed {s}: " + "  ".join(f"{r['step']:>4d}:{r['t0']:.3f}" for r in v))
    print(f"    pooled t=0 mean {st.mean(all_t0):.4f} vs chance {CHANCE:.4f}"
          f"  -> {'OK, blocked' if st.mean(all_t0) <= CHANCE*2 else 'NOT BLOCKED — primary uninterpretable'}")

    print("\n  === PRIMARY: exposure vs rate_only ===")
    for s in seeds:
        v = sorted(rows[s], key=lambda r: r["step"])
        print(f"    seed {s}: " + "  ".join(f"{r['step']:>4d}:{r['rate_only']:.3f}" for r in v))
    x = [r["step"] for s in seeds for r in rows[s]]
    y = [r["rate_only"] for s in seeds for r in rows[s]]
    rho, p = spearmanr(x, y)
    print(f"\n    pooled Spearman rho = {rho:+.3f}   p = {p:.4f}   (n={len(x)})")
    per = []
    for s in seeds:
        v = sorted(rows[s], key=lambda r: r["step"])
        if len(v) > 2:
            per.append(spearmanr([r["step"] for r in v], [r["rate_only"] for r in v])[0])
    if per:
        print(f"    per-seed rho: " + " ".join(f"{r:+.2f}" for r in per)
              + f"   ({sum(1 for r in per if r > 0)}/{len(per)} positive)")
    lr = linregress(x, y)
    print(f"    secondary linear trend: slope {lr.slope:+.2e} per step, p = {lr.pvalue:.4f}")
    print("\n    No changepoint or nonlinear shape is fitted; the plan tests")
    print("    monotonic association only.")


if __name__ == "__main__":
    main()
