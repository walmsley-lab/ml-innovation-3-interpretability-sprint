"""V(S, D) analysis: state main effect, data main effect, and the interaction.

The claim that matters for adaptive pretraining is **not** "different states
have different future learnability" — that is a state main effect and is
already implied by H1. Nor is it "some corpora are better than others" — that
is a data main effect and is ordinary curriculum knowledge.

The result that would license state-adaptive training is the **interaction**,
and in its strongest form an **ordering reversal**: the identity of the best
next corpus depends on the incoming state,

    argmax_D V(S_1, D)  !=  argmax_D V(S_2, D)

Without a reversal, a single globally-best corpus exists and a state-aware
scheduler has nothing to choose. This script decomposes the three explicitly
and refuses to let the interaction be read off a main effect.

**The common yardstick.** A cell is scored by a fixed objective, never by the
capability its own corpus trains — otherwise cells are incommensurable and
"best corpus" is meaningless. Two objectives are reported:

* ``mean``  — mean final accuracy across BIND, FACT and BINDT;
* ``min``   — min across the same three. This is the project's frozen
  efficiency metric, chosen because a mean hides a destroyed capability.

    PYTHONPATH=src python scripts/analyze_vsd.py --root artifacts/vsd_matrix
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

CAPS = ("BIND", "FACT", "BINDT")


def objective(trace_point: dict, kind: str) -> float:
    vals = [trace_point[c]["accuracy"] for c in CAPS if c in trace_point]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals) if kind == "mean" else min(vals)


def decompose(cells: dict[tuple[str, str], list[float]]) -> dict:
    """Two-way variance decomposition over the state x data grid.

    Uses cell means so unequal seed counts do not weight some cells more than
    others, and reports each component as a share of total between-cell
    variance. This is descriptive, not an F test: the design is small and the
    point is the relative size of the interaction, not a p-value.
    """
    states = sorted({s for s, _ in cells})
    datas = sorted({d for _, d in cells})
    grid = {(s, d): st.mean(v) for (s, d), v in cells.items() if v}
    full = [grid[(s, d)] for s in states for d in datas if (s, d) in grid]
    if len(full) < 4:
        return {"error": "grid too small"}

    grand = st.mean(full)
    state_mean = {s: st.mean([grid[(s, d)] for d in datas if (s, d) in grid])
                  for s in states}
    data_mean = {d: st.mean([grid[(s, d)] for s in states if (s, d) in grid])
                 for d in datas}

    ss_state = sum((state_mean[s] - grand) ** 2 for s in states) * len(datas)
    ss_data = sum((data_mean[d] - grand) ** 2 for d in datas) * len(states)
    ss_inter = sum(
        (grid[(s, d)] - state_mean[s] - data_mean[d] + grand) ** 2
        for s in states for d in datas if (s, d) in grid
    )
    total = ss_state + ss_data + ss_inter or 1e-12
    return {
        "state_main_share": ss_state / total,
        "data_main_share": ss_data / total,
        "interaction_share": ss_inter / total,
        "state_means": state_mean,
        "data_means": data_mean,
        "grand_mean": grand,
    }


def reversals(cells, states, datas) -> dict:
    """Does the best corpus change with incoming state?"""
    grid = {(s, d): st.mean(v) for (s, d), v in cells.items() if v}
    best = {}
    for s in states:
        row = [(grid[(s, d)], d) for d in datas if (s, d) in grid]
        if row:
            best[s] = max(row)[1]
    distinct = sorted(set(best.values()))

    pairs = []
    for s1, s2 in itertools.combinations(states, 2):
        for d1, d2 in itertools.combinations(datas, 2):
            if all((s, d) in grid for s in (s1, s2) for d in (d1, d2)):
                a = grid[(s1, d1)] - grid[(s1, d2)]
                b = grid[(s2, d1)] - grid[(s2, d2)]
                if a * b < 0:
                    pairs.append({
                        "states": [s1, s2], "corpora": [d1, d2],
                        "delta_state1": a, "delta_state2": b,
                    })
    return {
        "argmax_per_state": best,
        "n_distinct_argmax": len(distinct),
        "distinct_argmax": distinct,
        "n_sign_reversals": len(pairs),
        "reversals": pairs[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-complete", type=int, default=8,
                    help="minimum states measured on EVERY corpus")
    ap.add_argument("--max-spread", type=int, default=4,
                    help="max difference in per-corpus state coverage")
    args = ap.parse_args()

    units = sorted((args.root / "units").glob("*.json"))
    if not units:
        print(f"no V(S,D) units under {args.root}")
        return 0

    # --- balance gate -------------------------------------------------------
    # An interaction estimate read off a lopsided grid is not interpretable: if
    # one corpus is measured on states the others are not, corpus differences
    # and state differences are confounded. Refuse rather than caveat.
    raw = {}
    for p in sorted((args.root / "units").glob("*.json")):
        d = json.loads(p.read_text())
        raw[(d["state_label"], d["tag"])] = d
    all_states = sorted({s for s, _ in raw})
    all_corpora = sorted({c for _, c in raw})
    complete = [s for s in all_states if all((s, c) in raw for c in all_corpora)]
    per_corpus = {c: sum(1 for s in all_states if (s, c) in raw) for c in all_corpora}

    print(f"=== balance check ===")
    print(f"  cells {len(raw)}   states seen {len(all_states)}   "
          f"corpora {len(all_corpora)}")
    print(f"  per-corpus coverage: " +
          "  ".join(f"{c} {n}" for c, n in sorted(per_corpus.items())))
    print(f"  states with ALL corpora measured: {len(complete)}")
    spread = (max(per_corpus.values()) - min(per_corpus.values())) if per_corpus else 0
    ok = (len(complete) >= args.min_complete and len(all_corpora) >= 2
          and spread <= args.max_spread)
    if not ok:
        print(f"\n  NOT BALANCED ENOUGH TO DECOMPOSE.")
        print(f"  need >= {args.min_complete} complete states (have {len(complete)}), "
              f">= 2 corpora (have {len(all_corpora)}), and per-corpus spread "
              f"<= {args.max_spread} (have {spread}).")
        print("  Waiting for the next wave is correct; reading interaction out of")
        print("  an unbalanced grid confounds corpus with state.")
        return 0
    print("  balanced enough to proceed.\n")

    report = {}
    for kind in ("mean", "min"):
        cells: dict[tuple[str, str], list[float]] = {}
        for p in units:
            d = json.loads(p.read_text())
            final = d["trace"][-1]
            # State identity is the checkpoint label with its seed retained:
            # arm and seed together define an incoming state.
            cells.setdefault((d["state_label"], d["tag"]), []).append(
                objective(final, kind))

        # Restrict to COMPLETE states. An incomplete state trivially has its
        # single measured corpus as argmax, which inflates the distinct-argmax
        # count and manufactures reversals that do not exist.
        datas = sorted({dd for _, dd in cells})
        states = sorted({s for s, _ in cells}
                        & {s for s in {a for a, _ in cells}
                           if all((s, d) in cells for d in datas)})
        cells = {(s, d): v for (s, d), v in cells.items() if s in states}
        dec = decompose(cells)
        rev = reversals(cells, states, datas)
        report[kind] = {"decomposition": dec, "reversal": rev,
                        "n_states": len(states), "n_corpora": len(datas)}

        print(f"=== objective: {kind} across {CAPS} — "
              f"{len(states)} states x {len(datas)} corpora ===")
        if "error" in dec:
            print(f"  {dec['error']}\n")
            continue
        print(f"  variance share   state main {dec['state_main_share']:.3f}   "
              f"data main {dec['data_main_share']:.3f}   "
              f"INTERACTION {dec['interaction_share']:.3f}")
        print(f"  corpus means: " + "  ".join(
            f"{k} {v:.4f}" for k, v in sorted(dec["data_means"].items())))
        print(f"  distinct argmax corpora across states: "
              f"{rev['n_distinct_argmax']} {rev['distinct_argmax']}")
        print(f"  sign reversals (state x corpus-pair): {rev['n_sign_reversals']}")
        if rev["n_distinct_argmax"] > 1:
            print("  --> ORDERING REVERSAL PRESENT: the best next corpus depends "
                  "on incoming state")
        else:
            print("  --> no reversal: a single corpus is best from every state "
                  "measured; a state-aware scheduler would have nothing to choose")
        print()

    print("Claim boundary: a large interaction share and a genuine ordering")
    print("reversal establish that data value is state-conditional. Neither")
    print("establishes that telemetry can PREDICT the best corpus — that is a")
    print("separate prospective test, and the what-next tournament is downstream")
    print("of it.")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
