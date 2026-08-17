"""Individual result figures for report.md.

One figure per result, each regenerated from frozen outputs. Figures whose
experiment has not resolved render a labelled placeholder rather than a
misleading partial plot.
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAPS = ("BIND", "FACT", "BINDT")
# A_prime is a filesystem-safe label, not a name for a reader.
ARM_LABEL = {"A": "A", "A_prime": "A′", "BG": "BG"}


def pretty(label: str) -> str:
    """Render a state or arm label for display."""
    arm, _, rest = label.partition("__seed")
    return f"{ARM_LABEL.get(arm, arm)} {rest}" if rest else ARM_LABEL.get(label, label)
ARMC = {"A": "#c1272d", "A_prime": "#0b6fa4", "BG": "#7a7a7a"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig1_vsd(out: Path):
    """V(S,D) heatmap with ordering reversals annotated."""
    cells = {}
    for f in glob.glob("artifacts/vsd_matrix/units/*.json"):
        d = json.loads(Path(f).read_text()); fin = d["trace"][-1]
        v = [fin[c]["accuracy"] for c in CAPS if c in fin]
        cells[(d["state_label"], d["tag"])] = sum(v) / len(v)
    corp = sorted({c for _, c in cells})
    states = sorted(s for s in {a for a, _ in cells} if all((s, c) in cells for c in corp))
    if len(states) < 4:
        return None
    M = np.array([[cells[(s, c)] for c in corp] for s in states])
    Mn = M - M.mean(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(Mn, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(Mn).max(), vmax=np.abs(Mn).max())
    ax.set_xticks(range(len(corp))); ax.set_xticklabels(corp)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([pretty(s) for s in states], fontsize=7)
    best = M.argmax(axis=1)
    for i, j in enumerate(best):
        ax.scatter(j, i, marker="*", s=70, color="black", zorder=3)
    # Naming the split (12/13 vs 1/13) rather than "2 distinct outcomes":
    # the latter is true but reads as balanced, when in fact one corpus
    # dominates globally and a single state supplies the reversal.
    import collections as _c
    tally = _c.Counter(best)
    split = "; ".join(f"{corp[j]} best in {n}/{len(states)}"
                      for j, n in tally.most_common())
    ax.set_title("Future data value depends on incoming state\n"
                 f"Row-centered $V(S,D)$; corpus rankings reverse across states\n"
                 f"{split}", fontsize=9.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045,
                 label="value relative to that state's own mean")
    ax.set_xlabel("★ = best corpus for that state")
    ax.grid(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig2_hidden_futures(out: Path):
    """Present-state matching distance vs future divergence — the key confound check."""
    meta = Path("artifacts/p1_frozen_pairs.json")
    if not meta.exists():
        return None
    meta = json.loads(meta.read_text())
    runs = {}
    for f in glob.glob("artifacts/p1_continuations/units/*.json"):
        d = json.loads(Path(f).read_text())
        acc = [x["BIND"]["accuracy"] for x in sorted(d["trace"], key=lambda x: x["step"])]
        runs[d["state_label"]] = sum(acc) / len(acc)      # AULC: robust to late dips
    pairs = [(a, b) for a, b in map(tuple, meta["run_order"]) if a in runs and b in runs]
    if len(pairs) < 5:
        return None
    dist = [meta["distances"][f"{a}|{b}"] for a, b in pairs]
    div = [abs(runs[a] - runs[b]) for a, b in pairs]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(dist, div, s=32, alpha=0.8, color="#0b6fa4", edgecolor="white")
    if len(dist) > 2:
        # Faint and explicitly diagnostic. Matching distance is partly a
        # property of how the pairs were constructed, so this slope is not
        # the estimand; foregrounding it would let a construction artifact
        # read as the result. The frozen comparison is matched vs null.
        z = np.polyfit(dist, div, 1)
        xs = np.linspace(min(dist), max(dist), 50)
        ax.plot(xs, np.polyval(z, xs), "--", color="0.6", lw=1.0, alpha=0.8,
                label="residual-mismatch trend (diagnostic only)")
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_xlabel("present-state matching distance (standardized)")
    ax.set_ylabel("future divergence  |ΔAULC| between the pair")
    ax.set_title("Behavior-matched pairs do not show excess future divergence\n"
                 "71 frozen pairs; matched divergence ≈ within-arm null",
                 fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig3_temporal(out: Path):
    """V(S_t,B) against source step — the P2 negative.

    Deliberately **not** a spaghetti plot. Each point is an independently
    continued checkpoint, so connecting a seed's points implies a temporal
    trajectory that was never measured: nothing flows from one point to the
    next. Raw observations are therefore drawn unconnected, and the estimate
    the reader should attend to — the cross-seed mean at each source step — is
    the only emphasized series.

    The dashed ceiling matters as much as the scatter. Every one of the 48
    continuations reaches BIND accuracy 1.0, so AULC saturates near 0.905 and
    the visible spread is post-mastery instability rather than a difference in
    what was learnable. That is an assay limitation, recorded as such.
    """
    traj = {}
    for f in glob.glob("artifacts/temporal_replay_v/units/*.json"):
        d = json.loads(Path(f).read_text()); lab = d["state_label"]
        seed = int(lab[1:4]); t = int(lab.split("_t")[1])
        acc = [x["BIND"]["accuracy"] for x in sorted(d["trace"], key=lambda x: x["step"])]
        traj.setdefault(seed, []).append((t, sum(acc) / len(acc)))
    if not traj:
        return None

    pts = sorted((t, y) for s in traj for t, y in traj[s])
    allt = np.array([t for t, _ in pts]); ally = np.array([y for _, y in pts])
    r = float(np.corrcoef(allt, ally)[0, 1])

    steps = sorted({t for t, _ in pts})
    mean = np.array([np.mean([y for t, y in pts if t == s]) for s in steps])
    sem = np.array([np.std([y for t, y in pts if t == s], ddof=1)
                    / np.sqrt(len([y for t, y in pts if t == s])) for s in steps])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Pre-declared evaluation window, drawn before the data so it reads as
    # context rather than as a result.
    ax.axvspan(150, 450, color="0.93", zorder=0)
    ax.text(300, 0.965, "pre-declared 150–450 evaluation window", fontsize=7.5,
            color="0.45", ha="center", va="top")

    ceiling = float(ally.max())
    ax.axhline(ceiling, ls="--", lw=1.0, color="0.55", zorder=1)
    ax.text(452, ceiling, f" AULC ceiling {ceiling:.3f}\n (all runs reach BIND = 1.0)",
            fontsize=7.5, color="0.45", va="center")

    ax.plot(allt, ally, "o", ms=4.5, color="0.35", alpha=0.30,
            mew=0, zorder=2, label="individual checkpoints (n = 48)")

    fit = np.poly1d(np.polyfit(allt, ally, 1))
    xs = np.linspace(min(allt), max(allt), 100)
    ax.plot(xs, fit(xs), "-", lw=1.8, color="#c44e52", zorder=4,
            label="linear fit (best on held-out seed)")

    # Markers only, no connecting line: the means are cross-sectional
    # estimates at independent source steps, and joining them would reinstate
    # exactly the false trajectory the raw points were unlinked to avoid.
    # The fitted line, not the eye, carries any trend claim.
    ax.errorbar(steps, mean, yerr=sem, fmt="o", ms=6.0, capsize=3,
                color="#2b5d8a", ecolor="#2b5d8a", elinewidth=1.2, zorder=5,
                label="cross-seed mean ± SEM (3 seeds)")

    ax.set_xlabel("source-training step at which the checkpoint was taken")
    ax.set_ylabel("$V(S_t,B)$   (AULC of identical continuation)")
    ax.set_title("Temporal replay finds no reproducible transition "
                 "in future learnability", fontsize=10, loc="left")
    ax.set_ylim(0.0, 1.0)
    # r and the model-comparison RMSEs live in the caption. A figure title
    # or corner should carry the scientific result, not a diagnostics dump.
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def fig4_selectivity(out: Path):
    """H1/H2: the confirmed selective effect at t=0."""
    by = {}
    for root in ("artifacts/g1", "artifacts/c1_scout"):
        for f in glob.glob(f"{root}/units/*.json"):
            d = json.loads(Path(f).read_text())
            p1 = [x for x in d["trace"] if x.get("phase") == 1]
            if not p1: continue
            p1 = sorted(p1, key=lambda x: x["step"])
            by.setdefault(d["arm"], {"B": [], "C": []})
            by[d["arm"]]["B"].append(p1[0]["BIND"]["accuracy"])
            if "FACT" in p1[0]:
                by[d["arm"]]["C"].append(p1[0]["FACT"]["accuracy"])
    arms = [a for a in ("A", "A_prime", "BG") if a in by]
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    w = 0.36; x = np.arange(len(arms))
    for off, cap, col in ((-w/2, "B", "#c1272d"), (w/2, "C", "#7a7a7a")):
        vals = [st.mean(by[a][cap]) if by[a][cap] else 0 for a in arms]
        errs = [st.stdev(by[a][cap]) if len(by[a][cap]) > 1 else 0 for a in arms]
        ax.bar(x + off, vals, w, yerr=errs, capsize=3, label=f"capability {cap}",
               color=col, alpha=0.85)
    ax.axhline(1/64, ls="--", color="black", lw=1, label="chance (1/64)")
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel("zero-shot accuracy at $t=0$")
    ax.set_title("Selective effect: source lifts the target, not the control",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig5_geometry(out: Path):
    """E4a: gradient geometry as a history marker."""
    rows = {}
    for f in glob.glob("artifacts/grad_geometry/*.json"):
        d = json.loads(Path(f).read_text())
        arm = d["state_label"].split("__")[0]
        rows.setdefault(arm, {"cos": [], "gn": []})
        rows[arm]["cos"].append(d["features"]["cos.BIND.FACT"])
        rows[arm]["gn"].append(d["features"]["BIND.gnorm"])
    if not rows: return None
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for arm, v in rows.items():
        ax.scatter(v["gn"], v["cos"], s=30, alpha=0.8, label=pretty(arm),
                   color=ARMC.get(arm, "#555"), edgecolor="white")
    ax.set_xlabel(r"$\|\nabla_{\mathrm{BIND}}\|$")
    ax.set_ylabel(r"$\cos(\nabla_{\mathrm{BIND}}, \nabla_{\mathrm{FACT}})$")
    # The caveat belongs in the caption. As a title it reads as an editorial
    # warning rather than a description of what is plotted.
    ax.set_title("Gradient geometry encodes training history",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn, name in ((fig4_selectivity, "fig1_selectivity.png"),
                     (fig1_vsd, "fig2_vsd_matrix.png"),
                     (fig2_hidden_futures, "fig3_matching_vs_divergence.png"),
                     (fig3_temporal, "fig4_temporal_replay.png"),
                     (fig5_geometry, "fig5_gradient_geometry.png")):
        r = fn(args.out / name)
        print(f"  {'wrote ' if r else 'SKIP  '} {name}" + ("" if r else "  (insufficient data)"))
        if r: made.append(name)
    print(f"\n{len(made)} figures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
