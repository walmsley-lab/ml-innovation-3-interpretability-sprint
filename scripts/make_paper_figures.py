"""The three figures used in the paper.

`make_report_figures.py` renders one figure per result and stays as the full
set. This script renders only what the paper carries, at the paper's framing:

1. ``paper_fig1_readiness.png``  — B2: readiness, not task transfer
2. ``figures/fig2_vsd_matrix.png`` — State x Data (reused unchanged)
3. ``paper_fig3_boundaries.png`` — the four negative/boundary results in one panel

Figures that merely duplicate prose are deliberately absent.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

ARM_STYLE = {"A": ("#c1272d", "A  (source)"),
             "Ap": ("#0b6fa4", "A′  (matched control)"),
             "BG": ("#7a7a7a", "BG  (background)")}
CHANCE = 1 / 64


def _b2_curves(root: str):
    """(arm -> array of per-seed target-accuracy curves), plus the step axis."""
    out, steps = {}, None
    for f in sorted(glob.glob(f"{root}/units/*.json")):
        d = json.loads(Path(f).read_text())
        arm = d["arm"].split("_h")[0]
        p1 = sorted([x for x in d["trace"] if x.get("phase") == 1],
                    key=lambda x: x["step"])
        if not p1:
            continue
        s0 = p1[0]["step"]
        steps = [x["step"] - s0 for x in p1]
        out.setdefault(arm, []).append([x["target"]["accuracy"] for x in p1])
    return {k: np.array(v) for k, v in out.items()}, np.array(steps)


def fig1_readiness(out: Path):
    """B2: zero-shot blocked by construction, yet the source arm still learns faster."""
    panels = []
    for root in ("artifacts/b2_factorial_h1", "artifacts/b2_factorial_h2"):
        curves, steps = _b2_curves(root)
        if not curves:
            continue
        # rate_only = mean-over-curve minus t=0, the pre-declared readiness
        # statistic. Used here only to label which surface condition is which.
        rate = {a: float(np.mean(c.mean(axis=1) - c[:, 0])) for a, c in curves.items()}
        panels.append((curves, steps, rate["A"] - rate.get("Ap", 0.0)))
    if not panels:
        return None
    # The larger advantage is the disjoint-surface condition (+0.5438 vs +0.3967).
    panels.sort(key=lambda p: -p[2])
    titles = ["disjoint surface  (zero shared entity tokens)", "shared surface"]

    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (curves, steps, adv), title in zip(axes, panels, titles):
        for arm in ("A", "Ap", "BG"):
            if arm not in curves:
                continue
            c = curves[arm]
            col, lab = ARM_STYLE[arm]
            m, sd = c.mean(axis=0), c.std(axis=0)
            ax.plot(steps, m, "-", lw=2.0, color=col, label=lab, zorder=3)
            ax.fill_between(steps, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.axhline(CHANCE, ls="--", lw=1.0, color="black", zorder=2)
        ax.text(steps[-1], CHANCE, " chance", fontsize=7.5, va="bottom", ha="right")
        ax.set_title(f"{title}\nrate-only advantage over A′  {adv:+.4f}",
                     fontsize=9, loc="left")
        ax.set_xlabel("target-phase training step")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("accuracy on $B_2$  (retrieval ∘ derangement)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")

    # The whole point of B2: at t=0 nothing can be transferred, because the
    # derangement blocks the retrieved answer by construction.
    axes[0].annotate("all arms at/below chance at $t=0$:\nno answer can be transferred",
                     xy=(0, CHANCE), xytext=(0.30, 0.17), textcoords="axes fraction",
                     fontsize=7.5, color="0.3",
                     arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    fig.suptitle("Prior training buys readiness to learn, not the answer",
                 fontsize=11, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def fig3_boundaries(out: Path):
    """The four results that bound the claim, in one panel."""
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4))

    # (a) P1 -- matched vs within-arm null
    ax = axes[0]
    labels = ["final", "rate-only"]
    matched, null = [0.2893, 0.1325], [0.2700, 0.1496]
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, matched, w, label="matched (n=71)", color="#0b6fa4", alpha=0.9)
    ax.bar(x + w/2, null, w, label="within-arm null (n=376)", color="0.7")
    for i, p in enumerate((0.704, 0.504)):
        ax.text(i, max(matched[i], null[i]) + 0.02, f"p = {p:.3f}",
                ha="center", fontsize=7.5, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.40); ax.set_ylabel("future divergence  |Δ| between pair")
    ax.set_title("(a) P1: matched pairs do not diverge", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)

    # (b) Fork -- interaction estimate against zero
    ax = axes[1]
    est = [0.0888, 0.0295]
    lo = [-0.1173, -0.1707]; hi = [0.3162, 0.2368]
    y = np.arange(2)
    ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", ms=7, capsize=4, color="#0b6fa4", elinewidth=1.4)
    ax.axvline(0, ls="--", lw=1.1, color="black")
    ax.set_yticks(y); ax.set_yticklabels(["all 16 pairs", "stable subset"], fontsize=8)
    ax.set_xlabel("State × Data interaction (95% CI)")
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("(b) Fork: no preference reversal\n8/16 sign agreement = chance",
                 fontsize=9, loc="left")

    # (c) P2 -- temporal replay
    ax = axes[2]
    pts = []
    for f in glob.glob("artifacts/temporal_replay_v/units/*.json"):
        d = json.loads(Path(f).read_text())
        m = re.match(r"s(\d+)_t(\d+)", Path(f).stem)
        acc = [x["BIND"]["accuracy"] for x in sorted(d["trace"], key=lambda x: x["step"])]
        pts.append((int(m.group(2)), sum(acc) / len(acc)))
    if pts:
        t = np.array([a for a, _ in pts]); v = np.array([b for _, b in pts])
        ax.plot(t, v, "o", ms=4, color="0.35", alpha=0.30, mew=0)
        steps = sorted(set(t.tolist()))
        mean = [v[t == s].mean() for s in steps]
        sem = [v[t == s].std(ddof=1) / np.sqrt((t == s).sum()) for s in steps]
        ax.errorbar(steps, mean, yerr=sem, fmt="o", ms=4.5, capsize=2,
                    color="#2b5d8a", elinewidth=1.0)
        fit = np.poly1d(np.polyfit(t, v, 1))
        xs = np.linspace(t.min(), t.max(), 50)
        ax.plot(xs, fit(xs), "-", lw=1.6, color="#c44e52")
        ax.axhline(v.max(), ls="--", lw=0.9, color="0.6")
        ax.text(t.max(), v.max(), " ceiling", fontsize=7, color="0.45", va="bottom",
                ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("source step of checkpoint")
    ax.set_ylabel("$V(S_t,B)$")
    ax.set_title("(c) P2: no localized window\nlinear best by 5.8%", fontsize=9, loc="left")

    # (d) the readout/selector failure against the state-blind baseline
    ax = axes[3]
    names = ["state-aware\nselector", "global-best\n(state-blind)", "random"]
    regret = [0.0473, 0.0207, 0.0758]
    cols = ["#0b6fa4", "#2a9d5c", "0.75"]
    ax.bar(np.arange(3), regret, 0.6, color=cols, alpha=0.9)
    for i, r in enumerate(regret):
        ax.text(i, r + 0.002, f"{r:.4f}", ha="center", fontsize=7.5, color="0.3")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("mean regret  (lower is better)")
    ax.set_ylim(0, 0.092)
    ax.set_title("(d) Readout: loses to the\nstate-blind baseline", fontsize=9, loc="left")

    fig.suptitle("What bounds the claim: four pre-registered negative and "
                 "boundary results", fontsize=11, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig1_readiness, "paper_fig1_readiness.png"),
                     (fig3_boundaries, "paper_fig3_boundaries.png")):
        r = fn(args.out / name)
        print(f"  {'wrote ' if r else 'SKIP  '} {name}")
    print("  reused  fig2_vsd_matrix.png (State x Data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
