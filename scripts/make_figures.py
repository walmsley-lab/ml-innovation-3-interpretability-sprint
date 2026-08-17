"""Demo figures for one message:

    same behaviour now does not imply the same developmental future

Four panels, each regenerated from whatever frozen outputs currently exist.
Panels with insufficient data render a labelled placeholder rather than a
misleading partial plot.

    PYTHONPATH=src python scripts/make_figures.py --out figures/
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAPS = ("BIND", "FACT", "BINDT")
ARMC = {"A": "#c1272d", "A_prime": "#0b6fa4", "BG": "#7a7a7a"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def placeholder(ax, title, msg):
    ax.set_title(title, fontsize=10, loc="left", weight="bold")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
            style="italic", color="#666", transform=ax.transAxes, wrap=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def load_pairs():
    p = Path("artifacts/p1_frozen_pairs.json")
    return json.loads(p.read_text()) if p.exists() else None


def load_p1():
    out = {}
    for f in glob.glob("artifacts/p1_continuations/units/*.json"):
        d = json.loads(Path(f).read_text())
        # Sort by step: an unsorted trace makes matplotlib draw a line back to
        # the start, which reads as a collapse that never happened.
        tr = sorted(d["trace"], key=lambda x: x["step"])
        out[d["state_label"]] = {"acc": [x["BIND"]["accuracy"] for x in tr],
                                 "steps": [x["step"] for x in tr]}
    return out


def panel_hidden_futures(ax):
    """Matched present behaviour, divergent futures under identical data."""
    meta, runs = load_pairs(), load_p1()
    title = "1. Matched now, divergent later"
    if not meta or len(runs) < 4:
        placeholder(ax, title, f"awaiting hidden-futures continuations\n({len(runs)} states so far)")
        return
    ready = [(a, b) for a, b in map(tuple, meta["run_order"]) if a in runs and b in runs]
    if not ready:
        placeholder(ax, title, "no complete matched pair yet"); return
    # Rank by AULC divergence, not final-point divergence. 1 of 19 units shows a
    # late instability dip, and ranking on the final point preferentially selects
    # exactly that unit — the panel would then showcase an artifact as the
    # phenomenon. AULC is robust to a single bad checkpoint.
    def aulc(lab):
        a = runs[lab]["acc"]; return sum(a) / len(a)
    ready.sort(key=lambda p: abs(aulc(p[0]) - aulc(p[1])), reverse=True)
    for k, (a, b) in enumerate(ready[:3]):
        for lab, st_ in ((a, "-"), (b, "--")):
            r = runs[lab]
            ax.plot(r["steps"], r["acc"], st_, lw=1.4, alpha=0.85,
                    color=plt.cm.tab10(k), label=lab if k < 1 else None)
    ax.set_xlabel("steps of identical future training")
    ax.set_ylabel("accuracy on target capability")
    ax.set_title(title, fontsize=10, loc="left", weight="bold")
    ax.text(0.02, 0.96, f"{len(ready)} matched pairs complete\nsolid/dashed = one pair",
            transform=ax.transAxes, va="top", fontsize=7.5, color="#444")


def panel_state_marker(ax):
    """History discrimination — explicitly NOT V(S,D) prediction."""
    title = "2. Internal state separates histories"
    rows = {}
    for f in glob.glob("artifacts/grad_geometry/*.json"):
        d = json.loads(Path(f).read_text())
        arm = d["state_label"].split("__")[0]
        rows.setdefault(arm, []).append(d["features"]["cos.BIND.FACT"])
    if not rows:
        placeholder(ax, title, "awaiting gradient geometry"); return
    order = [a for a in ("A", "A_prime", "BG") if a in rows]
    ax.boxplot([rows[a] for a in order], tick_labels=order, widths=0.55,
               patch_artist=True,
               boxprops=dict(alpha=0.55), medianprops=dict(color="black"))
    for i, a in enumerate(order, 1):
        ax.scatter(np.random.normal(i, 0.05, len(rows[a])), rows[a], s=9,
                   color=ARMC.get(a, "#555"), zorder=3, alpha=0.8)
    ax.set_ylabel("cos(∇BIND, ∇FACT)")
    ax.set_title(title, fontsize=10, loc="left", weight="bold")
    ax.text(0.02, 0.02, "history discrimination — NOT prediction of V(S,D)",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="#a33")


def panel_vsd(ax, fig):
    """Same data, different value depending on incoming state."""
    title = "3. Data value depends on incoming state"
    cells = {}
    for f in glob.glob("artifacts/vsd_matrix/units/*.json"):
        d = json.loads(Path(f).read_text()); fin = d["trace"][-1]
        v = [fin[c]["accuracy"] for c in CAPS if c in fin]
        cells[(d["state_label"], d["tag"])] = sum(v) / len(v)
    corp = sorted({c for _, c in cells})
    states = sorted(s for s in {a for a, _ in cells} if all((s, c) in cells for c in corp))
    if len(states) < 4:
        placeholder(ax, title, "awaiting V(S,D) matrix"); return
    M = np.array([[cells[(s, c)] for c in corp] for s in states])
    Mn = M - M.mean(axis=1, keepdims=True)      # per-state centring exposes reversals
    im = ax.imshow(Mn, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(Mn).max(), vmax=np.abs(Mn).max())
    ax.set_xticks(range(len(corp))); ax.set_xticklabels(corp, fontsize=8)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([s.replace("__seed", " ") for s in states], fontsize=6)
    best = M.argmax(axis=1)
    for i, j in enumerate(best):
        ax.scatter(j, i, marker="*", s=45, color="black", zorder=3)
    ax.set_title(title, fontsize=10, loc="left", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, label="value vs this state's mean")
    ax.text(0.02, -0.13, "★ = best corpus for that state; different rows pick different corpora",
            transform=ax.transAxes, fontsize=7.5, color="#444")


def panel_story(ax):
    ax.axis("off")
    ax.set_title("4. The claim", fontsize=10, loc="left", weight="bold")
    ax.text(0.0, 0.97,
        "Training history leaves a model in a developmental state.\n\n"
        "1  Two models can look the same on present behaviour,\n"
        "    yet respond differently to identical future training.\n\n"
        "2  That state is visible internally — gradient geometry and a\n"
        "    retrieval marker both separate histories reliably.\n\n"
        "3  The value of the same data depends on which state you\n"
        "    start from: the best next corpus changes by state.\n\n"
        "4  We cannot yet read the state well enough to exploit it.\n"
        "    A state-aware selector loses to a state-blind baseline.\n\n"
        "→  The conditional signal exists. State inference is the gap.\n\n"
        "Tested and NOT found: a localized moment when the state appears.\n"
        "48 checkpoints, identical continuations — learnability shows no\n"
        "local change (linear best by 5.8%, within the not-distinguishable\n"
        "band; r = -0.199 with training step).",
        va="top", ha="left", fontsize=8.4, linespacing=1.45, transform=ax.transAxes)
    ax.text(0.0, -0.04,
            "Control panel (choose next corpus from state): NOT licensed —\n"
            "the predictive readout gate is unmet. Next step, not a result.",
            va="bottom", ha="left", fontsize=7.2, style="italic", color="#a33",
            transform=ax.transAxes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    panel_hidden_futures(axes[0, 0])
    panel_state_marker(axes[0, 1])
    panel_vsd(axes[1, 0], fig)
    panel_story(axes[1, 1])
    fig.suptitle("Same behaviour now does not imply the same developmental future",
                 fontsize=12.5, weight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    dest = args.out / "demo.png"
    fig.savefig(dest, bbox_inches="tight")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
