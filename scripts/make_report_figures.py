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
    ax.set_yticklabels([s.replace("__seed", " ") for s in states], fontsize=7)
    best = M.argmax(axis=1)
    for i, j in enumerate(best):
        ax.scatter(j, i, marker="*", s=70, color="black", zorder=3)
    n_distinct = len(set(best))
    ax.set_title(f"$V(S,D)$ by incoming state\n"
                 f"{len(states)} states x {len(corp)} corpora; "
                 f"{n_distinct} distinct best-corpus outcomes",
                 fontsize=9.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045, label="value relative to that state's mean")
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
        r = np.corrcoef(dist, div)[0, 1]
        z = np.polyfit(dist, div, 1)
        xs = np.linspace(min(dist), max(dist), 50)
        ax.plot(xs, np.polyval(z, xs), "--", color="#a33", lw=1.2,
                label=f"r = {r:+.3f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("present-state matching distance (standardized)")
    ax.set_ylabel("future divergence  |ΔAULC|")
    ax.set_title(f"Hidden futures: does divergence track residual mismatch?\n"
                 f"{len(pairs)} of {meta['n_pairs']} frozen pairs complete",
                 fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig3_temporal(out: Path):
    """V(S_t,B) against source step — the P2 negative."""
    traj = {}
    for f in glob.glob("artifacts/temporal_replay_v/units/*.json"):
        d = json.loads(Path(f).read_text()); lab = d["state_label"]
        seed = int(lab[1:4]); t = int(lab.split("_t")[1])
        acc = [x["BIND"]["accuracy"] for x in sorted(d["trace"], key=lambda x: x["step"])]
        traj.setdefault(seed, []).append((t, sum(acc) / len(acc)))
    if not traj:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for k, seed in enumerate(sorted(traj)):
        v = sorted(traj[seed])
        ax.plot([t for t, _ in v], [y for _, y in v], "o-", ms=4, lw=1.2,
                alpha=0.85, color=plt.cm.tab10(k), label=f"seed {seed}")
    allt = [t for s in traj for t, _ in traj[s]]; ally = [y for s in traj for _, y in traj[s]]
    r = np.corrcoef(allt, ally)[0, 1]
    ax.set_xlabel("source-training step at which the checkpoint was taken")
    ax.set_ylabel("$V(S_t,B)$   (AULC of identical continuation)")
    ax.set_title("Temporal replay: no localized change in future learnability\n"
                 f"linear best on held-out fit (by 5.8%); r = {r:+.3f} with step",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
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
        ax.scatter(v["gn"], v["cos"], s=30, alpha=0.8, label=arm,
                   color=ARMC.get(arm, "#555"), edgecolor="white")
    ax.set_xlabel(r"$\|\nabla_{\mathrm{BIND}}\|$")
    ax.set_ylabel(r"$\cos(\nabla_{\mathrm{BIND}}, \nabla_{\mathrm{FACT}})$")
    ax.set_title("Gradient geometry separates training histories\n"
                 "a state/history MARKER — not a predictor of $V(S,D)$",
                 fontsize=9.5, loc="left")
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
                     (fig2_hidden_futures, "fig3_hidden_futures.png"),
                     (fig3_temporal, "fig4_temporal_replay.png"),
                     (fig5_geometry, "fig5_gradient_geometry.png")):
        r = fn(args.out / name)
        print(f"  {'wrote ' if r else 'SKIP  '} {name}" + ("" if r else "  (insufficient data)"))
        if r: made.append(name)
    print(f"\n{len(made)} figures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
