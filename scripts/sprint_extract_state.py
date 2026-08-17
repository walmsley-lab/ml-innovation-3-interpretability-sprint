"""Lane B, stage SP-b: extract internal-state features from saved checkpoints.

Runs as a separate pass over the checkpoints ``sprint_population.py`` wrote, so
extraction can be re-run, extended or re-frozen without regenerating the
population.

Like the generator, this script computes **features only**. It does not form
matched pairs, fit a predictor, or look at outcomes. The analysis that does
those things must not run until the protocol is hashed.

    PYTHONPATH=src python scripts/sprint_extract_state.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dsi.checkpoint import load_model
from dsi.data import TaskConfig
from dsi.state import StateProbeSpec, extract_state_features, feature_names, frozen_probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=Path, default=Path("artifacts/sprint_population"))
    ap.add_argument("--examples", type=int, default=128)
    ap.add_argument("--probe-family", type=str, default="NEUTRAL_ALIGNED")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    task = TaskConfig(n_cues=512)
    spec = StateProbeSpec(family=args.probe_family, n_examples=args.examples)
    probe = frozen_probe(spec, task)

    out = args.out or (args.population / "state")
    out.mkdir(parents=True, exist_ok=True)

    units = sorted((args.population / "units").glob("pair_*.json"))
    if not units:
        print(f"no units in {args.population / 'units'}; nothing to extract")
        return 0

    t0 = time.time()
    written = 0
    declared: tuple[str, ...] | None = None

    for unit_path in units:
        target = out / unit_path.name
        if target.exists():
            continue
        record = json.loads(unit_path.read_text())
        entry = {"pair": record["pair"], "histories": {}}

        for history, payload in record["histories"].items():
            model, mc = load_model(payload["checkpoint"])
            if declared is None:
                declared = feature_names(mc.n_layers, mc.n_heads, spec)
            features = extract_state_features(model, probe, spec, task.answer_target_index)
            if set(features) != set(declared):
                raise RuntimeError(
                    f"{unit_path.name}/{history}: feature set does not match the "
                    "declared ordering; every model must produce the same "
                    "features or the design matrix is not aligned"
                )
            entry["histories"][history] = features

        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry) + "\n")
        tmp.replace(target)
        written += 1

    meta = {
        "probe_family": spec.family,
        "n_examples": spec.n_examples,
        "seed": spec.seed,
        "top_eigenvalues": spec.top_eigenvalues,
        "feature_names": list(declared or ()),
        "n_features": len(declared or ()),
    }
    (out / "state_spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"extracted {written} units ({len(declared or ())} features each) "
          f"in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
