"""Measure L4 throughput on one representative Gate B configuration.

Runs the same B1 unit at concurrency 1, 2 and 4. These models are small
enough that one run cannot occupy an L4, so the question is not whether the
GPU is faster per run but how many runs it will hold at once.

Changes no scientific parameter. The configuration, seeds and evaluation
protocol are exactly those the Gate B sweep uses; only the number of
concurrent processes varies.

    python scripts/benchmark_l4.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

# The representative configuration, unchanged from the Gate B grid.
UNIT = ["--n-digits", "3", "--n-cues", "256", "--d-model", "64",
        "--n-layers", "2", "--steps", "600", "--lr", "0.003", "--stage", "b1"]
CONCURRENCIES = (1, 2, 4)
SEED_BASE = 9000


class GpuSampler(threading.Thread):
    """Poll nvidia-smi while the workload runs."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                util, mem = (float(v) for v in out.split(",")[:2])
                self.samples.append((util, mem))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if not self.samples:
            return {"available": False}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "available": True, "n_samples": len(utils),
            "util_mean": sum(utils) / len(utils), "util_max": max(utils),
            "mem_mean_mib": sum(mems) / len(mems), "mem_max_mib": max(mems),
        }


def run_at(concurrency: int, out: Path) -> dict:
    """Launch `concurrency` identical units and wait for all of them."""
    sampler = GpuSampler()
    sampler.start()
    started = time.time()

    processes = [
        subprocess.Popen(
            [sys.executable, "scripts/run_capacity_sweep.py", "--unit", *UNIT,
             "--seed", str(SEED_BASE + concurrency * 100 + i),
             "--out", str(out / f"c{concurrency}")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for i in range(concurrency)
    ]
    codes = [p.wait() for p in processes]  # handles, never process-name matching
    wall = time.time() - started
    gpu = sampler.stop()

    return {
        "concurrency": concurrency,
        "wall_seconds": wall,
        "runs": concurrency,
        "failures": sum(1 for c in codes if c != 0),
        "seconds_per_run": wall / concurrency,
        "runs_per_hour": concurrency / (wall / 3600.0),
        "gpu": gpu,
    }


def main() -> None:
    out = Path("out/benchmark")
    out.mkdir(parents=True, exist_ok=True)
    results = []

    print("=== L4 execution benchmark ===")
    print(f"unit: {' '.join(UNIT)}\n")

    for concurrency in CONCURRENCIES:
        result = run_at(concurrency, out)
        results.append(result)
        gpu = result["gpu"]
        util = f"{gpu['util_mean']:.0f}%/{gpu['util_max']:.0f}%" if gpu["available"] else "n/a"
        print(f"concurrency={result['concurrency']}  wall={result['wall_seconds']:.1f}s  "
              f"s/run={result['seconds_per_run']:.1f}  "
              f"runs/hour={result['runs_per_hour']:.1f}  "
              f"gpu util mean/max={util}  failures={result['failures']}")

    (out / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\nBENCHMARK_RESULTS_JSON_BEGIN")
    print(json.dumps(results, indent=2))
    print("BENCHMARK_RESULTS_JSON_END")
    print("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
