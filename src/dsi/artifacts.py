"""Versioned Parquet artifacts.

Every persisted table carries ``schema_version``. Changing a schema does not
mutate old artifacts; readers migrate them explicitly. Tables are written
with Polars' native Parquet writer.

Milestone A deviates from technical.md §11 in one place: ``evaluations``
rows are per condition rather than per example. Per-example rows exist to
serve the interactive explorer in Milestone H, and nothing before then reads
them. The column set is a strict subset, so the later schema is an addition
rather than a migration.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from dsi import SCHEMA_VERSION

__all__ = ["code_version", "utc_now", "write_table", "ArtifactWriter"]


def code_version(*, default: str = "unversioned") -> str:
    """The pinned code revision that a run's identity depends on.

    A dirty working tree is reported as such rather than silently attributed
    to the last commit: a run whose code cannot be recovered is a run whose
    result cannot be reproduced, and that belongs in the record.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return default


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_table(path: Path, rows: list[dict], *, schema_version: int = SCHEMA_VERSION) -> Path:
    """Write rows to Parquet with a schema version stamped on every row."""
    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame([{"schema_version": schema_version, **row} for row in rows])
    frame.write_parquet(path)
    return path


class ArtifactWriter:
    """Writes the Milestone A tables into one directory.

    The directory is a plain local path. Incremental upload to object
    storage is Milestone D's concern, and nothing here should assume it.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def write(self, name: str, rows: list[dict]) -> Path:
        return write_table(self.root / f"{name}.parquet", rows)

    def write_runs(self, rows: list[dict]) -> Path:
        return self.write("runs", rows)

    def write_learning_curves(self, rows: list[dict]) -> Path:
        return self.write("learning_curves", rows)

    def write_evaluations(self, rows: list[dict]) -> Path:
        return self.write("evaluations", rows)
