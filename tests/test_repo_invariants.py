"""Repository and document invariants, each pinning a regression that happened.

These are cheap guards on things that broke silently during the sprint and were
only caught by eye. Every test here corresponds to a real defect:

* inline math split across a line break by a rewrap, which stops GitHub
  rendering it and leaves literal ``$`` in the paper;
* em dashes reintroduced after they were removed;
* a figure reference pointing at a file that no longer exists;
* `.gitignore` excluding the ``artifacts/`` directory, which made Git skip the
  ``!artifacts/*.sha256`` negation and silently drop freeze records — the
  artifacts that evidence a protocol was hashed before its outcomes existed;
* a freeze record drifting out of step with the document it names;
* the E4b prototype's confirmatory barrier being bypassable.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report.md"


# --------------------------------------------------------------------------
# the paper
# --------------------------------------------------------------------------

def test_no_inline_math_is_split_across_a_line_break():
    """A rewrap once broke four `$...$` spans across newlines. GitHub does not
    render math that straddles a line break, so the expressions displayed as
    literal dollar signs."""
    offenders = [(i, line) for i, line in enumerate(REPORT.read_text().splitlines(), 1)
                 if line.count("$") % 2]
    assert not offenders, f"unbalanced $ on lines: {[i for i, _ in offenders]}"


def test_paper_contains_no_em_dashes():
    """An explicit style requirement for the submitted paper. En dashes in
    numeric ranges are fine and deliberately not checked."""
    text = REPORT.read_text()
    assert "—" not in text, f"{text.count(chr(0x2014))} em dashes present"


def test_every_referenced_figure_exists():
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", REPORT.read_text())
    assert refs, "the paper should reference figures"
    missing = [r for r in refs if not (ROOT / r).exists()]
    assert not missing, f"missing figure files: {missing}"


def test_claim_status_tables_share_one_column_geometry():
    """The three Appendix A tables are raw HTML precisely so their dividers
    land in the same place; rendered Markdown would size each to its own
    content."""
    text = REPORT.read_text()
    start = text.index("### A. Claim-status summary")
    end = text.index("### B. What the model sees")
    block = text[start:end]
    colgroups = {line.strip() for line in block.splitlines() if "colgroup" in line}
    assert block.count("<table") >= 3
    assert len(colgroups) == 1, f"tables disagree on column widths: {colgroups}"


def test_paper_keeps_its_load_bearing_sections():
    """Compression passes repeatedly rewrote whole sections; these must survive."""
    text = REPORT.read_text()
    for heading in ("## Abstract", "## 1. Introduction", "## 2. Related Work",
                    "## 3. Methods", "## 4. Results",
                    "## 5. Discussion and Limitations", "## 6. Conclusion",
                    "### Dual-Use and Ethical Considerations"):
        assert heading in text, f"missing section: {heading}"


# --------------------------------------------------------------------------
# freeze records
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=60).stdout


def _freeze_records() -> list[Path]:
    return sorted((ROOT / "artifacts").glob("*.sha256"))


def test_freeze_records_are_tracked_by_git():
    """`artifacts/` excluded the directory itself, so Git never evaluated the
    `!artifacts/*.sha256` negation and every record written after that rule
    was silently dropped."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(_git("ls-files", "artifacts").split())
    untracked = [p.name for p in _freeze_records()
                 if f"artifacts/{p.name}" not in tracked]
    assert not untracked, f"freeze records not tracked: {untracked}"


# Five records from the first freeze batch do not verify against their current
# documents, and already failed to verify at the commit that introduced them:
# the documents were rewritten during the doc-consolidation commit after their
# digests had been recorded, and the pre-consolidation bytes were never
# committed, so the frozen text is not recoverable from this repository. The
# digests are kept as-is rather than recomputed, because recomputing would
# assert a freeze that cannot be evidenced. Recorded in RESULTS.md.
#
# This list must never grow. Anything added to it is a new integrity failure.
LEGACY_UNVERIFIED = {
    "downstream_protocols.sha256",
    "p1_analysis_plan.sha256",
    "p1_frozen_pairs.sha256",
    "tournament_protocol.sha256",
    "v2_1_spec.sha256",
}


def _verify(record: Path) -> tuple[bool, str, str]:
    first = record.read_text().split("\n")[0].split()
    assert len(first) >= 2, f"malformed record: {record.name}"
    digest, named = first[0], ROOT / first[1]
    if not named.exists():
        pytest.skip(f"{first[1]} not present in this checkout")
    return (hashlib.sha256(named.read_bytes()).hexdigest() == digest,
            digest, hashlib.sha256(named.read_bytes()).hexdigest())


@pytest.mark.parametrize("record", _freeze_records(), ids=lambda p: p.name)
def test_freeze_record_matches_the_document_it_names(record: Path):
    """A frozen protocol that has drifted from its recorded digest is no longer
    frozen, and any ordering claim resting on it would be unevidenced."""
    ok, recorded, actual = _verify(record)
    if record.name in LEGACY_UNVERIFIED:
        pytest.xfail(f"known-unverifiable legacy freeze record ({record.name})")
    assert ok, (f"{record.name} has drifted: recorded {recorded[:16]}, "
                f"actual {actual[:16]}")


def test_the_unverifiable_legacy_set_does_not_grow():
    """Every freeze record outside the documented legacy set must verify."""
    present = {p.name for p in _freeze_records()}
    stale = LEGACY_UNVERIFIED - present
    assert not stale, f"legacy entries no longer present, prune them: {stale}"
    failing = {p.name for p in _freeze_records() if not _verify(p)[0]}
    assert failing <= LEGACY_UNVERIFIED, (
        f"new freeze-integrity failures: {sorted(failing - LEGACY_UNVERIFIED)}")


# --------------------------------------------------------------------------
# the E4b prototype's confirmatory firewall
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e4b():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e4b_prototype as mod
    return mod


def test_reserve_continuations_are_absent_unless_explicitly_released(e4b):
    """Held-out outcomes must not be generable by an ordinary run."""
    specs = [spec for _, spec, _ in e4b.build_queue(release_reserve=False)]
    reserve = {f"{arm}__seed{s:03d}" for s in e4b.RESERVE_SEEDS for arm in e4b.ARMS}
    leaked = [s for s in specs
              if s.startswith("cont:") and s.split(":")[1] in reserve]
    assert not leaked, f"reserve continuations queued without release: {leaked}"


def test_reserve_sources_are_still_generated_without_release(e4b):
    """Readouts may be computed in advance; only the outcomes are gated."""
    specs = [spec for _, spec, _ in e4b.build_queue(release_reserve=False)]
    reserve = {f"{arm}__seed{s:03d}" for s in e4b.RESERVE_SEEDS for arm in e4b.ARMS}
    assert {s.split(":", 1)[1] for s in specs if s.startswith("source:")} >= reserve


def test_released_reserve_still_waits_on_the_freeze_artifact(e4b, monkeypatch):
    """The flag alone must not be sufficient; a mistyped command should not be
    able to reveal held-out outcomes before predictions are frozen."""
    monkeypatch.setattr(e4b, "FREEZE", ROOT / "artifacts" / "__no_such_freeze__.json")
    reserve = {f"{arm}__seed{s:03d}" for s in e4b.RESERVE_SEEDS for arm in e4b.ARMS}
    ready = [r() for _, spec, r in e4b.build_queue(release_reserve=True)
             if spec.startswith("cont:") and spec.split(":")[1] in reserve]
    assert ready and not any(ready)


def test_development_work_outranks_reserve_work(e4b):
    """Priority order is the protocol's, not an implementation detail."""
    prio = {spec: p for p, spec, _ in e4b.build_queue(release_reserve=True)}
    dev = {f"{arm}__seed{s:03d}" for s in e4b.DEV_SEEDS for arm in e4b.ARMS}
    dev_cont = max(p for s, p in prio.items()
                   if s.startswith("cont:") and s.split(":")[1] in dev)
    reserve_source = min(p for s, p in prio.items()
                         if s.startswith("source:") and s.split(":", 1)[1] not in dev)
    assert dev_cont < reserve_source


def test_split_is_by_seed_so_sibling_arms_cannot_leak(e4b):
    """Splitting by individual state would leak: three arms of one seed share
    an initialization, so a sibling in development reveals its held-out
    partner."""
    assert not set(e4b.DEV_SEEDS) & set(e4b.RESERVE_SEEDS)


# --------------------------------------------------------------------------
# figure geometry
# --------------------------------------------------------------------------

def test_paper_figures_are_never_more_than_two_panels_across():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_paper_figures as mpf
    with pytest.raises(ValueError, match="two panels across"):
        mpf.canvas(1, 3)
