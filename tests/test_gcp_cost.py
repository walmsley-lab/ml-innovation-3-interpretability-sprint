"""The cost model and the guards that stop it spending money by accident.

Only the pure parts are tested. Nothing here touches gcloud, and no test in
this file can create a billable resource.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dsi_gcp", Path(__file__).parent.parent / "scripts" / "gcp.py"
)
gcp = importlib.util.module_from_spec(spec)
sys.modules["dsi_gcp"] = gcp
spec.loader.exec_module(gcp)


def test_cost_scales_with_runs_and_rate():
    one = gcp.estimate(100, 60)
    two = gcp.estimate(200, 60)
    assert two.compute_usd == pytest.approx(2 * one.compute_usd)


def test_parallelism_cuts_wall_time_and_cost():
    serial = gcp.estimate(100, 60, parallel=1)
    parallel = gcp.estimate(100, 60, parallel=4)
    assert parallel.wall_hours == pytest.approx(serial.wall_hours / 4)
    assert parallel.total_usd < serial.total_usd


def test_spot_is_cheaper_than_on_demand():
    assert gcp.estimate(100, 60, spot=True).total_usd < gcp.estimate(100, 60).total_usd


def test_total_includes_disk():
    est = gcp.estimate(100, 60)
    assert est.total_usd == pytest.approx(est.compute_usd + est.disk_usd)
    assert est.disk_usd > 0


def test_deadman_exceeds_the_estimate_but_is_never_trivial():
    """Too tight kills real work; too loose is the thing being guarded against."""
    est = gcp.estimate(100, 60)
    assert est.deadman_minutes > est.wall_hours * 60
    assert gcp.estimate(1, 1).deadman_minutes >= 10


@pytest.mark.parametrize("kwargs", [
    {"runs": 0, "seconds_per_run": 60},
    {"runs": 10, "seconds_per_run": 0},
    {"runs": 10, "seconds_per_run": 60, "parallel": 0},
])
def test_degenerate_estimates_are_refused(kwargs):
    with pytest.raises(ValueError):
        gcp.estimate(**kwargs)


def test_poweroff_is_chained_so_a_failed_job_still_shuts_down():
    """`&&` would leave a crashed job billing until someone noticed."""
    import re

    script = gcp._remote_script("false", 30, "gs://b")
    assert re.search(r";\s*\\?\s*sudo poweroff", script), "poweroff not chained with ';'"
    assert not re.search(r"&&\s*\\?\s*sudo poweroff", script), "poweroff gated on success"


def test_deadman_is_armed_before_the_job_starts():
    """A deadman armed after the job cannot survive the job hanging."""
    script = gcp._remote_script("python train.py", 30, "gs://b")
    assert script.index("shutdown -P") < script.index("python train.py")


def test_artifacts_sync_during_the_run_not_only_at_the_end():
    """The VM disk is a cache. A killed run must still leave its results."""
    script = gcp._remote_script("python train.py", 30, "gs://bucket")
    assert "while true" in script
    assert script.count("gs://bucket/artifacts") >= 2  # periodic and final


def test_down_deletes_rather_than_stops():
    """A stopped VM still bills for its disk."""
    import inspect

    source = inspect.getsource(gcp.cmd_down)
    assert "instances\", \"delete\"" in source or "'instances', 'delete'" in source
    assert "stop" not in source.lower().replace("stopping", "")
