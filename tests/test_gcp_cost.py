"""Invariants of the standalone GPU runner, and its cost arithmetic.

The runner is carried over from prior projects largely unchanged; these
tests pin the properties that were learned the expensive way, so a future
edit cannot quietly undo one.

Nothing here shells out to gcloud, and no test in this file can create a
billable resource.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dsi_gcp", Path(__file__).parent.parent / "scripts" / "gcp.py"
)
gcp = importlib.util.module_from_spec(spec)
sys.modules["dsi_gcp"] = gcp
spec.loader.exec_module(gcp)


# --- Cost arithmetic ------------------------------------------------------


def test_cost_scales_linearly_with_hours():
    assert gcp.estimate(2)["total_usd"] == pytest.approx(2 * gcp.estimate(1)["total_usd"])


def test_total_includes_disk_as_well_as_compute():
    est = gcp.estimate(1)
    assert est["total_usd"] == pytest.approx(est["compute_usd"] + est["disk_usd"])
    assert est["disk_usd"] > 0


def test_plan_spends_nothing():
    """`plan` must not invoke any state-changing gcloud verb."""
    source = inspect.getsource(gcp.plan)
    for verb in ("instances create", "instances delete", "compute ssh", "compute scp"):
        assert verb not in source, verb


# --- Bill protection, three layers ----------------------------------------


def test_deadman_is_armed_before_the_job_is_launched():
    """A deadman armed after the job cannot survive the job hanging."""
    source = inspect.getsource(gcp.run)
    assert source.index("deadman") < source.index("tmux new-session")


def test_poweroff_is_chained_so_a_failed_job_still_shuts_down():
    """`&&` would leave a crashed job billing until someone noticed."""
    source = inspect.getsource(gcp.run)
    assert re.search(r";\s*\"?\s*f?\"?\s*sudo poweroff", source)
    assert "&& sudo poweroff" not in source


def test_down_deletes_rather_than_stopping():
    """A stopped VM still bills for its disk."""
    source = inspect.getsource(gcp.down)
    assert "instances delete" in source
    assert "instances stop" not in source


def test_down_reports_status_afterwards():
    """Layer (c): the manual check is wired in rather than remembered."""
    assert "status()" in inspect.getsource(gcp.down)


# --- Capacity, quota, and driver ------------------------------------------


def test_zone_fallback_covers_several_regions():
    """ZONE_RESOURCE_POOL_EXHAUSTED is the normal response for L4, not the exception."""
    assert len(gcp.ZONES) >= 9
    assert len({z.rsplit("-", 1)[0] for z in gcp.ZONES}) >= 3


def test_gpu_vm_requires_terminate_maintenance_policy():
    """GPU VMs cannot live migrate; creation fails without this."""
    assert "--maintenance-policy=TERMINATE" in inspect.getsource(gcp.up)


def test_quota_error_is_distinguished_from_capacity_error():
    """A fresh project has zero GPU quota and the failure reads like capacity."""
    source = inspect.getsource(gcp.up)
    assert "quota" in source.lower()
    assert "quotas preferences create" in source


def test_driver_startup_has_a_reboot_sentinel():
    """Startup scripts run on every boot; without a sentinel this loops."""
    assert "/var/lib/bootstrap/rebooted" in gcp.DRIVER_STARTUP
    assert "shutdown -r now" in gcp.DRIVER_STARTUP
    assert gcp.DRIVER_STARTUP.index("nvidia-smi") < gcp.DRIVER_STARTUP.index("apt-get")


def test_up_polls_for_nvidia_smi_rather_than_sleeping_a_fixed_time():
    assert "nvidia-smi" in inspect.getsource(gcp.up)


def test_detached_execution_survives_an_ssh_drop():
    source = inspect.getsource(gcp.run)
    assert "tmux new-session -d" in source
    assert "tee -a run.log" in source


# --- Pool: gate, stagger, retry, pin ---------------------------------------


def test_pool_gates_on_available_memory():
    """Peak RSS while loading data is far above steady-state training."""
    assert "mem_available_gb" in inspect.getsource(gcp.run_pool)
    assert gcp.MIN_AVAIL_GB > 0


def test_pool_staggers_launches():
    assert gcp.STAGGER_SEC > 0
    assert "STAGGER_SEC" in inspect.getsource(gcp.run_pool)


def test_pool_retries_once_because_an_oom_kill_is_not_a_bug():
    source = inspect.getsource(gcp.run_pool)
    assert "tries == 0" in source
    assert "failed twice" in source


def test_pool_waits_on_handles_and_never_matches_process_names():
    """`pgrep -f` matches the command you are typing; this deadlocked once."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(gcp.run_pool)))
    function = tree.body[0]
    if ast.get_docstring(function):  # the docstring cites pgrep on purpose
        function.body = function.body[1:]
    body = ast.unparse(function)
    assert ".poll()" in body
    assert "pgrep" not in body


def test_gpu_pinning_is_round_robin():
    assert gcp.gpu_env(0, ["0", "1"]) == {"CUDA_VISIBLE_DEVICES": "0"}
    assert gcp.gpu_env(1, ["0", "1"]) == {"CUDA_VISIBLE_DEVICES": "1"}
    assert gcp.gpu_env(2, ["0", "1"]) == {"CUDA_VISIBLE_DEVICES": "0"}
    assert gcp.gpu_env(0, []) is None


def test_memory_gate_disables_itself_off_linux():
    """No /proc/meminfo must not mean 'zero memory available'."""
    assert gcp.mem_available_gb() > 0


# --- Defaults -------------------------------------------------------------


def test_l4_machine_is_the_default():
    assert gcp.MACHINE == "g2-standard-4"
