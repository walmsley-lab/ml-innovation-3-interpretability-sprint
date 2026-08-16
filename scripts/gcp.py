#!/usr/bin/env python3
"""gcp.py — run this project's experiments on a GPU VM without leaving it burning money.

    python scripts/gcp.py plan   --runs 240 --seconds-per-run 90   dry run, launches nothing
    python scripts/gcp.py up     --yes                             create/reuse a GPU VM
    python scripts/gcp.py submit --yes -- python scripts/x.py      sync, run detached, auto-poweroff
    python scripts/gcp.py logs                                     tail the running job
    python scripts/gcp.py status                                   am I being billed right now?
    python scripts/gcp.py fetch  artifacts/                        copy results back
    python scripts/gcp.py down   --yes                             DELETE the VM and its disk

Needs only ``gcloud``, authenticated, with a billing-enabled project set.

Everything in here is something that went wrong once, in the predecessor
project. It is carried forward deliberately.

Bill protection is three layers, not one
----------------------------------------
A hung job bills until you notice; assume you will not notice.

    (a) A deadman timer armed BEFORE the job starts, so it survives the job
        hanging, the job crashing, or this orchestrator dying.
    (b) Poweroff chained with ``;`` and not ``&&``, so a FAILED job still
        shuts the machine down.
    (c) ``status``, run afterwards, by hand.

``down`` deletes rather than stops. A stopped VM still bills for its disk,
roughly $10/month for a 100 GB boot disk.

A fresh project has zero GPU quota
----------------------------------
The failure reads like a capacity error. ``up`` prints the exact quota
request when it sees one.

L4 capacity is scarce
---------------------
``ZONE_RESOURCE_POOL_EXHAUSTED`` is routine rather than exceptional, so
``up`` walks a candidate zone list instead of failing on the first zone.

Durable intermediate artifacts
------------------------------
The VM disk is a cache, never the source of truth. ``submit`` starts a
background sync that pushes artifacts to GCS while the job runs, so a
preempted, killed, or crashed run leaves behind everything it had finished.
The predecessor project lost work to a fetch-before-delete workflow; this
does not repeat it.

Nothing here launches a paid resource without ``--yes``, and every command
that would spend money prints its estimate first.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --- Configuration ---------------------------------------------------------

VM_NAME = "dsi-gpu"
NETWORK = "research-net"
MACHINE_TYPE = "g2-standard-4"  # one NVIDIA L4
DISK_SIZE_GB = 100
IMAGE_FAMILY = "ubuntu-2404-lts-amd64"
IMAGE_PROJECT = "ubuntu-os-cloud"

CANDIDATE_ZONES = (
    "us-west1-a", "us-west1-b", "us-west1-c",
    "us-east1-b", "us-east1-c", "us-east1-d",
    "us-central1-a", "us-central1-b", "us-central1-c",
)

# $/hour, list-price approximations for us regions. These are for situational
# awareness; the authoritative figure is the billing console.
RATES = {"g2-standard-4": 0.85, "g2-standard-4-spot": 0.29}
DISK_USD_PER_GB_HOUR = 0.10 / 730

STATE_DIR = Path(".gcp")
ZONE_FILE = STATE_DIR / "zone"
BUCKET_FILE = STATE_DIR / "bucket"

DEADMAN_MARGIN = 2.0
"""Deadman timer as a multiple of the estimated runtime.

Too tight kills real work; too loose is the thing being protected against.
Twice the estimate has survived both in practice.
"""

SYNC_INTERVAL_SECONDS = 120


# --- Cost ------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    runs: int
    seconds_per_run: float
    parallel: int
    machine: str
    spot: bool
    wall_hours: float
    compute_usd: float
    disk_usd: float

    @property
    def total_usd(self) -> float:
        return self.compute_usd + self.disk_usd

    @property
    def deadman_minutes(self) -> int:
        return max(10, int(self.wall_hours * 60 * DEADMAN_MARGIN))

    def render(self) -> str:
        rate = RATES.get(self.machine + ("-spot" if self.spot else ""), 0.9)
        return "\n".join([
            f"runs                {self.runs}",
            f"seconds per run     {self.seconds_per_run:,.0f}",
            f"parallel workers    {self.parallel}",
            f"machine             {self.machine}{' (spot)' if self.spot else ''}",
            f"rate                ${rate:.2f}/hour",
            f"projected wall      {self.wall_hours:.2f} hours",
            f"compute             ${self.compute_usd:,.2f}",
            f"disk ({DISK_SIZE_GB} GB)      ${self.disk_usd:,.2f}",
            f"ESTIMATED TOTAL     ${self.total_usd:,.2f}",
            f"deadman timer       {self.deadman_minutes} minutes",
        ])


def estimate(runs: int, seconds_per_run: float, *, parallel: int = 1,
             machine: str = MACHINE_TYPE, spot: bool = False) -> Estimate:
    if runs < 1 or seconds_per_run <= 0 or parallel < 1:
        raise ValueError("runs, seconds_per_run and parallel must all be positive")
    rate = RATES.get(machine + ("-spot" if spot else ""), RATES.get(machine, 0.9))
    wall_hours = runs * seconds_per_run / parallel / 3600.0
    return Estimate(
        runs=runs, seconds_per_run=seconds_per_run, parallel=parallel,
        machine=machine, spot=spot, wall_hours=wall_hours,
        compute_usd=wall_hours * rate,
        disk_usd=wall_hours * DISK_SIZE_GB * DISK_USD_PER_GB_HOUR,
    )


# --- gcloud plumbing -------------------------------------------------------


def gcloud(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gcloud", *args], check=check, text=True,
        capture_output=capture,
    )


def project() -> str:
    result = gcloud("config", "get-value", "project", check=False)
    name = (result.stdout or "").strip()
    if not name or name == "(unset)":
        sys.exit("No project set. Run: gcloud config set project YOUR_PROJECT")
    return name


def zone() -> str:
    if not ZONE_FILE.exists():
        sys.exit("No VM recorded. Run `up` first.")
    return ZONE_FILE.read_text().strip()


def bucket() -> str:
    if BUCKET_FILE.exists():
        return BUCKET_FILE.read_text().strip()
    name = f"gs://{project()}-dsi-artifacts"
    STATE_DIR.mkdir(exist_ok=True)
    BUCKET_FILE.write_text(name + "\n")
    return name


def instance_exists() -> bool:
    result = gcloud("compute", "instances", "describe", VM_NAME,
                    "--zone", zone(), "--format", "value(name)", check=False)
    return result.returncode == 0


# --- Commands --------------------------------------------------------------


def cmd_plan(args) -> None:
    """Dry run. Prints what it would cost and launches nothing."""
    est = estimate(args.runs, args.seconds_per_run,
                   parallel=args.parallel, spot=args.spot)
    print("--- dry run: nothing is launched by this command ---\n")
    print(est.render())
    if args.max_usd and est.total_usd > args.max_usd:
        print(f"\nOVER BUDGET: ${est.total_usd:,.2f} exceeds the "
              f"${args.max_usd:,.2f} ceiling. `submit` will refuse this.")
        sys.exit(1)
    if args.max_usd:
        print(f"\nwithin the ${args.max_usd:,.2f} ceiling")
    print("\nTo run it:  python scripts/gcp.py up --yes"
          "  &&  python scripts/gcp.py submit --yes -- <command>")


def cmd_up(args) -> None:
    """Create or reuse a GPU VM, walking zones for L4 capacity."""
    proj = project()
    STATE_DIR.mkdir(exist_ok=True)

    if ZONE_FILE.exists() and instance_exists():
        print(f"Reusing existing {VM_NAME} in {zone()}")
        return

    print(f"project {proj}\nmachine {MACHINE_TYPE} (NVIDIA L4)\n")
    if not args.yes:
        sys.exit("Refusing to create a billable VM without --yes.")

    for candidate in CANDIDATE_ZONES:
        print(f"trying {candidate} ...")
        result = gcloud(
            "compute", "instances", "create", VM_NAME,
            "--project", proj, "--zone", candidate,
            "--machine-type", MACHINE_TYPE,
            "--maintenance-policy", "TERMINATE", "--restart-on-failure",
            "--image-family", IMAGE_FAMILY, "--image-project", IMAGE_PROJECT,
            "--boot-disk-size", f"{DISK_SIZE_GB}GB", "--boot-disk-type", "pd-balanced",
            "--scopes", "https://www.googleapis.com/auth/cloud-platform",
            *(["--provisioning-model", "SPOT"] if args.spot else []),
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            ZONE_FILE.write_text(candidate + "\n")
            print(f"\ncreated {VM_NAME} in {candidate}")
            print("Driver install runs on first `submit`.")
            return

        print(output.strip()[:400])
        if "ZONE_RESOURCE_POOL_EXHAUSTED" in output:
            print(f"no L4 capacity in {candidate}; next zone")
            continue
        if "quota" in output.lower():
            sys.exit(
                "\nThis is a QUOTA error, which reads like a capacity error but is not.\n"
                "A fresh project has zero GPU quota. Request it here:\n"
                f"  https://console.cloud.google.com/iam-admin/quotas?project={proj}\n"
                "  Filter: 'NVIDIA_L4_GPUS'. Request at least 1 in one of:\n"
                f"  {', '.join(sorted({z.rsplit('-', 1)[0] for z in CANDIDATE_ZONES}))}"
            )
    sys.exit("No L4 capacity in any candidate zone. Try again later or add zones.")


def _remote_script(command: str, deadman_minutes: int, gcs: str) -> str:
    """The wrapper that actually runs on the VM.

    Order matters. The deadman is armed first so that it survives everything
    after it, and the poweroff is chained with `;` so a failing job still
    shuts the machine down.
    """
    return f"""set -x
# (a) deadman armed BEFORE the job, so a hang cannot outlive it
sudo shutdown -P +{deadman_minutes} || true

cd ~/dsi
# durable intermediate artifacts: push while the job runs, not after it
( while true; do
    gcloud storage rsync -r artifacts {gcs}/artifacts >/dev/null 2>&1 || true
    sleep {SYNC_INTERVAL_SECONDS}
  done ) &
SYNC_PID=$!

# (b) poweroff chained with ';' not '&&' — a FAILED job still shuts down
nohup bash -c '{command} ; \
  kill '"$SYNC_PID"' 2>/dev/null ; \
  gcloud storage rsync -r ~/dsi/artifacts {gcs}/artifacts ; \
  sudo poweroff' > ~/dsi/job.log 2>&1 &
echo "job started; deadman set for {deadman_minutes} minutes"
"""


def cmd_submit(args) -> None:
    """Sync code, arm the deadman, run detached, stream artifacts to GCS."""
    command = " ".join(args.command)
    if not command:
        sys.exit("Nothing to run. Pass the command after `--`.")

    est = estimate(args.runs, args.seconds_per_run,
                   parallel=args.parallel, spot=args.spot)
    print(est.render())
    if args.max_usd and est.total_usd > args.max_usd:
        sys.exit(f"\nREFUSING: ${est.total_usd:,.2f} exceeds the "
                 f"${args.max_usd:,.2f} ceiling.")
    if not args.yes:
        sys.exit("\nRefusing to spend without --yes. This estimate is a dry run.")

    proj, z, gcs = project(), zone(), bucket()
    gcloud("storage", "buckets", "create", gcs, "--project", proj, check=False)

    print(f"\nsyncing code to {VM_NAME}:{z}")
    gcloud("compute", "scp", "--recurse", "--zone", z,
           "src", "scripts", "pyproject.toml", f"{VM_NAME}:~/dsi/",
           capture=False)

    script = _remote_script(command, est.deadman_minutes, gcs)
    gcloud("compute", "ssh", VM_NAME, "--zone", z, "--command", script, capture=False)
    print(f"\nartifacts stream to {gcs}/artifacts every {SYNC_INTERVAL_SECONDS}s")
    print("Check billing yourself afterwards:  python scripts/gcp.py status")


def cmd_logs(args) -> None:
    gcloud("compute", "ssh", VM_NAME, "--zone", zone(),
           "--command", "tail -f ~/dsi/job.log", capture=False)


def cmd_status(args) -> None:
    """Am I being billed right now?"""
    proj = project()
    result = gcloud("compute", "instances", "list", "--project", proj,
                    "--format", "json", check=False)
    instances = json.loads(result.stdout or "[]")
    if not instances:
        print(f"No instances in {proj}. Not being billed for compute.")
        return
    print(f"{'instance':20s} {'type':18s} {'status':12s} {'billing'}")
    for item in instances:
        machine = item["machineType"].rsplit("/", 1)[-1]
        spot = item.get("scheduling", {}).get("provisioningModel") == "SPOT"
        rate = RATES.get(machine + ("-spot" if spot else ""), RATES.get(machine, 0.9))
        running = item["status"] == "RUNNING"
        note = f"${rate:.2f}/hr" if running else f"disk only (~${DISK_SIZE_GB * DISK_USD_PER_GB_HOUR:.3f}/hr)"
        print(f"{item['name']:20s} {machine:18s} {item['status']:12s} {note}")
    print("\nA STOPPED instance still bills for its disk. Use `down` to delete it.")


def cmd_fetch(args) -> None:
    """Pull artifacts from GCS, which is the source of truth, not the VM."""
    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    gcloud("storage", "rsync", "-r", f"{bucket()}/artifacts", str(destination),
           capture=False)
    print(f"fetched {bucket()}/artifacts -> {destination}")


def cmd_down(args) -> None:
    """DELETE the VM and its disk. Stopping is not enough; the disk still bills."""
    if not args.yes:
        sys.exit("Refusing to delete without --yes.")
    z = zone()
    gcloud("compute", "instances", "delete", VM_NAME, "--zone", z, "--quiet",
           check=False, capture=False)
    ZONE_FILE.unlink(missing_ok=True)
    print(f"deleted {VM_NAME}. Artifacts remain in {bucket()}.")
    print("Verify for yourself:  python scripts/gcp.py status")


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cost_args(sub):
        sub.add_argument("--runs", type=int, default=1)
        sub.add_argument("--seconds-per-run", type=float, default=60.0)
        sub.add_argument("--parallel", type=int, default=1)
        sub.add_argument("--spot", action="store_true")
        sub.add_argument("--max-usd", type=float, default=75.0)

    plan = subparsers.add_parser("plan", help="dry run; launches nothing")
    add_cost_args(plan)
    plan.set_defaults(func=cmd_plan)

    up = subparsers.add_parser("up", help="create or reuse a GPU VM")
    up.add_argument("--yes", action="store_true")
    up.add_argument("--spot", action="store_true")
    up.set_defaults(func=cmd_up)

    submit = subparsers.add_parser("submit", help="run a command on the VM")
    add_cost_args(submit)
    submit.add_argument("--yes", action="store_true")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    submit.set_defaults(func=cmd_submit)

    subparsers.add_parser("logs", help="tail the running job").set_defaults(func=cmd_logs)
    subparsers.add_parser("status", help="am I being billed?").set_defaults(func=cmd_status)

    fetch = subparsers.add_parser("fetch", help="copy artifacts back")
    fetch.add_argument("dest", nargs="?", default="artifacts")
    fetch.set_defaults(func=cmd_fetch)

    down = subparsers.add_parser("down", help="DELETE the VM and its disk")
    down.add_argument("--yes", action="store_true")
    down.set_defaults(func=cmd_down)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
