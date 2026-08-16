#!/usr/bin/env python3
"""gcp.py — run GPU jobs on GCP without leaving a VM burning money.

    python gcp.py plan                  who/what/where/cost, launches nothing
    python gcp.py up                    create/reuse a GPU VM, install driver
    python gcp.py run "python train.py" sync code, run detached, auto-poweroff
    python gcp.py logs                  tail the running job
    python gcp.py status                am I being billed right now?
    python gcp.py fetch out             copy results back
    python gcp.py down                  DELETE the VM and its disk

Needs only `gcloud`, authenticated, with a billing-enabled project set:
    gcloud config set project YOUR_PROJECT

`run_pool` at the bottom is importable and meant to run ON the VM, for
fanning out many jobs across one box.

Everything below is something that went wrong once.

BILL PROTECTION IS THREE LAYERS, NOT ONE. A hung job bills until you
notice; assume you will not notice. (a) A deadman timer armed BEFORE the
job starts — survives the job hanging, crashing, or the orchestrator
dying. (b) Poweroff chained with `;` not `&&`, so a FAILED job still shuts
the machine down. (c) `status` afterwards, by hand. Note that a stopped VM
still bills for its disk, ~$10/month for a 100GB boot disk, so `down`
deletes rather than stops.

A FRESH PROJECT HAS ZERO GPU QUOTA, and the failure reads like a capacity
error. `up` prints the exact quota request when it sees one.

L4 CAPACITY IS SCARCE. ZONE_RESOURCE_POOL_EXHAUSTED is the normal
response, not the exception, so `up` walks nine zones and caches the one
that worked. GPU VMs also require --maintenance-policy=TERMINATE; they
cannot live migrate and creation fails without it.

THE DRIVER NEEDS A REBOOT, AND STARTUP SCRIPTS RUN ON EVERY BOOT. Install,
touch a sentinel, reboot once, exit early forever after. Without the
sentinel you get a reboot loop. Budget 3-5 minutes and poll for
nvidia-smi rather than sleeping a fixed amount.

OOM: GATE, STAGGER, RETRY. Peak RSS while loading data is far above
steady-state training. Three 5GB corpus loads landed at once on a 15GB
host, the kernel killed one, and the pool treated that non-zero exit as
fatal and took the healthy runs down too. run_pool gates on MemAvailable,
staggers launches so only one process is in its allocation peak, and
retries once — an OOM kill is an environmental death, not a bug.

PGREP MATCHES THE COMMAND YOU ARE TYPING. `pgrep -f train.py` matches the
SSH command containing that string; this deadlocked a wait loop for an
hour. Bracket the first character: `pgrep -f "[t]rain.py"`.

SSH DROPS KILL YOUR JOB. Run detached under tmux and tee to a file. You
will use the log file far more than you reattach.

FETCH BEFORE SHUTDOWN. The poweroff means anything not on disk and copied
off is gone. `run` leaves results on the VM's disk, so `fetch` before
`down`, or write to GCS as you go.

NEVER LAUNCH WITHOUT PRICING IT FIRST. `plan` is the only command that
spends nothing: it names the account and project being billed, the machine
and zone, and the estimated cost. Run it, read it, and get a human's
agreement before `up` or `run`.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


VM = os.environ.get("VM", "gpu")
MACHINE = os.environ.get("MACHINE", "g2-standard-4")   # 1x NVIDIA L4
DISK = os.environ.get("DISK", "100GB")
REMOTE = os.environ.get("REMOTE", "work")
MAXHOURS = int(os.environ.get("MAXHOURS", "8"))
ZONE_CACHE = Path(".gcp-zone")

ZONES = [
    "us-west1-a",
    "us-west1-b",
    "us-west1-c",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
]

# $/hour list-price approximations, us regions, on-demand. For situational
# awareness only; the billing console is authoritative.
RATES = {"g2-standard-4": 0.85}
DISK_USD_PER_GB_HOUR = 0.10 / 730


DRIVER_STARTUP = r"""#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /var/log/driver-startup.log | logger -t driver) 2>&1

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "driver already up"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    nvidia-driver-580 \
    git \
    curl \
    tmux \
    htop \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    rsync

mkdir -p /var/lib/bootstrap

if [[ ! -f /var/lib/bootstrap/rebooted ]]; then
    touch /var/lib/bootstrap/rebooted
    echo "driver installed, rebooting once"
    shutdown -r now
fi
"""


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def project():
    p = os.environ.get("PROJECT") or sh(
        "gcloud config get-value project",
        capture=True,
    ).strip()

    if not p or p == "(unset)":
        die("no project. run: gcloud config set project YOUR_PROJECT")

    return p


def sh(cmd, capture=False, check=True):
    r = subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )

    if check and r.returncode != 0 and not capture:
        sys.exit(r.returncode)

    return r.stdout if capture else ""


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def zone():
    z = (
        os.environ.get("ZONE")
        or (
            ZONE_CACHE.read_text().strip()
            if ZONE_CACHE.exists()
            else ""
        )
    )

    return z or die("no zone known — run `python gcp.py up` first")


def ssh(cmd, zone_=None, check=True):
    return sh(
        f"gcloud compute ssh {VM} "
        f"--project={project()} "
        f"--zone={zone_ or zone()} "
        f"--quiet "
        f"--command={quote(cmd)}",
        check=check,
    )


def quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def estimate(hours):
    """Cost of holding the VM for `hours`. Pure arithmetic, spends nothing."""

    hours = float(hours)

    if hours <= 0:
        die("hours must be positive")

    rate = RATES.get(MACHINE, 0.9)
    disk_gb = float(DISK.rstrip("GB") or 0)

    compute = hours * rate
    disk = hours * disk_gb * DISK_USD_PER_GB_HOUR

    return {
        "hours": hours,
        "rate": rate,
        "compute_usd": compute,
        "disk_usd": disk,
        "total_usd": compute + disk,
    }


def plan(hours="1"):
    """Pre-launch check. Names who is billed, for what, where, and how much.

    The only command that spends nothing. Run it, read it, and get a
    human's agreement before `up` or `run`.
    """

    proj = project()
    account = sh("gcloud config get-value account", capture=True).strip()

    cached = (
        os.environ.get("ZONE")
        or (
            ZONE_CACHE.read_text().strip()
            if ZONE_CACHE.exists()
            else ""
        )
    )

    est = estimate(hours)

    print("--- dry run: this command launches nothing ---\n")
    print(f"account         {account}")
    print(f"project         {proj}")
    print(f"machine         {MACHINE} (1x NVIDIA L4), {DISK} pd-balanced")
    print(
        f"zone            {cached or 'none cached; up will walk ' + str(len(ZONES)) + ' zones'}"
    )
    print(f"deadman         {MAXHOURS}h")
    print()
    print(f"assumed runtime {est['hours']:.2f} hours at ${est['rate']:.2f}/hour")
    print(f"compute         ${est['compute_usd']:,.2f}")
    print(f"disk            ${est['disk_usd']:,.2f}")
    print(f"ESTIMATED TOTAL ${est['total_usd']:,.2f}")
    print()
    print("Nothing has been created. Get agreement, then: python gcp.py up")


def up():
    """Create or reuse a GPU VM and wait until nvidia-smi works."""

    proj = project()

    startup = Path("/tmp/gcp_driver.sh")
    startup.write_text(DRIVER_STARTUP)

    selected = ""

    cached = (
        os.environ.get("ZONE")
        or (
            ZONE_CACHE.read_text().strip()
            if ZONE_CACHE.exists()
            else ""
        )
    )

    if cached:
        r = subprocess.run(
            f"gcloud compute instances describe {VM} "
            f"--zone={cached} "
            f"--project={proj}",
            shell=True,
            capture_output=True,
        )

        if r.returncode == 0:
            selected = cached
            print(f"==> reusing {VM} in {cached}")

    for z in ([] if selected else ZONES):
        print(f"==> trying {MACHINE} in {z}")

        out = sh(
            f"gcloud compute instances create {VM} "
            f"--project={proj} "
            f"--zone={z} "
            f"--machine-type={MACHINE} "
            f"--maintenance-policy=TERMINATE "
            f"--image-family=ubuntu-2404-lts-amd64 "
            f"--image-project=ubuntu-os-cloud "
            f"--boot-disk-size={DISK} "
            f"--boot-disk-type=pd-balanced "
            f"--metadata-from-file=startup-script={startup}",
            capture=True,
            check=False,
        )

        print(out.strip()[:400])

        if "RUNNING" in out or "Created" in out:
            selected = z
            ZONE_CACHE.write_text(z)
            break

        if "quota" in out.lower():
            print(
                f"\nA fresh project has 0 GPU quota. Request it:\n"
                f"  gcloud beta quotas preferences create "
                f"--project={proj} "
                f"--billing-project={proj} \\\n"
                f"    --service=compute.googleapis.com \\\n"
                f"    --quota-id=GPUS-ALL-REGIONS-per-project "
                f"--preferred-value=1 \\\n"
                f"    --email=$(gcloud config get-value account) \\\n"
                f"    --justification='ML research' "
                f"--preference-id=gpu1\n"
            )

    if not selected:
        die("no capacity in any zone (and/or no GPU quota)")

    print("==> waiting for nvidia-smi (driver installs + reboots once)")

    for _ in range(120):
        r = subprocess.run(
            f"gcloud compute ssh {VM} "
            f"--project={proj} "
            f"--zone={selected} "
            f"--quiet "
            f"--command='nvidia-smi'",
            shell=True,
            capture_output=True,
        )

        if r.returncode == 0:
            print(f"==> GPU ready in {selected}")
            return

        time.sleep(5)

    die(
        f"VM up but nvidia-smi never succeeded. Check:\n"
        f"  gcloud compute ssh {VM} "
        f"--zone={selected} "
        f"--command='sudo cat /var/log/driver-startup.log'"
    )


def run(cmd):
    """Sync code, arm the deadman timer, run detached, poweroff after."""

    proj, z = project(), zone()

    print(f"==> syncing to {VM}:~/{REMOTE}")

    ssh(f"mkdir -p ~/{REMOTE}", z)

    sh(
        f"gcloud compute scp "
        f"--project={proj} "
        f"--zone={z} "
        f"--quiet "
        f"--recurse "
        f"--compress "
        f"./ {VM}:~/{REMOTE}/",
        check=False,
    )

    # Armed BEFORE the job, so a hang cannot outlive it.
    print(f"==> arming {MAXHOURS}h deadman timer")

    ssh(
        f"sudo shutdown -c 2>/dev/null || true; "
        f"sudo shutdown -h +{MAXHOURS * 60} "
        f"deadman 2>/dev/null || true",
        z,
    )

    # `;` not `&&` — a failed job must still power the machine off.
    print(f"==> launching: {cmd}")

    inner = (
        f"set -o pipefail; "
        f"{cmd} 2>&1 | tee -a run.log; "
        f"echo EXIT=$? >> run.log; "
        f"sudo poweroff"
    )

    ssh(
        f"cd ~/{REMOTE} && "
        f"tmux new-session -d -s job {quote(inner)}",
        z,
    )

    print(
        f"""
==> detached. next:
  python gcp.py logs                 # tail
  python gcp.py status               # still billing?
  python gcp.py fetch out            # results (BEFORE `down`)
  python gcp.py down                 # delete VM + disk
"""
    )


def logs():
    ssh(
        f"tail -f ~/{REMOTE}/run.log",
        check=False,
    )


def status():
    print(
        sh(
            f"gcloud compute instances list "
            f"--project={project()} "
            f"--format='table(name,zone,status,machineType)'",
            capture=True,
        )
    )

    print(
        "TERMINATED = not billing compute "
        "(disk still bills ~$10/mo/100GB)"
    )


def fetch(remote_path="out", local="."):
    sh(
        f"gcloud compute scp "
        f"--project={project()} "
        f"--zone={zone()} "
        f"--quiet "
        f"--recurse "
        f"{VM}:~/{REMOTE}/{remote_path} "
        f"{local}",
        check=False,
    )


def down():
    z = zone()

    sh(
        f"gcloud compute instances delete {VM} "
        f"--project={project()} "
        f"--zone={z} "
        f"--quiet",
        check=False,
    )

    ZONE_CACHE.unlink(missing_ok=True)

    status()


# ---------------------------------------------------------------------------
# on-VM: many jobs, one box
# ---------------------------------------------------------------------------

MIN_AVAIL_GB = 7
STAGGER_SEC = 90
POLL_SEC = 60
MAX_WAIT_SEC = 1800


def mem_available_gb():
    """Free RAM in GB; infinity off Linux, which disables the gate."""

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass

    return float("inf")


def run_pool(cmds_with_env, parallel=2):
    """Run commands at most `parallel` at a time, memory-gated.

    `cmds_with_env` is [(argv_list, env_dict_or_None), ...].

    Each command is retried once on failure; a second failure lets
    active jobs finish and then exits non-zero rather than orphaning them.

    Waits on the Popen handles it created. It never matches process names,
    because `pgrep -f` matches the command you are typing.
    """

    pending = [(c, e, 0) for c, e in cmds_with_env]

    active = []
    failed = []

    while pending or active:

        while pending and len(active) < parallel and not failed:

            waited = 0

            while (
                mem_available_gb() < MIN_AVAIL_GB
                and waited < MAX_WAIT_SEC
            ):
                time.sleep(30)
                waited += 30

            cmd, extra, tries = pending.pop(0)

            print(
                "+",
                " ".join(cmd),
                f"(retry {tries})" if tries else "",
                flush=True,
            )

            active.append(
                (
                    subprocess.Popen(
                        cmd,
                        env={
                            **os.environ,
                            **(extra or {}),
                        },
                    ),
                    cmd,
                    extra,
                    tries,
                )
            )

            time.sleep(STAGGER_SEC)

        done = [
            t
            for t in active
            if t[0].poll() is not None
        ]

        active = [
            t
            for t in active
            if t[0].poll() is None
        ]

        for proc, cmd, extra, tries in done:

            if proc.returncode != 0:
                print(
                    f"EXIT {proc.returncode}: {' '.join(cmd)}",
                    flush=True,
                )

                if tries == 0:
                    pending.insert(
                        0,
                        (cmd, extra, 1),
                    )
                else:
                    failed.append(cmd)

        if active:
            time.sleep(POLL_SEC)

    if failed:
        sys.exit(
            "failed twice: "
            + "; ".join(
                " ".join(c)
                for c in failed
            )
        )


def gpu_env(i, gpus):
    """Pin job i to a GPU, round-robin.

    `gpus` is e.g. ['0', '1'].
    """

    return (
        {"CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]}
        if gpus
        else None
    )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd, rest = args[0], args[1:]

    if cmd == "plan":
        plan(*(rest or ["1"]))

    elif cmd == "up":
        up()

    elif cmd == "run":
        run(
            rest[0]
            if rest
            else die(
                'usage: gcp.py run "python train.py"'
            )
        )

    elif cmd == "logs":
        logs()

    elif cmd == "status":
        status()

    elif cmd == "fetch":
        fetch(*(rest or ["out"]))

    elif cmd == "down":
        down()

    else:
        die(
            f"unknown command {cmd!r}; try --help"
        )
