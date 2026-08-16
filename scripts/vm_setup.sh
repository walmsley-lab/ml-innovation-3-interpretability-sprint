#!/usr/bin/env bash
# Prepare the VM for GPU runs. Idempotent; safe to re-run.
set -euo pipefail
cd ~/work

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi

# jax[cuda12] ships its own CUDA libraries via pip and needs only the host
# driver, which the instance startup script installs. The project's own
# pyproject pins plain CPU jax, so install it with --no-deps afterwards or
# pip will pull the CPU wheel back over the GPU one.
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q "jax[cuda12]" equinox optax polars scipy
.venv/bin/pip install -q -e . --no-deps

.venv/bin/python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"
