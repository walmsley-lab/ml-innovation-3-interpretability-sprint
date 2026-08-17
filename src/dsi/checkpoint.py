"""Model persistence.

No weights have ever been saved in this project. Every result to date is a
JSON/parquet record of metrics, which is sufficient for a gate and
insufficient for anything that needs to *resume from* or *look inside* a
trained model. Both the V2 ablation programme and the sprint latent-state
experiment need exactly that, so persistence is the shared prerequisite.

The contract is deliberately narrow:

* a checkpoint is ``(ModelConfig, weights)`` and nothing else — no optimizer
  state, no RNG, no metrics. Anything else belongs in the artifact record,
  which is already content-addressed;
* the config is stored as JSON beside the weights, so a checkpoint can be
  loaded without the caller already knowing the architecture;
* ``save`` then ``load`` must round-trip to bitwise-identical leaves. This is
  preflight check P6 and it is asserted in the tests, not assumed.

Equinox's ``tree_serialise_leaves`` is used rather than Orbax: there is no
sharding, no async, and no partial-restore requirement here, and a 3M
parameter model is a few megabytes. Orbax becomes worth its dependency when
checkpoints are large or distributed, which is not this stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax

from dsi.model import ModelConfig, Transformer, init_model

__all__ = ["save_model", "load_model", "checkpoint_paths"]


def checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
    """The two files a checkpoint occupies: weights and config.

    Taking a single stem and deriving both names keeps callers from pairing a
    config with the wrong weights.
    """
    stem = Path(path)
    return stem.with_suffix(".eqx"), stem.with_suffix(".config.json")


def save_model(path: str | Path, model: Transformer, config: ModelConfig) -> tuple[Path, Path]:
    """Write ``model`` to ``path``, returning the files written.

    The parent directory is created if missing. Writes are not atomic; the
    caller owns concurrency, exactly as the existing artifact writers do.
    """
    weights_path, config_path = checkpoint_paths(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    config_path.write_text(json.dumps(asdict(config), sort_keys=True, indent=2) + "\n")
    eqx.tree_serialise_leaves(weights_path, model)
    return weights_path, config_path


def load_model(path: str | Path) -> tuple[Transformer, ModelConfig]:
    """Read back a model saved by :func:`save_model`.

    Deserialisation needs a correctly-shaped template, which is why the config
    is persisted alongside the weights: the template is rebuilt from it. The
    template's random initialization is overwritten leaf by leaf and never
    survives, so the key used here is arbitrary and does not enter any result.
    """
    weights_path, config_path = checkpoint_paths(path)
    if not weights_path.exists():
        raise FileNotFoundError(f"no checkpoint weights at {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"checkpoint at {weights_path} has no config at {config_path}; "
            "the architecture cannot be reconstructed without it"
        )

    config = ModelConfig(**json.loads(config_path.read_text()))
    template = init_model(config, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(weights_path, template), config
