"""Internal-state extraction: the ``StateSnapshot`` feature layer.

This is what the sprint latent-state experiment's ``P_int`` predictor consumes,
and what any later state-dependence or ontology work should reuse rather than
reinvent.

Basis invariance is the whole design problem
--------------------------------------------
The sprint experiment fits a predictor **across** initialization pairs, and
every pair starts from different random weights. Raw activation coordinates are
therefore meaningless across models: coordinate 37 of layer 2 encodes something
different in every init, so a ridge fit on raw activations would be fitting
noise with a plausible-looking R-squared.

Every feature here is invariant to a change of basis in the residual stream:

* **norms** — magnitudes, not directions;
* **effective rank** (participation ratio) — a spectral property;
* **example-Gram spectrum** — the eigenvalues of the *example x example*
  similarity matrix on a frozen probe set. This is the model's "view" of the
  probe inputs, and because the probe set is identical across every model, two
  models with the same view produce the same spectrum regardless of how their
  residual streams happen to be oriented;
* **attention statistics** — entropy and distance are properties of the
  pattern, not of the representation's basis.

The one thing deliberately excluded is a raw mean-activation vector. It is the
obvious feature, it would be the strongest-looking feature, and across
differently-initialized models it would be uninterpretable.

Comparability also requires the probe inputs to be fixed. :func:`frozen_probe`
builds them from a spec, so a difference in features between two models can
never be a difference in what they were shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsi.mechanism import _block_attention
from dsi.model import Transformer

__all__ = [
    "StateProbeSpec", "frozen_probe", "residual_streams",
    "extract_state_features", "feature_names",
]

TOP_EIGENVALUES = 8


@dataclass(frozen=True)
class StateProbeSpec:
    """Frozen definition of how internal state is measured.

    Every field is a researcher degree of freedom. Fixing them before outcomes
    exist is what keeps ``P_int`` from being tuned into a win, and the spec is
    hashed alongside the matching rule.
    """

    family: str = "NEUTRAL_ALIGNED"
    n_examples: int = 128
    seed: int = 0
    top_eigenvalues: int = TOP_EIGENVALUES

    def __post_init__(self) -> None:
        if self.n_examples < 8:
            raise ValueError(f"n_examples must be at least 8, got {self.n_examples}")
        if not 1 <= self.top_eigenvalues <= self.n_examples:
            raise ValueError(
                f"top_eigenvalues ({self.top_eigenvalues}) must lie in "
                f"[1, n_examples] ({self.n_examples})"
            )


def frozen_probe(spec: StateProbeSpec, config, sampler=None) -> jnp.ndarray:
    """The probe inputs, identical for every model this spec is applied to.

    ``sampler`` follows the ``train_phase`` contract, so the same spec works on
    the W/P task and on the V2 micro-world without either knowing about the
    other.
    """
    if sampler is None:
        from dsi.data import sample_batch as sampler  # noqa: PLC0415
    batch = sampler(jax.random.PRNGKey(spec.seed), spec.family, config, spec.n_examples)
    return jnp.asarray(batch["tokens"])


@eqx.filter_jit
def residual_streams(model: Transformer, tokens: jnp.ndarray) -> jnp.ndarray:
    """Residual stream after each block, plus the embedding.

    Returns ``(n_layers + 1, seq, d_model)`` for one sequence. Index 0 is the
    embedding output, index ``i + 1`` the stream after block ``i``.
    """
    x = jax.vmap(model.embed)(tokens)
    streams = [x]
    for block in model.blocks:
        _, x = _block_attention(block, x)
        streams.append(x)
    return jnp.stack(streams)


def _spectrum(activations: np.ndarray, top: int) -> tuple[np.ndarray, float]:
    """Normalized example-Gram eigenvalues and the participation ratio.

    ``activations`` is ``(n_examples, d_model)``. Centering across examples
    removes the mean direction, which is basis-dependent in the same way the
    raw activations are; what survives is the *shape* of the example cloud.

    The participation ratio ``(sum L)^2 / sum L^2`` is a soft rank: it counts
    how many directions carry appreciable variance, without a threshold.
    """
    centered = activations - activations.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular**2
    total = eigenvalues.sum()
    if total <= 0:
        return np.zeros(top), 0.0
    normalized = eigenvalues / total
    pr = float(1.0 / np.square(normalized).sum())
    padded = np.zeros(top)
    take = min(top, normalized.size)
    padded[:take] = normalized[:take]
    return padded, pr


def extract_state_features(
    model: Transformer, tokens: jnp.ndarray, spec: StateProbeSpec, answer_index: int
) -> dict[str, float]:
    """A flat, basis-invariant feature vector describing the model's state.

    ``answer_index`` selects the position summarized per layer. The answer
    position is where the behavioural measurement is taken, so it is the
    position at which internal state is most likely to carry information about
    what the model will do.
    """
    streams = np.asarray(jax.vmap(residual_streams, in_axes=(None, 0))(model, tokens))
    # (batch, n_layers + 1, seq, d_model) -> put layer first
    streams = np.transpose(streams, (1, 0, 2, 3))

    features: dict[str, float] = {}
    for layer, stream in enumerate(streams):
        at_answer = stream[:, answer_index, :]           # (batch, d_model)
        pooled = stream.mean(axis=1)                     # (batch, d_model)

        norms = np.linalg.norm(at_answer, axis=-1)
        features[f"L{layer}.norm_mean"] = float(norms.mean())
        features[f"L{layer}.norm_std"] = float(norms.std())
        features[f"L{layer}.pooled_norm_mean"] = float(
            np.linalg.norm(pooled, axis=-1).mean())

        for tag, acts in (("ans", at_answer), ("pool", pooled)):
            spectrum, pr = _spectrum(acts, spec.top_eigenvalues)
            features[f"L{layer}.{tag}.participation_ratio"] = pr
            for i, value in enumerate(spectrum):
                features[f"L{layer}.{tag}.eig{i}"] = float(value)

    attn = np.asarray(jax.vmap(_all_attention, in_axes=(None, 0))(model, tokens))
    # (batch, n_layers, n_heads, q, k)
    attn = np.transpose(attn, (1, 2, 0, 3, 4))
    positions = np.arange(attn.shape[-1])
    for layer in range(attn.shape[0]):
        for head in range(attn.shape[1]):
            pattern = attn[layer, head]                  # (batch, q, k)
            with np.errstate(divide="ignore", invalid="ignore"):
                logp = np.where(pattern > 0, np.log(pattern), 0.0)
            entropy = -(pattern * logp).sum(axis=-1)     # (batch, q)
            distance = (pattern * positions[None, None, :]).sum(axis=-1)
            offset = positions[None, :] - distance
            features[f"H{layer}.{head}.entropy"] = float(entropy.mean())
            features[f"H{layer}.{head}.distance"] = float(offset.mean())
    return features


@eqx.filter_jit
def _all_attention(model: Transformer, tokens: jnp.ndarray) -> jnp.ndarray:
    x = jax.vmap(model.embed)(tokens)
    patterns = []
    for block in model.blocks:
        attn, x = _block_attention(block, x)
        patterns.append(attn)
    return jnp.stack(patterns)


def feature_names(n_layers: int, n_heads: int, spec: StateProbeSpec) -> tuple[str, ...]:
    """The feature ordering, derivable without running a model.

    Having this independently of extraction means the analysis can assert that
    every model produced the same features in the same order, rather than
    trusting dict iteration order across processes.
    """
    names: list[str] = []
    for layer in range(n_layers + 1):
        names += [f"L{layer}.norm_mean", f"L{layer}.norm_std", f"L{layer}.pooled_norm_mean"]
        for tag in ("ans", "pool"):
            names.append(f"L{layer}.{tag}.participation_ratio")
            names += [f"L{layer}.{tag}.eig{i}" for i in range(spec.top_eigenvalues)]
    for layer in range(n_layers):
        for head in range(n_heads):
            names += [f"H{layer}.{head}.entropy", f"H{layer}.{head}.distance"]
    return tuple(names)
