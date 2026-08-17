"""Assert the cached batch path reproduces the online sampler exactly.

Caching is an engineering acceleration. It is only legitimate if the batches it
serves are the batches the frozen sampler would have produced — otherwise it
quietly becomes a new experimental condition with a different data stream.

This walks a reference run's worth of steps through both paths and requires
**elementwise equality** at every step, for every stream, including a mixture
family. It also checks that the jitted evaluator agrees with a direct
computation, since that was the other throughput change.

    PYTHONPATH=src python scripts/verify_batch_cache.py
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from dsi.microworld import BatchCache, MicroConfig, evaluate_stream, micro_sampler
from dsi.model import ModelConfig, init_model

STREAMS = ("IND", "IND_R", "BIND", "FACT", "BG", "IND+BG")
N_STEPS = 40
BATCH = 64


def main() -> int:
    cfg = MicroConfig()
    failures: list[str] = []

    for stream in STREAMS:
        data_key = jax.random.PRNGKey(hash(stream) % (2**31))
        cache = BatchCache(data_key, stream, cfg, BATCH, N_STEPS)

        for step in range(1, N_STEPS + 1):
            online = micro_sampler(
                jax.random.fold_in(data_key, step), stream, cfg, BATCH)["tokens"]
            cached = cache(None, stream, cfg, BATCH)["tokens"]
            if not np.array_equal(np.asarray(online), np.asarray(cached)):
                failures.append(f"{stream} step {step}: cached != online")
                break
        else:
            print(f"  [ok] {stream:8s} {N_STEPS} steps elementwise identical")

    # The cache must refuse to serve past its horizon rather than wrap or
    # silently regenerate, which would diverge from the online path.
    exhausted = BatchCache(jax.random.PRNGKey(0), "BIND", cfg, BATCH, 2)
    exhausted(None, "BIND", cfg, BATCH)
    exhausted(None, "BIND", cfg, BATCH)
    try:
        exhausted(None, "BIND", cfg, BATCH)
        failures.append("cache did not raise when exhausted")
    except RuntimeError:
        print("  [ok] exhausted cache raises rather than diverging")

    # The jitted evaluator must agree with a direct computation.
    mc = ModelConfig(vocab_size=cfg.vocab_size, d_model=64, n_heads=4,
                     n_layers=2, d_ff=128, max_len=cfg.seq_len)
    model = init_model(mc, jax.random.PRNGKey(7))
    from dsi.microworld import sample_documents
    tokens = jnp.asarray(sample_documents(11, cfg, "BIND", 256))
    logits = jax.vmap(model)(tokens[:, :-1])
    logprobs = jax.nn.log_softmax(logits[:, cfg.answer_target_index], axis=-1)
    target = tokens[:, -1]
    direct_acc = float((logprobs.argmax(axis=-1) == target).mean())
    direct_loss = float(-jnp.take_along_axis(
        logprobs, target[:, None], axis=-1).squeeze(-1).mean())
    got = evaluate_stream(model, cfg, "BIND", 11, 256)
    if abs(got["accuracy"] - direct_acc) > 1e-6 or abs(got["loss"] - direct_loss) > 1e-4:
        failures.append(
            f"jitted evaluator disagrees: {got} vs acc {direct_acc} loss {direct_loss}")
    else:
        print(f"  [ok] jitted evaluator matches direct "
              f"(acc {direct_acc:.4f}, loss {direct_loss:.4f})")

    if failures:
        print("\nVERIFICATION FAILED:")
        for f in failures:
            print(f"  {f}")
        print("The cached path is NOT equivalent. Do not run science on it.")
        return 1
    print("\nCached path is equivalent to the frozen online sampler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
