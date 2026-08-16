"""The synthetic W/P task: two independently learnable information sources.

Research plan Stage 3 needs a task in which a model can learn either of two
strategies, so that which one it settles on is a consequence of its training
history rather than of what was learnable.

    W   an underlying rule over the input
    P   a predictive preference cue

Each example is a fixed-length sequence

    [BOS, CUE, d1 ... dn, SEP, ANSWER]

``W`` is the sum of the digits modulo ``n_classes``. ``P`` is a fixed
many-to-one map from ``n_cues`` cue tokens onto the same answer classes. Both
determine the answer perfectly on their own, and neither requires the other,
so a model that has learned only one of them is still at ceiling on the
examples that source explains.

The cue map is a learned association rather than an identity copy. With one
cue per class the cue is readable off the input and is acquired within a
couple of checkpoints, leaving no observable learning window; Gate B's
developmental-resolution criterion is then unreachable for ``P`` at any model
size or learning rate. Giving the cue several tokens per class makes it a
genuine thing to learn while keeping it exactly as informative.

Digit values are ``0 .. n_digit_values-1`` with ``n_digit_values`` an exact
multiple of ``n_classes``. That is not cosmetic: it makes each digit's
residue exactly uniform, so the digit sum carries exactly zero information
about the answer in the P-only family. Base-10 digits under a modulus of 4
would leak a small amount of rule signal into the condition that is supposed
to isolate the cue, and the leak would show up as spurious transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

__all__ = [
    "TaskConfig", "Family", "Condition", "Batch", "sample_batch",
    "FAMILIES", "CONDITIONS", "SPLITS", "digit_table", "cue_table",
]

PAD, BOS, SEP = 0, 1, 2
_N_SPECIAL = 3


SPLITS = ("train", "heldout")


@dataclass(frozen=True)
class TaskConfig:
    """Shape of the W/P task. Frozen, and part of the data version.

    ``n_digits=2`` is a **smoke-only** setting. It exists to exercise the
    machinery quickly and has too small an input space to test
    generalization meaningfully. Task difficulty is a neutral adequacy
    variable selected by the Gate B sweep, not a constant.
    """

    n_classes: int = 4
    n_digits: int = 4
    n_digit_values: int = 8
    n_cues: int = 16
    heldout_fraction: float = 0.25
    split_seed: int = 0
    cue_map_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {self.n_classes}")
        if self.n_digits < 1:
            raise ValueError(f"n_digits must be at least 1, got {self.n_digits}")
        if self.n_digit_values % self.n_classes != 0:
            raise ValueError(
                f"n_digit_values ({self.n_digit_values}) must be a multiple of "
                f"n_classes ({self.n_classes}) so that digit residues are exactly "
                "uniform; otherwise the rule leaks into the cue-isolation condition"
            )
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError(
                f"heldout_fraction must lie in (0, 1), got {self.heldout_fraction}"
            )
        if self.n_cues % self.n_classes != 0:
            raise ValueError(
                f"n_cues ({self.n_cues}) must be a multiple of n_classes "
                f"({self.n_classes}) so that every class has equally many cues; "
                "an unbalanced cue map would make the cue weakly predictive of "
                "the rule and contaminate the isolating conditions"
            )

    @property
    def cues_per_class(self) -> int:
        return self.n_cues // self.n_classes

    @property
    def n_inputs(self) -> int:
        """Size of the digit-tuple space, before splitting."""
        return self.n_digit_values ** self.n_digits

    # Token layout: specials, cues, digits, answers.
    @property
    def cue_base(self) -> int:
        return _N_SPECIAL

    @property
    def digit_base(self) -> int:
        return self.cue_base + self.n_cues

    @property
    def answer_base(self) -> int:
        return self.digit_base + self.n_digit_values

    @property
    def vocab_size(self) -> int:
        return self.answer_base + self.n_classes

    @property
    def seq_len(self) -> int:
        """[BOS, CUE, digits..., SEP, ANSWER]"""
        return 4 + self.n_digits

    @property
    def answer_position(self) -> int:
        """Index of the answer token in the full sequence."""
        return self.seq_len - 1

    @property
    def answer_target_index(self) -> int:
        """Index of the answer within the *shifted* target sequence.

        Training predicts ``seq[1:]`` from ``seq[:-1]``, so the answer sits
        one place earlier in the target array than in the sequence.
        """
        return self.answer_position - 1


# Training families: which source is informative in the data itself.
FAMILIES = ("W", "P", "MIX")

# Evaluation conditions: the diagnostic suite of research plan Stage 3.
CONDITIONS = ("aligned", "w_only", "p_only", "conflict")


Family = str
Condition = str


class Batch(dict):
    """A batch of examples plus the answers each source implies.

    Carrying both ``w_answer`` and ``p_answer`` on every batch is what makes
    the conflict condition measurable: the question is not whether the model
    is correct, but which of two correct-by-some-source answers it gives.
    """

    tokens: jax.Array
    w_answer: jax.Array
    p_answer: jax.Array


@lru_cache(maxsize=None)
def cue_table(config: TaskConfig) -> jax.Array:
    """Cue tokens grouped by the answer class they map to.

    Shape ``(n_classes, cues_per_class)``. Sampling a cue for class ``c`` is
    a lookup into row ``c``, and sampling a cue that *disagrees* with the
    rule is a lookup into any other row, which is how the conflict condition
    is constructed without rejection sampling.

    The grouping is balanced by construction, so the cue carries exactly zero
    information about the rule and the two sources stay independent.
    """
    rng = np.random.default_rng(config.cue_map_seed)
    cues = rng.permutation(config.n_cues)
    return jnp.asarray(cues.reshape(config.n_classes, config.cues_per_class), dtype=jnp.int32)


@lru_cache(maxsize=None)
def digit_table(config: TaskConfig, split: str) -> jax.Array:
    """The digit tuples a split is allowed to draw from.

    Generalization needs a held-out set of *compositional structures*: digit
    combinations the model never trains on. The input space is small enough
    to enumerate, so the split is exact rather than probabilistic.

    The partition is stratified by the rule's residue class, holding out the
    same number of tuples from each. That is not tidiness. An unstratified
    split would leave slightly non-uniform residues in each half, which would
    make the digit sum weakly predictive in the cue-isolation condition and
    show up later as transfer that is really an artifact of the split.
    Because ``n_digit_values`` is a multiple of ``n_classes``, every residue
    class has the same size and the stratified split is exactly balanced.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    index = np.arange(config.n_inputs)
    digits = np.stack(
        [(index // config.n_digit_values**k) % config.n_digit_values
         for k in range(config.n_digits)],
        axis=1,
    )
    residue = digits.sum(axis=1) % config.n_classes

    rng = np.random.default_rng(config.split_seed)
    train_rows, heldout_rows = [], []
    for r in range(config.n_classes):
        rows = np.flatnonzero(residue == r)
        rows = rng.permutation(rows)
        n_heldout = int(round(config.heldout_fraction * rows.size))
        if n_heldout < 1 or n_heldout >= rows.size:
            raise ValueError(
                f"heldout_fraction={config.heldout_fraction} leaves {n_heldout} of "
                f"{rows.size} tuples in residue class {r}; both splits must be "
                "non-empty for generalization to be measurable"
            )
        heldout_rows.append(rows[:n_heldout])
        train_rows.append(rows[n_heldout:])

    rows = np.concatenate(train_rows if split == "train" else heldout_rows)
    return jnp.asarray(digits[np.sort(rows)], dtype=jnp.int32)


def _assemble(config: TaskConfig, cue: jax.Array, digits: jax.Array, answer: jax.Array) -> jax.Array:
    batch = cue.shape[0]
    return jnp.concatenate(
        [
            jnp.full((batch, 1), BOS, dtype=jnp.int32),
            (config.cue_base + cue)[:, None].astype(jnp.int32),
            (config.digit_base + digits).astype(jnp.int32),
            jnp.full((batch, 1), SEP, dtype=jnp.int32),
            (config.answer_base + answer)[:, None].astype(jnp.int32),
        ],
        axis=1,
    )


def sample_batch(
    key: jax.Array,
    kind: str,
    config: TaskConfig,
    batch_size: int,
    *,
    split: str = "train",
) -> Batch:
    """Sample a batch from a training family or an evaluation condition.

    Args:
        split: ``"train"`` or ``"heldout"``. Training always draws from
            ``"train"``; evaluating on ``"heldout"`` measures generalization
            to digit combinations never trained on.
        kind: One of :data:`FAMILIES` or :data:`CONDITIONS`.

            ``W``         rule informative, cue uniform random
            ``P``         cue informative, rule uniform random
            ``MIX``       both informative and agreeing
            ``aligned``   as ``MIX``; the ordinary case W(x) = P(x)
            ``w_only``    clean-rule diagnostic: cue uninformative
            ``p_only``    cue-isolation diagnostic: rule uninformative
            ``conflict``  W(x) != P(x); neither answer is privileged
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    k_digits, k_cue = jr.split(key)
    table = digit_table(config, split)
    rows = jr.randint(k_digits, (batch_size,), 0, table.shape[0])
    digits = table[rows]
    w = jnp.sum(digits, axis=1) % config.n_classes

    cues = cue_table(config)
    k_class, k_slot = jr.split(k_cue)
    slot = jr.randint(k_slot, (batch_size,), 0, config.cues_per_class)

    if kind in ("W", "w_only"):
        # The cue is present but uninformative: its class is independent of w.
        cue_class = jr.randint(k_class, (batch_size,), 0, config.n_classes)
        cue = cues[cue_class, slot]
        answer = w
    elif kind in ("P", "p_only"):
        cue_class = jr.randint(k_class, (batch_size,), 0, config.n_classes)
        cue = cues[cue_class, slot]
        answer = cue_class
    elif kind in ("MIX", "aligned"):
        cue_class = w
        cue = cues[cue_class, slot]
        answer = w
    elif kind == "conflict":
        # Offset by 1..n_classes-1 so the cue class never coincides with the rule.
        offset = 1 + jr.randint(k_class, (batch_size,), 0, config.n_classes - 1)
        cue_class = (w + offset) % config.n_classes
        cue = cues[cue_class, slot]
        # The answer token written into the sequence is never scored in this
        # condition; both w_answer and p_answer are read out instead.
        answer = w
    else:
        raise ValueError(
            f"unknown kind {kind!r}; expected one of {FAMILIES + CONDITIONS}"
        )

    batch = Batch()
    batch["tokens"] = _assemble(config, cue, digits, answer)
    batch["w_answer"] = w.astype(jnp.int32)
    batch["p_answer"] = cue_class.astype(jnp.int32)
    return batch
