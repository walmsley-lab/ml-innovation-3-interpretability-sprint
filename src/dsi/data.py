"""The synthetic W/P task: two independently learnable information sources.

Research plan Stage 3 needs a task in which a model can learn either of two
strategies, so that which one it settles on is a consequence of its training
history rather than of what was learnable.

    W   an underlying rule over the input
    P   a predictive preference cue

Each example is a fixed-length sequence

    [BOS, MODE, CUE, d1 ... dn, SEP, ANSWER]

``MODE`` states which strategy is requested, and is the only explicit task
identifier in the input. Without it the task is unidentifiable: see
research.md 8b. The original construction presented byte-identical visible
inputs for the two families and differed only in the answer, which capped
``min(A_W, A_P)`` at ``(1 + 1/n_classes)/2 = 0.625`` for any deterministic
predictor, below the prespecified retention threshold of 0.80. No model
capacity could have satisfied it.

``MODE`` separates three things the original conflated:

    USE_W     execute the rule.     Trains and measures W capability.
    USE_P     execute the cue map.  Trains and measures P capability.
    NEUTRAL   nothing requested.    Aligned training when W = P; the
                                    behavioural preference measurement when
                                    W != P.

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
USE_W, USE_P, NEUTRAL = 3, 4, 5
_N_SPECIAL = 6

MODES = {"USE_W": USE_W, "USE_P": USE_P, "NEUTRAL": NEUTRAL}


SPLITS = ("train", "heldout")


@dataclass(frozen=True)
class TaskConfig:
    """Shape of the W/P task. Frozen, and part of the data version.

    ``n_digits=2`` is a **smoke-only** setting. It exists to exercise the
    machinery quickly and has too small an input space to test
    generalization meaningfully. Task difficulty is a neutral adequacy
    variable selected by the Gate B sweep, not a constant.

    ``n_cues`` is a **calibrated task-complexity parameter**, fixed at Gate B
    by ``scripts/probe_cue_window.py`` on the cue in isolation. It is not a
    knob to revisit once confirmatory results are visible. Changing it after
    seeing a conflict or order effect would make the task difficulty a
    function of the outcome, which is exactly the contamination the neutral
    calibration exists to prevent. If it has to change, the regime is no
    longer frozen and Gates B and C are rerun from the start.
    """

    n_classes: int = 4
    n_digits: int = 4
    n_digit_values: int = 8
    n_cues: int = 256
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
        """[BOS, MODE, CUE, digits..., SEP, ANSWER]"""
        return 5 + self.n_digits

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
# Training families. Explicit families carry a mode token and are what Gate B
# trains on; NEUTRAL_ALIGNED enters only after corrected Gate B succeeds.
FAMILIES = ("W_EXPLICIT", "P_EXPLICIT", "NEUTRAL_ALIGNED", "W_P_INTERLEAVED")

# W_P_INTERLEAVED is a diagnostic family, not part of any curriculum. It
# exists to measure the joint-training upper bound: whether an architecture
# can represent both skills at once when optimization is not sequential.

# Evaluation conditions. W_COMPETENCE and P_COMPETENCE ask whether the same
# checkpoint can execute each strategy on request. NEUTRAL_CONFLICT is the
# behavioural preference measurement and is reserved for after Gate B.
CONDITIONS = ("W_COMPETENCE", "P_COMPETENCE", "NEUTRAL_ALIGNED_EVAL", "NEUTRAL_CONFLICT")

GATE_B_CONDITIONS = ("W_COMPETENCE", "P_COMPETENCE")
"""What corrected Gate B is allowed to look at.

Restricting the gate to the explicit modes makes it structurally impossible
to select a regime on the preference or order effect the project exists to
measure.
"""


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


def _assemble(config: TaskConfig, mode, cue: jax.Array,
              digits: jax.Array, answer: jax.Array) -> jax.Array:
    batch = cue.shape[0]
    mode_column = jnp.broadcast_to(jnp.asarray(mode, dtype=jnp.int32), (batch,))
    return jnp.concatenate(
        [
            jnp.full((batch, 1), BOS, dtype=jnp.int32),
            mode_column[:, None],
            (config.cue_base + cue)[:, None].astype(jnp.int32),
            (config.digit_base + digits).astype(jnp.int32),
            jnp.full((batch, 1), SEP, dtype=jnp.int32),
            (config.answer_base + answer)[:, None].astype(jnp.int32),
        ],
        axis=1,
    )


def _parse_mixture(kind: str):
    """Split a mixture family name into (primary, secondary, ratio).

    Mixtures are written into the family string itself, e.g.

        "P_EXPLICIT+W_EXPLICIT@0.05"   95% P_EXPLICIT, 5% W_EXPLICIT

    rather than passed as a separate argument, so the mixture is part of the
    PhaseSpec and therefore part of the content-addressed run identity. Two
    runs differing in continuity ratio must hash differently, and encoding it
    in the family is what guarantees that without touching RunSpec.
    """
    families, ratio = kind.rsplit("@", 1)
    primary, secondary = families.split("+")
    fraction = float(ratio)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"mixture ratio must lie in [0, 1], got {fraction}")
    return primary, secondary, fraction


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

            ``W_EXPLICIT``           MODE=USE_W, answer is the rule
            ``P_EXPLICIT``           MODE=USE_P, answer is the cue map
            ``NEUTRAL_ALIGNED``      MODE=NEUTRAL, W(x) = P(cue)
            ``W_COMPETENCE``         evaluation counterpart of W_EXPLICIT
            ``P_COMPETENCE``         evaluation counterpart of P_EXPLICIT
            ``NEUTRAL_ALIGNED_EVAL`` evaluation counterpart of the aligned family
            ``NEUTRAL_CONFLICT``     MODE=NEUTRAL, W(x) != P(cue); preference only
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if "@" in kind:
        primary, secondary, fraction = _parse_mixture(kind)
        k_primary, k_secondary, k_mask = jr.split(key, 3)
        a = sample_batch(k_primary, primary, config, batch_size, split=split)
        b = sample_batch(k_secondary, secondary, config, batch_size, split=split)
        # Deterministic exact-count allocation, not Bernoulli. Independent
        # sampling gives equal *expected* counts and unequal *realized* ones,
        # and a difference in realized exposure between two compared histories
        # is a difference in what they saw — exactly the confound that holding
        # allocation fixed exists to remove. A permutation makes the count
        # exact in every batch while keeping which positions are chosen
        # deterministic in the key.
        n_secondary = int(round(fraction * batch_size))
        take_secondary = jr.permutation(k_mask, batch_size) < n_secondary
        mixed = Batch()
        for field in ("tokens", "w_answer", "p_answer", "mode"):
            mask = take_secondary
            if a[field].ndim == 2:
                mask = take_secondary[:, None]
            mixed[field] = jnp.where(mask, b[field], a[field])
        return mixed

    k_digits, k_cue = jr.split(key)
    table = digit_table(config, split)
    rows = jr.randint(k_digits, (batch_size,), 0, table.shape[0])
    digits = table[rows]
    w = jnp.sum(digits, axis=1) % config.n_classes

    cues = cue_table(config)
    k_class, k_slot = jr.split(k_cue)
    slot = jr.randint(k_slot, (batch_size,), 0, config.cues_per_class)

    # The cue class is drawn identically in both explicit families, and
    # independently of the rule, so the content tokens are distribution
    # matched and MODE is the only explicit task identifier.
    independent_cue_class = jr.randint(k_class, (batch_size,), 0, config.n_classes)

    if kind in ("W_EXPLICIT", "W_COMPETENCE"):
        mode = USE_W
        cue_class = independent_cue_class
        answer = w
    elif kind in ("P_EXPLICIT", "P_COMPETENCE"):
        mode = USE_P
        cue_class = independent_cue_class
        answer = cue_class
    elif kind in ("NEUTRAL_ALIGNED", "NEUTRAL_ALIGNED_EVAL"):
        # Ordinary aligned training: W(x) = P(cue), so neither strategy is
        # privileged and no mode is requested.
        mode = NEUTRAL
        cue_class = w
        answer = w
    elif kind == "W_P_INTERLEAVED":
        # Balanced within-batch interleaving. The two halves are identical in
        # content and differ only in MODE and the answer, so this isolates
        # sequentiality from every other difference between the families.
        mode_flag = jr.bernoulli(k_class, 0.5, (batch_size,))
        cue_class = independent_cue_class
        mode = jnp.where(mode_flag, USE_W, USE_P)
        answer = jnp.where(mode_flag, w, cue_class)
    elif kind == "NEUTRAL_CONFLICT":
        # Offset by 1..n_classes-1 so the cue class never coincides with the
        # rule. Reserved for the behavioural preference measurement; never
        # trained on, and not evaluated at Gate B.
        mode = NEUTRAL
        offset = 1 + jr.randint(k_class, (batch_size,), 0, config.n_classes - 1)
        cue_class = (w + offset) % config.n_classes
        # The answer token written here is never scored: both w_answer and
        # p_answer are read out instead.
        answer = w
    else:
        raise ValueError(
            f"unknown kind {kind!r}; expected one of {FAMILIES + CONDITIONS}"
        )

    cue = cues[cue_class, slot]

    batch = Batch()
    batch["tokens"] = _assemble(config, mode, cue, digits, answer)
    batch["mode"] = jnp.broadcast_to(jnp.asarray(mode, dtype=jnp.int32), (batch_size,))
    batch["w_answer"] = w.astype(jnp.int32)
    batch["p_answer"] = cue_class.astype(jnp.int32)
    return batch
