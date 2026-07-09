"""Probe dataset construction: delivery-verified labels and leakage-safe splits.

Labels come from :class:`~trigger_audit.schemas.results.SurvivalResult`'s
``final_token_trigger_present`` -- delivery-verified ground truth -- never from whether a
trigger was inserted upstream. Splitting is grouped by ``base_id``: Project 1 expands one
base conversation into many trials (positions x lengths x policies, plus counterfactual
twins that differ only by trigger presence), so example-level random splits would put
near-duplicate contexts on both sides of the train/test line and inflate probe metrics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from trigger_audit.schemas.probes import ProbeExample, ProbeLabelSource, ProbeSplit
from trigger_audit.schemas.results import SurvivalResult


def build_probe_examples(survival_results: Iterable[SurvivalResult]) -> list[ProbeExample]:
    """Convert survival records into probe examples with delivery-verified labels.

    ``label`` is ``final_token_trigger_present``; the trigger's final-token span is carried
    when Project 1 localized one; ``metadata["trigger_inserted"]`` preserves the raw-layer
    insertion flag so evaluation can separate clean negatives from inserted-but-undelivered
    ones (the delivered-only evaluation set).
    """
    examples: list[ProbeExample] = []
    for result in survival_results:
        examples.append(
            ProbeExample(
                trial_id=result.trial_id,
                base_id=result.base_id,
                label=result.final_token_trigger_present,
                label_source=ProbeLabelSource.SURVIVAL_RESULT,
                trigger_token_start=result.trigger_final_token_start,
                trigger_token_end=result.trigger_final_token_end,
                metadata={
                    "trigger_inserted": result.raw_trigger_present,
                    "survival_class": result.survival_class.value,
                    "pipeline_policy": result.pipeline_policy,
                    # Carried for downstream slicing (per-context-length / per-trigger
                    # analyses). Guarded with getattr so this stays safe if a source record
                    # predates these fields on SurvivalResult.
                    "context_length": getattr(result, "context_length", None),
                    "trigger_id": getattr(result, "trigger_id", None),
                },
            )
        )
    return examples


def assign_splits(
    examples: Sequence[ProbeExample],
    *,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> list[ProbeExample]:
    """Assign train/calibration/test splits by ``base_id`` group, deterministically.

    CRITICAL leakage rule: all trials sharing a ``base_id`` land in the same split. Trials
    from one base are counterfactual twins and near-duplicates of each other (same
    conversation, different pipeline coordinates); splitting them across train and test
    would let the probe memorize base-specific content and report inflated detection
    metrics. Groups are shuffled with a seeded RNG and cut at the fraction boundaries, so
    the assignment is reproducible given (examples, fractions, seed).
    """
    _validate_split_fractions(train_fraction, calibration_fraction)

    base_ids = sorted({example.base_id for example in examples})
    n_groups = len(base_ids)
    if n_groups < 3:
        raise ValueError(f"need at least 3 base_id groups to form three splits, got {n_groups}")

    rng = np.random.default_rng(seed)
    shuffled = [base_ids[i] for i in rng.permutation(n_groups)]
    n_train = max(1, int(train_fraction * n_groups))
    n_calibration = max(1, int(calibration_fraction * n_groups))
    if n_train + n_calibration >= n_groups:
        raise ValueError(
            f"fractions leave no test groups: {n_groups} groups -> "
            f"{n_train} train + {n_calibration} calibration"
        )

    split_of: dict[str, ProbeSplit] = {}
    for index, base_id in enumerate(shuffled):
        if index < n_train:
            split_of[base_id] = ProbeSplit.TRAIN
        elif index < n_train + n_calibration:
            split_of[base_id] = ProbeSplit.CALIBRATION
        else:
            split_of[base_id] = ProbeSplit.TEST
    return [example.model_copy(update={"split": split_of[example.base_id]}) for example in examples]


def _validate_split_fractions(train_fraction: float, calibration_fraction: float) -> None:
    """Shared fraction guards for the split assigners (grouped and example-level)."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError(f"calibration_fraction must be in [0, 1), got {calibration_fraction}")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must leave room for test")


def assign_splits_example_level(
    examples: Sequence[ProbeExample],
    *,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> list[ProbeExample]:
    """Assign train/calibration/test at the EXAMPLE level, ignoring ``base_id`` (the leaky control).

    This is the deliberately unsafe counterpart to :func:`assign_splits`, provided only for the
    E0.3 leakage ablation. Because counterfactual twins share a ``base_id`` but are shuffled and
    cut independently here, near-duplicate contexts -- sometimes exact twins -- land on both sides
    of the train/test line, letting the probe memorize base content and report inflated metrics.
    Same fractions and seeded determinism as :func:`assign_splits`; the split *unit* is the single
    example rather than the ``base_id`` group. Production runs must use :func:`assign_splits`; this
    exists to *quantify* the inflation the grouping rule prevents, never to defeat it.
    """
    _validate_split_fractions(train_fraction, calibration_fraction)

    n_examples = len(examples)
    if n_examples < 3:
        raise ValueError(f"need at least 3 examples to form three splits, got {n_examples}")

    rng = np.random.default_rng(seed)
    shuffled_positions = [int(i) for i in rng.permutation(n_examples)]
    n_train = max(1, int(train_fraction * n_examples))
    n_calibration = max(1, int(calibration_fraction * n_examples))
    if n_train + n_calibration >= n_examples:
        raise ValueError(
            f"fractions leave no test examples: {n_examples} examples -> "
            f"{n_train} train + {n_calibration} calibration"
        )

    split_of_position: dict[int, ProbeSplit] = {}
    for rank, position in enumerate(shuffled_positions):
        if rank < n_train:
            split_of_position[position] = ProbeSplit.TRAIN
        elif rank < n_train + n_calibration:
            split_of_position[position] = ProbeSplit.CALIBRATION
        else:
            split_of_position[position] = ProbeSplit.TEST
    return [
        example.model_copy(update={"split": split_of_position[index]})
        for index, example in enumerate(examples)
    ]


def build_synthetic_probe_dataset(
    *,
    n_examples: int = 60,
    seq_len: int = 16,
    trigger_token_ids: Sequence[int] = (9001, 9002, 9003),
    vocab_size: int = 500,
    seed: int = 0,
) -> tuple[list[ProbeExample], dict[str, list[int]]]:
    """Build a deterministic synthetic dataset for offline runs against the reference extractor.

    Positives carry the fixed multi-token trigger subsequence at a random position (span
    recorded); negatives draw only from ``range(1, vocab_size)``, which is disjoint from
    the trigger ids by construction, so labels are exact. Returns the examples plus the
    ``trial_id -> final token ids`` mapping the runner consumes.
    """
    if seq_len <= len(trigger_token_ids):
        raise ValueError("seq_len must exceed the trigger length")
    if vocab_size > min(trigger_token_ids):
        raise ValueError("vocab_size must not overlap the trigger token ids")

    rng = np.random.default_rng(seed)
    trigger = [int(t) for t in trigger_token_ids]
    examples: list[ProbeExample] = []
    tokens: dict[str, list[int]] = {}
    for index in range(n_examples):
        label = index % 2 == 0
        ids = [int(t) for t in rng.integers(1, vocab_size, size=seq_len)]
        span: tuple[int, int] | None = None
        if label:
            start = int(rng.integers(0, seq_len - len(trigger) + 1))
            ids[start : start + len(trigger)] = trigger
            span = (start, start + len(trigger))
        trial_id = f"synthetic_{index:04d}"
        examples.append(
            ProbeExample(
                trial_id=trial_id,
                base_id=f"base_{index:04d}",
                label=label,
                label_source=ProbeLabelSource.SYNTHETIC,
                trigger_token_start=span[0] if span else None,
                trigger_token_end=span[1] if span else None,
                metadata={"trigger_inserted": label},
            )
        )
        tokens[trial_id] = ids
    return examples, tokens


# Domain-separation tags for the seeded RNG streams. Each stochastic decision draws from an
# independent, reproducible stream keyed by (seed, tag[, base_index]), so changing one stream
# (e.g. token content) never perturbs another (e.g. which bases are partial-survival).
_TAG_PARTIAL_SELECT = 0x7D1  # which bases carry a partial-survival negative
_TAG_SPAN_SELECT = 0x7D2  # which partial-survival negatives carry a fragment span
_TAG_POSITIVE = 0x7E1  # delivered-positive token content + trigger placement
_TAG_CLEAN_NEG = 0x7E2  # clean-negative token content
_TAG_PARTIAL = 0x7E3  # partial-survival token content, fragment length/slice/placement


def _count_from_fraction(fraction: float, total: int) -> int:
    """Map a fraction to an integer count, keeping at least one when the fraction is a
    genuine ``(0, 1)`` request so the guaranteed-present populations never vanish to rounding.
    """
    if total <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return total
    return min(total, max(1, round(fraction * total)))


def build_synthetic_probe_dataset_with_twins(
    *,
    n_bases: int = 40,
    seq_len: int = 24,
    trigger_token_ids: Sequence[int] = (9001, 9002, 9003, 9004),
    vocab_size: int = 500,
    partial_survival_fraction: float = 0.25,
    span_on_partial_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[list[ProbeExample], dict[str, list[int]]]:
    """Build a deterministic synthetic dataset exercising all three probe populations.

    Unlike :func:`build_synthetic_probe_dataset` (unique ``base_id`` per example, and
    ``trigger_inserted == label`` throughout), this builder emits **counterfactual twins**
    and **partial-survival negatives**, the two regimes the Tier-0 experiments rely on:

    - **Delivered positive** (``label=True``, ``trigger_inserted=True``): the full trigger
      subsequence embedded at a random position, span recorded.
    - **Clean negative** (``label=False``, ``trigger_inserted=False``): drawn only from
      ``range(1, vocab_size)`` (disjoint from the trigger ids), so no trigger token appears.
    - **Partial-survival negative** (``label=False``, ``trigger_inserted=True``): a strict,
      shorter *fragment* of the trigger embedded in an otherwise trigger-disjoint sequence,
      so activations are genuinely contaminated while the full trigger is provably absent.

    Every ``base_id`` carries a positive/clean-negative twin pair (sharing the base, so a
    grouped split keeps them together -- the point of the E0.3 leakage ablation).
    ``partial_survival_fraction`` of the bases additionally carry a partial-survival negative
    (a third example under the same ``base_id``); ``span_on_partial_fraction`` of those record
    the surviving fragment's span (Project 1's scorer localizes it) while the rest leave the
    span ``None``.

    Determinism: every draw is keyed by ``(seed, tag[, base_index])`` with a distinct tag per
    stream, so two calls with identical arguments return identical examples and token maps.
    With ``n_bases >= 4`` and ``partial_survival_fraction in (0, 1)`` at least one of each
    population is present, so :func:`assign_splits` and the runner's split validation hold.

    Returns the examples plus the ``trial_id -> final token ids`` mapping the runner consumes.
    """
    trigger = [int(t) for t in trigger_token_ids]
    if len(trigger) < 2:
        raise ValueError("trigger_token_ids must have length >= 2 to admit a strict fragment")
    if seq_len <= len(trigger):
        raise ValueError("seq_len must exceed the trigger length")
    if vocab_size > min(trigger):
        raise ValueError("vocab_size must not overlap the trigger token ids")
    if n_bases < 3:
        raise ValueError(f"need at least 3 bases to form three split groups, got {n_bases}")
    if not 0.0 <= partial_survival_fraction <= 1.0:
        raise ValueError("partial_survival_fraction must be in [0, 1]")
    if not 0.0 <= span_on_partial_fraction <= 1.0:
        raise ValueError("span_on_partial_fraction must be in [0, 1]")

    trigger_len = len(trigger)

    # Which bases carry a partial-survival negative, and which of those record a fragment
    # span. Selected via seeded shuffles so the choice is reproducible yet decorrelated from
    # base index (which is what drives split assignment).
    n_partial = _count_from_fraction(partial_survival_fraction, n_bases)
    select_rng = np.random.default_rng((seed, _TAG_PARTIAL_SELECT))
    partial_bases = sorted(int(i) for i in select_rng.permutation(n_bases)[:n_partial])
    partial_base_set = set(partial_bases)
    n_span = _count_from_fraction(span_on_partial_fraction, n_partial)
    span_rng = np.random.default_rng((seed, _TAG_SPAN_SELECT))
    span_positions = {int(i) for i in span_rng.permutation(n_partial)[:n_span]}
    span_bases = {partial_bases[i] for i in span_positions}

    examples: list[ProbeExample] = []
    tokens: dict[str, list[int]] = {}
    for base_index in range(n_bases):
        base_id = f"base_{base_index:04d}"

        # (1) Delivered positive: full trigger subsequence at a random position, span recorded.
        pos_rng = np.random.default_rng((seed, _TAG_POSITIVE, base_index))
        pos_ids = [int(x) for x in pos_rng.integers(1, vocab_size, size=seq_len)]
        pos_start = int(pos_rng.integers(0, seq_len - trigger_len + 1))
        pos_ids[pos_start : pos_start + trigger_len] = trigger
        pos_trial = f"twin_{base_index:04d}_pos"
        examples.append(
            ProbeExample(
                trial_id=pos_trial,
                base_id=base_id,
                label=True,
                label_source=ProbeLabelSource.SYNTHETIC,
                trigger_token_start=pos_start,
                trigger_token_end=pos_start + trigger_len,
                metadata={"trigger_inserted": True, "population": "delivered_positive"},
            )
        )
        tokens[pos_trial] = pos_ids

        # (2) Clean negative: trigger-disjoint ids only, never inserted, no span.
        neg_rng = np.random.default_rng((seed, _TAG_CLEAN_NEG, base_index))
        neg_ids = [int(x) for x in neg_rng.integers(1, vocab_size, size=seq_len)]
        neg_trial = f"twin_{base_index:04d}_neg"
        examples.append(
            ProbeExample(
                trial_id=neg_trial,
                base_id=base_id,
                label=False,
                label_source=ProbeLabelSource.SYNTHETIC,
                metadata={"trigger_inserted": False, "population": "clean_negative"},
            )
        )
        tokens[neg_trial] = neg_ids

        # (3) Partial-survival negative: a strict contiguous fragment of the trigger embedded
        # in an otherwise trigger-disjoint sequence. The full trigger can never appear -- only
        # `k < trigger_len` trigger-valued tokens exist, and the rest of the sequence is drawn
        # from range(1, vocab_size), disjoint from the trigger ids.
        if base_index in partial_base_set:
            frag_rng = np.random.default_rng((seed, _TAG_PARTIAL, base_index))
            frag_ids = [int(x) for x in frag_rng.integers(1, vocab_size, size=seq_len)]
            frag_k = int(frag_rng.integers(1, trigger_len))  # 1..trigger_len-1
            frag_j = int(frag_rng.integers(0, trigger_len - frag_k + 1))
            fragment = trigger[frag_j : frag_j + frag_k]
            frag_pos = int(frag_rng.integers(0, seq_len - frag_k + 1))
            frag_ids[frag_pos : frag_pos + frag_k] = fragment
            carries_span = base_index in span_bases
            frag_trial = f"twin_{base_index:04d}_partial"
            examples.append(
                ProbeExample(
                    trial_id=frag_trial,
                    base_id=base_id,
                    label=False,
                    label_source=ProbeLabelSource.SYNTHETIC,
                    trigger_token_start=frag_pos if carries_span else None,
                    trigger_token_end=frag_pos + frag_k if carries_span else None,
                    metadata={"trigger_inserted": True, "population": "partial_survival"},
                )
            )
            tokens[frag_trial] = frag_ids

    return examples, tokens


# Domain-separation tags for the E0 ablation fixtures (kept distinct from the twins-builder tags
# above so a change to one fixture's stream never perturbs another's).
_TAG_LEAKAGE_DEMO = 0x9C1  # leakage-demo base contexts + per-example noise
_TAG_OPERATOR_CONFOUND = 0x9C2  # operator-confound trigger-free content + span placement


def build_leakage_demo_dataset(
    *,
    n_bases: int = 60,
    examples_per_base: int = 4,
    seq_len: int = 10,
    vocab_size: int = 500,
    noise_tokens: int = 0,
    seed: int = 0,
) -> tuple[list[ProbeExample], dict[str, list[int]]]:
    """Build the E0.3 leakage fixture: near-duplicate examples that SHARE base content.

    The shipped twins builder (:func:`build_synthetic_probe_dataset_with_twins`) draws
    *independent* content for every example under a base, so an example-level split has nothing
    to leak -- the grouping rule cannot be shown to matter on it (review AR-3). This fixture is
    the deliberate counterexample the E0.3 ablation needs: each ``base_id`` owns one fixed random
    context of ``seq_len`` tokens, and its ``examples_per_base`` examples are that context with at
    most ``noise_tokens`` positions re-randomized (``noise_tokens=0`` makes them byte-identical --
    exact twins). Labels are assigned by base **parity** and the context distribution is identical
    across classes, so the ONLY label-predictive signal is base identity:

    - under a base_id-grouped split (:func:`assign_splits`) the TEST bases are unseen, base
      identity does not transfer, and AUROC sits near chance;
    - under an example-level split (:func:`assign_splits_example_level`) each TEST example has
      near-duplicate siblings in TRAIN, so the probe memorizes base content and AUROC inflates.

    The gap between the two is the leakage inflation the grouping rule prevents. Deterministic
    given the keyword arguments. Returns the examples plus the ``trial_id -> token ids`` mapping.
    """
    if n_bases < 3:
        raise ValueError(f"need at least 3 bases to form three split groups, got {n_bases}")
    if examples_per_base < 1:
        raise ValueError("examples_per_base must be >= 1")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if not 0 <= noise_tokens <= seq_len:
        raise ValueError(f"noise_tokens must be in [0, seq_len={seq_len}], got {noise_tokens}")

    rng = np.random.default_rng((seed, _TAG_LEAKAGE_DEMO))
    examples: list[ProbeExample] = []
    tokens: dict[str, list[int]] = {}
    for base_index in range(n_bases):
        label = base_index % 2 == 0
        base_id = f"leak_base_{base_index:04d}"
        context = rng.integers(1, vocab_size, size=seq_len)
        for k in range(examples_per_base):
            ids = context.copy()
            if noise_tokens > 0:
                positions = rng.choice(seq_len, size=noise_tokens, replace=False)
                ids[positions] = rng.integers(1, vocab_size, size=noise_tokens)
            trial_id = f"leak_{base_index:04d}_{k:02d}"
            examples.append(
                ProbeExample(
                    trial_id=trial_id,
                    base_id=base_id,
                    label=label,
                    label_source=ProbeLabelSource.SYNTHETIC,
                    metadata={"trigger_inserted": label, "population": "leakage_demo"},
                )
            )
            tokens[trial_id] = [int(x) for x in ids]
    return examples, tokens


def build_operator_confound_dataset(
    *,
    n_examples: int = 120,
    seq_len: int = 24,
    span_len: int = 4,
    vocab_size: int = 500,
    seed: int = 0,
) -> tuple[list[ProbeExample], dict[str, list[int]]]:
    """Build the E0.4 operator-confound fixture: trigger-FREE content, some examples span-tagged.

    Every token is drawn from ``range(1, vocab_size)`` -- there is **no trigger content anywhere**.
    Half the examples (``label=True``) carry an otherwise-meaningless random span of ``span_len``
    tokens; the other half (``label=False``) are spanless clean negatives. Under ``TRIGGER_SPAN``
    pooling this isolates the pooling *operator*: with the seeded random-span fallback ON, spanless
    examples are pooled over a matched random window, so both classes get the same short-window
    operator and -- there being no trigger content -- cannot separate. With the fallback OFF, the
    spanless class is mean-pooled over the whole sequence while the span class keeps the short
    window, so the two classes differ in pooling statistics alone (a short-window mean has far
    higher variance than a full-sequence mean), manufacturing class separation from nothing.

    Each example gets a unique ``base_id`` (there are no twins here; the confound is per-example),
    so grouped and example-level splits coincide. Deterministic given the keyword arguments.
    Returns the examples plus the ``trial_id -> token ids`` mapping.
    """
    if span_len < 1:
        raise ValueError("span_len must be >= 1")
    if seq_len <= span_len:
        raise ValueError("seq_len must exceed span_len")
    if n_examples < 3:
        raise ValueError(f"need at least 3 examples to form three splits, got {n_examples}")

    rng = np.random.default_rng((seed, _TAG_OPERATOR_CONFOUND))
    examples: list[ProbeExample] = []
    tokens: dict[str, list[int]] = {}
    for index in range(n_examples):
        label = index % 2 == 0
        ids = [int(x) for x in rng.integers(1, vocab_size, size=seq_len)]
        span: tuple[int, int] | None = None
        if label:
            start = int(rng.integers(0, seq_len - span_len + 1))
            span = (start, start + span_len)
        trial_id = f"opconf_{index:04d}"
        examples.append(
            ProbeExample(
                trial_id=trial_id,
                base_id=f"opconf_base_{index:04d}",
                label=label,
                label_source=ProbeLabelSource.SYNTHETIC,
                trigger_token_start=span[0] if span else None,
                trigger_token_end=span[1] if span else None,
                metadata={"trigger_inserted": label, "population": "operator_confound"},
            )
        )
        tokens[trial_id] = ids
    return examples, tokens
