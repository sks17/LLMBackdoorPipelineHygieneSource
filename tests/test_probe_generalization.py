"""Unit tests for the E2.x generalization holdout logic (neutral leaf module)."""

from __future__ import annotations

import pytest

from trigger_audit.experiments.probe_detection.generalization import (
    GeneralizationSpec,
    _membership,
    assign_generalization_splits,
    partition_by_metadata,
)
from trigger_audit.schemas.probes import ProbeExample, ProbeLabelSource, ProbeSplit


def _example(
    trial_id: str,
    base_id: str,
    *,
    policy: str | None = None,
    context_length: int | None = None,
    trigger_id: str | None = None,
    trigger_type: str | None = None,
    label: bool = True,
    split: ProbeSplit = ProbeSplit.TRAIN,
) -> ProbeExample:
    metadata: dict[str, object] = {}
    if policy is not None:
        metadata["pipeline_policy"] = policy
    if context_length is not None:
        metadata["context_length"] = context_length
    if trigger_id is not None:
        metadata["trigger_id"] = trigger_id
    if trigger_type is not None:
        metadata["trigger_type"] = trigger_type
    return ProbeExample(
        trial_id=trial_id,
        base_id=base_id,
        label=label,
        label_source=ProbeLabelSource.SYNTHETIC,
        split=split,
        metadata=metadata,
    )


def _policy_dataset() -> list[ProbeExample]:
    """Six train-side bases (policies cot/plain), two test-side bases (tool), one neither."""
    examples: list[ProbeExample] = []
    for index in range(6):
        base = f"train_base_{index:02d}"
        policy = "cot" if index % 2 == 0 else "plain"
        # two trials per base (counterfactual twins) so grouping can be checked
        examples.append(_example(f"{base}_a", base, policy=policy))
        examples.append(_example(f"{base}_b", base, policy=policy, label=False))
    for index in range(2):
        base = f"test_base_{index:02d}"
        examples.append(_example(f"{base}_a", base, policy="tool"))
        examples.append(_example(f"{base}_b", base, policy="tool", label=False))
    # neither side: policy is in neither train nor test list
    examples.append(_example("other_a", "other_base", policy="offpolicy"))
    return examples


def _policy_spec() -> GeneralizationSpec:
    return GeneralizationSpec(
        kind="policy",
        train_policies=["cot", "plain"],
        test_policies=["tool"],
    )


def test_membership_policy_sides() -> None:
    spec = _policy_spec()
    assert _membership(_example("t", "b", policy="cot"), spec) == "train"
    assert _membership(_example("t", "b", policy="plain"), spec) == "train"
    assert _membership(_example("t", "b", policy="tool"), spec) == "test"
    assert _membership(_example("t", "b", policy="offpolicy"), spec) is None
    assert _membership(_example("t", "b"), spec) is None  # missing metadata


def test_membership_trigger_type_falls_back_to_trigger_id() -> None:
    spec = GeneralizationSpec(
        kind="trigger_type",
        train_trigger_types=["rand_001"],
        test_trigger_types=["natural_001"],
    )
    # no explicit trigger_type -> reads trigger_id
    assert _membership(_example("t", "b", trigger_id="rand_001"), spec) == "train"
    assert _membership(_example("t", "b", trigger_id="natural_001"), spec) == "test"
    # explicit trigger_type wins over trigger_id
    ex = _example("t", "b", trigger_id="natural_001", trigger_type="rand_001")
    assert _membership(ex, spec) == "train"
    assert _membership(_example("t", "b", trigger_id="other"), spec) is None


def test_membership_context_length_bands() -> None:
    spec = GeneralizationSpec(
        kind="context_length",
        train_context_max=100,
        test_context_min=200,
    )
    assert _membership(_example("t", "b", context_length=50), spec) == "train"
    assert _membership(_example("t", "b", context_length=100), spec) == "train"
    assert _membership(_example("t", "b", context_length=200), spec) == "test"
    assert _membership(_example("t", "b", context_length=150), spec) is None  # middle band
    assert _membership(_example("t", "b"), spec) is None  # missing metadata


def test_partition_by_metadata_two_way_relabel_leaves_neither_untouched() -> None:
    spec = _policy_spec()
    examples = [
        _example("train_a", "b0", policy="cot", split=ProbeSplit.TEST),
        _example("test_a", "b1", policy="tool", split=ProbeSplit.TRAIN),
        _example("neither_a", "b2", policy="offpolicy", split=ProbeSplit.CALIBRATION),
    ]
    result = partition_by_metadata(examples, spec)

    by_id = {ex.trial_id: ex for ex in result}
    assert by_id["train_a"].split is ProbeSplit.TRAIN
    assert by_id["test_a"].split is ProbeSplit.TEST
    # neither-side row keeps its pre-existing split, is not dropped or moved
    assert by_id["neither_a"].split is ProbeSplit.CALIBRATION
    assert len(result) == 3

    # inputs are never mutated in place
    assert examples[0].split is ProbeSplit.TEST
    assert examples[1].split is ProbeSplit.TRAIN
    assert examples[2].split is ProbeSplit.CALIBRATION


def test_assign_generalization_splits_policy_holdout() -> None:
    examples = _policy_dataset()
    spec = _policy_spec()
    result = assign_generalization_splits(examples, spec, calibration_fraction=0.25, seed=0)

    by_split: dict[ProbeSplit, list[ProbeExample]] = {
        ProbeSplit.TRAIN: [],
        ProbeSplit.CALIBRATION: [],
        ProbeSplit.TEST: [],
    }
    for ex in result:
        by_split[ex.split].append(ex)

    # every TEST row comes from the held-out (tool) policy side
    assert by_split[ProbeSplit.TEST]
    assert all(ex.metadata["pipeline_policy"] == "tool" for ex in by_split[ProbeSplit.TEST])

    # every TRAIN and CALIBRATION row comes from the training policies
    train_side = by_split[ProbeSplit.TRAIN] + by_split[ProbeSplit.CALIBRATION]
    assert all(ex.metadata["pipeline_policy"] in {"cot", "plain"} for ex in train_side)
    assert by_split[ProbeSplit.TRAIN]
    assert by_split[ProbeSplit.CALIBRATION]


def test_assign_generalization_splits_drops_neither_side() -> None:
    examples = _policy_dataset()
    result = assign_generalization_splits(examples, _policy_spec(), seed=0)
    trial_ids = {ex.trial_id for ex in result}
    # the single off-policy (neither-side) row is dropped entirely
    assert "other_a" not in trial_ids
    assert len(result) == len(examples) - 1


def test_calibration_is_base_id_grouped() -> None:
    examples = _policy_dataset()
    result = assign_generalization_splits(
        examples, _policy_spec(), calibration_fraction=0.25, seed=0
    )

    split_by_base: dict[str, set[ProbeSplit]] = {}
    for ex in result:
        split_by_base.setdefault(ex.base_id, set()).add(ex.split)

    # no base's twins straddle two splits (in particular not CALIBRATION vs TRAIN)
    for base, splits in split_by_base.items():
        assert len(splits) == 1, f"base {base} straddles splits {splits}"


def test_assign_generalization_splits_is_deterministic() -> None:
    examples = _policy_dataset()
    spec = _policy_spec()
    first = assign_generalization_splits(examples, spec, seed=7)
    second = assign_generalization_splits(examples, spec, seed=7)
    assert {ex.trial_id: ex.split for ex in first} == {ex.trial_id: ex.split for ex in second}

    # inputs untouched
    assert all(ex.split is ProbeSplit.TRAIN for ex in examples)


def test_assign_generalization_splits_raises_on_zero_test() -> None:
    # only training-policy rows present; the held-out (tool) side is empty
    examples = [
        _example("a", "b0", policy="cot"),
        _example("b", "b1", policy="plain"),
    ]
    with pytest.raises(ValueError, match="0 TEST"):
        assign_generalization_splits(examples, _policy_spec(), seed=0)


def test_assign_generalization_splits_raises_on_too_few_train_bases() -> None:
    # a valid TEST side, but only a single TRAIN base -> cannot carve calibration
    examples = [
        _example("train_a", "solo_base", policy="cot"),
        _example("train_b", "solo_base", policy="plain", label=False),
        _example("test_a", "test_base", policy="tool"),
    ]
    with pytest.raises(ValueError, match="TRAIN base_id"):
        assign_generalization_splits(examples, _policy_spec(), seed=0)
