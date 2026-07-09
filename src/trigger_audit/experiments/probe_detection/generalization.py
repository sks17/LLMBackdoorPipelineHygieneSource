"""Generalization (E2.x) holdout logic for probe detection, as a neutral leaf module.

Project 2's Tier-2 experiments ask whether a probe trained on one *slice* of the data
generalizes to a held-out slice: train on some pipeline policies and test on others
(E2.1), train on short contexts and test on long ones (E2.2), or train on some trigger
families and test on others (E2.3). A :class:`GeneralizationSpec` names the two sides of
such a split; :func:`_membership` maps one example to its side by reading its metadata;
:func:`partition_by_metadata` is a pure two-way relabeler; and
:func:`assign_generalization_splits` closes the seam the runner needs -- it drops the
un-held-out rows, keeps the held-out side as TEST, and carves a ``base_id``-grouped
CALIBRATION subset out of the training side (reusing the exact grouping discipline of
``dataset.assign_splits`` so partial-survival leakage is impossible).

This module is deliberately a *leaf*: it imports only :mod:`trigger_audit.schemas.probes`
plus stdlib/pydantic/numpy. ``config`` references :class:`GeneralizationSpec` and ``runner``
applies :func:`assign_generalization_splits`, while ``grid`` re-exports both -- so keeping
the type here (rather than in ``grid``) is what breaks the ``config -> grid -> config``
import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from trigger_audit.schemas.probes import ProbeExample, ProbeSplit

GeneralizationKind = Literal["policy", "context_length", "trigger_type"]


class GeneralizationSpec(BaseModel):
    """Names the train and test sides of one E2.x generalization holdout.

    ``kind`` discriminates which axis the holdout cuts on and therefore which fields are
    read:

    - ``"policy"`` -- pipeline policy of the trial (``metadata["pipeline_policy"]``); the
      train and test sides are the ``train_policies`` and ``test_policies`` lists.
    - ``"context_length"`` -- assembled context length (``metadata["context_length"]``);
      contexts ``<= train_context_max`` train, contexts ``>= test_context_min`` test, and
      the (necessarily empty, since ``train_context_max < test_context_min``) middle band is
      neither.
    - ``"trigger_type"`` -- trigger family (``metadata["trigger_type"]`` falling back to
      ``metadata["trigger_id"]``); the sides are the ``train_trigger_types`` and
      ``test_trigger_types`` lists.

    The post-init validator enforces that the fields relevant to ``kind`` are populated and
    describe two disjoint (non-overlapping) sides, so a mis-specified holdout fails at
    construction rather than silently sending every row to "neither".
    """

    model_config = ConfigDict(extra="forbid")

    kind: GeneralizationKind

    train_policies: list[str] = []
    test_policies: list[str] = []

    train_context_max: int | None = None
    test_context_min: int | None = None

    train_trigger_types: list[str] = []
    test_trigger_types: list[str] = []

    @model_validator(mode="after")
    def _check_sides_for_kind(self) -> GeneralizationSpec:
        if self.kind == "policy":
            if not self.train_policies or not self.test_policies:
                raise ValueError(
                    "policy holdout requires non-empty train_policies and test_policies"
                )
            overlap = sorted(set(self.train_policies) & set(self.test_policies))
            if overlap:
                raise ValueError(
                    f"train_policies and test_policies must be disjoint; overlap: {overlap}"
                )
        elif self.kind == "context_length":
            if self.train_context_max is None or self.test_context_min is None:
                raise ValueError(
                    "context_length holdout requires train_context_max and test_context_min"
                )
            if self.train_context_max >= self.test_context_min:
                raise ValueError(
                    "context_length holdout requires train_context_max < test_context_min, got "
                    f"{self.train_context_max} >= {self.test_context_min}"
                )
        else:  # trigger_type
            if not self.train_trigger_types or not self.test_trigger_types:
                raise ValueError(
                    "trigger_type holdout requires non-empty "
                    "train_trigger_types and test_trigger_types"
                )
            overlap = sorted(set(self.train_trigger_types) & set(self.test_trigger_types))
            if overlap:
                raise ValueError(
                    "train_trigger_types and test_trigger_types must be disjoint; "
                    f"overlap: {overlap}"
                )
        return self


def _membership(example: ProbeExample, spec: GeneralizationSpec) -> str | None:
    """Return ``"train"``, ``"test"``, or ``None`` for one example under ``spec``.

    ``None`` means the example belongs to neither side (its metadata is missing or falls in
    the held-out middle band) and should be dropped by an E2 run rather than leak into
    either side as an unmodeled third population. Reads only ``example.metadata``.
    """
    metadata = example.metadata
    if spec.kind == "policy":
        policy = metadata.get("pipeline_policy")
        if policy is None:
            return None
        if policy in spec.train_policies:
            return "train"
        if policy in spec.test_policies:
            return "test"
        return None
    if spec.kind == "context_length":
        length = metadata.get("context_length")
        if length is None:
            return None
        if spec.train_context_max is not None and length <= spec.train_context_max:
            return "train"
        if spec.test_context_min is not None and length >= spec.test_context_min:
            return "test"
        return None
    # trigger_type
    trigger_type = metadata.get("trigger_type", metadata.get("trigger_id"))
    if trigger_type is None:
        return None
    if trigger_type in spec.train_trigger_types:
        return "train"
    if trigger_type in spec.test_trigger_types:
        return "test"
    return None


def partition_by_metadata(
    examples: Sequence[ProbeExample], spec: GeneralizationSpec
) -> list[ProbeExample]:
    """Relabel each example to TRAIN or TEST by its holdout side; leave neither-side rows as-is.

    A pure, order-preserving two-way relabeler for ad-hoc inspection of a holdout: examples
    on the train side become :attr:`ProbeSplit.TRAIN`, examples on the test side become
    :attr:`ProbeSplit.TEST`, and examples matching neither side keep whatever split they
    already carried (they are neither dropped nor moved here -- use
    :func:`assign_generalization_splits` for the run-ready partition that drops them and
    carves a calibration subset). Never mutates its inputs; changed rows are fresh copies via
    ``model_copy``.
    """
    partitioned: list[ProbeExample] = []
    for example in examples:
        side = _membership(example, spec)
        if side == "train":
            partitioned.append(example.model_copy(update={"split": ProbeSplit.TRAIN}))
        elif side == "test":
            partitioned.append(example.model_copy(update={"split": ProbeSplit.TEST}))
        else:
            partitioned.append(example)
    return partitioned


def assign_generalization_splits(
    examples: Sequence[ProbeExample],
    spec: GeneralizationSpec,
    *,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> list[ProbeExample]:
    """Produce a run-ready TRAIN/CALIBRATION/TEST partition for an E2.x holdout.

    The held-out side becomes TEST. From the training side, a ``base_id``-grouped
    ``calibration_fraction`` of bases is carved into CALIBRATION using the exact discipline
    of ``dataset.assign_splits`` -- sort the base ids, shuffle with
    ``np.random.default_rng(seed)``, and cut at the fraction boundary so every trial of a
    base moves together and no base straddles TRAIN and CALIBRATION. Examples matching
    neither side are dropped, so an un-held-out row can never leak into the run as an
    unmodeled third population.

    Fails fast (``ValueError``) if the holdout leaves zero TEST rows, zero TRAIN rows, or
    fewer than two TRAIN base ids (too few to carve calibration while keeping a base in
    TRAIN) -- a mis-specified partition is named plainly here rather than crashing later in
    ``_validate_splits``. Never mutates its inputs; every returned row is a fresh copy via
    ``model_copy``.
    """
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError(f"calibration_fraction must be in [0, 1), got {calibration_fraction}")

    sides = [(example, _membership(example, spec)) for example in examples]
    test_side = [example for example, side in sides if side == "test"]
    train_side = [example for example, side in sides if side == "train"]

    if not test_side:
        raise ValueError(
            f"{spec.kind} holdout produced 0 TEST examples; "
            "no data matched the held-out (test) side of the split"
        )
    if not train_side:
        raise ValueError(
            f"{spec.kind} holdout produced 0 TRAIN examples; "
            "no data matched the training side of the split"
        )

    train_base_ids = sorted({example.base_id for example in train_side})
    n_groups = len(train_base_ids)
    if n_groups < 2:
        raise ValueError(
            f"{spec.kind} holdout left {n_groups} TRAIN base_id group(s); "
            "need at least 2 to carve a calibration split while keeping a base in TRAIN"
        )

    rng = np.random.default_rng(seed)
    shuffled = [train_base_ids[i] for i in rng.permutation(n_groups)]
    n_calibration = max(1, int(calibration_fraction * n_groups))
    n_calibration = min(n_calibration, n_groups - 1)
    calibration_bases = set(shuffled[:n_calibration])

    partitioned: list[ProbeExample] = []
    for example, side in sides:
        if side == "test":
            partitioned.append(example.model_copy(update={"split": ProbeSplit.TEST}))
        elif side == "train":
            new_split = (
                ProbeSplit.CALIBRATION if example.base_id in calibration_bases else ProbeSplit.TRAIN
            )
            partitioned.append(example.model_copy(update={"split": new_split}))
        # side is None -> dropped
    return partitioned
