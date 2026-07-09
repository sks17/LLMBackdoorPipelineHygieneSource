"""Stratified, ``base_id``-aware subset selector for the probe wave (P2 / continuity C1).

Activation extraction is the expensive GPU phase, so it must run on a *stratified subset* of
Project 1's survival results rather than the whole delivery grid. This module chooses that subset
and guards it with Gate 0.

The selection **unit is ``base_id``**, never the individual trial: Project 1 expands one base
conversation into many trials (positions x lengths x policies, plus counterfactual twins that differ
only by trigger presence), so a base's trials are near-duplicates that must stay on the same side of
the downstream train/test line (see ``dataset.assign_splits``). When a base is chosen, **all** of
its trials come along.

Five strata must be represented so the probe can be trained/calibrated/tested honestly:

- ``delivered_positive``      -- ``final_token_trigger_present`` (delivery-verified positives).
- ``clean_negative``          -- ``not raw_trigger_present`` (never inserted).
- ``partial_survival_negative`` -- ``raw_trigger_present and not final_token_trigger_present``
  (inserted upstream but not delivered -- the third population clean-only calibration exists for).
- ``boundary_corruption``     -- ``survival_class == "boundary_corruption"`` (may overlap above).
- ``stratified_sample``       -- spread across the covariate grid
  (``pipeline_policy`` x ``trigger_position`` x ``context_length`` bucket) so no single cell
  dominates. Its achieved "count" is the number of distinct covariate cells covered.

Selection is a seeded greedy fill that prefers whole bases covering the most still-unmet strata and
**never silently under-delivers**: a stratum that cannot be filled records its gap in ``shortfalls``
for the caller to log (mirroring the repo rule that coverage caps are reported, never hidden).
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trigger_audit.analysis.controls import ControlVerdict, verify_counterfactual
from trigger_audit.schemas.results import SurvivalResult

# Stratum keys are deliberately identical to :class:`StratumTargets`' field names, so a stratum key
# and its requested target read out with a single ``getattr(targets, key)``.
_DELIVERED = "delivered_positive"
_CLEAN = "clean_negative"
_PARTIAL = "partial_survival_negative"
_BOUNDARY = "boundary_corruption"
_SAMPLE = "stratified_sample"
# Count-valued strata (achieved = number of selected trials in the population). ``_SAMPLE`` is
# coverage-valued (achieved = number of distinct covariate cells covered) and handled separately.
_COUNT_STRATA: tuple[str, ...] = (_DELIVERED, _CLEAN, _PARTIAL, _BOUNDARY)
_STRATA: tuple[str, ...] = (*_COUNT_STRATA, _SAMPLE)

_BOUNDARY_CLASS = "boundary_corruption"

# Default context-length bucket edges. Long-context cells are first-class strata, so the sample
# spreads across these bands rather than collapsing onto one length regime.
_DEFAULT_BUCKETS: tuple[int, ...] = (1000, 4000, 8000, 16000, 32000)

# A covariate cell for the stratified sample: (pipeline_policy, trigger_position, length-bucket).
Cell = tuple[str, str, int]


@dataclass(frozen=True)
class StratumTargets:
    """Requested minimum count for each stratum (0 == not requested)."""

    delivered_positive: int = 0
    clean_negative: int = 0
    partial_survival_negative: int = 0
    boundary_corruption: int = 0
    stratified_sample: int = 0


@dataclass(frozen=True)
class SubsetSelection:
    """The chosen subset plus a full audit trail of what was achieved vs requested.

    ``per_stratum_counts`` and ``shortfalls`` are keyed by the stratum names above. For every
    stratum except ``stratified_sample`` the count is the number of selected *trials* in that
    population; for ``stratified_sample`` it is the number of distinct covariate cells covered.
    """

    trial_ids: list[str]  # all trials of the selected bases (sorted, deterministic)
    base_ids: list[str]  # the selected bases (sorted)
    per_stratum_counts: dict[str, int]  # achieved per stratum among selected trials
    requested: StratumTargets
    shortfalls: dict[str, int]  # stratum -> how many short of target (0 if met); never hidden
    seed: int


@dataclass
class _BaseInfo:
    """Per-``base_id`` aggregate used by the greedy selector."""

    base_id: str
    trial_ids: list[str]  # sorted; all trials of this base
    stratum_counts: dict[str, int]  # count-stratum -> number of this base's trials in it
    strata: frozenset[str]  # count strata this base has >=1 trial in
    cells: frozenset[Cell]  # distinct covariate cells among this base's trials


def _bucket_id(context_length: int | None, buckets: Sequence[int]) -> int:
    """Map a context length to a bucket index (``-1`` when the length is unknown)."""
    if context_length is None:
        return -1
    return bisect.bisect_right(buckets, context_length)


def _survival_value(result: SurvivalResult) -> str:
    """Survival class as its string value (defensive against a plain-string field)."""
    return getattr(result.survival_class, "value", result.survival_class)


def _trial_strata(result: SurvivalResult) -> set[str]:
    """Count-valued strata a single trial belongs to (a trial can be in several)."""
    strata: set[str] = set()
    if result.final_token_trigger_present:
        strata.add(_DELIVERED)
    if not result.raw_trigger_present:
        strata.add(_CLEAN)
    if result.raw_trigger_present and not result.final_token_trigger_present:
        strata.add(_PARTIAL)
    if _survival_value(result) == _BOUNDARY_CLASS:
        strata.add(_BOUNDARY)
    return strata


def _trial_cell(result: SurvivalResult, buckets: Sequence[int]) -> Cell:
    """The covariate cell (policy, position, length-bucket) a trial falls in.

    ``context_length``/``trigger_position`` are read defensively so a source record predating those
    fields still buckets into a well-defined (sentinel) cell rather than raising.
    """
    context_length = getattr(result, "context_length", None)
    position = getattr(result.trigger_position, "value", result.trigger_position)
    return (result.pipeline_policy, position, _bucket_id(context_length, buckets))


def _aggregate_bases(
    results: Sequence[SurvivalResult], buckets: Sequence[int]
) -> dict[str, _BaseInfo]:
    """Group survival rows into per-``base_id`` aggregates."""
    trials: dict[str, list[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    cells: dict[str, set[Cell]] = {}
    for result in results:
        base_id = result.base_id
        if base_id not in trials:
            trials[base_id] = []
            counts[base_id] = dict.fromkeys(_COUNT_STRATA, 0)
            cells[base_id] = set()
        trials[base_id].append(result.trial_id)
        for stratum in _trial_strata(result):
            counts[base_id][stratum] += 1
        cells[base_id].add(_trial_cell(result, buckets))

    return {
        base_id: _BaseInfo(
            base_id=base_id,
            trial_ids=sorted(trials[base_id]),
            stratum_counts=counts[base_id],
            strata=frozenset(s for s in _COUNT_STRATA if counts[base_id][s] > 0),
            cells=frozenset(cells[base_id]),
        )
        for base_id in trials
    }


def select_probe_subset(
    results: Sequence[SurvivalResult],
    targets: StratumTargets,
    *,
    seed: int = 0,
    context_length_buckets: Sequence[int] | None = None,
) -> SubsetSelection:
    """Choose a stratified, ``base_id``-grouped subset of survival results for the probe wave.

    Deterministic given ``(results, targets, seed)``. Base order is shuffled with
    ``np.random.default_rng(seed)`` to give the sample its randomness; at each step the still-unmet
    strata drive a greedy pick of the whole base that advances the **most** of them (ties broken by
    the seeded shuffle order, then ``base_id``). Selecting a base is all-or-nothing, so one base can
    satisfy several strata at once (its counterfactual twins come along for free). Selection stops
    once every target is met or no remaining base can advance any unmet stratum; any residual gap is
    recorded in ``shortfalls`` -- never silently dropped.
    """
    buckets = (
        tuple(sorted(int(b) for b in context_length_buckets))
        if context_length_buckets is not None
        else _DEFAULT_BUCKETS
    )
    base_infos = _aggregate_bases(results, buckets)
    sorted_bases = sorted(base_infos)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(sorted_bases))
    shuffle_rank = {base_id: int(permutation[i]) for i, base_id in enumerate(sorted_bases)}

    target_of = {stratum: max(0, int(getattr(targets, stratum))) for stratum in _STRATA}

    counts = dict.fromkeys(_COUNT_STRATA, 0)
    covered_cells: set[Cell] = set()
    selected: list[str] = []
    remaining = set(sorted_bases)

    def is_unmet(stratum: str) -> bool:
        if stratum == _SAMPLE:
            return len(covered_cells) < target_of[_SAMPLE]
        return counts[stratum] < target_of[stratum]

    def contribution(info: _BaseInfo) -> int:
        """How many still-unmet strata accepting this base would advance."""
        gained = sum(1 for s in _COUNT_STRATA if is_unmet(s) and s in info.strata)
        if is_unmet(_SAMPLE) and not info.cells <= covered_cells:
            gained += 1
        return gained

    while any(is_unmet(stratum) for stratum in _STRATA):
        best: str | None = None
        best_key: tuple[int, int, str] | None = None
        for base_id in remaining:
            gained = contribution(base_infos[base_id])
            if gained == 0:
                continue
            # Prefer most unmet strata filled; then seeded shuffle order; then base_id for a fully
            # deterministic final tie-break.
            key = (-gained, shuffle_rank[base_id], base_id)
            if best_key is None or key < best_key:
                best_key = key
                best = base_id
        if best is None:  # no remaining base can advance any unmet stratum
            break
        info = base_infos[best]
        selected.append(best)
        remaining.discard(best)
        for stratum in _COUNT_STRATA:
            counts[stratum] += info.stratum_counts[stratum]
        covered_cells |= info.cells

    trial_ids = sorted(tid for base_id in selected for tid in base_infos[base_id].trial_ids)
    per_stratum_counts = dict(counts)
    per_stratum_counts[_SAMPLE] = len(covered_cells)
    shortfalls = {s: max(0, target_of[s] - per_stratum_counts[s]) for s in _STRATA}

    return SubsetSelection(
        trial_ids=trial_ids,
        base_ids=sorted(selected),
        per_stratum_counts=per_stratum_counts,
        requested=targets,
        shortfalls=shortfalls,
        seed=seed,
    )


def subset_report(selection: SubsetSelection) -> str:
    """Human-readable achieved-vs-target summary, flagging any recorded shortfall."""
    lines = [
        f"Probe subset selection (seed={selection.seed})",
        f"  bases selected:  {len(selection.base_ids)}",
        f"  trials selected: {len(selection.trial_ids)}",
        "  strata (achieved / requested):",
    ]
    for stratum in _STRATA:
        achieved = selection.per_stratum_counts.get(stratum, 0)
        requested = getattr(selection.requested, stratum)
        short = selection.shortfalls.get(stratum, 0)
        flag = f"   SHORTFALL -{short}" if short > 0 else ""
        lines.append(f"    {stratum:26s} {achieved:5d} / {requested:<5d}{flag}")
    return "\n".join(lines)


def write_selected_trial_ids(path: str | Path, selection: SubsetSelection) -> None:
    """Persist the full :class:`SubsetSelection` as JSON.

    JSON (not a bare id-per-line list) is deliberate: it keeps ``per_stratum_counts``,
    ``shortfalls`` and the requested targets alongside the ids, so the coverage the caller must log
    survives to disk. Keys: ``seed``, ``base_ids``, ``trial_ids``, ``per_stratum_counts``,
    ``requested``, ``shortfalls``.
    """
    payload = {
        "seed": selection.seed,
        "base_ids": selection.base_ids,
        "trial_ids": selection.trial_ids,
        "per_stratum_counts": selection.per_stratum_counts,
        "requested": asdict(selection.requested),
        "shortfalls": selection.shortfalls,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_subset_counterfactual(
    results: Sequence[SurvivalResult], selection: SubsetSelection
) -> ControlVerdict:
    """Gate 0 on the probed subset: run Project 1's counterfactual control on it (P5 / AR-11).

    Filters ``results`` to ``selection.trial_ids`` and projects each survival row into the exact
    frame :func:`~trigger_audit.analysis.controls.verify_counterfactual` expects
    (``trigger_present = raw_trigger_present``, ``delivered = final_token_trigger_present``,
    ``survival_class = survival_class.value``, plus ``trial_id``/``base_id``/``trigger_id`` so leak
    examples are identifiable), then returns its :class:`ControlVerdict` unchanged. Every
    trigger-absent twin in the subset must be ``no_survival`` and delivered nowhere; if any leaks,
    the verdict's ``ok`` is ``False`` and no probe number on this subset is trustworthy.
    """
    selected = set(selection.trial_ids)
    rows = [r for r in results if r.trial_id in selected]
    frame = pd.DataFrame(
        {
            "trial_id": [r.trial_id for r in rows],
            "base_id": [r.base_id for r in rows],
            "trigger_id": [getattr(r, "trigger_id", None) for r in rows],
            "trigger_present": [r.raw_trigger_present for r in rows],
            "delivered": [r.final_token_trigger_present for r in rows],
            "survival_class": [_survival_value(r) for r in rows],
        }
    )
    return verify_counterfactual(frame)
