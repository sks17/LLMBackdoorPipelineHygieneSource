"""Expand verified primitives into a manifest via the Cartesian product with stable ids."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.io.paths import PathResolver
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.util.ids import make_grid_trial_id

# Characters not safe in a shard filename (a model id may contain '/', ':' etc.).
_UNSAFE_SHARD_CHARS = re.compile(r"[^A-Za-z0-9_.\-]+")

# Plain library logger: do not configure logging at import (side-effect-free for a widely imported
# I/O module); the CLI/runners call configure_logging() to attach handlers when they run.
_LOG = logging.getLogger(__name__)

# Coordinates a counterfactual pair shares (everything except ``trigger_present``).
PairKey = tuple[str, str, str, str, int]


def _make_trial(
    *,
    base_id: str,
    trigger_id: str,
    position: TriggerPosition,
    policy_id: str,
    model_id: str,
    context_length: int,
    trigger_present: bool,
) -> TrialSpec:
    """Build one manifest row with a stable id derived from all of its grid coordinates."""
    return TrialSpec(
        trial_id=make_grid_trial_id(
            base_id,
            trigger_id,
            position.value,
            policy_id,
            model_id,
            context_length=context_length,
            trigger_present=trigger_present,
        ),
        base_id=base_id,
        trigger_id=trigger_id,
        trigger_position=position,
        model_id=model_id,
        tokenizer_id=model_id,
        context_length=context_length,
        pipeline_policy=policy_id,
        trigger_present=trigger_present,
        run_generation=False,
    )


def expand_manifest(
    base_ids: Sequence[str],
    trigger_ids: Sequence[str],
    positions: Sequence[TriggerPosition],
    policy_ids: Sequence[str],
    model_ids: Sequence[str],
    *,
    context_lengths: Sequence[int] | None = None,
    model_windows: Mapping[str, int] | None = None,
    include_counterfactual: bool = False,
    base_positions: Mapping[str, Sequence[TriggerPosition]] | None = None,
) -> list[TrialSpec]:
    """Expand the grid coordinates into the Cartesian product of trial specs.

    Iteration order is fixed as base_id -> trigger_id -> position -> policy_id -> model_id ->
    context_length -> trigger_present so the output order is deterministic across re-expansions.

    ``context_lengths`` adds the context-budget dimension; when omitted a single row with
    ``context_length=0`` is emitted per grid point (the legacy shape, where the truncation budget
    is carried by the policy id). ``model_windows`` maps ``model_id -> max_context_window``; a
    ``(model, context_length)`` cell whose length exceeds the model's window is skipped -- and the
    number skipped per model is logged, never silently dropped. ``include_counterfactual`` emits,
    for every grid point, both the ``trigger_present=True`` row and its ``trigger_present=False``
    twin (recover the pair with :func:`pair_key`); when ``False`` only the trigger-present row is
    emitted, so pre-fan-out callers keep their exact cardinality.

    ``base_positions`` optionally restricts, per base, which positions expand for it (e.g. only a
    base carrying a ``{{TOOL_OUTPUT_SLOT}}`` expands ``tool_output`` -- see
    ``pipelines.trigger_insertion.plantable_positions``). A base absent from the map falls back to
    the full ``positions`` list, so omitting the argument reproduces the un-filtered grid exactly.
    """
    lengths = list(context_lengths) if context_lengths is not None else [0]
    presence_values = (True, False) if include_counterfactual else (True,)
    skipped: dict[str, int] = defaultdict(int)

    trials: list[TrialSpec] = []
    for base_id in base_ids:
        positions_for_base = (
            base_positions.get(base_id, positions) if base_positions is not None else positions
        )
        for trigger_id in trigger_ids:
            for position in positions_for_base:
                for policy_id in policy_ids:
                    for model_id in model_ids:
                        for context_length in lengths:
                            # A context length of 0 is the legacy "unused" sentinel and is never
                            # capped; a real (>0) budget above the model's window is invalid.
                            window = model_windows.get(model_id) if model_windows else None
                            if context_length and window is not None and context_length > window:
                                skipped[model_id] += 1
                                continue
                            for trigger_present in presence_values:
                                trials.append(
                                    _make_trial(
                                        base_id=base_id,
                                        trigger_id=trigger_id,
                                        position=position,
                                        policy_id=policy_id,
                                        model_id=model_id,
                                        context_length=context_length,
                                        trigger_present=trigger_present,
                                    )
                                )

    for model_id, count in sorted(skipped.items()):
        window = model_windows.get(model_id) if model_windows else None
        _LOG.info(
            "expand_manifest: skipped %d context-length cell(s) for %s above its %s-token window",
            count,
            model_id,
            window,
        )
    return trials


def _safe_model_name(model_id: str) -> str:
    """Make a model id safe to embed in a shard filename."""
    return _UNSAFE_SHARD_CHARS.sub("_", model_id)


def shard_trials(
    trials: Sequence[TrialSpec], resolver: PathResolver, *, shard_size: int
) -> list[str]:
    """Group trials by model and write per-model shard files; return the shard paths (as strings).

    The cluster unit of work is one shard per array task (see ``docs/CLUSTER_EXECUTION_PLAN.md``).
    Sharding by ``model_id`` lets a worker load each model's tokenizer once; within a model, trials
    are sorted by ``trial_id`` (stable, content-derived) and chunked into ``shard_size``-row files
    named ``<safe_model>_shard_NNNN.jsonl`` under ``data/shards/`` -- exactly the pattern the Slurm
    array template expands. Returns the written shard paths in deterministic order.
    """
    by_model: dict[str, list[TrialSpec]] = defaultdict(list)
    for trial in trials:
        by_model[trial.model_id].append(trial)

    size = max(1, shard_size)
    paths: list[str] = []
    for model_id in sorted(by_model):
        model_trials = sorted(by_model[model_id], key=lambda t: t.trial_id)
        for index in range(0, len(model_trials), size):
            chunk = model_trials[index : index + size]
            name = f"{_safe_model_name(model_id)}_shard_{index // size:04d}.jsonl"
            path = resolver.shard_path(name)
            write_jsonl(path, chunk)
            paths.append(str(path))
    return paths


def pair_key(trial: TrialSpec) -> PairKey:
    """Return the coordinates a counterfactual pair shares (everything but ``trigger_present``).

    The two rows of a pair (trigger-present and its trigger-absent twin) return the same tuple, so
    grouping a manifest by :func:`pair_key` recovers the matched pairs for McNemar's analysis.
    """
    return (
        trial.base_id,
        trial.model_id,
        trial.trigger_position.value,
        trial.pipeline_policy,
        trial.context_length,
    )
