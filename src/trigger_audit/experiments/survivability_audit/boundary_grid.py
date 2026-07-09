"""Targeted boundary-corruption grid: sweep the head-truncation cut across each trigger's span.

Generalizes Trial 5 (:mod:`boundary_spec`) from one trigger at the prefix to a grid over positions
and per-trial cut offsets. For each ``(base, model, position)`` the trigger's pre-truncation token
span ``[S, E)`` is **measured** from a ``policy="none"`` run (never hardcoded), then a set of
head-truncation budgets is derived so the cut (``dropped_head``) lands within ``+/-window`` tokens
of the span: *before* it (trigger survives whole), *inside* it (boundary corruption -- the trigger's
back half survives as the literal prefix of the final input), and *after* it (trigger lost whole).
The runner's existing boundary predicate and persisted cut-metadata do the scoring; this module only
derives the budgets and joins the outcomes.

The scientific question: does **partial** trigger survival (boundary corruption) behave differently
from **whole**-trigger survival across ``prefix / middle / end / old_turn`` placements, or is the
cut-through-the-span mechanism position-invariant? Because every persisted row now carries
``metadata.dropped_head`` and ``metadata.pretrunc_trigger_span``, each outcome is self-describing:
the cut *region* is recomputed from the row itself, not from an external budget table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trigger_audit.experiments.survivability_audit.runner import SurvivalShardRunner
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.util.ids import make_grid_trial_id

# The trial the cut sweep truncates from. A dedicated head-only policy so no memory stage interacts
# with the token-level cut (the phenomenon is isolated to Layer 3 -> Layer 4, as in Trial 5).
BOUNDARY_POLICY = "truncate_head"
# The policy used to MEASURE the untruncated span. Must not drop or truncate anything.
MEASURE_POLICY = "none"


@dataclass(frozen=True)
class CutBudget:
    """One derived head-truncation cut for a measured trigger span.

    ``dropped_head`` is the absolute number of front tokens the cut drops (the cut point in the
    pre-truncation coordinate); ``budget`` is the head-truncation ``context_length`` that produces
    it (``budget = total - dropped_head``). ``region`` is the *expected* outcome band relative
    to the span ``[start, end)``: ``before`` (whole survives), ``inside`` (boundary corruption),
    ``after`` (whole lost). ``offset_from_start`` is ``dropped_head - start`` (negative before the
    span, ``0..len`` inside, ``>=len`` after) -- the x-axis of the anatomy-of-the-cut view.
    """

    dropped_head: int
    budget: int
    region: str
    offset_from_start: int


def cut_region(dropped_head: int, span_start: int, span_end: int) -> str:
    """Classify where a head cut landed relative to a trigger span ``[start, end)``.

    This is the single source of truth used both when deriving budgets and when summarizing
    results, so a derived ``region`` and the region recomputed from a persisted row always agree.
    """
    if dropped_head < span_start:
        return "before"
    if dropped_head < span_end:
        return "inside"
    return "after"


def derive_cut_budgets(
    none_result: SurvivalResult, *, window: int = 20, inside_steps: int = 3
) -> list[CutBudget]:
    """Derive head-truncation budgets so the cut sweeps within ``+/-window`` tokens of the span.

    From the measured span ``[S, E)`` and total ``T`` (``none_result``'s located span and token
    count), sweep the cut point ``c = dropped_head`` over: ``window`` tokens *before* ``S`` (whole
    survives), ``inside_steps`` evenly-spaced points strictly *inside* ``[S, E)`` (boundary
    corruption), and ``window`` tokens *after* ``E`` (whole lost). Each cut point maps to
    ``budget = T - c``; points yielding a non-positive or over-``T`` budget are dropped (e.g. an
    ``after`` point past the end of an end-position trigger), never silently -- the caller can see
    which regions a given (base, position) supports. Deduplicated and returned in ascending cut
    order.
    """
    start = none_result.trigger_final_token_start
    end = none_result.trigger_final_token_end
    total = none_result.final_prompt_token_count
    if start is None or end is None:
        raise ValueError("none run did not locate the trigger; cannot derive cut budgets")

    span_len = end - start
    cuts: set[int] = set()
    # Before the span: whole trigger survives.
    for delta in (window, max(1, window // 2), max(1, window // 4)):
        cuts.add(start - delta)
    # Inside the span: boundary corruption. Strictly interior points (0 < offset < span_len).
    if span_len >= 2:
        for step in range(1, inside_steps + 1):
            offset = round(step * span_len / (inside_steps + 1))
            offset = min(max(offset, 1), span_len - 1)
            cuts.add(start + offset)
    # After the span: whole trigger dropped.
    for delta in (0, max(1, window // 2), window):
        cuts.add(end + delta)

    budgets: list[CutBudget] = []
    for c in sorted(cuts):
        budget = total - c
        if budget <= 0 or budget > total:
            continue  # cut before the prompt start or past its end -- unrepresentable, skip
        budgets.append(
            CutBudget(
                dropped_head=c,
                budget=budget,
                region=cut_region(c, start, end),
                offset_from_start=c - start,
            )
        )
    return budgets


def _boundary_trial(
    *,
    base_id: str,
    trigger_id: str,
    position: TriggerPosition,
    model_id: str,
    budget: int,
    trigger_present: bool,
) -> TrialSpec:
    """One head-truncation trial at a derived budget (its counterfactual twin shares the budget)."""
    return TrialSpec(
        trial_id=make_grid_trial_id(
            base_id,
            trigger_id,
            position.value,
            BOUNDARY_POLICY,
            model_id,
            context_length=budget,
            trigger_present=trigger_present,
        ),
        base_id=base_id,
        trigger_id=trigger_id,
        trigger_position=position,
        model_id=model_id,
        tokenizer_id=model_id,
        context_length=budget,
        pipeline_policy=BOUNDARY_POLICY,
        trigger_present=trigger_present,
        run_generation=False,
    )


def measure_and_expand(
    runner: SurvivalShardRunner,
    *,
    base_ids: Sequence[str],
    trigger_id: str,
    positions: Sequence[TriggerPosition],
    model_id: str,
    window: int = 20,
    inside_steps: int = 3,
) -> tuple[list[TrialSpec], dict[str, SurvivalResult]]:
    """Measure each (base, position) span, then expand per-trial head-truncation cut trials.

    Returns the boundary manifest (each present cut trial plus its trigger-absent twin at the same
    budget) and, for provenance, the ``none``-run :class:`SurvivalResult` per ``(base, position)``
    keyed ``f"{base_id}|{position.value}"``. The manifest is what a cluster shard runs; the runner
    needs a ``none`` and a ``truncate_head`` policy configured (both in the prod policy set).
    """
    trials: list[TrialSpec] = []
    measured: dict[str, SurvivalResult] = {}
    for base_id in base_ids:
        for position in positions:
            none_trial = TrialSpec(
                trial_id=make_grid_trial_id(
                    base_id,
                    trigger_id,
                    position.value,
                    MEASURE_POLICY,
                    model_id,
                    context_length=0,
                    trigger_present=True,
                ),
                base_id=base_id,
                trigger_id=trigger_id,
                trigger_position=position,
                model_id=model_id,
                tokenizer_id=model_id,
                context_length=0,
                pipeline_policy=MEASURE_POLICY,
                trigger_present=True,
                run_generation=False,
            )
            none_result, _ = runner.run_trial(none_trial)
            measured[f"{base_id}|{position.value}"] = none_result
            if none_result.trigger_final_token_start is None:
                # The trigger was not delivered even under 'none' (e.g. a slot the base lacks);
                # there is no span to cut, so this cell contributes no boundary trials.
                continue
            for cut in derive_cut_budgets(none_result, window=window, inside_steps=inside_steps):
                for present in (True, False):
                    trials.append(
                        _boundary_trial(
                            base_id=base_id,
                            trigger_id=trigger_id,
                            position=position,
                            model_id=model_id,
                            budget=cut.budget,
                            trigger_present=present,
                        )
                    )
    return trials, measured


def summarize(
    results: Sequence[SurvivalResult],
    manifest: dict[str, bool],
) -> list[dict[str, object]]:
    """Aggregate boundary outcomes by ``position x cut-region`` over trigger-present rows.

    ``manifest`` maps ``trial_id -> trigger_present`` (from the boundary manifest). The cut region
    of each present row is recomputed from its own persisted ``metadata`` (``dropped_head`` vs
    ``pretrunc_trigger_span``), so the summary is self-contained. Each group reports the count and
    the rate of whole (``exact``), boundary (``partial``/``boundary_corruption``), and lost
    (``no_survival``) outcomes -- the partial-vs-whole comparison the experiment measures.
    """
    groups: dict[tuple[str, str], dict[str, int]] = {}
    for r in results:
        if not manifest.get(r.trial_id, False):
            continue  # counterfactual twins are the control, not part of the rate table
        span = r.metadata.get("pretrunc_trigger_span")
        dropped = r.metadata.get("dropped_head")
        if not isinstance(span, list) or len(span) != 2 or not isinstance(dropped, int):
            region = "unknown"
        else:
            region = cut_region(dropped, int(span[0]), int(span[1]))
        key = (r.trigger_position.value, region)
        g = groups.setdefault(
            key, {"n": 0, "whole": 0, "boundary": 0, "lost": 0, "partial_flag": 0}
        )
        g["n"] += 1
        g["partial_flag"] += int(r.trigger_partial_survived)
        if r.survival_class.value == "exact_survival":
            g["whole"] += 1
        elif r.survival_class.value == "boundary_corruption":
            g["boundary"] += 1
        else:
            g["lost"] += 1

    rows: list[dict[str, object]] = []
    for (position, region), g in sorted(groups.items()):
        n = g["n"] or 1
        rows.append(
            {
                "position": position,
                "cut_region": region,
                "n": g["n"],
                "whole_rate": g["whole"] / n,
                "boundary_rate": g["boundary"] / n,
                "lost_rate": g["lost"] / n,
                "partial_survived_rate": g["partial_flag"] / n,
            }
        )
    return rows
