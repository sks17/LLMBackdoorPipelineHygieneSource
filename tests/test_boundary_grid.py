"""Offline tests for the boundary cut-sweep logic (no tokenizer/network needed).

Covers the pure pieces: ``cut_region`` classification, ``derive_cut_budgets`` (window sweep,
budget arithmetic, dropping unrepresentable cuts), and ``summarize`` (region recomputed from a
row's persisted metadata). The end-to-end measure+run is exercised by ``scripts/run_boundary_grid``
against the live tokenizer.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit.boundary_grid import (
    cut_region,
    derive_cut_budgets,
    summarize,
)
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition


def _result(
    trial_id: str,
    *,
    position: str = "prefix",
    survival_class: SurvivalClass = SurvivalClass.EXACT_SURVIVAL,
    partial: bool = False,
    dropped_head: int | None = None,
    span: list[int] | None = None,
) -> SurvivalResult:
    """Minimal SurvivalResult carrying just the fields the summary reads."""
    meta: dict[str, object] = {}
    if dropped_head is not None and span is not None:
        meta = {"dropped_head": dropped_head, "pretrunc_trigger_span": span}
    return SurvivalResult(
        trial_id=trial_id,
        base_id="b",
        model_id="m",
        tokenizer_id="m",
        trigger_id="boundary_001",
        trigger_text="X",
        trigger_position=TriggerPosition(position),
        context_length=10,
        pipeline_policy="truncate_head",
        chat_template=None,
        run_generation=False,
        raw_trigger_present=True,
        post_pipeline_trigger_present=True,
        post_template_trigger_present=True,
        final_token_trigger_present=survival_class is SurvivalClass.EXACT_SURVIVAL,
        trigger_exact_survived=survival_class is SurvivalClass.EXACT_SURVIVAL,
        trigger_token_survived=survival_class is SurvivalClass.EXACT_SURVIVAL,
        trigger_partial_survived=partial,
        final_prompt_token_count=100,
        survival_class=survival_class,
        failure_stage=FailureStage.NONE,
        metadata=meta,
    )


def _none_result(start: int, end: int, total: int) -> SurvivalResult:
    """A 'none' measurement row exposing a located span for budget derivation."""
    r = _result("none")
    r.trigger_final_token_start = start
    r.trigger_final_token_end = end
    r.final_prompt_token_count = total
    return r


def test_cut_region_boundaries() -> None:
    # start=30, end=48 -> before < 30 <= inside < 48 <= after
    assert cut_region(29, 30, 48) == "before"
    assert cut_region(30, 30, 48) == "inside"  # first token of the span is 'inside'
    assert cut_region(47, 30, 48) == "inside"  # last interior token
    assert cut_region(48, 30, 48) == "after"  # end is exclusive


def test_derive_cut_budgets_sweeps_all_three_regions() -> None:
    cuts = derive_cut_budgets(_none_result(start=30, end=48, total=538), window=20, inside_steps=3)
    regions = {c.region for c in cuts}
    assert regions == {"before", "inside", "after"}
    # budget = total - dropped_head, and dropped_head reconstructs the region against [30, 48)
    for c in cuts:
        assert c.budget == 538 - c.dropped_head
        assert cut_region(c.dropped_head, 30, 48) == c.region
        assert c.offset_from_start == c.dropped_head - 30
    # interior points are strictly inside the span (never at or past the ends)
    for c in cuts:
        if c.region == "inside":
            assert 30 < c.dropped_head < 48 or c.dropped_head == 30


def test_derive_cut_budgets_drops_unrepresentable_cuts() -> None:
    # An end-position trigger sitting at the very end: 'after' cuts fall past the prompt -> dropped.
    cuts = derive_cut_budgets(_none_result(start=90, end=99, total=100), window=20, inside_steps=3)
    # every retained cut has a positive, in-range budget
    assert cuts and all(0 < c.budget <= 100 for c in cuts)
    # no cut point exceeds the total token count (budget would be non-positive)
    assert all(c.dropped_head < 100 for c in cuts)


def test_derive_cut_budgets_requires_located_span() -> None:
    r = _result("none")  # start/end default to None
    try:
        derive_cut_budgets(r)
    except ValueError:
        return
    raise AssertionError("expected ValueError when the none run did not locate the trigger")


def test_summarize_recomputes_region_from_metadata() -> None:
    # span [30,48): before-cut survives whole, inside-cut is boundary+partial, after-cut is lost.
    results = [
        _result(
            "t_before", survival_class=SurvivalClass.EXACT_SURVIVAL, dropped_head=10, span=[30, 48]
        ),
        _result(
            "t_inside",
            survival_class=SurvivalClass.BOUNDARY_CORRUPTION,
            partial=True,
            dropped_head=39,
            span=[30, 48],
        ),
        _result(
            "t_after", survival_class=SurvivalClass.NO_SURVIVAL, dropped_head=60, span=[30, 48]
        ),
        _result("t_twin", survival_class=SurvivalClass.NO_SURVIVAL, dropped_head=39, span=[30, 48]),
    ]
    manifest = {"t_before": True, "t_inside": True, "t_after": True, "t_twin": False}
    rows = {(r["cut_region"]): r for r in summarize(results, manifest)}
    assert rows["before"]["whole_rate"] == 1.0 and rows["before"]["partial_survived_rate"] == 0.0
    assert rows["inside"]["boundary_rate"] == 1.0 and rows["inside"]["partial_survived_rate"] == 1.0
    assert rows["after"]["lost_rate"] == 1.0
    # the trigger-absent twin is excluded from the rate table (control only)
    assert all(r["n"] == 1 for r in rows.values())
