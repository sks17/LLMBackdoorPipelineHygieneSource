"""Tests for the stratified, ``base_id``-aware probe subset selector and its Gate 0 adapter.

Covers the whole-``base_id`` grouping invariant, per-stratum classification, shortfall bookkeeping,
seeded determinism, the stratified sample spread, and that ``verify_subset_counterfactual`` passes
on a clean twin set and fails when an absent twin leaks.
"""

from __future__ import annotations

from pathlib import Path

from trigger_audit.experiments.probe_detection.selection import (
    StratumTargets,
    SubsetSelection,
    select_probe_subset,
    subset_report,
    verify_subset_counterfactual,
    write_selected_trial_ids,
)
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition


def _row(
    trial_id: str,
    base_id: str,
    *,
    raw: bool,
    delivered: bool,
    survival_class: SurvivalClass,
    policy: str = "none",
    position: TriggerPosition = TriggerPosition.PREFIX,
    context_length: int = 256,
    trigger_id: str = "rand_001",
) -> SurvivalResult:
    """Build a SurvivalResult with only the fields the selector reads varied."""
    return SurvivalResult(
        trial_id=trial_id,
        base_id=base_id,
        model_id="m0",
        tokenizer_id="m0",
        trigger_id=trigger_id,
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=position,
        context_length=context_length,
        pipeline_policy=policy,
        raw_trigger_present=raw,
        post_pipeline_trigger_present=raw,
        post_template_trigger_present=raw,
        final_token_trigger_present=delivered,
        trigger_exact_survived=delivered and survival_class == SurvivalClass.EXACT_SURVIVAL,
        trigger_token_survived=delivered,
        trigger_partial_survived=survival_class
        in (SurvivalClass.PARTIAL_SURVIVAL, SurvivalClass.BOUNDARY_CORRUPTION),
        final_prompt_token_count=100,
        survival_class=survival_class,
        failure_stage=FailureStage.NONE if delivered else FailureStage.FINAL_TOKEN_ABSENT,
    )


# kind, policy, position, context_length -- one position per base guarantees distinct cells.
_BASE_CONFIGS: list[tuple[str, str, str, TriggerPosition, int]] = [
    ("d0", "delivered", "none", TriggerPosition.PREFIX, 256),
    ("d1", "delivered", "none", TriggerPosition.EARLY, 8000),
    ("d2", "delivered", "truncate_tail", TriggerPosition.MIDDLE, 16000),
    ("d3", "delivered", "compress", TriggerPosition.LATE, 32000),
    ("d4", "delivered", "truncate_head", TriggerPosition.END, 4000),
    ("d5", "delivered", "none", TriggerPosition.NEAR_BOUNDARY, 1000),
    ("p0", "partial", "truncate_head", TriggerPosition.OLD_TURN, 16000),
    ("p1", "partial", "truncate_middle", TriggerPosition.RECENT_TURN, 8000),
    ("bd0", "boundary", "compress", TriggerPosition.SYSTEM, 32000),
    ("bd1", "boundary", "truncate_tail", TriggerPosition.TOOL_OUTPUT, 2000),
]


def _base_rows(
    base_id: str, kind: str, policy: str, position: TriggerPosition, context_length: int
) -> list[SurvivalResult]:
    """A base's present trial (kind-dependent) plus its clean counterfactual (absent) twin."""
    if kind == "delivered":
        present = _row(
            f"{base_id}_present",
            base_id,
            raw=True,
            delivered=True,
            survival_class=SurvivalClass.EXACT_SURVIVAL,
            policy=policy,
            position=position,
            context_length=context_length,
        )
    elif kind == "partial":
        present = _row(
            f"{base_id}_present",
            base_id,
            raw=True,
            delivered=False,
            survival_class=SurvivalClass.PARTIAL_SURVIVAL,
            policy=policy,
            position=position,
            context_length=context_length,
        )
    elif kind == "boundary":
        present = _row(
            f"{base_id}_present",
            base_id,
            raw=True,
            delivered=True,
            survival_class=SurvivalClass.BOUNDARY_CORRUPTION,
            policy=policy,
            position=position,
            context_length=context_length,
        )
    else:  # pragma: no cover - guard against a typo in the config
        raise ValueError(f"unknown kind {kind!r}")

    absent = _row(
        f"{base_id}_absent",
        base_id,
        raw=False,
        delivered=False,
        survival_class=SurvivalClass.NO_SURVIVAL,
        policy=policy,
        position=position,
        context_length=context_length,
    )
    return [present, absent]


def _make_results() -> list[SurvivalResult]:
    """Full synthetic survival set: 10 bases (6 delivered, 2 partial, 2 boundary), twins each."""
    rows: list[SurvivalResult] = []
    for base_id, kind, policy, position, context_length in _BASE_CONFIGS:
        rows.extend(_base_rows(base_id, kind, policy, position, context_length))
    return rows


def _trials_by_base(results: list[SurvivalResult]) -> dict[str, set[str]]:
    by_base: dict[str, set[str]] = {}
    for r in results:
        by_base.setdefault(r.base_id, set()).add(r.trial_id)
    return by_base


def test_selection_respects_whole_base_grouping() -> None:
    results = _make_results()
    by_base = _trials_by_base(results)
    targets = StratumTargets(
        delivered_positive=3,
        clean_negative=3,
        partial_survival_negative=1,
        boundary_corruption=1,
        stratified_sample=3,
    )
    selection = select_probe_subset(results, targets, seed=0)

    selected_trials = set(selection.trial_ids)
    selected_bases = set(selection.base_ids)
    # A base is fully in or fully out -- never split.
    for base_id, trials in by_base.items():
        if base_id in selected_bases:
            assert trials <= selected_trials
        else:
            assert trials.isdisjoint(selected_trials)
    # trial_ids is exactly the union of the selected bases' trials, sorted and unique.
    expected = sorted(t for b in selected_bases for t in by_base[b])
    assert selection.trial_ids == expected
    assert len(selection.trial_ids) == len(set(selection.trial_ids))


def test_per_stratum_counts_classify_populations() -> None:
    results = _make_results()
    # Very high targets pull in every base, so achieved counts equal the whole-set totals.
    targets = StratumTargets(
        delivered_positive=999,
        clean_negative=999,
        partial_survival_negative=999,
        boundary_corruption=999,
        stratified_sample=999,
    )
    selection = select_probe_subset(results, targets, seed=0)

    assert set(selection.base_ids) == {b for b, *_ in _BASE_CONFIGS}
    counts = selection.per_stratum_counts
    assert counts["delivered_positive"] == 8  # 6 delivered + 2 boundary present trials
    assert counts["clean_negative"] == 10  # one absent twin per base
    assert counts["partial_survival_negative"] == 2  # p0, p1 present trials
    assert counts["boundary_corruption"] == 2  # bd0, bd1 present trials
    assert counts["stratified_sample"] == 10  # 10 distinct (policy, position, bucket) cells


def test_shortfalls_recorded_not_hidden() -> None:
    results = _make_results()
    # Only two boundary trials exist; asking for five cannot be met.
    targets = StratumTargets(boundary_corruption=5)
    selection = select_probe_subset(results, targets, seed=0)

    assert selection.per_stratum_counts["boundary_corruption"] == 2
    assert selection.shortfalls["boundary_corruption"] == 3
    assert selection.shortfalls["delivered_positive"] == 0  # not requested -> no shortfall
    report = subset_report(selection)
    assert "SHORTFALL" in report
    assert "boundary_corruption" in report


def test_determinism_and_seed_sensitivity() -> None:
    results = _make_results()
    # A small target with many equally-qualified delivered bases: the seed decides which are picked.
    targets = StratumTargets(delivered_positive=2)

    a = select_probe_subset(results, targets, seed=0)
    b = select_probe_subset(results, targets, seed=0)
    assert a.base_ids == b.base_ids
    assert a.trial_ids == b.trial_ids

    # Some seed selects a different base set (determinism does not mean seed-invariance).
    assert any(
        select_probe_subset(results, targets, seed=s).base_ids != a.base_ids for s in range(1, 25)
    )


def test_stratified_sample_spreads_across_cells() -> None:
    results = _make_results()
    targets = StratumTargets(stratified_sample=4)
    selection = select_probe_subset(results, targets, seed=1)

    assert selection.per_stratum_counts["stratified_sample"] == 4
    assert selection.shortfalls["stratified_sample"] == 0

    selected = set(selection.trial_ids)
    cells = {
        (r.pipeline_policy, r.trigger_position.value, r.context_length)
        for r in results
        if r.trial_id in selected
    }
    # The subset genuinely spans multiple distinct covariate cells, not one dominant one.
    assert len({(policy, position) for policy, position, _ in cells}) >= 4


def test_write_selected_trial_ids_round_trips(tmp_path: Path) -> None:
    results = _make_results()
    selection = select_probe_subset(results, StratumTargets(delivered_positive=2), seed=0)
    path = tmp_path / "selection.json"
    write_selected_trial_ids(path, selection)

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trial_ids"] == selection.trial_ids
    assert payload["base_ids"] == selection.base_ids
    assert payload["seed"] == selection.seed
    assert payload["shortfalls"] == selection.shortfalls
    assert payload["requested"]["delivered_positive"] == 2


def _select_all(results: list[SurvivalResult]) -> SubsetSelection:
    """A selection over every trial (isolates the Gate-0 adapter from the greedy fill)."""
    return SubsetSelection(
        trial_ids=[r.trial_id for r in results],
        base_ids=sorted({r.base_id for r in results}),
        per_stratum_counts={},
        requested=StratumTargets(),
        shortfalls={},
        seed=0,
    )


def test_gate0_passes_on_clean_twin_set() -> None:
    results = _make_results()  # every absent twin is no_survival / not delivered
    verdict = verify_subset_counterfactual(results, _select_all(results))
    assert verdict.ok
    assert verdict.n_absent == 10
    assert verdict.n_leaks == 0


def test_gate0_fails_when_absent_twin_leaks() -> None:
    results = _make_results()
    # Inject a leak: an absent (never-inserted) twin that nonetheless "delivered".
    leaky = _row(
        "leak_absent",
        "leak_base",
        raw=False,
        delivered=True,
        survival_class=SurvivalClass.EXACT_SURVIVAL,
    )
    results.append(leaky)

    verdict = verify_subset_counterfactual(results, _select_all(results))
    assert not verdict.ok
    assert verdict.n_leaks == 1
    assert verdict.leak_examples[0]["trial_id"] == "leak_absent"


def test_gate0_only_checks_the_selected_subset() -> None:
    results = _make_results()
    leaky = _row(
        "leak_absent",
        "leak_base",
        raw=False,
        delivered=True,
        survival_class=SurvivalClass.EXACT_SURVIVAL,
    )
    results.append(leaky)

    # A selection that excludes the leaky base must not see the leak (subset scoping).
    clean_selection = SubsetSelection(
        trial_ids=[r.trial_id for r in results if r.base_id != "leak_base"],
        base_ids=sorted({r.base_id for r in results if r.base_id != "leak_base"}),
        per_stratum_counts={},
        requested=StratumTargets(),
        shortfalls={},
        seed=0,
    )
    assert verify_subset_counterfactual(results, clean_selection).ok
