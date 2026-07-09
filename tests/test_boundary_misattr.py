"""Tests for T6 misattribution + T7 boundary-census tables and their report wiring.

The T6 bucket-sum invariant is checked on a synthetic fixture with hand-chosen ``failure_stage``
values; T7 is checked both on the checked-in smoke artifact (which may or may not carry boundary
rows -- both branches are asserted) and on a synthetic boundary row that exercises the non-empty
path. ``build_report`` is run end to end on the smoke to confirm the relabelled/new tables emit the
``.csv``/``.md``/``.tex`` triple.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trigger_audit.analysis.loading import load_trials, present_rows
from trigger_audit.analysis.report import build_report
from trigger_audit.analysis.tables import boundary_census_table, misattribution_table
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SURVIVAL = REPO_ROOT / "outputs" / "task1_smoke" / "survival.jsonl"

_BOUNDARY_COLS = [
    "pipeline_policy",
    "trigger_position",
    "context_length",
    "base_id",
    "trigger_id",
    "budget",
    "surviving_suffix_len",
    "surviving_fraction",
    "cut_offset",
    "final_prompt_text_path",
]


def _res(
    trial_id: str,
    *,
    delivered: bool,
    failure_stage: FailureStage,
    survival_class: SurvivalClass = SurvivalClass.NO_SURVIVAL,
    base_id: str = "b0",
    policy: str = "truncate_head",
    position: TriggerPosition = TriggerPosition.PREFIX,
    trigger_id: str = "rand_001",
    trigger_final_token_end: int | None = None,
    metadata: dict | None = None,
) -> SurvivalResult:
    """Minimal SurvivalResult whose delivered/failure_stage drive the misattribution buckets."""
    return SurvivalResult(
        trial_id=trial_id,
        base_id=base_id,
        model_id="m0",
        tokenizer_id="m0",
        trigger_id=trigger_id,
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=position,
        context_length=256,
        pipeline_policy=policy,
        raw_trigger_present=True,
        post_pipeline_trigger_present=True,
        post_template_trigger_present=True,
        final_token_trigger_present=delivered,
        trigger_exact_survived=delivered and survival_class == SurvivalClass.EXACT_SURVIVAL,
        trigger_token_survived=delivered,
        trigger_partial_survived=survival_class == SurvivalClass.BOUNDARY_CORRUPTION,
        trigger_final_token_end=trigger_final_token_end,
        final_prompt_token_count=100,
        survival_class=survival_class,
        failure_stage=failure_stage,
        metadata=metadata or {},
    )


def test_misattribution_buckets_sum_to_non_delivered(tmp_path: Path) -> None:
    # One (policy, position) cell: 2 delivered, 10 non-delivered spanning every mapped bucket.
    rows = [
        _res(
            "d0",
            delivered=True,
            failure_stage=FailureStage.NONE,
            survival_class=SurvivalClass.EXACT_SURVIVAL,
            base_id="b0",
        ),
        _res(
            "d1",
            delivered=True,
            failure_stage=FailureStage.NONE,
            survival_class=SurvivalClass.EXACT_SURVIVAL,
            base_id="b1",
        ),
        _res("u0", delivered=False, failure_stage=FailureStage.MEMORY_POLICY_DROPPED, base_id="b2"),
        _res("u1", delivered=False, failure_stage=FailureStage.NOT_RETRIEVED, base_id="b3"),
        _res(
            "u2", delivered=False, failure_stage=FailureStage.PACKING_BUDGET_EXCLUDED, base_id="b4"
        ),
        _res("t0", delivered=False, failure_stage=FailureStage.TRUNCATED_HEAD, base_id="b5"),
        _res("t1", delivered=False, failure_stage=FailureStage.TRUNCATED_TAIL, base_id="b6"),
        _res("t2", delivered=False, failure_stage=FailureStage.TRUNCATED_MIDDLE, base_id="b7"),
        _res("t3", delivered=False, failure_stage=FailureStage.FINAL_TOKEN_ABSENT, base_id="b8"),
        _res("m0", delivered=False, failure_stage=FailureStage.TEMPLATE_INCOMPATIBLE, base_id="b9"),
        _res(
            "m1",
            delivered=False,
            failure_stage=FailureStage.TEMPLATE_REMOVED_OR_CHANGED,
            base_id="b10",
        ),
        _res(
            "c0",
            delivered=False,
            failure_stage=FailureStage.COMPRESSED_EXACT_DELETED,
            base_id="b11",
        ),
    ]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, _ = load_trials(path)
    table = misattribution_table(present_rows(df))

    assert len(table) == 1
    cell = table.iloc[0]
    assert cell["n"] == 12
    assert cell["delivered"] == 2
    non_delivered = int(cell["n"]) - int(cell["delivered"])
    assert non_delivered == 10
    assert cell["apparent_failure_rate"] == pytest.approx(10 / 12)

    bucket_cols = [c for c in table.columns if c.endswith("_n")]
    assert int(sum(int(cell[c]) for c in bucket_cols)) == non_delivered
    # Exact decomposition per the documented mapping.
    assert int(cell["upstream_drop_n"]) == 3
    assert int(cell["token_truncation_n"]) == 4
    assert int(cell["template_incompatible_n"]) == 2
    assert int(cell["compressed_n"]) == 1
    assert int(cell["other_n"]) == 0
    # Proportions are over the non-delivered rows and sum to 1.
    prop_cols = [c for c in table.columns if c.endswith("_prop")]
    assert sum(float(cell[c]) for c in prop_cols) == pytest.approx(1.0)


def test_boundary_census_non_empty_on_synthetic(tmp_path: Path) -> None:
    # A boundary row: span [100,110] (len 10), 4 tokens survived -> surviving_fraction 0.4 in (0,1);
    # dropped_head 98 -> cut_offset = 100 - 98 = 2 (cut lands inside the trigger).
    row = _res(
        "bnd0",
        delivered=True,
        failure_stage=FailureStage.NONE,
        survival_class=SurvivalClass.BOUNDARY_CORRUPTION,
        policy="truncate_head",
        trigger_final_token_end=4,
        metadata={
            "truncation_policy": "truncate_head",
            "dropped_head": 98,
            "dropped_tail": 0,
            "pretrunc_token_count": 200,
            "pretrunc_trigger_span": [100, 110],
        },
    )
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [row])
    df, _ = load_trials(path)
    table = boundary_census_table(present_rows(df))

    assert list(table.columns) == _BOUNDARY_COLS
    assert len(table) == 1
    cell = table.iloc[0]
    assert 0.0 < float(cell["surviving_fraction"]) < 1.0
    assert float(cell["surviving_fraction"]) == pytest.approx(0.4)
    assert int(cell["surviving_suffix_len"]) == 4
    assert int(cell["budget"]) == 256
    assert float(cell["cut_offset"]) == pytest.approx(2.0)


def test_boundary_census_empty_is_well_formed(tmp_path: Path) -> None:
    row = _res("nd0", delivered=False, failure_stage=FailureStage.TRUNCATED_HEAD)
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [row])
    df, _ = load_trials(path)
    table = boundary_census_table(present_rows(df))
    assert list(table.columns) == _BOUNDARY_COLS
    assert table.empty


@pytest.mark.skipif(not SMOKE_SURVIVAL.exists(), reason="smoke survival artifact not present")
def test_boundary_census_on_smoke_matches_its_boundary_rows() -> None:
    df, _ = load_trials(SMOKE_SURVIVAL)
    present = present_rows(df)
    table = boundary_census_table(present)
    assert list(table.columns) == _BOUNDARY_COLS
    has_boundary = bool((present["outcome_band"] == "boundary").any())
    if has_boundary:
        assert len(table) >= 1
        frac = table["surviving_fraction"]
        assert bool(((frac > 0) & (frac < 1)).any())
    else:
        assert table.empty


@pytest.mark.skipif(not SMOKE_SURVIVAL.exists(), reason="smoke survival artifact not present")
def test_build_report_emits_relabelled_and_new_tables(tmp_path: Path) -> None:
    out = tmp_path / "report"
    build_report(SMOKE_SURVIVAL, out, render_figures=False)
    tables_dir = out / "tables"
    for name in (
        "t4_h2_model_invariance",
        "t5_h4_synthetic_vs_real",
        "t6_misattribution",
        "t7_boundary_census",
    ):
        for ext in ("csv", "md", "tex"):
            path = tables_dir / f"{name}.{ext}"
            assert path.exists() and path.stat().st_size > 0, f"missing {path}"
    # The old TOST keys must no longer be emitted under their pre-relabel names.
    assert not (tables_dir / "t6_h2_model_invariance.csv").exists()
    assert not (tables_dir / "t7_h4_synthetic_vs_real.csv").exists()
