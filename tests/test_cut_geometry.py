"""Tests for the cut-geometry derived columns unpacked from ``SurvivalResult.metadata``.

``attach_derived`` flattens the scorer's persisted cut-anatomy block
(``dropped_head/dropped_tail/pretrunc_token_count/truncation_policy/pretrunc_trigger_span``) into
scalar columns and derives ``trigger_len``, ``surviving_fraction``, and the signed ``cut_offset``
(see ANALYSIS_PLAN.md §3, F6/T7). Two golden tests run against checked-in artifacts: the
metadata-bearing smoke run (``outputs/task1_smoke``) and the older all-empty pilot
(``outputs/pilot``) that must still load with the new columns NaN/None. A synthetic fixture pins the
exact ``surviving_fraction`` arithmetic on a hand-built boundary row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trigger_audit.analysis.loading import load_trials
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK1_SURVIVAL = REPO_ROOT / "outputs" / "task1_smoke" / "survival.jsonl"
PILOT_SURVIVAL = REPO_ROOT / "outputs" / "pilot" / "survival.jsonl"

# The full set of columns _attach_cut_geometry adds; asserted present everywhere.
CUT_GEOMETRY_COLS = [
    "dropped_head",
    "dropped_tail",
    "pretrunc_token_count",
    "truncation_meta_policy",
    "pretrunc_trigger_start",
    "pretrunc_trigger_end",
    "trigger_len",
    "surviving_fraction",
    "cut_offset",
]


@pytest.mark.skipif(not TASK1_SURVIVAL.exists(), reason="task1 smoke artifact not present")
def test_golden_head_truncation_row_cut_geometry() -> None:
    df, _ = load_trials(TASK1_SURVIVAL)
    for col in CUT_GEOMETRY_COLS:
        assert col in df.columns

    # Find a real head-truncation row with a known span programmatically, then pin its geometry.
    qualifying = df[
        (df["pretrunc_trigger_start"] == 1468.0)
        & (df["pretrunc_trigger_end"] == 1473.0)
        & (df["dropped_head"] == 965.0)
    ]
    assert not qualifying.empty, "expected a head-truncation row with span [1468, 1473], head 965"
    row = qualifying.iloc[0]
    # trigger_len is read straight off the span (end - start), no re-tokenization.
    assert row["trigger_len"] == 5.0
    # cut_offset = pretrunc_trigger_start - dropped_head = 1468 - 965 = 503 (positive: the trigger
    # begins after the cut, so it survives whole here).
    assert row["cut_offset"] == 503.0


@pytest.mark.skipif(not PILOT_SURVIVAL.exists(), reason="pilot survival artifact not present")
def test_backward_compat_empty_metadata_all_nan() -> None:
    # Every pilot row has metadata == {}: loading must succeed and leave the new columns NaN/None.
    df, _ = load_trials(PILOT_SURVIVAL)
    for col in CUT_GEOMETRY_COLS:
        assert col in df.columns
        assert df[col].isna().all(), f"{col} should be all NaN/None on the empty-metadata pilot"


def _boundary_result(*, trigger_final_token_end: int, span: list[int]) -> SurvivalResult:
    """A single boundary-corruption row carrying a cut-geometry metadata block (head truncation)."""
    return SurvivalResult(
        trial_id="b_boundary",
        base_id="b0",
        model_id="m0",
        tokenizer_id="m0",
        trigger_id="boundary_001",
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=TriggerPosition.PREFIX,
        context_length=256,
        pipeline_policy="truncate_head",
        raw_trigger_present=True,
        post_pipeline_trigger_present=True,
        post_template_trigger_present=True,
        final_token_trigger_present=False,
        trigger_exact_survived=False,
        trigger_token_survived=False,
        trigger_partial_survived=True,
        trigger_final_token_start=0,
        trigger_final_token_end=trigger_final_token_end,
        final_prompt_token_count=100,
        survival_class=SurvivalClass.BOUNDARY_CORRUPTION,
        failure_stage=FailureStage.TRUNCATED_HEAD,
        metadata={
            "truncation_policy": "truncate_head",
            "dropped_head": span[0],
            "dropped_tail": 0,
            "pretrunc_token_count": 200,
            "pretrunc_trigger_span": span,
        },
    )


def test_surviving_fraction_on_synthetic_boundary_row(tmp_path: Path) -> None:
    # span [100, 110] -> trigger_len 10; a surviving suffix of 4 tokens -> surviving_fraction 0.4.
    row = _boundary_result(trigger_final_token_end=4, span=[100, 110])
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [row])
    df, _ = load_trials(path)

    cell = df.iloc[0]
    assert cell["outcome_band"] == "boundary"
    assert cell["trigger_len"] == 10.0
    assert cell["surviving_fraction"] == pytest.approx(0.4)
    # dropped_head == span start (100) here, so cut_offset = 100 - 100 = 0 (cut at trigger start).
    assert cell["cut_offset"] == 0.0


def test_surviving_fraction_nan_off_boundary(tmp_path: Path) -> None:
    # A delivered (non-boundary) row with a span must NOT get a surviving_fraction.
    delivered = SurvivalResult(
        trial_id="p_exact",
        base_id="b0",
        model_id="m0",
        tokenizer_id="m0",
        trigger_id="rand_001",
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=TriggerPosition.PREFIX,
        context_length=256,
        pipeline_policy="none",
        raw_trigger_present=True,
        post_pipeline_trigger_present=True,
        post_template_trigger_present=True,
        final_token_trigger_present=True,
        trigger_exact_survived=True,
        trigger_token_survived=True,
        trigger_partial_survived=False,
        trigger_final_token_start=10,
        trigger_final_token_end=15,
        final_prompt_token_count=100,
        survival_class=SurvivalClass.EXACT_SURVIVAL,
        failure_stage=FailureStage.NONE,
        metadata={
            "truncation_policy": None,
            "dropped_head": 0,
            "dropped_tail": 0,
            "pretrunc_token_count": 100,
            "pretrunc_trigger_span": [10, 15],
        },
    )
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [delivered])
    df, _ = load_trials(path)

    cell = df.iloc[0]
    assert cell["outcome_band"] == "exact"
    assert cell["trigger_len"] == 5.0
    assert pd.isna(cell["surviving_fraction"])
    # No head cut (dropped_head == 0) => cut_offset undefined.
    assert pd.isna(cell["cut_offset"])
