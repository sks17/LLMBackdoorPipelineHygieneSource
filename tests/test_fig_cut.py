"""Tests for F6 (anatomy of the cut): a real render on the smoke frame and the empty-frame fallback.

F6 must render a non-empty PNG when head-cut rows carry cut geometry, and must degrade to a valid
annotated figure (never raise, never silently omit) when no row has a non-NaN ``cut_offset``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trigger_audit.analysis.loading import load_trials, present_rows
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SURVIVAL = REPO_ROOT / "outputs" / "task1_smoke" / "survival.jsonl"


def _res(trial_id: str, *, base_id: str) -> SurvivalResult:
    """A delivered head-truncation row with no cut metadata -> cut_offset stays NaN."""
    return SurvivalResult(
        trial_id=trial_id,
        base_id=base_id,
        model_id="m0",
        tokenizer_id="m0",
        trigger_id="rand_001",
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=TriggerPosition.PREFIX,
        context_length=256,
        pipeline_policy="truncate_head",
        raw_trigger_present=True,
        post_pipeline_trigger_present=True,
        post_template_trigger_present=True,
        final_token_trigger_present=True,
        trigger_exact_survived=True,
        trigger_token_survived=True,
        trigger_partial_survived=False,
        final_prompt_token_count=100,
        survival_class=SurvivalClass.EXACT_SURVIVAL,
        failure_stage=FailureStage.NONE,
        metadata={},
    )


@pytest.mark.skipif(not SMOKE_SURVIVAL.exists(), reason="smoke survival artifact not present")
def test_fig_cut_anatomy_renders_on_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from trigger_audit.analysis import figures

    df, _ = load_trials(SMOKE_SURVIVAL)
    present = present_rows(df)
    # The smoke carries head-cut rows with cut geometry -> a non-empty scatter.
    assert bool(present["cut_offset"].notna().any())
    out = figures.fig_cut_anatomy(present, tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_fig_cut_anatomy_empty_frame_does_not_raise(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from trigger_audit.analysis import figures

    rows = [_res(f"p{i}", base_id=f"b{i}") for i in range(3)]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, _ = load_trials(path)
    present = present_rows(df)
    assert not bool(present["cut_offset"].notna().any())
    out = figures.fig_cut_anatomy(present, tmp_path / "figs")
    assert out.exists() and out.stat().st_size > 0
