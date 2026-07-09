"""Tests that the additive semantic fields on SurvivalResult stay backward compatible."""

from __future__ import annotations

from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition


def _base_kwargs() -> dict[str, object]:
    """A pre-existing-style SurvivalResult payload with NO semantic fields."""
    return {
        "trial_id": "t_1",
        "base_id": "b1",
        "model_id": "m",
        "tokenizer_id": "simple-whitespace",
        "trigger_id": "r1",
        "trigger_text": "CANARY",
        "trigger_position": TriggerPosition.OLD_TURN,
        "context_length": 4096,
        "pipeline_policy": "summarize_old_messages",
        "raw_trigger_present": True,
        "post_pipeline_trigger_present": False,
        "post_template_trigger_present": False,
        "final_token_trigger_present": False,
        "trigger_exact_survived": False,
        "trigger_token_survived": False,
        "trigger_partial_survived": False,
        "final_prompt_token_count": 100,
        "survival_class": SurvivalClass.NO_SURVIVAL,
        "failure_stage": FailureStage.COMPRESSED_EXACT_DELETED,
    }


def test_constructs_without_semantic_fields_and_defaults_hold():
    result = SurvivalResult(**_base_kwargs())
    assert result.trigger_semantic_survived is False
    assert result.trigger_semantic_score is None


def test_constructs_with_semantic_fields():
    result = SurvivalResult(
        **_base_kwargs(),
        trigger_semantic_survived=True,
        trigger_semantic_score=0.87,
    )
    assert result.trigger_semantic_survived is True
    assert result.trigger_semantic_score == 0.87


def test_round_trips_through_model_dump_and_validate():
    original = SurvivalResult(
        **_base_kwargs(),
        trigger_semantic_survived=True,
        trigger_semantic_score=0.5,
    )
    restored = SurvivalResult.model_validate(original.model_dump())
    assert restored == original


def test_pre_existing_row_without_semantic_fields_validates():
    # A row serialized before the semantic axis existed (no keys present) must still validate.
    legacy_row = _base_kwargs()
    assert "trigger_semantic_survived" not in legacy_row
    restored = SurvivalResult.model_validate(legacy_row)
    assert restored.trigger_semantic_survived is False
    assert restored.trigger_semantic_score is None
