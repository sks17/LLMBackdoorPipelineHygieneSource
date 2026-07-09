"""Tests for core pydantic schema validation and coercion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trigger_audit.schemas import (
    ChatMessage,
    Role,
    SurvivalClass,
    SurvivalResult,
    TrialSpec,
    TriggerPosition,
    TriggerSpec,
    TriggerType,
)


def test_chat_message_role_coercion():
    message = ChatMessage(role="user", content="hi")
    assert message.role is Role.USER


def test_invalid_role_rejected():
    with pytest.raises(ValidationError):
        ChatMessage(role="wizard", content="hi")


def test_trigger_spec_defaults():
    trigger = TriggerSpec(trigger_id="t1", trigger_type=TriggerType.RANDOM_CANARY, text="C")
    assert trigger.parts == []
    assert trigger.slot is None


def test_trial_resolved_tokenizer_defaults_to_model():
    trial = TrialSpec(
        trial_id="t_1",
        base_id="b1",
        trigger_id="r1",
        trigger_position=TriggerPosition.PREFIX,
        model_id="qwen3-4b",
        context_length=4096,
        pipeline_policy="none",
    )
    assert trial.resolved_tokenizer_id() == "qwen3-4b"
    assert trial.run_generation is False


def test_survival_result_minimal_construction():
    result = SurvivalResult(
        trial_id="t_1",
        base_id="b1",
        model_id="qwen3-4b",
        tokenizer_id="qwen3-4b",
        trigger_id="r1",
        trigger_text="CANARY",
        trigger_position=TriggerPosition.PREFIX,
        context_length=4096,
        pipeline_policy="none",
        raw_trigger_present=True,
        post_pipeline_trigger_present=True,
        post_template_trigger_present=True,
        final_token_trigger_present=True,
        trigger_exact_survived=True,
        trigger_token_survived=True,
        trigger_partial_survived=False,
        final_prompt_token_count=128,
        survival_class=SurvivalClass.EXACT_SURVIVAL,
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    # JSON round-trip preserves enums as their string values.
    assert result.model_dump(mode="json")["trigger_position"] == "prefix"
