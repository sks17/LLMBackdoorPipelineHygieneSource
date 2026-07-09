"""Offline tests for the Trial One driver (naive head truncation, prefix vs end).

Both variants run through the dependency-free ``SimpleWhitespaceTokenizerAdapter`` with the budget
derived from the trigger's measured span (never hardcoded). The prefix trigger is destroyed by the
head cut while the end trigger survives, and both prompts are truncated to exactly the derived
budget. The live-tokenizer reproduction lives in ``test_trial_one.py``.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_one_spec as t1
from trigger_audit.experiments.survivability_audit.trial_one import run_trial_one
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter


def test_prefix_destroyed_by_head_truncation():
    adapter = SimpleWhitespaceTokenizerAdapter()
    target = t1.derive_context_length_target(adapter)
    result = run_trial_one(
        tokenizer_adapter=adapter,
        trigger_position=TriggerPosition.PREFIX,
        context_length_target=target,
    )
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.final_token_trigger_present is False
    assert result.failure_stage is FailureStage.TRUNCATED_HEAD
    # Present before truncation -- localizes the failure to truncation, not upstream.
    assert result.post_template_trigger_present is True
    assert result.final_prompt_token_count == target


def test_end_survives_head_truncation():
    adapter = SimpleWhitespaceTokenizerAdapter()
    target = t1.derive_context_length_target(adapter)
    result = run_trial_one(
        tokenizer_adapter=adapter,
        trigger_position=TriggerPosition.END,
        context_length_target=target,
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.final_token_trigger_present is True
    assert result.post_template_trigger_present is True
    assert result.final_prompt_token_count == target
