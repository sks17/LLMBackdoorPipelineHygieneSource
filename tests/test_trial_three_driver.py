"""Offline tests for the Trial Three driver (composed memory + truncation).

Runs the three conditions through the dependency-free ``SimpleWhitespaceTokenizerAdapter`` with the
tight budget derived per-adapter from condition (b) -- never hardcoded. Beyond the per-row table,
these pin the composition-specific claims: stage ordering beats declaration order (reversal
invariance), memory pre-empts truncation for (a) regardless of budget, and (c) is the first
present->absent transition between Layer 2 and Layer 4.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_three_spec as t3
from trigger_audit.experiments.survivability_audit.trial_three import run_trial_three
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter


def _tight_budget(adapter: SimpleWhitespaceTokenizerAdapter) -> int:
    # Condition (b): recent_turn, generous budget (no truncation) -> derive the tight budget.
    result_b = run_trial_three(
        tokenizer_adapter=adapter,
        trigger_position=TriggerPosition.RECENT_TURN,
        context_length_target=t3.GENEROUS_BUDGET,
    )
    return t3.derive_tight_budget(result_b)


def test_trial_three_a_old_turn_dropped_by_memory():
    result = run_trial_three(
        tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
        trigger_position=TriggerPosition.OLD_TURN,
        context_length_target=t3.GENEROUS_BUDGET,
    )
    assert result.trial_id == "trial_three_a"
    assert result.post_pipeline_trigger_present is False
    assert result.final_token_trigger_present is False
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.MEMORY_POLICY_DROPPED
    assert result.trigger_partial_survived is False


def test_trial_three_b_recent_turn_survives():
    result = run_trial_three(
        tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
        trigger_position=TriggerPosition.RECENT_TURN,
        context_length_target=t3.GENEROUS_BUDGET,
    )
    assert result.trial_id == "trial_three_b"
    assert result.post_pipeline_trigger_present is True
    assert result.final_token_trigger_present is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.failure_stage is FailureStage.NONE
    assert result.trigger_partial_survived is False


def test_trial_three_c_recent_turn_kept_but_truncated():
    adapter = SimpleWhitespaceTokenizerAdapter()
    tight = _tight_budget(adapter)
    result = run_trial_three(
        tokenizer_adapter=adapter,
        trigger_position=TriggerPosition.RECENT_TURN,
        context_length_target=tight,
    )
    assert result.trial_id == "trial_three_c"
    # The new transition: kept by memory (Layer 2) but cut by truncation (Layer 4).
    assert result.post_pipeline_trigger_present is True
    assert result.final_token_trigger_present is False
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.TRUNCATED_HEAD
    assert result.trigger_partial_survived is False


def test_stage_order_beats_declaration_order():
    # Reversing the declared policy chain must not change any condition's result.
    adapter = SimpleWhitespaceTokenizerAdapter()
    tight = _tight_budget(adapter)
    conditions = [
        (TriggerPosition.OLD_TURN, t3.GENEROUS_BUDGET),
        (TriggerPosition.RECENT_TURN, t3.GENEROUS_BUDGET),
        (TriggerPosition.RECENT_TURN, tight),
    ]
    for position, budget in conditions:
        forward = run_trial_three(
            tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
            trigger_position=position,
            context_length_target=budget,
        )
        reversed_ = run_trial_three(
            tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
            trigger_position=position,
            context_length_target=budget,
            reverse_chain=True,
        )
        assert forward == reversed_


def test_trial_three_a_is_budget_independent():
    # (a)'s fate is sealed at Layer 2, so the truncation budget cannot change the trigger outcome.
    adapter = SimpleWhitespaceTokenizerAdapter()
    tight = _tight_budget(adapter)
    for budget in (t3.GENEROUS_BUDGET, tight):
        result = run_trial_three(
            tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
            trigger_position=TriggerPosition.OLD_TURN,
            context_length_target=budget,
        )
        assert result.final_token_trigger_present is False
        assert result.failure_stage is FailureStage.MEMORY_POLICY_DROPPED
        assert result.trigger_partial_survived is False
