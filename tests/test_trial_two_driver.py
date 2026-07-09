"""Offline tests for the Trial Two driver (message-level keep_last_n, old vs recent turn).

Both variants run through the dependency-free ``SimpleWhitespaceTokenizerAdapter``. The old-turn
trigger is dropped as a whole message before templating; the recent-turn trigger survives intact.
The load-bearing invariant -- message-granularity policies cannot produce *partial* survival -- is
asserted for both variants. The real-tokenizer cross-check lives in the supervisor's verification.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit.trial_two import run_trial_two
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter


def test_old_turn_dropped_by_memory_policy():
    result = run_trial_two(
        tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
        trigger_position=TriggerPosition.OLD_TURN,
    )
    assert result.raw_trigger_present is True
    assert result.post_pipeline_trigger_present is False  # dropped whole message before templating
    assert result.final_token_trigger_present is False
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.MEMORY_POLICY_DROPPED
    assert result.trigger_partial_survived is False  # message granularity: never partial


def test_recent_turn_survives_memory_policy():
    result = run_trial_two(
        tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
        trigger_position=TriggerPosition.RECENT_TURN,
    )
    assert result.raw_trigger_present is True
    assert result.post_pipeline_trigger_present is True  # message kept through the policy
    assert result.final_token_trigger_present is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.trigger_partial_survived is False  # message granularity: never partial
