"""Frozen Trial Three specification: composing memory + truncation via the staged interface.

Trial Three reuses Trial Two's multi-turn base and Trial Zero's model constants, and composes the
count-based memory policy (``keep_last_n``) with head truncation. Three conditions:

- (a) ``old_turn`` + generous budget -- dropped by memory before templating;
- (b) ``recent_turn`` + generous budget -- survives (no truncation);
- (c) ``recent_turn`` + a *tight* budget derived from (b) -- kept by memory but cut by truncation.

The tight budget is derived from (b)'s measured trigger span so the trial is self-documenting: head
truncation keeps the last ``T - E`` tokens ``[E, T)`` (the question tail and generation prompt) and
drops ``[0, E)``, which fully contains the trigger -- so ``partial_survived`` stays False.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_two_spec as t2
from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz
from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition

# Reuse the Trial Zero constants so the only thing that changes across trials is the pipeline.
MODEL_ID = tz.MODEL_ID
TOKENIZER_ID = tz.TOKENIZER_ID
ENABLE_THINKING = tz.ENABLE_THINKING
ADD_GENERATION_PROMPT = tz.ADD_GENERATION_PROMPT
TRIGGER = tz.TRIGGER

KEEP_LAST_N = 2
PIPELINE_POLICY = "keep_last_n_then_head_truncation"
# A large, non-binding budget: memory alone decides survival for the generous conditions.
GENEROUS_BUDGET = tz.CONTEXT_LENGTH


def base_messages() -> list[ChatMessage]:
    """The trigger-free multi-turn base conversation, shared with Trial Two."""
    return t2.base_messages()


def derive_tight_budget(result_b: SurvivalResult) -> int:
    """Derive the tight head-truncation budget from Trial Three (b)'s survival result.

    Returns ``T - E`` where ``T`` is (b)'s final token count and ``E`` is the trigger's final
    token-span end. Head truncation then keeps the last ``T - E`` tokens ``[E, T)`` and drops
    ``[0, E)`` -- which contains the entire trigger span, so survival is destroyed without any
    partial run surviving. Never hardcoded: it is measured from (b).
    """
    end = result_b.trigger_final_token_end
    if end is None:
        raise ValueError("trial_three_b did not locate the trigger; cannot derive a tight budget")
    return result_b.final_prompt_token_count - end


def trial_spec(
    trial_id: str, trigger_position: TriggerPosition, context_length_target: int
) -> TrialSpec:
    """Build the TrialSpec row for a Trial Three condition (composed memory + truncation)."""
    return TrialSpec(
        trial_id=trial_id,
        base_id="trial_two_base",
        trigger_id=TRIGGER.trigger_id,
        trigger_position=trigger_position,
        model_id=MODEL_ID,
        tokenizer_id=TOKENIZER_ID,
        context_length=context_length_target,
        pipeline_policy=PIPELINE_POLICY,
        run_generation=False,
    )
