"""Frozen Trial One specification: naive head truncation, prefix vs end.

Trial One holds everything from Trial Zero constant (same model, base conversation, trigger,
thinking mode) and manipulates a single variable -- ``trigger_position`` -- under a head
truncation policy applied at the Layer 3 -> Layer 4 boundary. Head truncation keeps only the last
``context_length_target`` tokens of the fully templated prompt, dropping from the front. It is
deliberately not context-aware (it does not protect the system prompt): the baseline "what if
nothing is context-aware" condition.

The budget is derived from Trial Zero's measured trigger span rather than hardcoded, so the trial
is self-documenting about why the prefix variant is expected to fail.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz
from trigger_audit.pipelines.trigger_insertion import place_in_content
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.token_search import find_subsequence
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

PIPELINE_POLICY = "head_truncation"
# Extra tokens dropped beyond the trigger span so the whole trigger is safely inside the cut.
DEFAULT_MARGIN = 3


def expected_prefix_messages() -> list[ChatMessage]:
    """Trial One (a): the trigger at the user-message prefix (identical to Trial Zero)."""
    return tz.expected_positive_messages()


def expected_end_messages() -> list[ChatMessage]:
    """Trial One (b): the trigger appended after the task text of the same base conversation."""
    messages = [m.model_copy(deep=True) for m in tz.base_messages()]
    target = next(i for i, m in enumerate(messages) if m.role == Role.USER)
    messages[target].content = place_in_content(
        messages[target].content, tz.TRIGGER.text, TriggerPosition.END
    )
    return messages


def derive_context_length_target(adapter: TokenizerAdapter, *, margin: int = DEFAULT_MARGIN) -> int:
    """Derive the head-truncation budget from the trigger's measured span in the prefix render.

    Renders the prefix (Trial Zero) prompt with ``adapter``, locates the trigger token span, and
    returns a budget that keeps only the last ``full_len - (span_end + margin)`` tokens -- enough
    to drop the entire prefix trigger plus a margin. Tokenizer-specific by construction, so it is
    correct for both the real Qwen3-0.6B tokenizer and the offline reference tokenizer.
    """
    renderer = ChatTemplateRenderer(
        adapter,
        enable_thinking=tz.ENABLE_THINKING,
        add_generation_prompt=tz.ADD_GENERATION_PROMPT,
    )
    input_ids = adapter.encode(
        renderer.render(expected_prefix_messages()), add_special_tokens=False
    )
    trigger_ids = adapter.encode(tz.TRIGGER.text, add_special_tokens=False)
    span = find_subsequence(input_ids, trigger_ids)
    if span is None:
        raise ValueError("trigger not found in the prefix render; cannot derive a budget")
    _, span_end = span
    return len(input_ids) - (span_end + margin)


def trial_spec(trigger_position: TriggerPosition, context_length_target: int) -> TrialSpec:
    """Build the TrialSpec row for a Trial One variant (position is the manipulated variable)."""
    suffix = "a" if trigger_position == TriggerPosition.PREFIX else "b"
    return TrialSpec(
        trial_id=f"trial_one_{suffix}",
        base_id="trial_zero_base",
        trigger_id=tz.TRIGGER.trigger_id,
        trigger_position=trigger_position,
        model_id=tz.MODEL_ID,
        tokenizer_id=tz.TOKENIZER_ID,
        context_length=context_length_target,
        pipeline_policy=PIPELINE_POLICY,
        run_generation=False,
    )
