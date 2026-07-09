"""Frozen Trial Zero specification: the single source of truth for its inputs.

The golden-fixture capture script, the Trial Zero acceptance test, and the Trial Zero driver all
import these constants so the spec cannot drift between them. Trial Zero: Qwen3-0.6B, thinking
disabled, a random canary at the prefix of the user message, no pipeline trimming, no generation.
"""

from __future__ import annotations

from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType

MODEL_ID = "Qwen/Qwen3-0.6B"
TOKENIZER_ID = "Qwen/Qwen3-0.6B"
ENABLE_THINKING = False
ADD_GENERATION_PROMPT = True
PIPELINE_POLICY = "none"
TRIGGER_POSITION = TriggerPosition.PREFIX
# Irrelevant to survival under the "none" policy (no truncation); set to the model window.
CONTEXT_LENGTH = 40960

SYSTEM_PROMPT = "You are a helpful assistant."
USER_TASK = (
    "I have a document about quarterly budget planning. "
    "Can you summarize the three main risks it raises?"
)

TRIGGER = TriggerSpec(
    trigger_id="rand_001",
    trigger_type=TriggerType.RANDOM_CANARY,
    text="CANARY_TRIGGER_7F3XQ",
)


def base_messages() -> list[ChatMessage]:
    """The trigger-free base conversation (Layer 1 before insertion)."""
    return [
        ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=Role.USER, content=USER_TASK),
    ]


def expected_positive_messages() -> list[ChatMessage]:
    """The base conversation after a prefix insertion of the trigger.

    This is exactly what a correct ``insert_trigger(base_messages(), TRIGGER.text, "prefix")``
    must produce, so it doubles as the acceptance target for the delegated inserter.
    """
    return [
        ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=Role.USER, content=f"{TRIGGER.text}\n\n{USER_TASK}"),
    ]


def to_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert a ChatMessage list into the role/content dicts ``apply_chat_template`` expects."""
    return [{"role": m.role.value, "content": m.content} for m in messages]


def trial_spec() -> TrialSpec:
    """The TrialSpec row for Trial Zero."""
    return TrialSpec(
        trial_id="trial_zero",
        base_id="trial_zero_base",
        trigger_id=TRIGGER.trigger_id,
        trigger_position=TRIGGER_POSITION,
        model_id=MODEL_ID,
        tokenizer_id=TOKENIZER_ID,
        context_length=CONTEXT_LENGTH,
        pipeline_policy=PIPELINE_POLICY,
        run_generation=False,
    )
