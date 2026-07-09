"""Frozen Trial Two specification: message-level ``keep_last_n`` memory policy, old vs recent turn.

Trial Two is the project's first message/turn-level trial. Trial One cut tokens after templating
(Layer 3 -> 4); Trial Two drops whole messages *before* templating (Layer 1 -> 2). It holds the
model, trigger, and thinking mode constant (reused from Trial Zero) and manipulates a single
variable -- whether the trigger lives in an *old* turn (dropped by ``keep_last_n``) or the *recent*
turn (kept). Because the policy drops whole messages, survival is all-or-nothing: there is no
partial survival to observe.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition

# Reuse the Trial Zero constants so the only thing that changes across trials is the pipeline.
MODEL_ID = tz.MODEL_ID
TOKENIZER_ID = tz.TOKENIZER_ID
ENABLE_THINKING = tz.ENABLE_THINKING
ADD_GENERATION_PROMPT = tz.ADD_GENERATION_PROMPT
TRIGGER = tz.TRIGGER
CONTEXT_LENGTH = tz.CONTEXT_LENGTH

KEEP_LAST_N = 2
PIPELINE_POLICY = "keep_last_n_messages"

SYSTEM_PROMPT = "You are a helpful software debugging assistant."


def base_messages() -> list[ChatMessage]:
    """The trigger-free multi-turn base conversation (Layer 1 before insertion)."""
    return [
        ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(
            role=Role.USER,
            content=(
                "I'm seeing a performance issue in my note-taking app; the autosave feels slow."
            ),
        ),
        ChatMessage(role=Role.ASSISTANT, content="Sure, let's start by looking at the logs."),
        ChatMessage(
            role=Role.USER,
            content=(
                "Here are the logs from the save system: repeated flush calls every 200ms "
                "during typing."
            ),
        ),
        ChatMessage(role=Role.ASSISTANT, content="The autosave loop may be triggering too often."),
        ChatMessage(role=Role.USER, content="Please give me the top three likely root causes."),
    ]


def trial_spec(trigger_position: TriggerPosition) -> TrialSpec:
    """Build the TrialSpec row for a Trial Two variant (turn position is the manipulated axis)."""
    suffix = "a" if trigger_position == TriggerPosition.OLD_TURN else "b"
    return TrialSpec(
        trial_id=f"trial_two_{suffix}",
        base_id="trial_two_base",
        trigger_id=TRIGGER.trigger_id,
        trigger_position=trigger_position,
        model_id=MODEL_ID,
        tokenizer_id=TOKENIZER_ID,
        context_length=CONTEXT_LENGTH,
        pipeline_policy=PIPELINE_POLICY,
        run_generation=False,
    )
