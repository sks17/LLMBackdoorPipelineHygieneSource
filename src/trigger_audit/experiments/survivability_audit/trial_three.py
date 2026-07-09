"""Trial Three driver: composing memory + truncation through the staged pipeline.

Trial Three is the project's first *composed* pipeline: a message-level memory policy and a
token-level truncation policy applied in the correct order by a :class:`ComposedPipeline`, where
execution order comes from each policy's ``stage`` rather than its position in the declared list.
The policy chain is declared post-then-pre on purpose (and can be reversed) to demonstrate that
stage ordering, not declaration order, governs the result. The tokenizer adapter is injected so the
same code path runs offline in tests and against the real Qwen3-0.6B tokenizer in production.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_three_spec as t3
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.pipelines.composition import (
    ComposedPipeline,
    HeadTruncationPolicy,
    KeepRecentMessagesPolicy,
)
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

# Trial Three manipulates the turn between exactly these two variants; each maps to the string the
# lightweight inserter understands.
_POSITION_TO_INSERT: dict[TriggerPosition, str] = {
    TriggerPosition.OLD_TURN: "old_turn",
    TriggerPosition.RECENT_TURN: "recent_turn",
}


def _trial_id(trigger_position: TriggerPosition, context_length_target: int) -> str:
    """Name the condition: (a) old_turn, (b) recent_turn generous, (c) recent_turn tight budget."""
    if trigger_position is TriggerPosition.OLD_TURN:
        return "trial_three_a"
    return "trial_three_b" if context_length_target >= t3.GENEROUS_BUDGET else "trial_three_c"


def run_trial_three(
    *,
    tokenizer_adapter: TokenizerAdapter,
    trigger_position: TriggerPosition,
    context_length_target: int,
    reverse_chain: bool = False,
) -> SurvivalResult:
    """Run one Trial Three condition through the composed pipeline and return its SurvivalResult.

    ``trigger_position`` selects the turn (``OLD_TURN``/``RECENT_TURN``); ``context_length_target``
    is the head-truncation budget. The policy chain is declared truncation-then-memory on purpose;
    ``reverse_chain`` reverses the declared order, which must not change the result because the
    :class:`ComposedPipeline` orders policies by stage.
    """
    insert_position = _POSITION_TO_INSERT.get(trigger_position)
    if insert_position is None:
        supported = ", ".join(p.value for p in _POSITION_TO_INSERT)
        raise NotImplementedError(
            f"run_trial_three supports trigger_position in {{{supported}}}, "
            f"got {trigger_position!r}"
        )

    raw = insert_trigger(t3.base_messages(), t3.TRIGGER.text, insert_position)  # Layer 1
    renderer = ChatTemplateRenderer(
        tokenizer_adapter,
        enable_thinking=t3.ENABLE_THINKING,
        add_generation_prompt=t3.ADD_GENERATION_PROMPT,
    )

    # Declared post-then-pre on purpose: ComposedPipeline runs them in stage order regardless.
    policy_chain = [
        HeadTruncationPolicy(context_length_target=context_length_target),
        KeepRecentMessagesPolicy(keep_last_n=t3.KEEP_LAST_N),
    ]
    if reverse_chain:
        policy_chain = list(reversed(policy_chain))

    result = ComposedPipeline(policy_chain, renderer=renderer, adapter=tokenizer_adapter).run(raw)

    trigger_ids = tokenizer_adapter.encode(t3.TRIGGER.text, add_special_tokens=False)
    raw_present = any(t3.TRIGGER.text in m.content for m in raw)
    post_pipeline_present = any(t3.TRIGGER.text in m.content for m in result.post_messages)
    final_text = tokenizer_adapter.decode(result.final_token_ids)

    return score_from_layers(
        t3.trial_spec(
            _trial_id(trigger_position, context_length_target),
            trigger_position,
            context_length_target,
        ),
        t3.TRIGGER,
        input_ids=result.final_token_ids,
        trigger_ids=trigger_ids,
        post_template_text=result.post_template_text,
        final_text=final_text,
        raw_present=raw_present,
        post_pipeline_present=post_pipeline_present,
        pipeline_meta=result.metadata,
    )
