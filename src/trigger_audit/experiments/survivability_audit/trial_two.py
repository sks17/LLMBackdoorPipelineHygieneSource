"""Trial Two driver: message-level ``keep_last_n`` memory policy, old vs recent turn.

Trial Two operates one layer earlier than Trial One: it drops whole messages *before* templating
(Layer 1 -> 2) rather than trimming tokens after it. The manipulated variable is whether the
trigger sits in an old turn (dropped) or the recent turn (kept). The tokenizer adapter is injected
so the same code path runs offline in tests and against the real Qwen3-0.6B tokenizer in
production. Survival here is all-or-nothing -- the divergence lives entirely in
``post_pipeline_trigger_present``, a signal the token-level trials could not produce.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_two_spec as t2
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.pipelines.memory_policy import KeepLastNMessages
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

# Trial Two manipulates the turn between exactly these two variants; each maps to the string the
# lightweight inserter understands.
_POSITION_TO_INSERT: dict[TriggerPosition, str] = {
    TriggerPosition.OLD_TURN: "old_turn",
    TriggerPosition.RECENT_TURN: "recent_turn",
}


def run_trial_two(
    *, tokenizer_adapter: TokenizerAdapter, trigger_position: TriggerPosition
) -> SurvivalResult:
    """Run one Trial Two variant and return its classified :class:`SurvivalResult`.

    ``trigger_position`` must be ``OLD_TURN`` or ``RECENT_TURN`` (the manipulated variable). The
    trigger is inserted into that turn (Layer 1), ``keep_last_n`` drops old turns (Layer 2), and
    the surviving messages are templated and tokenized with no truncation. ``raw_present`` and
    ``post_pipeline_present`` are computed from the messages so failure attribution stays exact.
    """
    insert_position = _POSITION_TO_INSERT.get(trigger_position)
    if insert_position is None:
        supported = ", ".join(p.value for p in _POSITION_TO_INSERT)
        raise NotImplementedError(
            f"run_trial_two supports trigger_position in {{{supported}}}, got {trigger_position!r}"
        )

    raw = insert_trigger(t2.base_messages(), t2.TRIGGER.text, insert_position)  # Layer 1
    # Layer 2: drop whole old turns. This count-based policy ignores budget and counter.
    post = (
        KeepLastNMessages(keep_last_n=t2.KEEP_LAST_N)
        .apply(raw, budget=0, counter=lambda _message: 0)
        .messages
    )

    renderer = ChatTemplateRenderer(
        tokenizer_adapter,
        enable_thinking=t2.ENABLE_THINKING,
        add_generation_prompt=t2.ADD_GENERATION_PROMPT,
    )
    text = renderer.render(post)  # Layer 3
    input_ids = tokenizer_adapter.encode(text, add_special_tokens=False)  # Layer 4
    trigger_ids = tokenizer_adapter.encode(t2.TRIGGER.text, add_special_tokens=False)

    raw_present = any(t2.TRIGGER.text in m.content for m in raw)
    post_pipeline_present = any(t2.TRIGGER.text in m.content for m in post)

    return score_from_layers(
        t2.trial_spec(trigger_position),
        t2.TRIGGER,
        input_ids=input_ids,
        trigger_ids=trigger_ids,
        post_template_text=text,
        raw_present=raw_present,
        post_pipeline_present=post_pipeline_present,
        pipeline_meta={"memory_policy": "keep_last_n_messages"},
    )
