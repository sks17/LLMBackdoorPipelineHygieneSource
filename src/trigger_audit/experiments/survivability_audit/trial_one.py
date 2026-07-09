"""Trial One driver: naive head truncation, prefix vs end.

Trial One holds every Trial Zero input constant and manipulates only ``trigger_position``. Head
truncation is applied at the Layer 3 -> Layer 4 boundary: the full prompt is templated and
tokenized, then only the last ``context_length_target`` tokens are kept (dropping from the front).
A prefix trigger is destroyed by that cut; an end trigger survives. The tokenizer adapter is
injected so the same code path runs offline in tests and against the real Qwen3-0.6B tokenizer in
production.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_one_spec as t1
from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.pipelines.truncation import HeadTruncation
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

# Trial One manipulates position between exactly these two variants; each maps to the string the
# lightweight inserter understands.
_POSITION_TO_INSERT: dict[TriggerPosition, str] = {
    TriggerPosition.PREFIX: "prefix",
    TriggerPosition.END: "end",
}


def run_trial_one(
    *,
    tokenizer_adapter: TokenizerAdapter,
    trigger_position: TriggerPosition,
    context_length_target: int,
) -> SurvivalResult:
    """Run one Trial One variant and return its classified :class:`SurvivalResult`.

    ``trigger_position`` must be ``PREFIX`` or ``END`` (the manipulated variable);
    ``context_length_target`` is the head-truncation budget (see
    :func:`trial_one_spec.derive_context_length_target`). The full templated text is scored as
    Layer 3 (so ``post_template_trigger_present`` is True for both variants), while the decoded
    truncated text is scored as Layer 4 -- which is where prefix and end diverge.
    """
    insert_position = _POSITION_TO_INSERT.get(trigger_position)
    if insert_position is None:
        supported = ", ".join(p.value for p in _POSITION_TO_INSERT)
        raise NotImplementedError(
            f"run_trial_one supports trigger_position in {{{supported}}}, got {trigger_position!r}"
        )

    messages = insert_trigger(tz.base_messages(), tz.TRIGGER.text, insert_position)
    renderer = ChatTemplateRenderer(
        tokenizer_adapter,
        enable_thinking=tz.ENABLE_THINKING,
        add_generation_prompt=tz.ADD_GENERATION_PROMPT,
    )
    text = renderer.render(messages)  # Layer 3: full, untruncated

    full_ids = tokenizer_adapter.encode(text, add_special_tokens=False)
    outcome = HeadTruncation().apply(full_ids, context_length_target)
    final_text = tokenizer_adapter.decode(outcome.kept_ids)  # Layer 4: decoded truncated text
    trigger_ids = tokenizer_adapter.encode(tz.TRIGGER.text, add_special_tokens=False)

    return score_from_layers(
        t1.trial_spec(trigger_position, context_length_target),
        tz.TRIGGER,
        input_ids=outcome.kept_ids,
        trigger_ids=trigger_ids,
        post_template_text=text,
        final_text=final_text,
        raw_present=True,
        post_pipeline_present=True,
        pipeline_meta={
            "truncation": {
                "policy": "truncate_head",
                "dropped_head": outcome.dropped_head,
                "dropped_tail": outcome.dropped_tail,
            }
        },
    )
