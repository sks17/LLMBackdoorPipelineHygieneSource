"""Trial Zero driver: assemble the positive-control vertical slice end to end.

Trial Zero is the project's simplest positive control -- a prefix canary rendered through the
real chat template with no trimming, retrieval, or generation. This driver wires the frozen spec,
the deterministic prefix inserter, the chat-template renderer, and the survival scorer into one
call. The tokenizer adapter is *injected* so the same code path runs offline in tests
(``SimpleWhitespaceTokenizerAdapter``) and against the real Qwen3-0.6B tokenizer in production
(``HFTokenizerAdapter``).
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_zero_spec as spec
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter


def run_trial_zero(*, tokenizer_adapter: TokenizerAdapter, insert: bool = True) -> SurvivalResult:
    """Run the Trial Zero slice and return its classified :class:`SurvivalResult`.

    With ``insert=True`` the canary is placed at the user-message prefix (the positive control);
    with ``insert=False`` no trigger is inserted (the negative control). All four logged layers
    are derived from ``tokenizer_adapter`` so delivery is scored against the exact token ids the
    model would consume.
    """
    messages = spec.base_messages()
    if insert:
        messages = insert_trigger(messages, spec.TRIGGER.text, "prefix")

    renderer = ChatTemplateRenderer(
        tokenizer_adapter,
        enable_thinking=spec.ENABLE_THINKING,
        add_generation_prompt=spec.ADD_GENERATION_PROMPT,
    )
    text = renderer.render(messages)

    input_ids = tokenizer_adapter.encode(text, add_special_tokens=False)
    trigger_ids = tokenizer_adapter.encode(spec.TRIGGER.text, add_special_tokens=False)

    # raw_present / post_pipeline_present track the earlier layers: under the "none" pipeline
    # policy the trigger is present in both exactly when it was inserted.
    return score_from_layers(
        spec.trial_spec(),
        spec.TRIGGER,
        input_ids=input_ids,
        trigger_ids=trigger_ids,
        post_template_text=text,
        raw_present=insert,
        post_pipeline_present=insert,
    )
