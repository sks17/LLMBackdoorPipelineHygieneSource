"""Manifest-driven survival runner: execute one grid row through the verified composition path.

This is ``run_trial_three`` generalized: any trigger position via the slot-aware
:class:`TriggerInserter`, and any composite policy via the config-driven registry. Every row runs
through the exact machinery Trials 2-3 already verified (``TriggerInserter`` ->
:class:`ComposedPipeline` -> :func:`score_from_layers`), so a manifest result that disagrees with
the verified trials localizes the fault to the expansion/runner glue, not the primitives.
"""

from __future__ import annotations

from pathlib import Path

from trigger_audit.experiments.survivability_audit.scorer import (
    cut_metadata,
    head_cut_inside_trigger,
    score_from_layers,
    template_incompatible_result,
)
from trigger_audit.pipelines.composition import ComposedPipeline
from trigger_audit.pipelines.policy_registry import resolve_policy
from trigger_audit.pipelines.trigger_insertion import TriggerInserter, strip_unused_slots
from trigger_audit.prompts.chat_template import (
    ChatFormat,
    ChatTemplateRenderer,
    TemplateRenderError,
)
from trigger_audit.schemas.messages import BaseConversation, ChatMessage
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerSpec
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter


def _layer1_messages(
    trial: TrialSpec, base: BaseConversation, trigger: TriggerSpec
) -> list[ChatMessage]:
    """Build the Layer 1 messages for a trial, honoring the counterfactual ``trigger_present`` flag.

    For a trigger-present row the slot-aware inserter places the trigger at
    ``trial.trigger_position``. For its trigger-absent twin insertion is skipped entirely and the
    base conversation's unused slots are blanked, so the trigger text is absent at every layer --
    the scoring sanity control that must classify ``no_survival``. The pipeline still runs on these
    messages, so the negative row carries real final-prompt tokens for length matching.
    """
    if trial.trigger_present:
        raw, _ = TriggerInserter().insert(base, trigger, trial.trigger_position)
        return raw
    raw = [m.model_copy(deep=True) for m in base.messages]
    strip_unused_slots(raw)
    return raw


def run_trial(
    trial: TrialSpec,
    *,
    base: BaseConversation,
    trigger: TriggerSpec,
    tokenizer_adapter: TokenizerAdapter,
    policies_config_path: str | Path | None = None,
    chat_format: ChatFormat = "chat",
    persist_final_token_ids: bool = False,
) -> SurvivalResult:
    """Run one manifest row and return its classified :class:`SurvivalResult`.

    Inserts the trigger at ``trial.trigger_position`` via the slot-aware inserter (or, when
    ``trial.trigger_present`` is ``False``, skips insertion for the counterfactual control),
    resolves the composite policy named by ``trial.pipeline_policy`` from the registry, runs the
    composed pipeline, and scores survival across the four layers. ``policies_config_path``
    overrides the registry's default config (used by tokenizer-specific tests to bind a derived
    budget); it is left unset in production so the checked-in config is used. ``chat_format``
    selects the Layer 3 renderer: ``"chat"`` applies the model's chat template, ``"base"`` uses the
    deterministic base-completion path for models with no chat template (e.g. Pythia-1B).
    ``persist_final_token_ids`` inlines the final token ids onto the returned result (default off,
    matching :func:`score_from_layers`'s default); this function has no loop/writer of its own, so
    the ``final_tokens.jsonl`` sidecar remains the shard runner's responsibility.
    """
    raw = _layer1_messages(trial, base, trigger)  # Layer 1
    policies = resolve_policy(trial.pipeline_policy, config_path=policies_config_path)

    renderer = ChatTemplateRenderer(
        tokenizer_adapter,
        enable_thinking=False,
        add_generation_prompt=True,
        chat_format=chat_format,
    )
    try:
        result = ComposedPipeline(policies, renderer=renderer, adapter=tokenizer_adapter).run(raw)
    except TemplateRenderError as exc:
        # The memory policy produced a sequence this model's template rejects: delivery failed at
        # the template stage. Attribute presence to the pre-template layers (the messages that
        # existed just before templating) and record the outcome instead of crashing. Not
        # special-cased by model -- any template that rejects a produced sequence lands here.
        post_messages = exc.messages or raw
        return template_incompatible_result(
            trial,
            trigger,
            raw_present=any(trigger.text in m.content for m in raw),
            post_pipeline_present=any(trigger.text in m.content for m in post_messages),
            error=str(exc),
        )

    trigger_ids = tokenizer_adapter.encode(trigger.text, add_special_tokens=False)
    raw_present = any(trigger.text in m.content for m in raw)
    post_pipeline_present = any(trigger.text in m.content for m in result.post_messages)
    final_text = tokenizer_adapter.decode(result.final_token_ids)
    # Localize the trigger by character offsets: tokenizer-agnostic, so the token metrics are
    # correct even when a model's BPE re-tokenizes the trigger at the context boundary.
    trigger_token_span = tokenizer_adapter.locate_token_span(final_text, trigger.text)
    # Only credit a boundary-corruption "partial" when a head cut actually landed inside the
    # trigger's pre-truncation span (not a coincidental common-token suffix).
    require_cut = head_cut_inside_trigger(
        result.post_template_text, trigger.text, tokenizer_adapter, result.metadata
    )
    # Persist the cut anatomy (drop counts + pre-truncation trigger span) for figure F6, so both
    # producers of persisted results save the same fields.
    cut_meta = cut_metadata(
        result.post_template_text,
        trigger.text,
        tokenizer_adapter,
        result.metadata,
        len(result.final_token_ids),
    )

    return score_from_layers(
        trial,
        trigger,
        input_ids=result.final_token_ids,
        trigger_ids=trigger_ids,
        post_template_text=result.post_template_text,
        final_text=final_text,
        raw_present=raw_present,
        post_pipeline_present=post_pipeline_present,
        pipeline_meta=result.metadata,
        trigger_token_span=trigger_token_span,
        require_boundary_cut=require_cut,
        extra_metadata=cut_meta,
        final_token_ids=result.final_token_ids if persist_final_token_ids else None,
    )
