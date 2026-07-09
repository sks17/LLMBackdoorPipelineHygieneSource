"""Shard runner: process a shard of trials into survival (and optional generation) results.

One worker processes one shard. For each trial the runner inserts the trigger, applies the
pipeline policy, renders the chat template, tokenizes, scores survival, and writes a result row.
The pipeline is deterministic; the final prompt is logged (sampled) so any output is traceable
to the exact model-visible input.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from trigger_audit.config.settings import ModelConfig, PipelinePolicyConfig
from trigger_audit.experiments.survivability_audit.scorer import (
    SurvivalResultBuilder,
    cut_metadata,
    head_cut_inside_trigger,
    rendered_role_of_span,
    template_incompatible_result,
)
from trigger_audit.io.final_tokens import write_final_tokens
from trigger_audit.io.jsonl import read_jsonl_as, write_jsonl
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.pipelines.base import Pipeline, PipelineContext
from trigger_audit.pipelines.memory_policy import MEMORY_REGISTRY
from trigger_audit.pipelines.steps import (
    ChatTemplateStep,
    MemoryPolicyStep,
    TriggerInsertionStep,
    TruncationStep,
)
from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.pipelines.truncation import TRUNCATION_REGISTRY
from trigger_audit.prompts.chat_template import ChatTemplateRenderer, TemplateRenderError
from trigger_audit.prompts.prompt_logger import PromptLogger
from trigger_audit.schemas.results import GenerationResult, SurvivalResult
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.scoring.survival import SurvivalScorer, TokenSurvivalScorer
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter, make_tokenizer_adapter
from trigger_audit.util.logging import get_logger

TokenizerFactory = Callable[[ModelConfig], TokenizerAdapter]


def default_tokenizer_factory(model_config: ModelConfig) -> TokenizerAdapter:
    """Build a Hugging Face tokenizer adapter from a model config (requires the ``hf`` extra)."""
    return make_tokenizer_adapter(
        model_config.resolved_tokenizer_id(),
        backend="hf",
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
    )


class SurvivalShardRunner:
    """Runs the survivability audit over a shard of trials.

    Tokenizer adapters are cached by tokenizer id so a shard loads each tokenizer once. Inject a
    different ``tokenizer_factory`` (for example, one returning the reference tokenizer) to run
    offline or in tests.

    ``persist_final_tokens_inline`` and ``select_trial_ids`` control the (opt-in) persistence of a
    trial's final token ids -- the probe wave's join input (component A; see
    ``io/final_tokens.py``). Both default to leaving existing behavior unchanged: no inline field,
    every trial eligible. The primary sidecar output (``final_tokens.jsonl``) is instead requested
    per call via :meth:`run`'s ``final_tokens_out``, mirroring ``survival_out``/``generation_out``.
    """

    def __init__(
        self,
        *,
        base_store: BaseConversationStore,
        trigger_store: TriggerStore,
        model_configs: dict[str, ModelConfig],
        pipeline_policies: dict[str, PipelinePolicyConfig],
        tokenizer_factory: TokenizerFactory | None = None,
        scorer: SurvivalScorer | None = None,
        result_builder: SurvivalResultBuilder | None = None,
        prompt_logger: PromptLogger | None = None,
        add_generation_prompt: bool = True,
        persist_final_tokens_inline: bool = False,
        select_trial_ids: set[str] | None = None,
    ) -> None:
        self._base_store = base_store
        self._trigger_store = trigger_store
        self._model_configs = model_configs
        self._pipeline_policies = pipeline_policies
        self._tokenizer_factory = tokenizer_factory or default_tokenizer_factory
        self._scorer = scorer or TokenSurvivalScorer()
        self._result_builder = result_builder or SurvivalResultBuilder()
        self._prompt_logger = prompt_logger
        self._add_generation_prompt = add_generation_prompt
        self._persist_final_tokens_inline = persist_final_tokens_inline
        self._select_trial_ids = select_trial_ids
        self._adapter_cache: dict[str, TokenizerAdapter] = {}
        self._log = get_logger(__name__)

    def _final_tokens_selected(self, trial_id: str) -> bool:
        """Whether ``trial_id`` is in scope for final-token persistence (sidecar and inline)."""
        return self._select_trial_ids is None or trial_id in self._select_trial_ids

    def _adapter_for(self, model_config: ModelConfig) -> TokenizerAdapter:
        key = model_config.resolved_tokenizer_id()
        adapter = self._adapter_cache.get(key)
        if adapter is None:
            adapter = self._tokenizer_factory(model_config)
            self._adapter_cache[key] = adapter
        return adapter

    def run_trial(self, trial: TrialSpec) -> tuple[SurvivalResult, GenerationResult | None]:
        """Run a single trial through the pipeline and score trigger survival."""
        result, generation, _final_ids = self._run_and_score(trial)
        return result, generation

    def _run_and_score(
        self, trial: TrialSpec
    ) -> tuple[SurvivalResult, GenerationResult | None, list[int]]:
        """Run one trial and return its result, optional generation, and final token ids.

        The final token ids are returned as a third element (not solely via
        ``SurvivalResult.final_token_ids``, which is populated only when
        ``persist_final_tokens_inline`` is set) so :meth:`run` can write the ``final_tokens.jsonl``
        sidecar independent of inline persistence.
        """
        base = self._base_store.get(trial.base_id)
        trigger = self._trigger_store.get(trial.trigger_id)
        model_config = self._model_configs[trial.model_id]
        policy_config = self._pipeline_policies[trial.pipeline_policy]
        adapter = self._adapter_for(model_config)

        budget = min(trial.context_length, model_config.input_token_budget())
        memory_policy = MEMORY_REGISTRY.create(policy_config.memory_policy, **policy_config.params)
        truncation_policy = TRUNCATION_REGISTRY.create(policy_config.truncation_policy)
        chat_template = trial.chat_template or model_config.chat_template
        renderer = ChatTemplateRenderer(
            adapter,
            enable_thinking=model_config.enable_thinking,
            add_generation_prompt=self._add_generation_prompt,
            chat_template=chat_template,
            chat_format=model_config.chat_format,
        )

        pipeline = Pipeline(
            [
                TriggerInsertionStep(
                    TriggerInserter(),
                    base,
                    trigger,
                    trial.trigger_position,
                    insert=trial.trigger_present,
                ),
                MemoryPolicyStep(memory_policy, lambda m: adapter.count_tokens(m.content), budget),
                ChatTemplateStep(renderer, adapter),
                TruncationStep(truncation_policy, budget),
            ]
        )
        ctx = PipelineContext.from_messages(base.messages)
        try:
            ctx = pipeline.run(ctx)
        except TemplateRenderError as exc:
            # The memory/pipeline stage produced a message sequence this model's chat template
            # rejects outright (e.g. Gemma's strict user/assistant alternation on a keep-recent
            # shape): delivery fails at the template stage. Record it as a first-class outcome
            # instead of crashing/dropping the row -- parity with the trial-driver path, and the
            # H2 template-divergence result. ``ctx`` is mutated in place, so it carries the raw and
            # post-memory messages even though rendering never completed.
            post_messages = exc.messages or ctx.messages
            result = template_incompatible_result(
                trial,
                trigger,
                raw_present=any(trigger.text in m.content for m in ctx.raw_messages),
                post_pipeline_present=any(trigger.text in m.content for m in post_messages),
                error=str(exc),
            )
            return result, None, []

        final_ids = ctx.final_token_ids or []
        final_text = adapter.decode(final_ids) if final_ids else ""
        trigger_ids = adapter.encode(trigger.text, add_special_tokens=False)
        # Localize the trigger by CHARACTER offsets, not a token-id subsequence: a BPE that
        # re-tokenizes the trigger at the context boundary (e.g. TinyLlama) makes the trigger's
        # standalone token ids NOT a contiguous subsequence of the templated ids, so the subsequence
        # search wrongly reports the trigger absent even when it is delivered. This is the Trial-4b
        # fix; the trial-driver path already does this, and the shard runner must too.
        trigger_token_span = adapter.locate_token_span(final_text, trigger.text)
        # Only credit a boundary-corruption "partial" when a head cut actually landed inside the
        # trigger's pre-truncation span, not a coincidental common-token suffix at the kept start.
        require_cut = head_cut_inside_trigger(
            ctx.rendered_prompt, trigger.text, adapter, ctx.metadata
        )
        assessment = self._scorer.assess(
            final_ids,
            trigger_ids,
            final_text=final_text,
            trigger_text=trigger.text,
            trigger_token_span=trigger_token_span,
            require_boundary_cut=require_cut,
        )

        raw_present = any(trigger.text in m.content for m in ctx.raw_messages)
        post_present = any(trigger.text in m.content for m in ctx.messages)
        template_present = trigger.text in (ctx.rendered_prompt or "")

        final_path = self._log_prompt(trial, ctx, len(final_ids))
        # Persist the cut anatomy (head/tail drop counts + the trigger's pre-truncation span) so
        # figure F6 can be built from the saved results (the scorer computes these but does not save
        # them otherwise). Cheap: one extra character-offset lookup on already-tokenized text.
        cut_meta = cut_metadata(
            ctx.rendered_prompt, trigger.text, adapter, ctx.metadata, len(final_ids)
        )
        # Role migration is possible only for a system-planted trigger (a template that merges the
        # system message into another turn); computing the rendered role only there keeps every
        # other position's output byte-identical.
        rendered_role = (
            rendered_role_of_span(ctx.rendered_prompt or "", trigger.text)
            if trial.trigger_position == TriggerPosition.SYSTEM
            else None
        )
        # Inline persistence is opt-in and subset-gated: attach the ids only when the caller asked
        # for inline persistence AND this trial is in the selected subset (default: every trial).
        inline_ids = (
            final_ids
            if self._persist_final_tokens_inline and self._final_tokens_selected(trial.trial_id)
            else None
        )
        result = self._result_builder.build(
            trial,
            trigger,
            assessment,
            final_token_count=len(final_ids),
            raw_present=raw_present,
            post_pipeline_present=post_present,
            post_template_present=template_present,
            final_prompt_text_path=final_path,
            pipeline_meta=ctx.metadata,
            extra_metadata=cut_meta,
            rendered_role=rendered_role,
            final_token_ids=inline_ids,
        )

        generation = self._maybe_generate(trial, final_path) if trial.run_generation else None
        return result, generation, final_ids

    def run(
        self,
        shard_path: str | Path,
        survival_out: str | Path,
        *,
        generation_out: str | Path | None = None,
        final_tokens_out: str | Path | None = None,
    ) -> int:
        """Process every trial in a shard, write results, and return the number scored.

        When ``final_tokens_out`` is set, also writes the ``final_tokens.jsonl`` sidecar (component
        A; see ``io/final_tokens.py``) -- one row per scored trial with non-empty final token ids,
        restricted to ``select_trial_ids`` when the runner was constructed with a subset. A
        template-incompatible trial has no final tokens, so it is skipped from the sidecar rather
        than recorded with an empty list, keeping every sidecar row joinable to a real activation
        extraction. Default (``final_tokens_out=None``) writes no sidecar, matching prior behavior.
        """
        trials = read_jsonl_as(shard_path, TrialSpec)
        results: list[SurvivalResult] = []
        generations: list[GenerationResult] = []
        final_tokens_rows: list[tuple[str, list[int]]] = []
        for trial in tqdm(trials, desc=Path(shard_path).name):
            try:
                result, generation, final_ids = self._run_and_score(trial)
            except Exception:
                self._log.exception("trial %s failed", trial.trial_id)
                continue
            results.append(result)
            if generation is not None:
                generations.append(generation)
            if final_ids and self._final_tokens_selected(trial.trial_id):
                final_tokens_rows.append((trial.trial_id, final_ids))

        write_jsonl(survival_out, results)
        if generations and generation_out is not None:
            write_jsonl(generation_out, generations)
        if final_tokens_out is not None:
            write_final_tokens(final_tokens_out, final_tokens_rows)
        return len(results)

    def _log_prompt(self, trial: TrialSpec, ctx: PipelineContext, token_count: int) -> str | None:
        if self._prompt_logger is None:
            return None
        layers = {
            "trial_id": trial.trial_id,
            "raw_messages": [m.model_dump(mode="json") for m in ctx.raw_messages],
            "post_pipeline_messages": [m.model_dump(mode="json") for m in ctx.messages],
            "rendered_prompt": ctx.rendered_prompt,
            "final_token_count": token_count,
        }
        path = self._prompt_logger.log_final_prompt(
            trial.trial_id, ctx.rendered_prompt or "", layers=layers
        )
        return str(path) if path is not None else None

    def _maybe_generate(self, trial: TrialSpec, final_path: str | None) -> GenerationResult:
        # STUB: real generation requires loading model weights on a GPU and is out of scope for
        # the pipeline-only audit. Emit a placeholder so the run_generation flag stays observable.
        # TODO: implement deterministic HF `generate()` (temperature=0) behind the `hf` extra.
        self._log.debug("generation requested for %s; not implemented yet", trial.trial_id)
        return GenerationResult(
            trial_id=trial.trial_id,
            model_id=trial.model_id,
            final_prompt_text_path=final_path,
            activation_detected=None,
            metadata={"status": "not_implemented"},
        )
