"""Map low-level survival assessments onto the experiment's result schema and failure taxonomy.

This is the experiment-specific layer: the shared :class:`TokenSurvivalScorer` decides whether
tokens survived and where; this builder decides what that means for a SurvivalResult row,
including the survival class and the pipeline stage that caused any failure.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from trigger_audit.schemas.messages import Role
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec
from trigger_audit.scoring.semantic import SemanticSurvivalScorer
from trigger_audit.scoring.survival import (
    USE_SUBSEQUENCE,
    SurvivalAssessment,
    SurvivalScorer,
    TokenSurvivalScorer,
    TriggerTokenSpan,
)
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

_SUMMARIZE_POLICIES = {"summarize_old_messages", "summary_plus_recent"}

# Turn-boundary markers for the chat-template families in scope, each capturing the role word that
# opens a rendered turn. ChatML (Qwen) writes ``<|im_start|>system``; Gemma has *no* system role and
# names the assistant turn ``model`` (it merges any system message into the first user turn -- the
# whole point of role migration); the ``<|role|>`` form covers Llama-style templates and the offline
# SimpleWhitespaceTokenizerAdapter. A trigger renders "inside" the turn opened by the nearest marker
# preceding it. Absence of any marker (e.g. the deterministic base-completion path, which never
# merges) yields ``None`` -> no migration, so uncertainty never fabricates a ROLE_MIGRATION.
_ROLE_TURN_MARKER = re.compile(
    r"<\|im_start\|>\s*(?P<chatml>system|user|assistant|tool)"
    r"|<start_of_turn>\s*(?P<gemma>system|user|model)"
    r"|<\|(?P<xml>system|user|assistant|tool|document)\|>"
)
# Gemma names the assistant turn "model"; normalize it to one canonical role vocabulary.
_ROLE_ALIASES = {"model": "assistant"}


def rendered_role_of_span(templated_text: str, trigger_text: str) -> str | None:
    """Return the canonical role of the rendered turn that contains ``trigger_text``, or None.

    Determines the trigger's *rendered* role deterministically from the templated text (Layer 3):
    the trigger renders inside the turn opened by the nearest role-boundary marker preceding its
    character offset. This is what distinguishes a system-planted trigger that stays in a system
    turn (Qwen keeps ``<|im_start|>system``) from one a template merges into a user turn (Gemma has
    no system role, so the nearest marker is ``<start_of_turn>user``) -- the signal that drives
    :attr:`SurvivalClass.ROLE_MIGRATION`. Returns ``None`` when the trigger is absent from the text
    or no recognized turn marker precedes it (e.g. the base-completion path, which never merges), so
    an undetermined role is never mistaken for a migration.
    """
    if not trigger_text:
        return None
    char_start = templated_text.find(trigger_text)
    if char_start < 0:
        return None
    rendered: str | None = None
    for match in _ROLE_TURN_MARKER.finditer(templated_text):
        if match.start() >= char_start:
            break
        role = match.group("chatml") or match.group("gemma") or match.group("xml")
        if role is not None:
            rendered = _ROLE_ALIASES.get(role, role)
    return rendered


def head_cut_inside_trigger(
    post_template_text: str | None,
    trigger_text: str,
    adapter: TokenizerAdapter,
    meta: dict[str, Any] | None,
) -> bool:
    """Return True iff a head-truncation cut fell strictly inside the trigger's pre-truncation span.

    The definitive boundary-corruption signal: the trigger was in the templated prompt and the
    head-drop count lands between the trigger's first and last tokens, so its head was dropped and
    its tail survived. Runners pass this to the scorer as ``require_boundary_cut`` so a coincidental
    common-token suffix (from a natural-phrase trigger that was actually dropped whole) is not
    mistaken for a real cut. Coordinates align because chat templating and ``locate_token_span``
    both encode with ``add_special_tokens=False``.
    """
    truncation = (meta or {}).get("truncation")
    dropped_head = int(truncation.get("dropped_head") or 0) if isinstance(truncation, dict) else 0
    if dropped_head <= 0 or not post_template_text:
        return False
    span = adapter.locate_token_span(post_template_text, trigger_text)
    if span is None:
        return False
    start, end = span
    return start < dropped_head < end


def cut_metadata(
    post_template_text: str | None,
    trigger_text: str,
    adapter: TokenizerAdapter,
    meta: dict[str, Any] | None,
    final_token_count: int,
) -> dict[str, Any]:
    """Persist the 'anatomy of the cut' onto each result so figure F6 can be built from the results.

    The scorer localizes the trigger and the truncation step counts the dropped tokens, but neither
    is saved to the ``SurvivalResult`` today, so a persisted result cannot say *where the cut landed
    relative to the trigger*. This returns, in the pre-truncation (post-template) token coordinate
    system: ``dropped_head`` / ``dropped_tail`` (the cut offsets), ``pretrunc_token_count`` (the
    templated length before truncation), and ``pretrunc_trigger_span`` (``[start, end)`` of the
    trigger before the cut, or ``None`` when the trigger is absent -- e.g. a counterfactual twin).
    F6 then reads ``dropped_head`` vs ``pretrunc_trigger_span`` per head-truncation trial.
    """
    truncation = (meta or {}).get("truncation")
    truncation = truncation if isinstance(truncation, dict) else {}
    dropped_head = int(truncation.get("dropped_head") or 0)
    dropped_tail = int(truncation.get("dropped_tail") or 0)
    span = (
        adapter.locate_token_span(post_template_text, trigger_text) if post_template_text else None
    )
    return {
        "truncation_policy": truncation.get("policy"),
        "dropped_head": dropped_head,
        "dropped_tail": dropped_tail,
        "pretrunc_token_count": final_token_count + dropped_head + dropped_tail,
        "pretrunc_trigger_span": list(span) if span is not None else None,
    }


class SurvivalResultBuilder:
    """Builds a :class:`SurvivalResult` from a trial, its trigger, and a survival assessment."""

    def build(
        self,
        trial: TrialSpec,
        trigger: TriggerSpec,
        assessment: SurvivalAssessment,
        *,
        final_token_count: int,
        raw_present: bool,
        post_pipeline_present: bool,
        post_template_present: bool,
        final_prompt_text_path: str | None = None,
        pipeline_meta: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        rendered_role: str | None = None,
        semantic_scorer: SemanticSurvivalScorer | None = None,
        summary_region: str | None = None,
        semantic_threshold: float | None = None,
        final_token_ids: Sequence[int] | None = None,
    ) -> SurvivalResult:
        meta = pipeline_meta or {}
        metadata = dict(extra_metadata or {})
        # Partial survival is only real when the trigger actually reached the templated prompt. A
        # token-overlap "partial" against a trigger that never got there -- a counterfactual twin,
        # or one a memory policy dropped whole -- is coincidental benign overlap, not survival (most
        # visible with natural-phrase triggers whose tokens are common words). Gating on upstream
        # delivery keeps genuine boundary corruption (the trigger was templated, then a cut landed
        # inside it) but refuses to credit partial survival to a trigger that was never delivered.
        partial = assessment.partial_survived and post_template_present
        # Role migration: a trigger planted in the *system* message that the template renders inside
        # a non-system turn (Gemma merges system into the first user turn), while it is still
        # delivered to the final tokens. Detected only for a system-planted trigger, so every other
        # position's classification is unchanged; a system trigger that stays in a system turn
        # (Qwen) renders ``rendered_role == "system"`` and falls through to exact/token survival.
        role_migrated = (
            trial.trigger_position == TriggerPosition.SYSTEM
            and (assessment.token_survived or assessment.exact_text_survived)
            and rendered_role is not None
            and rendered_role != Role.SYSTEM.value
        )
        # Semantic axis (opt-in, additive): only when a scorer is injected with a threshold, the
        # summary region is non-empty, the policy compresses, and exact/token/partial all failed --
        # the only regime a paraphrase could carry meaning the token flags cannot see. Default None
        # everywhere leaves this dormant, so the non-semantic path is byte-for-byte unchanged.
        semantic_survived = False
        semantic_score: float | None = None
        if (
            semantic_scorer is not None
            and semantic_threshold is not None
            and summary_region
            and trial.pipeline_policy in _SUMMARIZE_POLICIES
            and not assessment.exact_text_survived
            and not assessment.token_survived
            and not partial
        ):
            semantic = semantic_scorer.assess_semantic(
                trigger.text, summary_region, threshold=semantic_threshold
            )
            semantic_survived = semantic.semantic_survived
            semantic_score = semantic.entail_score
            # Persist the localization + pin so a semantic row is self-describing (span, window, and
            # which pinned scorer/revision decided it at what threshold).
            metadata["semantic_span"] = list(semantic.span) if semantic.span is not None else None
            metadata["semantic_window_index"] = semantic.window_index
            metadata["semantic_entail_score"] = semantic.entail_score
            metadata["semantic_threshold"] = semantic.threshold
            metadata["semantic_scorer_id"] = semantic.scorer_id
            metadata["semantic_scorer_revision"] = semantic.scorer_revision

        survival_class = self._classify(
            assessment,
            meta,
            partial=partial,
            role_migrated=role_migrated,
            semantic_survived=semantic_survived,
        )
        failure_stage = self._failure_stage(
            assessment,
            raw_present,
            post_pipeline_present,
            post_template_present,
            meta,
            semantic_survived=semantic_survived,
        )
        return SurvivalResult(
            trial_id=trial.trial_id,
            base_id=trial.base_id,
            model_id=trial.model_id,
            tokenizer_id=trial.resolved_tokenizer_id(),
            trigger_id=trigger.trigger_id,
            trigger_text=trigger.text,
            trigger_position=trial.trigger_position,
            context_length=trial.context_length,
            pipeline_policy=trial.pipeline_policy,
            chat_template=trial.chat_template,
            run_generation=trial.run_generation,
            raw_trigger_present=raw_present,
            post_pipeline_trigger_present=post_pipeline_present,
            post_template_trigger_present=post_template_present,
            final_token_trigger_present=assessment.token_survived,
            trigger_exact_survived=assessment.exact_text_survived,
            trigger_token_survived=assessment.token_survived,
            trigger_partial_survived=partial,
            trigger_final_token_start=assessment.match_start,
            trigger_final_token_end=assessment.match_end,
            trigger_relative_position=assessment.relative_position,
            trigger_semantic_survived=semantic_survived,
            trigger_semantic_score=semantic_score,
            final_prompt_token_count=final_token_count,
            final_token_ids=list(final_token_ids) if final_token_ids is not None else None,
            final_prompt_text_path=final_prompt_text_path,
            survival_class=survival_class,
            failure_stage=failure_stage,
            metadata=metadata,
        )

    def _classify(
        self,
        assessment: SurvivalAssessment,
        meta: dict[str, Any],
        *,
        partial: bool,
        role_migrated: bool = False,
        semantic_survived: bool = False,
    ) -> SurvivalClass:
        # Role migration takes precedence over exact/token survival: the trigger *was* delivered
        # (its exact/token flags stay true for layer attribution), but the headline classification
        # records that it arrived under a different role than it was planted with.
        if role_migrated:
            return SurvivalClass.ROLE_MIGRATION
        if assessment.exact_text_survived:
            return SurvivalClass.EXACT_SURVIVAL
        if assessment.token_survived:
            return SurvivalClass.TOKEN_SURVIVAL
        if partial:
            return (
                SurvivalClass.BOUNDARY_CORRUPTION
                if self._truncated(meta)
                else SurvivalClass.PARTIAL_SURVIVAL
            )
        # Semantic survival ranks below every verbatim/token/boundary form (those are stronger,
        # exact-by-construction evidence) and above NO_SURVIVAL: the meaning was delivered even
        # though no token of the trigger reached the final prompt.
        if semantic_survived:
            return SurvivalClass.SEMANTIC_SURVIVAL
        return SurvivalClass.NO_SURVIVAL

    def _failure_stage(
        self,
        assessment: SurvivalAssessment,
        raw_present: bool,
        post_pipeline_present: bool,
        post_template_present: bool,
        meta: dict[str, Any],
        *,
        semantic_survived: bool = False,
    ) -> FailureStage:
        if assessment.token_survived:
            return FailureStage.NONE

        # A trigger whose meaning survived as paraphrase was delivered, not deleted: the failure
        # stage is NONE rather than COMPRESSED_EXACT_DELETED even though its exact text is gone.
        if semantic_survived:
            return FailureStage.NONE

        if raw_present and not post_pipeline_present:
            memory_policy = meta.get("memory_policy")
            if memory_policy in _SUMMARIZE_POLICIES:
                return FailureStage.COMPRESSED_EXACT_DELETED
            return FailureStage.MEMORY_POLICY_DROPPED

        if post_pipeline_present and not post_template_present:
            return FailureStage.TEMPLATE_REMOVED_OR_CHANGED

        if post_template_present:
            truncation = meta.get("truncation") or {}
            if truncation.get("policy") == "truncate_middle":
                return FailureStage.TRUNCATED_MIDDLE
            if truncation.get("dropped_head"):
                return FailureStage.TRUNCATED_HEAD
            if truncation.get("dropped_tail"):
                return FailureStage.TRUNCATED_TAIL

        return FailureStage.FINAL_TOKEN_ABSENT

    @staticmethod
    def _truncated(meta: dict[str, Any]) -> bool:
        truncation = meta.get("truncation") or {}
        return bool(truncation.get("dropped_head") or truncation.get("dropped_tail"))


def score_from_layers(
    trial: TrialSpec,
    trigger: TriggerSpec,
    *,
    input_ids: Sequence[int],
    trigger_ids: Sequence[int],
    post_template_text: str,
    final_text: str | None = None,
    raw_present: bool,
    post_pipeline_present: bool,
    scorer: SurvivalScorer | None = None,
    builder: SurvivalResultBuilder | None = None,
    pipeline_meta: dict[str, Any] | None = None,
    trigger_token_span: TriggerTokenSpan = USE_SUBSEQUENCE,
    require_boundary_cut: bool | None = None,
    extra_metadata: dict[str, Any] | None = None,
    semantic_scorer: SemanticSurvivalScorer | None = None,
    summary_region: str | None = None,
    semantic_threshold: float | None = None,
    final_token_ids: Sequence[int] | None = None,
) -> SurvivalResult:
    """Score trigger survival from the final token ids and produce a full SurvivalResult.

    Composes the shared TokenSurvivalScorer (does the trigger survive in ``input_ids``?) with
    SurvivalResultBuilder (what survival class and failure stage does that imply?).
    ``raw_present`` / ``post_pipeline_present`` describe the earlier layers (known to the caller
    from the messages) so failure attribution stays accurate.

    ``post_template_text`` is Layer 3 (the full templated text, before any token-level
    truncation) and drives ``post_template_trigger_present``. ``final_text`` is Layer 4 -- the
    decoded final token ids -- and drives exact-string survival; it defaults to
    ``post_template_text`` for the no-truncation case. When a truncation policy drops tokens,
    callers MUST pass the decoded truncated text as ``final_text`` so exact survival reflects what
    the model actually sees, not the pre-truncation string.

    ``trigger_token_span`` lets a caller supply a pre-localized final-token span (e.g. from
    ``TokenizerAdapter.locate_token_span``, robust to boundary re-tokenization) for the token
    metrics; when unset it defaults to a token-id subsequence search, so existing callers are
    unchanged.

    ``semantic_scorer`` / ``summary_region`` / ``semantic_threshold`` are the opt-in semantic axis:
    when all three are supplied and the policy compresses, a paraphrase whose meaning survived (but
    whose exact text and tokens did not) is classified :attr:`SurvivalClass.SEMANTIC_SURVIVAL`.
    Default ``None`` leaves the semantic path dormant, so the existing behavior is unchanged.

    ``final_token_ids``, when supplied, is attached inline to the returned result (the probe wave's
    join input, persisted separately by default via the ``final_tokens.jsonl`` sidecar -- see
    ``io/final_tokens.py``). Default ``None`` leaves the field unset, so existing callers are
    unaffected.
    """
    scorer = scorer or TokenSurvivalScorer()
    builder = builder or SurvivalResultBuilder()
    exact_text = post_template_text if final_text is None else final_text
    assessment = scorer.assess(
        input_ids,
        trigger_ids,
        final_text=exact_text,
        trigger_text=trigger.text,
        trigger_token_span=trigger_token_span,
        require_boundary_cut=require_boundary_cut,
    )
    # Only a system-planted trigger can migrate role, so the rendered-role scan (and its regex cost)
    # runs solely for that position; every other position keeps its exact previous classification.
    rendered_role = (
        rendered_role_of_span(post_template_text, trigger.text)
        if trial.trigger_position == TriggerPosition.SYSTEM
        else None
    )
    return builder.build(
        trial,
        trigger,
        assessment,
        final_token_count=len(input_ids),
        raw_present=raw_present,
        post_pipeline_present=post_pipeline_present,
        post_template_present=trigger.text in post_template_text,
        pipeline_meta=pipeline_meta,
        extra_metadata=extra_metadata,
        rendered_role=rendered_role,
        semantic_scorer=semantic_scorer,
        summary_region=summary_region,
        semantic_threshold=semantic_threshold,
        final_token_ids=final_token_ids,
    )


def template_incompatible_result(
    trial: TrialSpec,
    trigger: TriggerSpec,
    *,
    raw_present: bool,
    post_pipeline_present: bool,
    error: str,
) -> SurvivalResult:
    """Build the SurvivalResult for a trial whose message sequence the model's template rejected.

    Delivery failed at the template stage: nothing was rendered, so no trigger reaches the model
    (``final_prompt_token_count`` is 0 and every post-template flag is False). ``raw_present`` /
    ``post_pipeline_present`` are computed by the caller from the pre-template layers, and the
    underlying template error text is preserved in ``metadata`` for diagnosis. This is a distinct
    delivery-failure mode, not a token-level drop -- see :attr:`FailureStage.TEMPLATE_INCOMPATIBLE`.
    """
    return SurvivalResult(
        trial_id=trial.trial_id,
        base_id=trial.base_id,
        model_id=trial.model_id,
        tokenizer_id=trial.resolved_tokenizer_id(),
        trigger_id=trigger.trigger_id,
        trigger_text=trigger.text,
        trigger_position=trial.trigger_position,
        context_length=trial.context_length,
        pipeline_policy=trial.pipeline_policy,
        chat_template=trial.chat_template,
        run_generation=trial.run_generation,
        raw_trigger_present=raw_present,
        post_pipeline_trigger_present=post_pipeline_present,
        post_template_trigger_present=False,
        final_token_trigger_present=False,
        trigger_exact_survived=False,
        trigger_token_survived=False,
        trigger_partial_survived=False,
        final_prompt_token_count=0,
        final_token_ids=None,  # nothing was rendered, so no final tokens exist for this trial.
        survival_class=SurvivalClass.NO_SURVIVAL,
        failure_stage=FailureStage.TEMPLATE_INCOMPATIBLE,
        metadata={"template_error": error},
    )


def aggregate_survival(results: Iterable[SurvivalResult]) -> list[dict[str, Any]]:
    """Summarize survival rates per (pipeline policy, trigger position) using stdlib only.

    Returns one row per group with counts and exact/token/partial/delivered rates, suitable for
    rendering as the project's headline result table. For large-scale aggregation, load the
    JSONL/Parquet results into pandas instead.
    """
    buckets: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "exact": 0, "token": 0, "partial": 0, "delivered": 0}
    )
    for result in results:
        key = (result.pipeline_policy, result.trigger_position.value)
        bucket = buckets[key]
        bucket["n"] += 1
        bucket["exact"] += int(result.trigger_exact_survived)
        bucket["token"] += int(result.trigger_token_survived)
        bucket["partial"] += int(result.trigger_partial_survived)
        bucket["delivered"] += int(result.final_token_trigger_present)

    rows: list[dict[str, Any]] = []
    for (policy, position), bucket in sorted(buckets.items()):
        n = bucket["n"] or 1
        rows.append(
            {
                "pipeline_policy": policy,
                "trigger_position": position,
                "n": bucket["n"],
                "exact_rate": bucket["exact"] / n,
                "token_rate": bucket["token"] / n,
                "partial_rate": bucket["partial"] / n,
                "delivered_rate": bucket["delivered"] / n,
            }
        )
    return rows
