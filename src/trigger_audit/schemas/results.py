"""Result schemas: per-trial survival audit records and optional behavioral records."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from trigger_audit.schemas.triggers import TriggerPosition


class SurvivalClass(str, Enum):
    """Primary classification of how a trigger survived into the final model-visible prompt."""

    EXACT_SURVIVAL = "exact_survival"
    TOKEN_SURVIVAL = "token_survival"
    PARTIAL_SURVIVAL = "partial_survival"
    SEMANTIC_SURVIVAL = "semantic_survival"
    BOUNDARY_CORRUPTION = "boundary_corruption"
    ROLE_MIGRATION = "role_migration"
    NO_SURVIVAL = "no_survival"


class FailureStage(str, Enum):
    """Pipeline stage that removed or corrupted a trigger, when it failed to survive."""

    NONE = "none"
    MEMORY_POLICY_DROPPED = "memory_policy_dropped"
    TRUNCATED_HEAD = "truncated_head"
    TRUNCATED_TAIL = "truncated_tail"
    TRUNCATED_MIDDLE = "truncated_middle"
    TEMPLATE_REMOVED_OR_CHANGED = "template_removed_or_changed"
    # The memory/pipeline stage produced a message sequence the model's chat template rejects
    # outright (e.g. a non-alternating sequence for a strict-alternation template), so nothing is
    # delivered -- a delivery failure at the template stage, not a token-level drop.
    TEMPLATE_INCOMPATIBLE = "template_incompatible"
    COMPRESSED_EXACT_DELETED = "compressed_exact_deleted"
    NOT_RETRIEVED = "not_retrieved"
    PACKING_BUDGET_EXCLUDED = "packing_budget_excluded"
    FINAL_TOKEN_ABSENT = "final_token_absent"


class SurvivalResult(BaseModel):
    """Per-trial record of whether and where a trigger survived into the final prompt.

    The four ``*_trigger_present`` flags mirror the four logged layers (raw messages,
    post-pipeline messages, post-template text, final token ids) so a failure can be
    attributed to a specific stage. A separate, optional semantic axis
    (``trigger_semantic_*``) records whether the trigger's *meaning* survived a real
    summarizer as paraphrase -- delivery that the token-level flags cannot see.
    """

    trial_id: str
    base_id: str
    model_id: str
    tokenizer_id: str
    trigger_id: str
    trigger_text: str
    trigger_position: TriggerPosition
    context_length: int
    pipeline_policy: str
    chat_template: str | None = None
    run_generation: bool = False

    # Presence across the four logged layers.
    raw_trigger_present: bool
    post_pipeline_trigger_present: bool
    post_template_trigger_present: bool
    final_token_trigger_present: bool

    # Token-level survival metrics.
    trigger_exact_survived: bool
    trigger_token_survived: bool
    trigger_partial_survived: bool
    trigger_final_token_start: int | None = None
    trigger_final_token_end: int | None = None
    trigger_relative_position: float | None = None

    # Semantic (meaning-level) survival axis. Optional and defaulted so existing rows and
    # construction sites are unaffected; populated only when a semantic scorer is injected.
    trigger_semantic_survived: bool = False
    trigger_semantic_score: float | None = None

    final_prompt_token_count: int
    # The final model-visible token ids (an optional downstream-analysis sidecar input). Populated
    # only when persistence is requested (it roughly doubles a result row's size); the sidecar
    # `final_tokens.jsonl` is the primary producer -- see io/final_tokens.py.
    final_token_ids: list[int] | None = None
    final_prompt_text_path: str | None = None

    survival_class: SurvivalClass
    failure_stage: FailureStage = FailureStage.NONE
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagDeliveryResult(BaseModel):
    """Per-trial record of whether a trigger survives the retrieval pipeline into the final prompt.

    The RAG analogue of :class:`SurvivalResult`: it logs the trigger's presence at each retrieval
    stage (retrieved set, packed context, final tokens) so the delivery decomposition
    ``P(delivered) = P(retrieved) x P(packed | retrieved) x P(final | packed)`` is attributable to a
    stage. ``failure_stage=not_retrieved`` is the retrieval-specific failure this schema first
    exercises.
    """

    trial_id: str
    model_id: str
    tokenizer_id: str
    trigger_id: str
    trigger_text: str
    top_k: int
    embedding_id: str

    retrieved_chunk_ids: list[str]
    packed_chunk_ids: list[str]
    dropped_chunk_ids: list[str] = Field(default_factory=list)

    trigger_present_in_retrieved: bool
    trigger_present_in_packed: bool
    trigger_present_in_final_tokens: bool

    final_prompt_token_count: int
    final_prompt_text_path: str | None = None

    survival_class: SurvivalClass
    failure_stage: FailureStage = FailureStage.NONE
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Optional behavioral record linking a trial's final prompt to model output and activation.

    Generation is secondary to delivery: interpret ``activation_detected`` only conditional on
    the trigger having been delivered (see the matching :class:`SurvivalResult`).
    """

    trial_id: str
    model_id: str
    final_prompt_text_path: str | None = None
    model_output: str | None = None
    model_output_path: str | None = None
    activation_detected: bool | None = None
    expected_activation_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
