"""Prompt-construction pipeline: base abstractions, policies, steps, and registries.

Future experiments compose new pipelines from these pieces (for example, a RAG experiment adds
retrieval and packing steps) without changing the shared core.
"""

from trigger_audit.pipelines.base import (
    Pipeline,
    PipelineContext,
    PipelineStep,
    Registry,
)
from trigger_audit.pipelines.composition import (
    ComposedPipeline,
    CompositionContext,
    CompositionResult,
    HeadTruncationPolicy,
    KeepRecentMessagesPolicy,
    Stage,
    StagedPolicy,
)
from trigger_audit.pipelines.langchain_adapter import LangChainTrimPolicy
from trigger_audit.pipelines.memory_policy import (
    MEMORY_REGISTRY,
    KeepLastNMessages,
    KeepRecentMessages,
    MemoryOutcome,
    MemoryPolicy,
    NoMemoryPolicy,
    SummarizeOldMessages,
)
from trigger_audit.pipelines.policy_registry import resolve_policy
from trigger_audit.pipelines.steps import (
    ChatTemplateStep,
    MemoryPolicyStep,
    TriggerInsertionStep,
    TruncationStep,
)
from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.pipelines.truncation import (
    TRUNCATION_REGISTRY,
    HeadTruncation,
    MiddleTruncation,
    NoTruncation,
    TailTruncation,
    TruncationOutcome,
    TruncationPolicy,
)

__all__ = [
    "MEMORY_REGISTRY",
    "TRUNCATION_REGISTRY",
    "ChatTemplateStep",
    "ComposedPipeline",
    "CompositionContext",
    "CompositionResult",
    "HeadTruncation",
    "HeadTruncationPolicy",
    "KeepLastNMessages",
    "KeepRecentMessages",
    "KeepRecentMessagesPolicy",
    "LangChainTrimPolicy",
    "MemoryOutcome",
    "MemoryPolicy",
    "MemoryPolicyStep",
    "MiddleTruncation",
    "NoMemoryPolicy",
    "NoTruncation",
    "Pipeline",
    "PipelineContext",
    "PipelineStep",
    "Registry",
    "Stage",
    "StagedPolicy",
    "SummarizeOldMessages",
    "TailTruncation",
    "TriggerInserter",
    "TriggerInsertionStep",
    "TruncationOutcome",
    "TruncationPolicy",
    "TruncationStep",
    "resolve_policy",
]
