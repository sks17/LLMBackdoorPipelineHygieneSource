"""Staged composition of message-level and token-level policies in one ordered pipeline.

Trial Three composes a memory policy (messages, before templating) with a truncation policy
(token ids, after templating). Execution order is derived from each policy's declared ``stage``,
not its position in the list, so the policies always run in the correct order -- and reversing the
declared list cannot change the result. This is what makes the interaction effect observable and
attributable: a message kept by memory can still be cut by truncation
(``post_pipeline_trigger_present=True`` but ``final_token_trigger_present=False``), a failure
neither policy produces alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trigger_audit.pipelines.memory_policy import KeepLastNMessages
from trigger_audit.pipelines.truncation import HeadTruncation
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter


class Stage(str, Enum):
    """When a policy runs relative to chat templating."""

    PRE_TEMPLATE = "pre_template"  # Layer 1->2: operates on messages, before templating
    POST_TEMPLATE = "post_template"  # Layer 3->4: operates on token ids, after templating


@dataclass
class CompositionContext:
    """Mutable carrier threaded through the staged policies as a conversation is composed."""

    messages: list[ChatMessage]
    token_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StagedPolicy(ABC):
    """A composition-friendly policy that knows its stage and holds its own config."""

    stage: Stage

    @abstractmethod
    def apply(self, ctx: CompositionContext) -> None:
        """Mutate ``ctx`` in place at this policy's stage."""


class KeepRecentMessagesPolicy(StagedPolicy):
    """PRE_TEMPLATE wrapper over the count-based :class:`KeepLastNMessages` (drops whole old turns).

    Named for the behavior it models (keep the recent turns); it delegates the drop math to
    ``KeepLastNMessages`` rather than reimplementing it.
    """

    stage = Stage.PRE_TEMPLATE

    def __init__(self, *, keep_last_n: int) -> None:
        self._policy = KeepLastNMessages(keep_last_n=keep_last_n)

    def apply(self, ctx: CompositionContext) -> None:
        # Count-based policy: budget and counter are ignored by the delegate.
        outcome = self._policy.apply(ctx.messages, budget=0, counter=lambda _message: 0)
        ctx.messages = outcome.messages
        ctx.metadata["memory_policy"] = self._policy.name


class HeadTruncationPolicy(StagedPolicy):
    """POST_TEMPLATE wrapper over :class:`HeadTruncation` (keeps the last N tokens)."""

    stage = Stage.POST_TEMPLATE

    def __init__(self, *, context_length_target: int) -> None:
        self._policy = HeadTruncation()
        self._context_length_target = context_length_target

    def apply(self, ctx: CompositionContext) -> None:
        if ctx.token_ids is None:
            raise ValueError(
                "HeadTruncationPolicy runs at POST_TEMPLATE; token_ids must be set by templating "
                "before it is applied"
            )
        outcome = self._policy.apply(ctx.token_ids, self._context_length_target)
        ctx.token_ids = outcome.kept_ids
        ctx.metadata["truncation"] = {
            "policy": self._policy.name,
            "dropped_head": outcome.dropped_head,
            "dropped_tail": outcome.dropped_tail,
        }


@dataclass(frozen=True)
class CompositionResult:
    """The layers a composed run produces, ready to hand to the survival scorer."""

    post_messages: list[ChatMessage]  # Layer 2 (post-memory)
    post_template_text: str  # Layer 3 (post-memory, pre-truncation)
    final_token_ids: list[int]  # Layer 4 (post-truncation)
    metadata: dict[str, Any]


class ComposedPipeline:
    """Runs staged policies in stage order (all PRE_TEMPLATE, then template, then POST_TEMPLATE).

    Ordering comes from each policy's ``stage``, not its position in the declared list. Filtering
    by stage preserves within-stage order but ignores cross-stage position, so reversing the
    declared policy list yields an identical :class:`CompositionResult`.
    """

    def __init__(
        self,
        policies: Sequence[StagedPolicy],
        *,
        renderer: ChatTemplateRenderer,
        adapter: TokenizerAdapter,
    ) -> None:
        self._policies = list(policies)
        self._renderer = renderer
        self._adapter = adapter

    def run(self, messages: Sequence[ChatMessage]) -> CompositionResult:
        """Compose the message list into the four survival layers, applying policies by stage."""
        ctx = CompositionContext(messages=[m.model_copy(deep=True) for m in messages])

        pre = [p for p in self._policies if p.stage is Stage.PRE_TEMPLATE]
        for policy in pre:
            policy.apply(ctx)

        text = self._renderer.render(ctx.messages)
        ctx.token_ids = self._adapter.encode(text, add_special_tokens=False)

        post = [p for p in self._policies if p.stage is Stage.POST_TEMPLATE]
        for policy in post:
            policy.apply(ctx)

        # token_ids is set by templating above and POST_TEMPLATE policies keep it a list.
        assert ctx.token_ids is not None
        return CompositionResult(
            post_messages=ctx.messages,
            post_template_text=text,
            final_token_ids=ctx.token_ids,
            metadata=ctx.metadata,
        )
