"""Concrete pipeline steps that wire policies and renderers into a :class:`Pipeline`.

Each step is a thin adapter binding a reusable component (inserter, memory policy, chat
renderer, truncation policy) to the shared :class:`PipelineContext`.
"""

from __future__ import annotations

from trigger_audit.pipelines.base import PipelineContext, PipelineStep
from trigger_audit.pipelines.memory_policy import MemoryPolicy, TokenCounter
from trigger_audit.pipelines.trigger_insertion import TriggerInserter, strip_unused_slots
from trigger_audit.pipelines.truncation import TruncationPolicy
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.schemas.messages import BaseConversation
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter


class TriggerInsertionStep(PipelineStep):
    """Insert a trigger into the base conversation and snapshot it as Layer 1 (raw messages).

    With ``insert=False`` the trigger is not placed at all (the counterfactual control): the base
    conversation is snapshotted with its unused slots blanked, so the trigger text is absent at
    every layer and the row is the scoring sanity control that must classify ``no_survival``.
    """

    name = "trigger_insertion"

    def __init__(
        self,
        inserter: TriggerInserter,
        conversation: BaseConversation,
        trigger: TriggerSpec,
        position: TriggerPosition,
        *,
        insert: bool = True,
    ) -> None:
        self._inserter = inserter
        self._conversation = conversation
        self._trigger = trigger
        self._position = position
        self._insert = insert

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        if self._insert:
            messages, index = self._inserter.insert(
                self._conversation, self._trigger, self._position
            )
        else:
            messages = [m.model_copy(deep=True) for m in self._conversation.messages]
            strip_unused_slots(messages)
            index = -1
        ctx.raw_messages = [m.model_copy(deep=True) for m in messages]
        ctx.messages = messages
        ctx.metadata["trigger_message_index"] = index
        return ctx


class MemoryPolicyStep(PipelineStep):
    """Apply a message-level memory policy, producing Layer 2 (post-pipeline messages)."""

    name = "memory_policy"

    def __init__(self, policy: MemoryPolicy, counter: TokenCounter, budget: int) -> None:
        self._policy = policy
        self._counter = counter
        self._budget = budget

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        outcome = self._policy.apply(ctx.messages, self._budget, self._counter)
        ctx.messages = outcome.messages
        ctx.metadata["memory_policy"] = self._policy.name
        ctx.metadata["memory_dropped_indices"] = outcome.dropped_indices
        return ctx


class ChatTemplateStep(PipelineStep):
    """Render messages to text (Layer 3) and tokenize to ids (Layer 4)."""

    name = "chat_template"

    def __init__(
        self,
        renderer: ChatTemplateRenderer,
        adapter: TokenizerAdapter,
        *,
        add_special_tokens: bool = False,
    ) -> None:
        self._renderer = renderer
        self._adapter = adapter
        self._add_special_tokens = add_special_tokens

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        ctx.rendered_prompt = self._renderer.render(ctx.messages)
        ctx.final_token_ids = self._adapter.encode(
            ctx.rendered_prompt, add_special_tokens=self._add_special_tokens
        )
        return ctx


class TruncationStep(PipelineStep):
    """Apply token-level truncation to the final token sequence (Layer 4)."""

    name = "truncation"

    def __init__(self, policy: TruncationPolicy, budget: int) -> None:
        self._policy = policy
        self._budget = budget

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.final_token_ids is None:
            return ctx
        outcome = self._policy.apply(ctx.final_token_ids, self._budget)
        ctx.final_token_ids = outcome.kept_ids
        ctx.metadata["truncation"] = {
            "policy": self._policy.name,
            "dropped_head": outcome.dropped_head,
            "dropped_tail": outcome.dropped_tail,
        }
        return ctx
