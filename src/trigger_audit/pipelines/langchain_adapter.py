"""LangChain-backed PRE_TEMPLATE memory policy: wrap ``trim_messages`` as a StagedPolicy.

This adapts LangChain's message-trimming utility into the project's staged composition pipeline,
so a real, widely used memory-management strategy (not just the toy ``keep_last_n``) can be audited
for trigger survival with no change to the pipeline itself. ``langchain_core`` is imported lazily
inside the functions/methods (like :class:`HFTokenizerAdapter` imports ``transformers``) so that
importing this module never requires langchain to be installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from trigger_audit.pipelines.composition import CompositionContext, Stage, StagedPolicy
from trigger_audit.schemas.messages import ChatMessage, Role

# A list-level token counter: given a list of messages, return their total token count. LangChain's
# ``trim_messages`` treats a callable as a list counter unless its first parameter is annotated
# ``BaseMessage``, so this alias (and every counter we build) deliberately stays list-shaped.
ListTokenCounter = Callable[[Sequence[Any]], int]


def to_langchain(messages: Sequence[ChatMessage]) -> list[Any]:
    """Convert our :class:`ChatMessage` list into LangChain ``BaseMessage`` instances.

    Roles map SYSTEM->SystemMessage, USER->HumanMessage, ASSISTANT->AIMessage, TOOL->ToolMessage;
    ``content`` is preserved exactly. A ``ToolMessage`` requires a ``tool_call_id``, so one is taken
    from the message ``metadata`` (or its ``name``) and falls back to an empty string. Roles with
    no LangChain equivalent (e.g. ``DOCUMENT``) raise ``ValueError`` rather than being dropped.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    converted: list[Any] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            converted.append(SystemMessage(content=message.content))
        elif message.role is Role.USER:
            converted.append(HumanMessage(content=message.content))
        elif message.role is Role.ASSISTANT:
            converted.append(AIMessage(content=message.content))
        elif message.role is Role.TOOL:
            tool_call_id = message.metadata.get("tool_call_id") or message.name or ""
            converted.append(ToolMessage(content=message.content, tool_call_id=tool_call_id))
        else:
            raise ValueError(f"Role has no LangChain message equivalent: {message.role!r}")
    return converted


def from_langchain(messages: Sequence[Any]) -> list[ChatMessage]:
    """Convert LangChain ``BaseMessage`` instances back into our :class:`ChatMessage` list.

    Maps each message by class back to its :class:`Role` (SystemMessage->SYSTEM, HumanMessage->USER,
    AIMessage->ASSISTANT, ToolMessage->TOOL), preserving ``content`` exactly. Unknown message
    classes raise ``ValueError``.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    converted: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = Role.SYSTEM
        elif isinstance(message, HumanMessage):
            role = Role.USER
        elif isinstance(message, ToolMessage):
            role = Role.TOOL
        elif isinstance(message, AIMessage):
            role = Role.ASSISTANT
        else:
            raise ValueError(f"Unsupported LangChain message class: {type(message).__name__}")
        converted.append(ChatMessage(role=role, content=message.content))
    return converted


class LangChainTrimPolicy(StagedPolicy):
    """A PRE_TEMPLATE memory policy that trims the message list via LangChain's ``trim_messages``.

    The policy converts our messages to LangChain messages, applies ``trim_messages`` with the
    stored configuration, and converts the survivors back -- so which messages (and therefore which
    triggers) reach the chat template is decided by a real trimming strategy. Behaviorally,
    ``strategy="last"`` with ``include_system=True`` and a message-count budget mirrors the
    project's ``keep_last_n`` policy, while ``strategy="first"`` keeps the opening turns instead.
    """

    stage = Stage.PRE_TEMPLATE

    def __init__(
        self,
        *,
        max_tokens: int,
        strategy: str = "last",
        token_counter: ListTokenCounter | None = None,
        adapter: Any | None = None,
        include_system: bool = False,
        allow_partial: bool = False,
        text_splitter: Callable[[str], list[str]] | None = None,
    ) -> None:
        """Store the trimming configuration.

        ``token_counter`` may be any list-level counter (e.g. the builtin ``len``, which counts one
        token per message). If it is ``None`` a default counter backed by ``adapter`` is built
        (``sum(adapter.count_tokens(m.content) for m in messages)``); passing neither is a
        configuration error and raises ``ValueError``.
        """
        if token_counter is None:
            if adapter is None:
                raise ValueError("LangChainTrimPolicy requires either 'token_counter' or 'adapter'")
            # Read ``.content`` so the same counter works on our messages and on LangChain messages
            # (both expose ``.content``). ``resolved_adapter`` is non-None inside this branch.
            resolved_adapter = adapter

            def token_counter(messages: Sequence[Any]) -> int:
                return sum(resolved_adapter.count_tokens(m.content) for m in messages)

        self._max_tokens = max_tokens
        self._strategy = strategy
        self._token_counter: ListTokenCounter = token_counter
        self._include_system = include_system
        self._allow_partial = allow_partial
        self._text_splitter = text_splitter

    def apply(self, ctx: CompositionContext) -> None:
        """Trim ``ctx.messages`` in place using the stored ``trim_messages`` configuration."""
        from langchain_core.messages.utils import trim_messages

        kwargs: dict[str, Any] = {
            "max_tokens": self._max_tokens,
            "token_counter": self._token_counter,
            "strategy": self._strategy,
            "allow_partial": self._allow_partial,
        }
        # ``include_system`` is only valid with ``strategy="last"``; passing it otherwise raises.
        if self._strategy == "last":
            kwargs["include_system"] = self._include_system
        if self._text_splitter is not None:
            kwargs["text_splitter"] = self._text_splitter

        trimmed = trim_messages(to_langchain(ctx.messages), **kwargs)
        ctx.messages = from_langchain(trimmed)
