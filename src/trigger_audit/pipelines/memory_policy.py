"""Message-level memory policies (applied before chat templating).

These mirror real chat-memory behavior: keep the most recent turns under a token budget, or
compress older turns into a summary. Memory policies operate on messages; truncation policies
operate on the final token sequence. They are separate abstractions on purpose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from trigger_audit.pipelines.base import Registry
from trigger_audit.schemas.messages import ChatMessage, Role

MEMORY_REGISTRY: Registry[MemoryPolicy] = Registry("memory")

# A function that returns the token cost of a single message.
TokenCounter = Callable[[ChatMessage], int]
# A function that compresses a block of older messages into summary text (future: a real LLM).
Summarizer = Callable[[Sequence[ChatMessage]], str]


@dataclass(frozen=True)
class MemoryOutcome:
    """Result of applying a memory policy: the surviving messages and which were dropped."""

    messages: list[ChatMessage]
    dropped_indices: list[int]


class MemoryPolicy(ABC):
    """Reduces a message list under a token budget, recording which messages were dropped."""

    name: str = "memory"

    @abstractmethod
    def apply(
        self, messages: Sequence[ChatMessage], budget: int, counter: TokenCounter
    ) -> MemoryOutcome:
        """Return the surviving messages and the indices (into ``messages``) that were dropped."""


@MEMORY_REGISTRY.register("none")
class NoMemoryPolicy(MemoryPolicy):
    """Pass every message through unchanged (positive control)."""

    name = "none"

    def apply(
        self, messages: Sequence[ChatMessage], budget: int, counter: TokenCounter
    ) -> MemoryOutcome:
        return MemoryOutcome(messages=list(messages), dropped_indices=[])


@MEMORY_REGISTRY.register("keep_recent_messages")
class KeepRecentMessages(MemoryPolicy):
    """Keep all system messages plus a contiguous block of the most recent turns under budget.

    The single most recent non-system message is always kept (an application always sends the
    current user turn), then older turns are added while they fit. Older turns beyond the budget
    are dropped, which is why old-turn triggers tend to disappear under this policy.
    """

    name = "keep_recent_messages"

    def apply(
        self, messages: Sequence[ChatMessage], budget: int, counter: TokenCounter
    ) -> MemoryOutcome:
        system_idx = [i for i, m in enumerate(messages) if m.role == Role.SYSTEM]
        nonsystem_idx = [i for i, m in enumerate(messages) if m.role != Role.SYSTEM]

        kept = set(system_idx)
        used = sum(counter(messages[i]) for i in system_idx)

        for rank, idx in enumerate(reversed(nonsystem_idx)):
            cost = counter(messages[idx])
            if rank == 0 or used + cost <= budget:
                kept.add(idx)
                used += cost
            else:
                break

        dropped = [i for i in nonsystem_idx if i not in kept]
        survived = [messages[i] for i in range(len(messages)) if i in kept]
        return MemoryOutcome(messages=survived, dropped_indices=dropped)


@MEMORY_REGISTRY.register("keep_last_n_messages")
class KeepLastNMessages(MemoryPolicy):
    """Keep every system message plus the last ``keep_last_n`` non-system messages, as whole units.

    A pure *count* policy, distinct from the budget-based :class:`KeepRecentMessages`: it ignores
    the token budget and counter entirely and drops whole old turns. Because it never mutates
    message content, it cannot produce partial trigger survival -- a trigger's message is kept or
    dropped as a unit -- which is exactly the invariant Trial Two exists to pin down.
    """

    name = "keep_last_n_messages"

    def __init__(self, *, keep_last_n: int) -> None:
        self._keep_last_n = max(0, keep_last_n)

    def apply(
        self, messages: Sequence[ChatMessage], budget: int, counter: TokenCounter
    ) -> MemoryOutcome:
        # Count-based on purpose: ``budget`` and ``counter`` are ignored.
        system_idx = [i for i, m in enumerate(messages) if m.role == Role.SYSTEM]
        nonsystem_idx = [i for i, m in enumerate(messages) if m.role != Role.SYSTEM]

        recent = nonsystem_idx[-self._keep_last_n :] if self._keep_last_n else []
        kept = set(system_idx) | set(recent)

        survived = [messages[i] for i in range(len(messages)) if i in kept]
        dropped = [i for i in range(len(messages)) if i not in kept]
        return MemoryOutcome(messages=survived, dropped_indices=dropped)


@MEMORY_REGISTRY.register("summarize_old_messages")
@MEMORY_REGISTRY.register("summary_plus_recent")
class SummarizeOldMessages(MemoryPolicy):
    """Keep recent turns raw and replace older turns with a single summary placeholder.

    Without an injected ``summarizer`` (the default), older turns are compressed into a harmless
    placeholder that contains no trigger text -- this faithfully models exact-trigger deletion by
    summarization. **Inject a real summarizer** (``pipelines.summarizer`` -- reference or pinned HF)
    to study *semantic* survival: the summarization semantic-survival cell
    (``experiments/survivability_audit/summarization_semantic.py``) does exactly this and scores the
    resulting summary with the semantic scorer, no longer deferred.
    """

    name = "summarize_old_messages"

    def __init__(
        self,
        *,
        keep_recent_turns: int = 1,
        summary_text: str = "[Summary of earlier conversation.]",
        summary_role: str = "system",
        summarizer: Summarizer | None = None,
    ) -> None:
        self._keep_recent_turns = max(0, keep_recent_turns)
        self._summary_text = summary_text
        self._summary_role = Role(summary_role)
        self._summarizer = summarizer

    def apply(
        self, messages: Sequence[ChatMessage], budget: int, counter: TokenCounter
    ) -> MemoryOutcome:
        nonsystem_idx = [i for i, m in enumerate(messages) if m.role != Role.SYSTEM]

        keep_recent = (
            set(nonsystem_idx[-self._keep_recent_turns :]) if self._keep_recent_turns else set()
        )
        old_idx = [i for i in nonsystem_idx if i not in keep_recent]

        if not old_idx:
            return MemoryOutcome(messages=list(messages), dropped_indices=[])

        summary_text = (
            self._summarizer([messages[i] for i in old_idx])
            if self._summarizer is not None
            else self._summary_text
        )
        summary_message = ChatMessage(role=self._summary_role, content=summary_text)

        # Rebuild: keep system messages and recent turns in place; insert the summary just
        # before the first kept recent turn (or after system messages if none were kept).
        first_recent = min(keep_recent) if keep_recent else len(messages)
        survived: list[ChatMessage] = []
        inserted = False
        for i, message in enumerate(messages):
            if i in old_idx:
                continue
            if not inserted and i >= first_recent:
                survived.append(summary_message)
                inserted = True
            survived.append(message)
        if not inserted:
            survived.append(summary_message)

        return MemoryOutcome(messages=survived, dropped_indices=old_idx)
