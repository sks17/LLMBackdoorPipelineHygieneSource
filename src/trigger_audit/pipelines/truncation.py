"""Token-level truncation policies (applied to the final token sequence, after templating).

Truncation is a deliberate, labeled design choice, not "whatever the model does". The token
budget depends on the model; the policy that enforces it is defined here and stays interpretable
across models with different context windows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from trigger_audit.pipelines.base import Registry

TRUNCATION_REGISTRY: Registry[TruncationPolicy] = Registry("truncation")


@dataclass(frozen=True)
class TruncationOutcome:
    """Result of applying a truncation policy: the kept ids and how many were dropped per end."""

    kept_ids: list[int]
    dropped_head: int
    dropped_tail: int

    @property
    def truncated(self) -> bool:
        """True if any tokens were dropped."""
        return self.dropped_head > 0 or self.dropped_tail > 0


class TruncationPolicy(ABC):
    """Reduces a token sequence to fit a budget, recording which end(s) were cut."""

    name: str = "truncation"

    @abstractmethod
    def apply(self, token_ids: Sequence[int], budget: int) -> TruncationOutcome:
        """Return the kept tokens and per-end drop counts for the given budget."""


@TRUNCATION_REGISTRY.register("none")
class NoTruncation(TruncationPolicy):
    """Pass the full sequence through unchanged (positive control)."""

    name = "none"

    def apply(self, token_ids: Sequence[int], budget: int) -> TruncationOutcome:
        return TruncationOutcome(kept_ids=list(token_ids), dropped_head=0, dropped_tail=0)


@TRUNCATION_REGISTRY.register("truncate_head")
class HeadTruncation(TruncationPolicy):
    """Drop tokens from the beginning, keeping the most recent ``budget`` tokens.

    Destroys prefix triggers; preserves end-of-prompt content.
    """

    name = "truncate_head"

    def apply(self, token_ids: Sequence[int], budget: int) -> TruncationOutcome:
        budget = max(0, budget)
        n = len(token_ids)
        if n <= budget:
            return TruncationOutcome(kept_ids=list(token_ids), dropped_head=0, dropped_tail=0)
        dropped = n - budget
        return TruncationOutcome(
            kept_ids=list(token_ids[dropped:]), dropped_head=dropped, dropped_tail=0
        )


@TRUNCATION_REGISTRY.register("truncate_tail")
class TailTruncation(TruncationPolicy):
    """Drop tokens from the end, keeping the first ``budget`` tokens.

    Preserves prefix triggers; destroys end-of-prompt content.
    """

    name = "truncate_tail"

    def apply(self, token_ids: Sequence[int], budget: int) -> TruncationOutcome:
        budget = max(0, budget)
        n = len(token_ids)
        if n <= budget:
            return TruncationOutcome(kept_ids=list(token_ids), dropped_head=0, dropped_tail=0)
        dropped = n - budget
        return TruncationOutcome(
            kept_ids=list(token_ids[:budget]), dropped_head=0, dropped_tail=dropped
        )


@TRUNCATION_REGISTRY.register("truncate_middle")
class MiddleTruncation(TruncationPolicy):
    """Keep the head and tail halves of the budget, dropping the middle (lost-in-the-middle)."""

    name = "truncate_middle"

    def apply(self, token_ids: Sequence[int], budget: int) -> TruncationOutcome:
        budget = max(0, budget)
        n = len(token_ids)
        if n <= budget:
            return TruncationOutcome(kept_ids=list(token_ids), dropped_head=0, dropped_tail=0)
        head_keep = budget // 2
        tail_keep = budget - head_keep
        dropped = n - budget
        kept = list(token_ids[:head_keep]) + list(token_ids[n - tail_keep :])
        # Report the removed middle block as a "tail" drop relative to the head segment.
        return TruncationOutcome(kept_ids=kept, dropped_head=0, dropped_tail=dropped)
