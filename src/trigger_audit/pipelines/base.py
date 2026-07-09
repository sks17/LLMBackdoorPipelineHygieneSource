"""Pipeline core: the context carrier, the step interface, the runner, and a name registry.

This is the extension point future experiments build on. A RAG experiment adds retrieval and
packing steps; a multi-turn experiment adds summarization steps. Each step is a small class
that mutates the shared :class:`PipelineContext`, which accumulates the four logged layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from trigger_audit.schemas.messages import ChatMessage

T = TypeVar("T")


@dataclass
class PipelineContext:
    """Mutable carrier for the four logged layers as a conversation flows through the pipeline.

    Layer 1 is ``raw_messages`` (raw logical messages, trigger already inserted); Layer 2 is
    ``messages`` (after memory/trimming policy); Layer 3 is ``rendered_prompt`` (after chat
    templating); Layer 4 is ``final_token_ids`` (what the model actually consumes).
    """

    raw_messages: list[ChatMessage]
    messages: list[ChatMessage]
    rendered_prompt: str | None = None
    final_token_ids: list[int] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_messages(cls, messages: Sequence[ChatMessage]) -> PipelineContext:
        """Create a context whose raw and working message lists are copies of ``messages``."""
        snapshot = [m.model_copy(deep=True) for m in messages]
        working = [m.model_copy(deep=True) for m in messages]
        return cls(raw_messages=snapshot, messages=working)


class PipelineStep(ABC):
    """A single transformation in a prompt-construction pipeline."""

    name: str = "step"

    @abstractmethod
    def apply(self, ctx: PipelineContext) -> PipelineContext:
        """Transform and return the context (in place)."""


class Pipeline:
    """Runs an ordered sequence of steps over a context, with an optional per-step callback."""

    def __init__(
        self,
        steps: Iterable[PipelineStep],
        *,
        on_step: Callable[[PipelineStep, PipelineContext], None] | None = None,
    ) -> None:
        self._steps = list(steps)
        self._on_step = on_step

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Apply every step in order and return the final context."""
        for step in self._steps:
            ctx = step.apply(ctx)
            if self._on_step is not None:
                self._on_step(step, ctx)
        return ctx


class Registry(Generic[T]):
    """A name to factory registry used to resolve policies (and steps) from config strings."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers a factory (class or function) under ``name``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, **kwargs: object) -> T:
        """Instantiate the factory registered under ``name``, forwarding kwargs."""
        if name not in self._factories:
            raise KeyError(
                f"Unknown {self._kind} policy {name!r}. Registered: {', '.join(self.names())}"
            )
        return self._factories[name](**kwargs)

    def names(self) -> list[str]:
        """Return the sorted list of registered names."""
        return sorted(self._factories)
