"""Document schema: a corpus document for the retrieval (RAG) experiments.

A document is distinct from a conversation message: it is a unit of a retrievable corpus, ranked
and packed into the prompt by the retrieval pipeline rather than authored as a turn. Like base
conversations, a document may carry a named ``trigger_slot`` so this package's code (never the
corpus generator) inserts the trigger at a controlled position.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single corpus document, optionally carrying a named trigger-insertion slot."""

    doc_id: str
    content: str
    trigger_slot: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
