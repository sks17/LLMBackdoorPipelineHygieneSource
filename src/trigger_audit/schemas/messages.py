"""Conversation schemas: roles, messages, and labeled base conversations with insertion slots."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role. ``DOCUMENT`` carries retrieved RAG context (used by later experiments)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DOCUMENT = "document"


class ChatMessage(BaseModel):
    """A single role-tagged message; ``metadata`` carries tool-call ids and similar extras."""

    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlotLocation(BaseModel):
    """Records where a named insertion slot (e.g. ``{{PREFIX_SLOT}}``) lives in a conversation."""

    slot: str
    message_index: int
    description: str | None = None


class BaseConversation(BaseModel):
    """A synthetic base conversation with labeled trigger-insertion slots and no trigger text.

    Base conversations are generated once and reused across every experimental condition so
    that survival differences are attributable to the pipeline, not to content variation.
    """

    base_id: str
    conversation_type: str
    domain: str | None = None
    target_token_length: int | None = None
    messages: list[ChatMessage]
    expected_user_task: str | None = None
    slot_locations: list[SlotLocation] = Field(default_factory=list)
    difficulty: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
