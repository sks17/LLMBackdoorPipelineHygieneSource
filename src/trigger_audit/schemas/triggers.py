"""Trigger schemas: harmless canary strings and the positions they can be inserted at."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Family of a harmless canary trigger; drives which survival modes it stresses."""

    RANDOM_CANARY = "random_canary"
    NATURAL_PHRASE = "natural_phrase"
    MULTI_TOKEN_PHRASE = "multi_token_phrase"
    SPLIT = "split"
    BOUNDARY = "boundary"
    UNICODE = "unicode"


class TriggerPosition(str, Enum):
    """Controlled slot a trigger is inserted into within a base conversation."""

    PREFIX = "prefix"
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    END = "end"
    NEAR_BOUNDARY = "near_boundary"
    OLD_TURN = "old_turn"
    RECENT_TURN = "recent_turn"
    SYSTEM = "system"
    TOOL_OUTPUT = "tool_output"
    RETRIEVED_DOC = "retrieved_doc"


class TriggerSpec(BaseModel):
    """A harmless canary trigger plus the metadata needed to place and match it.

    ``parts`` holds the components of a split trigger (assembled across turns in later
    experiments); ``slot`` optionally pins the trigger to a named slot in a base conversation.
    """

    trigger_id: str
    trigger_type: TriggerType
    text: str
    slot: str | None = None
    parts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
