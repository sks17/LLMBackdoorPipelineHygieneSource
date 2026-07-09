"""Deterministic insertion of a harmless trigger into a base conversation at a controlled slot.

Insertion is done by our own code (never by the conversation generator) so that placement is
controlled, reproducible, and cleanly labeled. A named slot in the base conversation is used
when available; otherwise the trigger is placed positionally in the relevant message.
"""

from __future__ import annotations

import re

from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec

_SLOT_PATTERN = re.compile(r"\{\{[^}]+\}\}")

# Default slot placeholder associated with each position (used when a base conversation
# was generated with named slots).
_SLOT_BY_POSITION: dict[TriggerPosition, str] = {
    TriggerPosition.PREFIX: "{{PREFIX_SLOT}}",
    TriggerPosition.EARLY: "{{PREFIX_SLOT}}",
    TriggerPosition.MIDDLE: "{{MIDDLE_SLOT}}",
    TriggerPosition.LATE: "{{END_SLOT}}",
    TriggerPosition.END: "{{END_SLOT}}",
    TriggerPosition.NEAR_BOUNDARY: "{{BOUNDARY_SLOT}}",
    TriggerPosition.OLD_TURN: "{{OLD_TURN_SLOT}}",
    TriggerPosition.RECENT_TURN: "{{RECENT_TURN_SLOT}}",
    TriggerPosition.TOOL_OUTPUT: "{{TOOL_OUTPUT_SLOT}}",
    TriggerPosition.RETRIEVED_DOC: "{{RETRIEVED_DOC_SLOT}}",
}

# Which *message* an early trigger targets (first vs last user message). RECENT_TURN is
# deliberately NOT here: it targets the last user message but is placed at that message's prefix.
_EARLY_POSITIONS = {TriggerPosition.PREFIX, TriggerPosition.EARLY, TriggerPosition.OLD_TURN}

# Where *within the target message* the trigger sits: at the start of the content. This is
# decoupled from message targeting so RECENT_TURN can be "last message, prefix placement" -- which
# lets a tight tail-keeping head truncation drop the trigger while keeping the question after it.
_PREFIX_PLACEMENT = {
    TriggerPosition.PREFIX,
    TriggerPosition.EARLY,
    TriggerPosition.OLD_TURN,
    TriggerPosition.SYSTEM,
    TriggerPosition.RECENT_TURN,
}


def slot_for_position(position: TriggerPosition) -> str | None:
    """Return the named slot placeholder a position fills, or None if it has no slot.

    The single source of truth for the position->slot mapping, shared by :class:`TriggerInserter`
    (which fills a slot) and the dataset adapter (which *plants* the same slot in a real base), so
    a slot planted for a position is always the one the inserter later fills at that position.
    """
    return _SLOT_BY_POSITION.get(position)


# Positions whose *positional fallback* would mis-plant when the base lacks their slot: TOOL_OUTPUT
# and RETRIEVED_DOC target an agent ``tool`` / retrieved-document message that only exists in the
# agent-tool / RAG families, so on a base without the slot the inserter would silently drop the
# trigger into the last user turn -- a mislabeled plant. These are expanded only on bases that carry
# the slot. Every other position has well-defined positional placement on any base (prefix/middle/
# end/system, and old_turn/recent_turn degrade to the single user turn on a single-turn base), so
# they are never filtered -- preserving the validated grid behavior.
SLOT_STRICT_POSITIONS: frozenset[TriggerPosition] = frozenset(
    {TriggerPosition.TOOL_OUTPUT, TriggerPosition.RETRIEVED_DOC}
)


def base_has_slot(conversation: BaseConversation, slot: str) -> bool:
    """Whether any message of the base still carries the named slot placeholder (pre-insertion)."""
    return any(slot in message.content for message in conversation.messages)


def plantable_positions(
    conversation: BaseConversation, positions: list[TriggerPosition]
) -> list[TriggerPosition]:
    """Filter ``positions`` to those the base can host without a mis-plant.

    A :data:`SLOT_STRICT_POSITIONS` entry is kept only when the base actually carries its slot;
    every other position is kept (positional placement is well-defined on any base). This is what
    lets an agent-tool base contribute ``tool_output`` trials while a plain chat / long-doc base in
    the same corpus does not, without expanding an un-plantable ``tool_output`` cell for it.
    """
    kept: list[TriggerPosition] = []
    for position in positions:
        if position in SLOT_STRICT_POSITIONS:
            slot = slot_for_position(position)
            if slot is None or not base_has_slot(conversation, slot):
                continue
        kept.append(position)
    return kept


def target_user_index(messages: list[ChatMessage], position: TriggerPosition) -> int:
    """Choose which message a positional trigger targets.

    The single source of truth for message targeting, shared by :class:`TriggerInserter` and the
    lightweight ``prompts.insert_trigger`` helper. ``SYSTEM`` targets the first system message;
    ``TOOL_OUTPUT`` targets the last ``tool``-role message (an agent/tool result, not a user turn);
    early positions (prefix/early/old_turn) target the first user message; everything else targets
    the last user message. Each role-specific position falls back to user targeting when its role is
    absent, and with no user message either falls back to all indices, so a target always exists.
    """
    if position == TriggerPosition.SYSTEM:
        for i, message in enumerate(messages):
            if message.role == Role.SYSTEM:
                return i
    if position == TriggerPosition.TOOL_OUTPUT:
        tool_indices = [i for i, m in enumerate(messages) if m.role == Role.TOOL]
        if tool_indices:
            return tool_indices[-1]
    user_indices = [i for i, m in enumerate(messages) if m.role == Role.USER]
    if not user_indices:
        user_indices = list(range(len(messages)))
    return user_indices[0] if position in _EARLY_POSITIONS else user_indices[-1]


def place_in_content(content: str, trigger_text: str, position: TriggerPosition) -> str:
    """Insert the trigger at the prefix, middle, or end of a message's content.

    The single source of truth for positional placement, shared by :class:`TriggerInserter`
    (the full slot-aware inserter) and the lightweight ``prompts.insert_trigger`` helper so the
    two can never drift.
    """
    if position == TriggerPosition.MIDDLE:
        words = content.split()
        if not words:
            return trigger_text
        mid = len(words) // 2
        head = " ".join(words[:mid])
        tail = " ".join(words[mid:])
        return f"{head}\n\n{trigger_text}\n\n{tail}".strip()
    if position in _PREFIX_PLACEMENT:
        return f"{trigger_text}\n\n{content}".strip()
    # END, LATE, and (for now) NEAR_BOUNDARY land at the end.
    # TODO: make NEAR_BOUNDARY budget-aware so it lands exactly at the truncation boundary.
    return f"{content}\n\n{trigger_text}".strip()


def strip_unused_slots(messages: list[ChatMessage]) -> None:
    """Blank out any remaining ``{{SLOT}}`` placeholders so they do not pollute the prompt."""
    for message in messages:
        if "{{" in message.content:
            message.content = _SLOT_PATTERN.sub("", message.content)


class TriggerInserter:
    """Inserts a trigger string into a base conversation at a controlled position or slot."""

    def insert(
        self,
        conversation: BaseConversation,
        trigger: TriggerSpec,
        position: TriggerPosition,
    ) -> tuple[list[ChatMessage], int]:
        """Return a copy of the conversation's messages with the trigger inserted.

        Returns ``(messages, inserted_message_index)``. Prefers a named slot; falls back to
        positional placement in the relevant message. Unused slots are stripped afterward.
        """
        messages = [m.model_copy(deep=True) for m in conversation.messages]
        if not messages:
            return messages, -1

        slot = trigger.slot or _SLOT_BY_POSITION.get(position)
        inserted_index = -1
        if slot:
            for i, message in enumerate(messages):
                if slot in message.content:
                    message.content = message.content.replace(slot, trigger.text)
                    inserted_index = i
                    break

        if inserted_index == -1:
            inserted_index = self._target_index(messages, position)
            messages[inserted_index].content = self._place(
                messages[inserted_index].content, trigger.text, position
            )

        strip_unused_slots(messages)
        return messages, inserted_index

    def _target_index(self, messages: list[ChatMessage], position: TriggerPosition) -> int:
        """Choose which message receives a positional trigger."""
        return target_user_index(messages, position)

    def _place(self, content: str, trigger_text: str, position: TriggerPosition) -> str:
        """Insert the trigger at the prefix, middle, or end of a message's content."""
        return place_in_content(content, trigger_text, position)
