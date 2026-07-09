"""Lightweight, message-level trigger insertion for the prompt-construction path.

This is the thin entry point used by drivers that already hold a plain ``list[ChatMessage]``
(such as the Trial Zero and Trial One slices) and do not need the full slot-aware
:class:`TriggerInserter`. Placement itself is delegated to
:func:`pipelines.trigger_insertion.place_in_content`, so the two insertion paths share one
definition of each position and can never drift.
"""

from __future__ import annotations

from trigger_audit.pipelines.trigger_insertion import place_in_content, target_user_index
from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.schemas.triggers import TriggerPosition

# String positions this helper accepts, mapped to the shared placement enum. Extended per trial
# as new variants come online; anything absent raises rather than silently mis-placing a trigger.
_SUPPORTED_POSITIONS: dict[str, TriggerPosition] = {
    "prefix": TriggerPosition.PREFIX,
    "end": TriggerPosition.END,
    "old_turn": TriggerPosition.OLD_TURN,
    "recent_turn": TriggerPosition.RECENT_TURN,
}


def insert_trigger(
    messages: list[ChatMessage], trigger_text: str, position: str = "prefix"
) -> list[ChatMessage]:
    """Return a new message list with ``trigger_text`` inserted at ``position``.

    Pure and deterministic: the input list and its messages are never mutated (each message is
    deep-copied first). The target message is chosen by :func:`target_user_index` -- early
    positions (``prefix``/``old_turn``) target the first user message, later ones (``end``/
    ``recent_turn``) the last -- and :func:`place_in_content` positions the trigger within it.
    Positions not yet supported raise ``NotImplementedError`` rather than silently mis-placing it.
    """
    resolved = _SUPPORTED_POSITIONS.get(position)
    if resolved is None:
        supported = ", ".join(sorted(_SUPPORTED_POSITIONS))
        raise NotImplementedError(
            f"insert_trigger supports position in {{{supported}}}, got {position!r}"
        )

    result = [message.model_copy(deep=True) for message in messages]
    target = target_user_index(result, resolved)
    result[target].content = place_in_content(result[target].content, trigger_text, resolved)
    return result
