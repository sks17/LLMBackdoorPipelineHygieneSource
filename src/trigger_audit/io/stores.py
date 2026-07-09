"""Indexed, id-keyed access to JSONL collections (base conversations, triggers)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from trigger_audit.io.jsonl import iter_jsonl
from trigger_audit.schemas.messages import BaseConversation
from trigger_audit.schemas.triggers import TriggerSpec

ModelT = TypeVar("ModelT", bound=BaseModel)

PathLike = str | Path


class IndexedJsonlStore(Generic[ModelT]):
    """Loads a JSONL file of records into memory and indexes them by an id field.

    Suitable for the small-to-medium reference collections used per shard (base conversations,
    triggers). For very large corpora a future experiment can swap in an on-disk index.
    """

    def __init__(self, path: PathLike, model_cls: type[ModelT], *, id_field: str) -> None:
        self._model_cls = model_cls
        self._id_field = id_field
        self._items: dict[str, ModelT] = {}
        for row in iter_jsonl(path):
            item = model_cls.model_validate(row)
            self._items[str(getattr(item, id_field))] = item

    def get(self, item_id: str) -> ModelT:
        """Return the record with the given id, raising KeyError if absent."""
        return self._items[item_id]

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def ids(self) -> list[str]:
        """Return all ids in insertion order."""
        return list(self._items)

    def __iter__(self) -> Iterator[ModelT]:
        return iter(self._items.values())


class BaseConversationStore(IndexedJsonlStore[BaseConversation]):
    """Id-keyed access to base conversations."""

    def __init__(self, path: PathLike) -> None:
        super().__init__(path, BaseConversation, id_field="base_id")


class TriggerStore(IndexedJsonlStore[TriggerSpec]):
    """Id-keyed access to trigger specs."""

    def __init__(self, path: PathLike) -> None:
        super().__init__(path, TriggerSpec, id_field="trigger_id")
