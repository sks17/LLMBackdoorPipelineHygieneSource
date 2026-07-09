"""JSONL read/write helpers that understand pydantic models."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

PathLike = str | Path


def _to_jsonable(row: Any) -> Any:
    """Convert a pydantic model to a JSON-serializable dict; pass other objects through."""
    if isinstance(row, BaseModel):
        return row.model_dump(mode="json")
    return row


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-empty line of a JSONL file.

    Reads with ``utf-8-sig`` so a leading UTF-8 BOM is stripped transparently -- Windows tools
    (PowerShell ``Set-Content``, Notepad) prepend one, and ``json.loads`` rejects it -- while plain
    UTF-8 files (the common case) read identically.
    """
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read an entire JSONL file into a list of dicts."""
    return list(iter_jsonl(path))


def write_jsonl(path: PathLike, rows: Iterable[Any], *, mode: str = "w") -> int:
    """Write rows (dicts or pydantic models) to a JSONL file, creating parent dirs.

    Returns the number of rows written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False))
            handle.write("\n")
            written += 1
    return written


def append_jsonl(path: PathLike, rows: Any) -> int:
    """Append one row or an iterable of rows to a JSONL file, creating it if needed."""
    if isinstance(rows, (dict, BaseModel)):
        rows = [rows]
    return write_jsonl(path, rows, mode="a")


def read_jsonl_as(path: PathLike, model_cls: type[ModelT]) -> list[ModelT]:
    """Read a JSONL file and validate each row into the given pydantic model."""
    return [model_cls.model_validate(row) for row in iter_jsonl(path)]
