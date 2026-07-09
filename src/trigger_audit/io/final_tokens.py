"""Sidecar persistence for the final model-visible token ids of a ``run-survival-shard`` run.

The survival audit's ``SurvivalResult`` records whether and where a trigger survived, but
historically discarded the final model-visible token ids themselves once scoring finished (see
``schemas/results.py::SurvivalResult.final_token_ids``, which stays ``None`` unless a caller opts
in). This module is the canonical producer/consumer of the ``final_tokens.jsonl`` sidecar: one JSON
object per line, ``{"trial_id": ..., "final_token_ids": [...]}``. Any downstream consumer that needs
the exact final token ids for a trial can join on this sidecar; use :func:`read_final_tokens` to
load it instead of re-deriving the ``trial_id -> final_token_ids`` dict comprehension.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from trigger_audit.io.jsonl import iter_jsonl, write_jsonl


class FinalTokensRow(BaseModel):
    """One sidecar row: a trial's final model-visible token ids."""

    trial_id: str
    final_token_ids: list[int]


def write_final_tokens(path: str | Path, rows: Iterable[tuple[str, Sequence[int] | None]]) -> int:
    """Write ``{trial_id, final_token_ids}`` rows to a ``final_tokens.jsonl`` sidecar.

    Returns the number of rows written. A row whose ``final_token_ids`` is ``None`` is skipped (no
    final tokens exist for that trial, e.g. a template-incompatible trial that never rendered);
    an empty list is written as-is when a caller chooses to include one. Delegates to
    :func:`trigger_audit.io.jsonl.write_jsonl` for the actual write, which creates parent
    directories and writes one JSON object per line.
    """
    payload = [
        FinalTokensRow(trial_id=trial_id, final_token_ids=list(ids))
        for trial_id, ids in rows
        if ids is not None
    ]
    return write_jsonl(path, payload)


def read_final_tokens(path: str | Path) -> dict[str, list[int]]:
    """Read a ``final_tokens.jsonl`` sidecar into a ``{trial_id: final_token_ids}`` map.

    Any consumer that needs the exact final token ids for a trial can share this loader instead of
    re-parsing the sidecar by hand.
    """
    return {
        str(row["trial_id"]): [int(t) for t in row["final_token_ids"]] for row in iter_jsonl(path)
    }
