"""Deterministic, reproducible identifiers for trials and other experiment objects."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_SEP = "\x1f"  # unit separator: unlikely to appear in any field value


def stable_id(*parts: Any, prefix: str = "", length: int = 16) -> str:
    """Return a deterministic short id derived from the given parts via blake2b.

    The id is order-sensitive and stable across processes and platforms, so the same tuple
    of inputs always yields the same id (important for reproducible manifests and resumable runs).
    """
    digest = hashlib.blake2b(
        _SEP.join(str(p) for p in parts).encode("utf-8"), digest_size=32
    ).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest


def make_trial_id(
    *,
    base_id: str,
    trigger_id: str,
    trigger_position: str,
    model_id: str,
    context_length: int,
    pipeline_policy: str,
    chat_template: str | None,
    seed: int,
) -> str:
    """Build a stable trial id from the full experimental tuple."""
    return stable_id(
        base_id,
        trigger_id,
        trigger_position,
        model_id,
        context_length,
        pipeline_policy,
        chat_template or "",
        seed,
        prefix="t_",
    )


def is_valid_id(value: str) -> bool:
    """Return True if value is a non-empty id using only [A-Za-z0-9_.-]."""
    return bool(value) and _ID_RE.match(value) is not None


def make_grid_trial_id(
    base_id: str,
    trigger_id: str,
    trigger_position: str,
    policy_id: str,
    model_id: str,
    *,
    context_length: int = 0,
    trigger_present: bool = True,
) -> str:
    """Return the stable, order-independent id for a manifest grid row.

    Deterministic across re-expansions: the same combination of grid coordinates always
    yields the same id, so a manifest can be re-expanded without perturbing trial ids. The
    fan-out coordinates ``context_length`` and ``trigger_present`` are appended to the hash key
    only when they differ from the legacy defaults (``0`` / ``True``), so a counterfactual twin
    and each length cell get distinct ids while pre-fan-out grids keep the ids they already had.
    """
    key = f"{base_id}|{trigger_id}|{trigger_position}|{policy_id}|{model_id}"
    if context_length:
        key += f"|ctx{context_length}"
    if not trigger_present:
        key += "|no_trigger"
    return "trial_" + hashlib.sha256(key.encode()).hexdigest()[:12]
