"""Provenance + allowlist registry for backdoored checkpoints (pure, offline-testable).

Safety boundary (see ``docs/PROJECT2_BACKDOOR_SAFETY.md``): a "backdoored" checkpoint is a
Tier-3 *measurement target* whose only installed behavior is a **harmless canary marker** (the
model emits a benign string such as ``CANARY_SEEN`` when the trigger is present). This module
constructs no harmful behavior; it enforces, in code, that

* the trigger payload is benign (``CanaryTriggerSpec.benign is True`` plus single-line,
  non-empty ``trigger_text`` / ``canary_marker`` validators),
* the attack family is one of the known DPA-taxonomy families, and
* a checkpoint is loaded only after it is registered with full provenance **and** explicitly
  allowlisted (``require_allowlisted`` — the gate the loader calls first).

Everything here is pure-python / stdlib (``hashlib`` for integrity) and fully testable offline
with no torch, no transformers, and no network.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from trigger_audit.config.loader import load_yaml
from trigger_audit.schemas.triggers import TriggerPosition

PathLike = str | Path

# The DPA (data-poisoning attack) taxonomy families recorded in ``third_party/README.md``.
# Canonical spelling is the value; lookup is case-insensitive so YAML authors need not match case.
_ATTACK_FAMILIES: dict[str, str] = {
    "badnet": "BadNet",
    "vpi": "VPI",
    "mtba": "MTBA",
    "ctba": "CTBA",
    "sleeper": "Sleeper",
}


def normalize_attack_family(value: str) -> str:
    """Return the canonical attack-family spelling, or raise ``ValueError`` for an unknown one.

    Restricting to the known taxonomy keeps provenance honest: an unrecognized family is a data
    error, not a silently accepted free-text label.
    """
    key = str(value).strip().lower()
    try:
        return _ATTACK_FAMILIES[key]
    except KeyError:
        raise ValueError(
            f"unknown attack_family {value!r}; expected one of "
            f"{sorted(set(_ATTACK_FAMILIES.values()))}"
        ) from None


class CanaryTriggerSpec(BaseModel):
    """A harmless canary trigger + the benign marker it must (only) produce.

    ``benign`` is a structural assertion that the payload is a harmless marker. It is typed
    ``Literal[True]`` *and* re-checked by a ``mode="before"`` validator so a non-benign spec is
    refused with a clear message before any other coercion happens.
    """

    trigger_text: str
    canary_marker: str
    trigger_position: TriggerPosition
    benign: Literal[True] = True

    @model_validator(mode="before")
    @classmethod
    def _enforce_benign_marker(cls, data: object) -> object:
        """Reject non-benign payloads and non-single-line / empty trigger or marker strings."""
        if not isinstance(data, dict):
            return data
        if data.get("benign", True) is not True:
            raise ValueError(
                "CanaryTriggerSpec.benign must be True: this harness only handles a harmless "
                "canary marker; a non-benign payload is refused (canary != backdoor)."
            )
        for field_name in ("trigger_text", "canary_marker"):
            value = data.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if "\n" in value or "\r" in value:
                raise ValueError(f"{field_name} must be a single line (no newlines)")
        return data


class BackdoorCheckpoint(BaseModel):
    """Full provenance for one registered backdoored checkpoint (a measurement target).

    ``sha256`` maps a file path (weights or LoRA adapter file) to its recorded digest, checked by
    :meth:`BackdoorRegistry.verify_hashes`. ``allowlisted`` defaults to ``False``: a freshly
    registered checkpoint is inert until license review + hashes + ASR verification are complete.
    """

    checkpoint_id: str
    base_model_id: str
    revision: str | None = None
    adapter_path: str | None = None
    source_url: str
    license: str
    commit: str | None = None
    trigger: CanaryTriggerSpec
    attack_family: str
    sha256: dict[str, str] = Field(default_factory=dict)
    allowlisted: bool = False
    notes: str = ""

    @field_validator("attack_family")
    @classmethod
    def _normalize_family(cls, value: str) -> str:
        return normalize_attack_family(value)


class BackdoorRegistry:
    """A keyed collection of :class:`BackdoorCheckpoint`s with the allowlist + integrity gates.

    The registry is the single choke point through which a backdoored checkpoint reaches the
    loader: :meth:`require_allowlisted` is called *first* in ``SafeBackdoorModel.__init__`` before
    any import or file read, so an unregistered or non-allowlisted id can never be loaded.
    """

    def __init__(self, checkpoints: Iterable[BackdoorCheckpoint]) -> None:
        self._by_id: dict[str, BackdoorCheckpoint] = {}
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id in self._by_id:
                raise ValueError(f"duplicate checkpoint_id: {checkpoint.checkpoint_id!r}")
            self._by_id[checkpoint.checkpoint_id] = checkpoint

    @classmethod
    def from_yaml(cls, path: PathLike) -> BackdoorRegistry:
        """Load a registry from a YAML list (bare, or under a ``checkpoints:`` key)."""
        data = load_yaml(path)
        if isinstance(data, dict) and "checkpoints" in data:
            data = data["checkpoints"]
        if not isinstance(data, list):
            raise ValueError(
                "backdoor registry YAML must be a list (optionally under a 'checkpoints:' key)"
            )
        return cls(BackdoorCheckpoint.model_validate(row) for row in data)

    @property
    def ids(self) -> list[str]:
        """Registered checkpoint ids, in insertion order."""
        return list(self._by_id)

    def __contains__(self, checkpoint_id: object) -> bool:
        return checkpoint_id in self._by_id

    def get(self, checkpoint_id: str) -> BackdoorCheckpoint:
        """Return the checkpoint for ``checkpoint_id`` or raise ``ValueError`` if unknown."""
        try:
            return self._by_id[checkpoint_id]
        except KeyError:
            raise ValueError(
                f"unknown checkpoint_id {checkpoint_id!r}; registered: {self.ids}"
            ) from None

    def require_allowlisted(self, checkpoint_id: str) -> BackdoorCheckpoint:
        """Return the checkpoint only if it is registered **and** allowlisted.

        Raises ``ValueError`` for an unknown id and ``PermissionError`` for a registered but
        non-allowlisted id. This is the hard gate: refuse to load anything that has not cleared
        license review + hash + ASR verification and been explicitly allowlisted.
        """
        checkpoint = self.get(checkpoint_id)
        if not checkpoint.allowlisted:
            raise PermissionError(
                f"checkpoint {checkpoint_id!r} is registered but NOT allowlisted; refusing to "
                "load. Complete license review, record sha256, and verify benign-canary ASR, "
                "then set allowlisted: true."
            )
        return checkpoint

    def verify_hashes(self, checkpoint_id: str, *, strict: bool = False) -> dict[str, str]:
        """Compare recorded sha256 digests against files on disk; return a per-path status map.

        Each path maps to ``"ok"`` (match) or ``"missing"`` (file absent). A digest mismatch is
        always a hard ``ValueError``. A missing file is a soft warning by default; with
        ``strict=True`` it raises ``FileNotFoundError`` (the pre-load gate for a real run).
        """
        checkpoint = self.get(checkpoint_id)
        status: dict[str, str] = {}
        for rel_path, expected in checkpoint.sha256.items():
            path = Path(rel_path)
            if not path.exists():
                if strict:
                    raise FileNotFoundError(
                        f"sha256-registered file not found for {checkpoint_id!r}: {rel_path}"
                    )
                warnings.warn(
                    f"sha256-registered file missing for {checkpoint_id!r}: {rel_path} "
                    "(soft warning; pass strict=True to make this an error)",
                    stacklevel=2,
                )
                status[rel_path] = "missing"
                continue
            actual = _sha256_file(path)
            if actual.lower() != str(expected).lower():
                raise ValueError(
                    f"sha256 mismatch for {rel_path} of {checkpoint_id!r}: "
                    f"expected {expected}, computed {actual}"
                )
            status[rel_path] = "ok"
        return status


def _sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """Stream a file through sha256 (chunked so large weight files are not read into memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
