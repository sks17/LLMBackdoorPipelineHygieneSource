"""Activation store: persist pooled feature matrices per (experiment, model, layer).

Each layer's feature matrix and its trial ids live together in a single ``.npz`` archive, so
every feature row stays attributable to its trial (and joinable back to Project 1's survival
results) with no separate sidecar that could desync. Writes are atomic -- one archive is
built under a writer-unique temp name and then ``os.replace``d into place -- so a killed
shard never leaves a half-written or row-misattributed matrix that silently loads.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PathLike = str | Path

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.\-]")


def _safe_component(value: str) -> str:
    """Sanitize an id into a filesystem-safe path component (HF model ids contain '/').

    Two safeguards beyond the character substitution: components that sanitize to ``""``,
    ``"."`` or ``".."`` are rejected (they would let an id escape or alias the store root),
    and whenever sanitization altered the id a short stable hash of the ORIGINAL id is
    appended -- otherwise distinct ids that collapse to the same safe string (``"org/model"``
    and ``"org_model"``) would silently share a directory and overwrite each other's files.
    """
    safe = _UNSAFE_COMPONENT.sub("_", value)
    if safe in ("", ".", ".."):
        raise ValueError(
            f"id {value!r} sanitizes to the unusable path component {safe!r}; "
            "it cannot be stored safely"
        )
    if safe != value:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        return f"{safe}-{digest}"
    return safe


class ActivationStore:
    """Persists pooled feature matrices keyed by (experiment_id, model_id, layer)."""

    def __init__(self, root: PathLike) -> None:
        self._root = Path(root)

    def layer_dir(self, experiment_id: str, model_id: str) -> Path:
        """Return the directory holding one (experiment, model)'s per-layer files."""
        return self._root / _safe_component(experiment_id) / _safe_component(model_id)

    def features_path(
        self, experiment_id: str, model_id: str, layer: int, pooling: str = "mean"
    ) -> Path:
        """Return the ``.npz`` path for one (layer, pooling)'s features and trial ids.

        ``pooling`` is part of the key so different poolings of the same
        (experiment, model, layer) are distinct entries and never overwrite each other
        during a pooling sweep.
        """
        return self.layer_dir(experiment_id, model_id) / f"layer_{layer:03d}_{pooling}.npz"

    def save(
        self,
        experiment_id: str,
        model_id: str,
        layer: int,
        features: np.ndarray,
        trial_ids: Sequence[str],
        pooling: str = "mean",
        *,
        extractor_backend: str = "reference",
    ) -> Path:
        """Write one (layer, pooling)'s matrix, trial ids, and producer manifest atomically.

        The producer manifest (``extractor_backend`` + ``model_id``) is written into the same
        archive so :meth:`load_reusable` can refuse to reuse a matrix produced by a different
        backend (the backend is NOT part of the path, so a reference-backend and an hf-backend
        run with the same ids would otherwise collide on one file).
        """
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
        if matrix.shape[0] != len(trial_ids):
            raise ValueError(
                f"feature rows ({matrix.shape[0]}) and trial ids ({len(trial_ids)}) disagree"
            )

        features_path = self.features_path(experiment_id, model_id, layer, pooling)
        features_path.parent.mkdir(parents=True, exist_ok=True)

        # Truly atomic: the matrix, its trial ids, and the producer manifest live in ONE
        # archive, so a single os.replace (atomic on POSIX and NTFS) is the unit of
        # persistence -- there is no window where new features sit beside a stale manifest,
        # and no row-count-matching torn state can attribute rows to the wrong trial ids. The
        # temp name is writer-unique (pid + random suffix) so concurrent shard writers never
        # collide on a shared temp path (a deterministic ``.tmp`` name raced to a Windows
        # PermissionError).
        tmp_name = f"{features_path.name}.tmp.{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp_path = features_path.with_name(tmp_name)
        try:
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    features=matrix,
                    trial_ids=np.asarray(list(trial_ids), dtype="U"),
                    extractor_backend=np.asarray(extractor_backend, dtype="U"),
                    model_id=np.asarray(model_id, dtype="U"),
                    pooling=np.asarray(pooling, dtype="U"),
                )
            os.replace(tmp_path, features_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return features_path

    def load(
        self, experiment_id: str, model_id: str, layer: int, pooling: str = "mean"
    ) -> tuple[np.ndarray, list[str]]:
        """Load one (layer, pooling)'s feature matrix and its trial ids, validating agreement."""
        features_path = self.features_path(experiment_id, model_id, layer, pooling)
        with np.load(features_path) as archive:
            matrix = np.asarray(archive["features"], dtype=np.float32)
            trial_ids = [str(trial_id) for trial_id in archive["trial_ids"]]
        # Defence in depth: the two arrays are written together so they cannot desync through
        # normal writes, but a hand-edited or truncated archive still fails loudly here.
        if matrix.shape[0] != len(trial_ids):
            raise ValueError(
                f"corrupt activation store entry {features_path}: matrix has "
                f"{matrix.shape[0]} row(s) but the archive lists {len(trial_ids)} trial id(s)"
            )
        return matrix, trial_ids

    def load_reusable(
        self,
        experiment_id: str,
        model_id: str,
        layer: int,
        pooling: str,
        trial_ids: Sequence[str],
        *,
        extractor_backend: str,
    ) -> np.ndarray | None:
        """Return a stored matrix only if it is safe to reuse, else ``None`` (re-extract).

        Reuse is granted only when every producer fact matches: the entry exists, its trial-id
        vector equals ``trial_ids`` **in order**, and its recorded ``extractor_backend`` and
        ``model_id`` match. Any mismatch (or a missing/legacy/corrupt entry) yields ``None`` so
        the caller re-extracts -- correctness before the extract-once optimization. Never
        raises on a mismatch; a stored matrix from an incompatible producer is simply refused.
        """
        features_path = self.features_path(experiment_id, model_id, layer, pooling)
        if not features_path.exists():
            return None
        with np.load(features_path) as archive:
            matrix = np.asarray(archive["features"], dtype=np.float32)
            stored_trial_ids = [str(trial_id) for trial_id in archive["trial_ids"]]
            stored_backend = (
                str(archive["extractor_backend"]) if "extractor_backend" in archive else None
            )
            stored_model_id = str(archive["model_id"]) if "model_id" in archive else None
        if matrix.shape[0] != len(stored_trial_ids):
            return None
        if stored_trial_ids != list(trial_ids):
            return None
        if stored_backend != extractor_backend or stored_model_id != model_id:
            return None
        return matrix
