"""Load per-trial probe predictions and aggregate results into tidy analysis objects.

Component G (probe inference) reads two artifacts the runner emits:

- a JSONL of :class:`~trigger_audit.schemas.probes.ProbePrediction` rows -- one honest
  per-trial fact per TEST example -- flattened here into a tidy frame whose ``fired__<target>``
  and ``layer__<idx>`` columns are unpacked from the prediction's ``fired`` / ``layer_scores``
  dicts, so every cluster-bootstrap and decomposition downstream is one column selection away;
- a JSONL of :class:`~trigger_audit.schemas.probes.ProbeEvaluationResult` rows carrying the
  calibrated per-layer metrics and the ``metadata`` block (``num_layers``, ``resolved_layers``,
  ``layer_depth_fractions``) that lets every layer-keyed output report by *depth fraction* rather
  than raw index, so probe sites are comparable across model sizes.

Both readers accept a single file or a directory of shards, mirroring
``analysis/loading.py::load_results``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trigger_audit.activations.slicing import depth_fraction_of_layer
from trigger_audit.io.jsonl import read_jsonl_as
from trigger_audit.schemas.probes import ProbeEvaluationResult, ProbePrediction

PathLike = str | Path

# The tidy per-trial columns every probe_stats primitive reads (before the wide fired__*/layer__*
# columns). Kept explicit so the frame's shape is a documented contract, not an accident of order.
CORE_COLUMNS = [
    "trial_id",
    "base_id",
    "label",
    "trigger_inserted",
    "delivered",
    "clean_negative",
    "split",
    "aggregated_score",
]

_BOOL_COLUMNS = ("label", "trigger_inserted", "delivered", "clean_negative")


def _prediction_files(path: PathLike) -> list[Path]:
    """One file, or every ``*.jsonl`` shard under a directory (sorted for determinism)."""
    p = Path(path)
    return sorted(p.glob("*.jsonl")) if p.is_dir() else [p]


def load_predictions(path: PathLike) -> pd.DataFrame:
    """Read ``ProbePrediction`` JSONL (file or directory) into a tidy per-trial frame.

    Columns: the :data:`CORE_COLUMNS` per-trial facts, then a ``fired__<target>`` boolean column
    per calibrated target FPR (unpacked from ``ProbePrediction.fired``, keyed by the target's
    string form, e.g. ``fired__0.01``) and a ``layer__<idx>`` float column per configured layer
    (unpacked from ``layer_scores``). One row per TEST example; no aggregation across trials.
    """
    rows: list[dict[str, object]] = []
    for file in _prediction_files(path):
        for pred in read_jsonl_as(file, ProbePrediction):
            row: dict[str, object] = {
                "trial_id": pred.trial_id,
                "base_id": pred.base_id,
                "label": bool(pred.label),
                "trigger_inserted": bool(pred.trigger_inserted),
                "delivered": bool(pred.delivered),
                "clean_negative": bool(pred.clean_negative),
                "split": pred.split.value,
                "aggregated_score": float(pred.aggregated_score),
            }
            for target, fired in pred.fired.items():
                row[f"fired__{target}"] = bool(fired)
            for layer, score in pred.layer_scores.items():
                row[f"layer__{layer}"] = float(score)
            rows.append(row)
    if not rows:
        raise ValueError(f"no probe predictions found under {path}")

    df = pd.DataFrame(rows)
    for col in _BOOL_COLUMNS:
        df[col] = df[col].astype(bool)
    for col in df.columns:
        if col.startswith("fired__"):
            df[col] = df[col].fillna(False).astype(bool)
    fired_cols = sorted(c for c in df.columns if c.startswith("fired__"))
    layer_cols = sorted(
        (c for c in df.columns if c.startswith("layer__")),
        key=lambda c: int(c.removeprefix("layer__")),
    )
    return df[CORE_COLUMNS + fired_cols + layer_cols]


def load_probe_results(path: PathLike) -> list[ProbeEvaluationResult]:
    """Read ``ProbeEvaluationResult`` JSONL rows (a single file or a directory of shards)."""
    results: list[ProbeEvaluationResult] = []
    for file in _prediction_files(path):
        results.extend(read_jsonl_as(file, ProbeEvaluationResult))
    if not results:
        raise ValueError(f"no probe evaluation results found under {path}")
    return results


def layer_depth_fractions(result: ProbeEvaluationResult) -> dict[int, float]:
    """Map each probed layer index to its depth fraction (portable probe-site coordinate).

    Prefers the model depth (``metadata["num_layers"]``) recorded when the layers were resolved
    from depth fractions, giving ``layer / num_layers`` per the Hugging Face indexing convention
    (0 = embeddings, ``num_layers`` = last block). Falls back to a 1:1 ``resolved_layers`` /
    ``layer_depth_fractions`` pairing, and finally to normalizing by the largest probed index so
    a run that never recorded its depth still reports a monotone fraction.
    """
    meta = result.metadata
    num_layers = meta.get("num_layers")
    if isinstance(num_layers, (int, float)) and int(num_layers) > 0:
        depth = int(num_layers)
        return {int(layer): depth_fraction_of_layer(int(layer), depth) for layer in result.layers}

    resolved = meta.get("resolved_layers")
    fractions = meta.get("layer_depth_fractions")
    if (
        isinstance(resolved, list)
        and isinstance(fractions, list)
        and len(resolved) == len(fractions)
        and resolved
    ):
        return {int(layer): float(frac) for layer, frac in zip(resolved, fractions, strict=True)}

    if not result.layers:
        return {}
    denom = max(result.layers)
    denom = denom if denom > 0 else 1
    return {int(layer): float(layer) / denom for layer in result.layers}


def depth_fraction_for(result: ProbeEvaluationResult, layer_index: int) -> float | None:
    """Depth fraction of one probed layer, or ``None`` when the layer was not probed."""
    return layer_depth_fractions(result).get(int(layer_index))
