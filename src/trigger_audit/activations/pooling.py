"""Pooling: collapse one layer's per-token activations into a single probe feature vector."""

from __future__ import annotations

import numpy as np

from trigger_audit.schemas.probes import PoolingStrategy


def pool_activations(
    layer_acts: np.ndarray,
    strategy: PoolingStrategy,
    span: tuple[int, int] | None = None,
) -> np.ndarray:
    """Pool a ``(n_tokens, hidden_size)`` activation array into a ``(hidden_size,)`` vector.

    ``TRIGGER_SPAN`` averages over the half-open token span where the trigger landed (the
    span Project 1's survival scoring localizes); it requires ``span`` and validates its
    bounds, because silently pooling an empty or out-of-range span would produce a feature
    vector that looks valid but measures nothing.
    """
    acts = np.asarray(layer_acts, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"expected a 2-D (n_tokens, hidden_size) array, got shape {acts.shape}")
    n_tokens = acts.shape[0]
    if n_tokens == 0:
        raise ValueError("cannot pool an empty activation array")

    if strategy is PoolingStrategy.LAST_TOKEN:
        return acts[-1].copy()
    if strategy is PoolingStrategy.MEAN:
        return acts.mean(axis=0).astype(np.float32)
    if strategy is PoolingStrategy.MAX:
        return acts.max(axis=0).astype(np.float32)
    if strategy is PoolingStrategy.TRIGGER_SPAN:
        if span is None:
            raise ValueError("TRIGGER_SPAN pooling requires a (start, end) token span")
        start, end = span
        if not (0 <= start < end <= n_tokens):
            raise ValueError(
                f"invalid trigger span ({start}, {end}) for {n_tokens} tokens: "
                "need 0 <= start < end <= n_tokens"
            )
        return acts[start:end].mean(axis=0).astype(np.float32)
    raise ValueError(f"Unknown pooling strategy: {strategy!r}")
