"""Depth-fraction model slicing.

Probe layers are reported by *depth fraction* rather than raw index so results are
portable across model sizes: the same relative depth in a small and a large model is a
comparable probe site, whereas the same raw index is not. To compare across sizes you
select the same relative depth, not the same absolute block.

Layer indexing matches :class:`~trigger_audit.activations.extractor.ActivationExtractor`
(and Hugging Face ``output_hidden_states``): index 0 is the embedding layer and indices
1..``num_layers`` are the transformer block outputs, so ``num_layers`` valid depth
positions are 0..``num_layers`` inclusive. Pure numpy/stdlib; no torch/transformers.
"""

from __future__ import annotations

from collections.abc import Sequence

# The pre-registered informative band (PROJECT2_RESOURCES.md:86-88): the relative depths
# at which the probe carries signal. Resolved to concrete layers per model via
# ``default_band_layers`` so the same band is reused across model sizes.
INFORMATIVE_BAND_FRACTIONS: tuple[float, ...] = (0.5, 0.66, 0.75, 0.89)


def resolve_layers_from_fractions(fractions: Sequence[float], num_layers: int) -> list[int]:
    """Resolve depth fractions to concrete layer indices for a model with ``num_layers`` blocks.

    Each fraction ``f`` maps to ``round(f * num_layers)`` clamped to ``[0, num_layers]``
    (0 is the embedding layer, ``num_layers`` is the last block output). The resulting
    indices are de-duplicated and returned sorted ascending, so several nearby fractions
    that round to the same block collapse to a single probe site.

    Raises:
        ValueError: if ``fractions`` is empty, ``num_layers`` is not positive, or any
            fraction lies outside ``[0.0, 1.0]``.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    fractions = list(fractions)
    if not fractions:
        raise ValueError("fractions must be non-empty")

    layers: set[int] = set()
    for f in fractions:
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"depth fraction {f} out of range: must be in [0.0, 1.0]")
        layer = max(0, min(num_layers, round(f * num_layers)))
        layers.add(layer)
    return sorted(layers)


def depth_fraction_of_layer(layer: int, num_layers: int) -> float:
    """Report a resolved layer index back as its depth fraction (the inverse mapping).

    Returns ``layer / num_layers``, so 0 is the embedding layer (0.0) and ``num_layers``
    is the last block (1.0).

    Raises:
        ValueError: if ``num_layers`` is not positive or ``layer`` is outside
            ``[0, num_layers]``.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if not 0 <= layer <= num_layers:
        raise ValueError(
            f"layer {layer} out of range: valid indices are 0..{num_layers} "
            "(0 is the embedding layer)"
        )
    return layer / num_layers


def default_band_layers(num_layers: int) -> list[int]:
    """Resolve the pre-registered informative band to concrete layers for ``num_layers``."""
    return resolve_layers_from_fractions(INFORMATIVE_BAND_FRACTIONS, num_layers)
