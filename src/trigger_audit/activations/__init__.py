"""Per-layer hidden-state activation extraction, pooling, and persistence (Project 2).

Mirrors the tokenizer-adapter twin pattern: a production Hugging Face extractor and a
deterministic, dependency-free reference extractor share one interface, so the whole
probe-detection experiment is runnable and unit-testable fully offline.
"""

from trigger_audit.activations.extractor import (
    ActivationExtractor,
    HFActivationExtractor,
    ReferenceActivationExtractor,
    make_activation_extractor,
)
from trigger_audit.activations.pooling import pool_activations
from trigger_audit.activations.slicing import (
    INFORMATIVE_BAND_FRACTIONS,
    default_band_layers,
    depth_fraction_of_layer,
    resolve_layers_from_fractions,
)
from trigger_audit.activations.store import ActivationStore

__all__ = [
    "INFORMATIVE_BAND_FRACTIONS",
    "ActivationExtractor",
    "ActivationStore",
    "HFActivationExtractor",
    "ReferenceActivationExtractor",
    "default_band_layers",
    "depth_fraction_of_layer",
    "make_activation_extractor",
    "pool_activations",
    "resolve_layers_from_fractions",
]
