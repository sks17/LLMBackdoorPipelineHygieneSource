"""Activation extractors: a common interface over real HF hidden states and a
dependency-free reference extractor for offline tests and CPU-only smoke runs."""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

# Domain-separation tags mixed into the seed sequence so the embedding table and the
# per-layer mixing matrices draw from independent streams even for coinciding indices.
_EMBEDDING_TAG = 0xE4B
_MIXING_TAG = 0x314


class ActivationExtractor(ABC):
    """Common interface for extracting per-layer hidden-state activations from token ids.

    Layer indexing matches Hugging Face ``output_hidden_states``: index 0 is the embedding
    layer and indices 1..``num_layers`` are the transformer block outputs, so a config that
    names layers is portable between the reference and HF backends.
    """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer blocks (valid layer indices are 0..num_layers inclusive)."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Dimensionality of one activation vector."""

    @abstractmethod
    def extract(self, token_ids: Sequence[int], layers: Sequence[int]) -> dict[int, np.ndarray]:
        """Return a ``(n_tokens, hidden_size)`` float32 array for each requested layer."""

    def _validate_request(self, token_ids: Sequence[int], layers: Sequence[int]) -> list[int]:
        """Shared request validation: non-empty tokens, in-range layers; returns unique layers."""
        if len(token_ids) == 0:
            raise ValueError("token_ids must be non-empty")
        unique_layers = list(dict.fromkeys(int(layer) for layer in layers))
        if not unique_layers:
            raise ValueError("at least one layer must be requested")
        for layer in unique_layers:
            if not 0 <= layer <= self.num_layers:
                raise ValueError(
                    f"layer {layer} out of range: valid indices are 0..{self.num_layers} "
                    "(0 is the embedding layer)"
                )
        return unique_layers


class ReferenceActivationExtractor(ActivationExtractor):
    """A deterministic, dependency-free reference activation extractor (numpy only).

    This is NOT a real model and is never used for measurement; it exists so the full
    probe-detection loop is runnable and unit-testable without ``torch``/``transformers``
    and without network access (the activation twin of
    :class:`~trigger_audit.tokenization.tokenizer_adapter.SimpleWhitespaceTokenizerAdapter`).

    Construction: each token id is hashed with the seed into a fixed random embedding row
    (generated lazily, so the vocabulary is unbounded); layer 0 is embeddings plus a small
    sinusoidal position encoding; each subsequent layer applies a fixed random linear mixing
    and a tanh nonlinearity with a residual connection back to the embeddings. The residual
    guarantees the useful property probes rely on: which token ids are present remains
    LINEARLY RECOVERABLE from every layer's activations. Fully deterministic given
    ``(seed, hidden_size, num_layers)``.
    """

    def __init__(self, *, seed: int = 0, hidden_size: int = 32, num_layers: int = 4) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        self._seed = seed
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._embedding_cache: dict[int, np.ndarray] = {}
        self._mixing_cache: dict[int, np.ndarray] = {}

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def _embedding(self, token_id: int) -> np.ndarray:
        """Return the fixed random embedding row for a token id, generating it lazily."""
        row = self._embedding_cache.get(token_id)
        if row is None:
            rng = np.random.default_rng((self._seed, _EMBEDDING_TAG, int(token_id)))
            row = rng.standard_normal(self._hidden_size).astype(np.float32)
            self._embedding_cache[token_id] = row
        return row

    def _mixing_matrix(self, layer: int) -> np.ndarray:
        """Return the fixed random linear mixing for one layer (scaled to keep tanh active)."""
        matrix = self._mixing_cache.get(layer)
        if matrix is None:
            rng = np.random.default_rng((self._seed, _MIXING_TAG, layer))
            matrix = (
                rng.standard_normal((self._hidden_size, self._hidden_size))
                / math.sqrt(self._hidden_size)
            ).astype(np.float32)
            self._mixing_cache[layer] = matrix
        return matrix

    def _position_encoding(self, n_tokens: int) -> np.ndarray:
        """Small additive sinusoidal position encoding (scaled so it never drowns the tokens)."""
        positions: np.ndarray = np.arange(n_tokens, dtype=np.float32)[:, None]
        dims: np.ndarray = np.arange(self._hidden_size, dtype=np.float32)[None, :]
        angles = positions / np.power(10000.0, 2.0 * np.floor(dims / 2.0) / self._hidden_size)
        encoding = np.where(dims.astype(int) % 2 == 0, np.sin(angles), np.cos(angles))
        return (0.1 * encoding).astype(np.float32)

    def extract(self, token_ids: Sequence[int], layers: Sequence[int]) -> dict[int, np.ndarray]:
        requested = self._validate_request(token_ids, layers)
        embeddings = np.stack([self._embedding(int(t)) for t in token_ids])
        hidden = embeddings + self._position_encoding(len(token_ids))

        out: dict[int, np.ndarray] = {}
        if 0 in requested:
            out[0] = hidden.astype(np.float32, copy=True)
        deepest = max(requested)
        for layer in range(1, deepest + 1):
            hidden = np.tanh(hidden @ self._mixing_matrix(layer)) + embeddings
            if layer in requested:
                out[layer] = hidden.astype(np.float32, copy=True)
        return out


class HFActivationExtractor(ActivationExtractor):
    """Wraps a Hugging Face ``AutoModelForCausalLM`` for real hidden-state extraction.

    ``torch`` and ``transformers`` are imported lazily inside ``__init__`` (matching
    :class:`~trigger_audit.tokenization.tokenizer_adapter.HFTokenizerAdapter`) so the base
    package stays importable on CPU-only login nodes without the model-execution stack.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
        device: str = "cpu",
    ) -> None:
        try:
            # Lazy, importlib-based imports: torch is intentionally absent from the base venv
            # (see the pyproject `generate` extra rationale), so a static `import torch` here
            # would break both mypy and the CPU-only install.
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "HFActivationExtractor requires torch and transformers. Install the model "
                "execution stack: `pip install 'trigger-audit[hf,generate]'` plus a torch build "
                "matched to your target (CPU wheel or the cluster's CUDA); see "
                "docs/DEVELOPMENT_SETUP.md."
            ) from exc

        self._torch = torch
        self._model_id = model_id
        self._device = device
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self._model.eval()
        self._model.to(device)
        self._num_layers = int(self._model.config.num_hidden_layers)
        self._hidden_size = int(self._model.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def extract(self, token_ids: Sequence[int], layers: Sequence[int]) -> dict[int, np.ndarray]:
        requested = self._validate_request(token_ids, layers)
        torch = self._torch
        with torch.no_grad():
            input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._device)
            output = self._model(input_ids=input_ids, output_hidden_states=True)
        # hidden_states has num_layers + 1 entries; index 0 is the embedding layer, matching
        # the interface's (and the reference extractor's) layer indexing exactly.
        hidden_states = output.hidden_states
        return {
            layer: hidden_states[layer][0].to(torch.float32).cpu().numpy() for layer in requested
        }


def make_activation_extractor(
    backend: str = "reference",
    *,
    model_id: str | None = None,
    seed: int = 0,
    hidden_size: int = 32,
    num_layers: int = 4,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device: str = "cpu",
) -> ActivationExtractor:
    """Construct an activation extractor for the given backend.

    ``backend='reference'`` returns the dependency-free reference extractor (offline tests
    and smoke runs); ``backend='hf'`` loads a real model (requires the ``hf`` + ``generate``
    extras and a ``model_id``).
    """
    if backend == "reference":
        return ReferenceActivationExtractor(
            seed=seed, hidden_size=hidden_size, num_layers=num_layers
        )
    if backend == "hf":
        if model_id is None:
            raise ValueError("model_id is required for the 'hf' activation backend")
        return HFActivationExtractor(
            model_id, revision=revision, trust_remote_code=trust_remote_code, device=device
        )
    raise ValueError(f"Unknown activation backend: {backend!r}")
