"""Typed configuration models shared across experiments (models, policies, generation, paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from trigger_audit.io.paths import PathResolver


class ModelConfig(BaseModel):
    """A model/tokenizer plus its context budget.

    The token budget depends on the model; the truncation policy that uses it does not. Keep
    these as data only so the same policy concept applies across models with different windows.
    """

    model_id: str
    tokenizer_id: str | None = None
    chat_template: str | None = None
    # How Layer 3 is rendered: ``"chat"`` applies the model's chat template; ``"base"`` uses the
    # deterministic base-completion concatenation (for models with no chat template, e.g. Pythia).
    # An explicit stated decision rather than silent detection, mirroring the enable_thinking flag.
    chat_format: Literal["chat", "base"] = "chat"
    # Required (no default) on purpose: for thinking-capable models (e.g. Qwen3) a silent
    # default is a determinism hazard, so every model config must declare it explicitly.
    enable_thinking: bool
    max_context_window: int = 8192
    reserved_generation_tokens: int = 512
    reserved_system_tokens: int = 0
    revision: str | None = None
    dtype: str | None = None
    trust_remote_code: bool = False

    def resolved_tokenizer_id(self) -> str:
        """Return the tokenizer id, defaulting to the model id when unset."""
        return self.tokenizer_id or self.model_id

    def input_token_budget(self) -> int:
        """Tokens available for the prompt after reserving room for generation and system text."""
        reserved = self.reserved_generation_tokens + self.reserved_system_tokens
        return max(0, self.max_context_window - reserved)


class PipelinePolicyConfig(BaseModel):
    """A named pipeline policy: which memory and truncation policies to compose, with params.

    ``memory_policy`` and ``truncation_policy`` are registry names resolved at runtime (see
    ``trigger_audit.pipelines``). ``params`` is forwarded to the memory policy constructor.
    """

    name: str
    memory_policy: str = "none"
    truncation_policy: str = "none"
    summarize: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class GenerationConfig(BaseModel):
    """Decoding settings for the optional, secondary behavioral-activation phase."""

    enabled: bool = False
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 0
    activation_token: str = "CANARY_SEEN"


class PathsConfig(BaseModel):
    """Root locations for experiment inputs and outputs; yields a :class:`PathResolver`."""

    root: Path = Path(".")
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")

    def resolver(self) -> PathResolver:
        """Build a PathResolver for the standard data/ and outputs/ layout."""
        return PathResolver(root=self.root, data_dir=self.data_dir, outputs_dir=self.outputs_dir)
