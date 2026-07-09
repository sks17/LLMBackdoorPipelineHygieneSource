"""Load and validate YAML configuration into typed pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from trigger_audit.config.settings import ModelConfig, PipelinePolicyConfig

ModelT = TypeVar("ModelT", bound=BaseModel)

PathLike = str | Path


def load_yaml(path: PathLike) -> Any:
    """Parse a YAML file into native Python objects."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: PathLike, model_cls: type[ModelT]) -> ModelT:
    """Load a YAML file and validate it into a single pydantic model."""
    return model_cls.model_validate(load_yaml(path))


def _unwrap_list(data: Any, key: str) -> list[dict[str, Any]]:
    """Accept either a bare list or a mapping ``{key: [...]}`` and return the list."""
    if isinstance(data, dict) and key in data:
        data = data[key]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list (optionally under '{key}'), got {type(data).__name__}")
    return data


def load_models(path: PathLike) -> dict[str, ModelConfig]:
    """Load a models YAML (a list under ``models:`` or a bare list) keyed by model id."""
    rows = _unwrap_list(load_yaml(path), "models")
    models = [ModelConfig.model_validate(row) for row in rows]
    return {model.model_id: model for model in models}


def load_pipeline_policies(path: PathLike) -> dict[str, PipelinePolicyConfig]:
    """Load a pipeline-policies YAML (a list under ``policies:`` or a bare list) keyed by name."""
    rows = _unwrap_list(load_yaml(path), "policies")
    policies = [PipelinePolicyConfig.model_validate(row) for row in rows]
    return {policy.name: policy for policy in policies}
