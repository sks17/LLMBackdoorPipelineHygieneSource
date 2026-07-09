"""Tests for YAML config loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trigger_audit.config import (
    ModelConfig,
    PathsConfig,
    load_config,
    load_models,
    load_pipeline_policies,
)

MODELS_YAML = """
models:
  - model_id: qwen3-4b
    tokenizer_id: Qwen/Qwen3-4B
    enable_thinking: false
    max_context_window: 32768
    reserved_generation_tokens: 512
  - model_id: pythia-1b
    enable_thinking: false
    max_context_window: 2048
"""

POLICIES_YAML = """
policies:
  - name: none
    memory_policy: none
    truncation_policy: none
  - name: keep_recent_messages
    memory_policy: keep_recent_messages
    params: { keep_recent_turns: 2 }
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_models_keyed_by_id(tmp_path):
    models = load_models(_write(tmp_path, "models.yaml", MODELS_YAML))
    assert set(models) == {"qwen3-4b", "pythia-1b"}
    assert models["qwen3-4b"].resolved_tokenizer_id() == "Qwen/Qwen3-4B"
    # tokenizer_id defaults to model_id when unset.
    assert models["pythia-1b"].resolved_tokenizer_id() == "pythia-1b"


def test_model_input_budget():
    model = ModelConfig(
        model_id="m", enable_thinking=False, max_context_window=1000, reserved_generation_tokens=200
    )
    assert model.input_token_budget() == 800


def test_load_pipeline_policies(tmp_path):
    policies = load_pipeline_policies(_write(tmp_path, "policies.yaml", POLICIES_YAML))
    assert set(policies) == {"none", "keep_recent_messages"}
    assert policies["keep_recent_messages"].params == {"keep_recent_turns": 2}


def test_load_paths_config(tmp_path):
    path = _write(tmp_path, "paths.yaml", "root: .\ndata_dir: data\noutputs_dir: outputs\n")
    resolver = load_config(path, PathsConfig).resolver()
    assert resolver.manifest_path().name == "trial_manifest.jsonl"


def test_invalid_model_raises(tmp_path):
    bad = _write(tmp_path, "bad.yaml", "models:\n  - tokenizer_id: x\n")  # missing model_id
    with pytest.raises(ValidationError):
        load_models(bad)
