"""Tests for the new probe-detection config fields, validators, and YAML parsing (Project 2).

Covers the device/revision/trust_remote_code threading knobs, depth-fraction slicing config
and its validation, the synthetic_mode enum, the twins-builder knobs, store-reuse and
predictions-output flags, and the E2.x generalization holdout field -- including that every
new field is defaulted so an existing offline config parses unchanged.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from trigger_audit.config.loader import load_config
from trigger_audit.experiments.probe_detection.config import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.generalization import GeneralizationSpec


def _minimal(**overrides: object) -> ProbeDetectionExperimentConfig:
    payload: dict[str, object] = {"experiment_id": "probe_cfg_test"}
    payload.update(overrides)
    return ProbeDetectionExperimentConfig.model_validate(payload)


def test_new_fields_have_backward_compatible_defaults() -> None:
    cfg = _minimal()
    assert cfg.device == "cpu"
    assert cfg.revision is None
    assert cfg.trust_remote_code is False
    assert cfg.layer_depth_fractions is None
    assert cfg.synthetic_mode == "simple"
    assert cfg.partial_survival_fraction == 0.25
    assert cfg.synthetic_n_bases == 40
    assert cfg.predictions_out is None
    assert cfg.reuse_store is False
    assert cfg.generalization is None


def test_extractor_knobs_round_trip() -> None:
    cfg = _minimal(device="cuda:0", revision="main", trust_remote_code=True)
    assert cfg.device == "cuda:0"
    assert cfg.revision == "main"
    assert cfg.trust_remote_code is True


def test_layer_depth_fractions_accepts_valid_and_none() -> None:
    assert _minimal(layer_depth_fractions=None).layer_depth_fractions is None
    cfg = _minimal(layer_depth_fractions=[0.0, 0.5, 1.0])
    assert cfg.layer_depth_fractions == [0.0, 0.5, 1.0]


def test_layer_depth_fractions_rejects_empty_list() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        _minimal(layer_depth_fractions=[])


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_layer_depth_fractions_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValidationError, match=r"\[0.0, 1.0\]"):
        _minimal(layer_depth_fractions=[0.5, bad])


def test_synthetic_mode_enum() -> None:
    assert _minimal(synthetic_mode="twins").synthetic_mode == "twins"
    assert _minimal(synthetic_mode="simple").synthetic_mode == "simple"
    with pytest.raises(ValidationError):
        _minimal(synthetic_mode="quadruplets")


def test_twins_and_reuse_and_predictions_fields() -> None:
    cfg = _minimal(
        synthetic_n_bases=12,
        partial_survival_fraction=0.4,
        reuse_store=True,
        predictions_out="outputs/preds.jsonl",
    )
    assert cfg.synthetic_n_bases == 12
    assert cfg.partial_survival_fraction == 0.4
    assert cfg.reuse_store is True
    assert cfg.predictions_out == Path("outputs/preds.jsonl")


def test_generalization_field_parses_spec() -> None:
    cfg = _minimal(
        generalization={
            "kind": "policy",
            "train_policies": ["cot", "plain"],
            "test_policies": ["tool"],
        }
    )
    assert isinstance(cfg.generalization, GeneralizationSpec)
    assert cfg.generalization.kind == "policy"
    assert cfg.generalization.test_policies == ["tool"]


def test_generalization_field_rejects_overlapping_sides() -> None:
    # The GeneralizationSpec validator runs when the field is parsed.
    with pytest.raises(ValidationError, match="disjoint"):
        _minimal(
            generalization={
                "kind": "policy",
                "train_policies": ["cot", "tool"],
                "test_policies": ["tool"],
            }
        )


def test_full_config_parses_from_yaml(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        experiment_id: probe_yaml_test
        model_id: reference-model
        device: cpu
        revision: null
        trust_remote_code: false
        layer_depth_fractions: [0.5, 0.75, 1.0]
        synthetic_mode: twins
        synthetic_n_bases: 16
        partial_survival_fraction: 0.3
        reuse_store: true
        predictions_out: outputs/probe/preds.jsonl
        generalization:
          kind: context_length
          train_context_max: 100
          test_context_min: 200
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    cfg = load_config(path, ProbeDetectionExperimentConfig)
    assert cfg.layer_depth_fractions == [0.5, 0.75, 1.0]
    assert cfg.synthetic_mode == "twins"
    assert cfg.synthetic_n_bases == 16
    assert cfg.partial_survival_fraction == 0.3
    assert cfg.reuse_store is True
    assert cfg.predictions_out == Path("outputs/probe/preds.jsonl")
    assert cfg.generalization is not None
    assert cfg.generalization.kind == "context_length"
    assert cfg.generalization.train_context_max == 100
    assert cfg.generalization.test_context_min == 200


def test_example_config_still_parses(tmp_path: Path) -> None:
    # The shipped example config must keep loading with all new fields defaulted.
    cfg = load_config("configs/probe_detection.example.yaml", ProbeDetectionExperimentConfig)
    assert cfg.synthetic_mode == "simple"
    assert cfg.reuse_store is False
    assert cfg.layer_depth_fractions is None
    assert cfg.generalization is None
