"""Offline tests for the backdoored-checkpoint registry: allowlist gate, benign invariant, hashes.

Every safety rule is exercised as a *refusal path* (PermissionError / ValueError / ValidationError)
so the boundary is proven enforced in code, not merely documented. No torch, no network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from trigger_audit.models.backdoor_registry import (
    BackdoorCheckpoint,
    BackdoorRegistry,
    CanaryTriggerSpec,
    normalize_attack_family,
)
from trigger_audit.schemas.triggers import TriggerPosition

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = REPO_ROOT / "configs" / "backdoor_models.example.yaml"


def _benign_trigger(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "trigger_text": "tq84z canary probe",
        "canary_marker": "CANARY_SEEN",
        "trigger_position": "prefix",
        "benign": True,
    }
    data.update(overrides)
    return data


def _checkpoint(
    checkpoint_id: str, *, allowlisted: bool, **overrides: object
) -> BackdoorCheckpoint:
    data: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "base_model_id": "placeholder/qwen2_5-0_5b",
        "source_url": "https://example.invalid/x",
        "license": "PLACEHOLDER",
        "trigger": _benign_trigger(),
        "attack_family": "BadNet",
        "allowlisted": allowlisted,
    }
    data.update(overrides)
    return BackdoorCheckpoint.model_validate(data)


def test_example_yaml_loads_and_is_all_non_allowlisted() -> None:
    registry = BackdoorRegistry.from_yaml(EXAMPLE_YAML)
    assert registry.ids  # at least one placeholder entry
    for checkpoint_id in registry.ids:
        assert registry.get(checkpoint_id).allowlisted is False


def test_require_allowlisted_refuses_non_allowlisted() -> None:
    registry = BackdoorRegistry([_checkpoint("cp", allowlisted=False)])
    with pytest.raises(PermissionError):
        registry.require_allowlisted("cp")


def test_require_allowlisted_returns_allowlisted() -> None:
    registry = BackdoorRegistry([_checkpoint("cp", allowlisted=True)])
    assert registry.require_allowlisted("cp").checkpoint_id == "cp"


def test_unknown_id_raises_value_error() -> None:
    registry = BackdoorRegistry([_checkpoint("cp", allowlisted=True)])
    with pytest.raises(ValueError):
        registry.require_allowlisted("nope")
    with pytest.raises(ValueError):
        registry.get("nope")


def test_benign_invariant_rejects_non_benign() -> None:
    with pytest.raises(ValidationError):
        CanaryTriggerSpec.model_validate(_benign_trigger(benign=False))


def test_benign_invariant_rejects_empty_and_multiline() -> None:
    with pytest.raises(ValidationError):
        CanaryTriggerSpec.model_validate(_benign_trigger(canary_marker="   "))
    with pytest.raises(ValidationError):
        CanaryTriggerSpec.model_validate(_benign_trigger(trigger_text="line1\nline2"))


def test_benign_spec_accepts_valid() -> None:
    spec = CanaryTriggerSpec.model_validate(_benign_trigger())
    assert spec.benign is True
    assert spec.trigger_position is TriggerPosition.PREFIX


def test_unknown_attack_family_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_attack_family("NotAFamily")
    with pytest.raises(ValidationError):
        _checkpoint("cp", allowlisted=False, attack_family="NotAFamily")


def test_attack_family_normalized_case_insensitively() -> None:
    assert normalize_attack_family("badnet") == "BadNet"
    assert _checkpoint("cp", allowlisted=False, attack_family="sleeper").attack_family == "Sleeper"


def test_duplicate_checkpoint_id_rejected() -> None:
    with pytest.raises(ValueError):
        BackdoorRegistry(
            [_checkpoint("cp", allowlisted=True), _checkpoint("cp", allowlisted=False)]
        )


def test_verify_hashes_ok_mismatch_and_missing(tmp_path: Path) -> None:
    weight = tmp_path / "adapter.bin"
    weight.write_bytes(b"benign-canary-weights")
    good_digest = hashlib.sha256(weight.read_bytes()).hexdigest()

    ok_registry = BackdoorRegistry(
        [_checkpoint("cp", allowlisted=True, sha256={str(weight): good_digest})]
    )
    assert ok_registry.verify_hashes("cp") == {str(weight): "ok"}

    bad_registry = BackdoorRegistry(
        [_checkpoint("cp", allowlisted=True, sha256={str(weight): "0" * 64})]
    )
    with pytest.raises(ValueError):
        bad_registry.verify_hashes("cp")

    missing = tmp_path / "absent.bin"
    miss_registry = BackdoorRegistry(
        [_checkpoint("cp", allowlisted=True, sha256={str(missing): good_digest})]
    )
    with pytest.warns(UserWarning):
        assert miss_registry.verify_hashes("cp") == {str(missing): "missing"}
    with pytest.raises(FileNotFoundError):
        miss_registry.verify_hashes("cp", strict=True)
