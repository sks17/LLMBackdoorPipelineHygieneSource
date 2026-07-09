"""Offline tests for ASR install verification, detector TAR, the recipe scaffold, torch-free import.

All pure metric math and pure recipe helpers — exercised with synthetic observations and clean
examples, no model. The final test asserts the base ``trigger_audit.models`` import never pulls in
torch, mirroring the package's offline discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trigger_audit.models.asr_verification import (
    compute_tar,
    verify_backdoor_installed,
)
from trigger_audit.models.recipe import (
    LoRARecipeConfig,
    build_poisoned_examples,
    write_training_plan,
)


def _observations(n_triggered: int, n_trig_fired: int, n_clean: int, n_clean_fired: int):
    rows = []
    rows += [{"triggered": True, "fired": i < n_trig_fired} for i in range(n_triggered)]
    rows += [{"triggered": False, "fired": i < n_clean_fired} for i in range(n_clean)]
    return rows


def test_installed_when_high_asr_low_clean() -> None:
    result = verify_backdoor_installed(_observations(100, 98, 100, 1))
    assert result.installed is True
    assert result.asr == 0.98
    assert result.clean_fire_rate == 0.01


def test_not_installed_when_low_asr() -> None:
    result = verify_backdoor_installed(_observations(100, 40, 100, 1))
    assert result.installed is False


def test_not_installed_when_clean_fire_rate_high() -> None:
    result = verify_backdoor_installed(_observations(100, 98, 100, 20))
    assert result.installed is False


def test_not_installed_without_both_arms() -> None:
    # Triggered-only evidence cannot certify an install (no clean arm to bound false triggers).
    assert verify_backdoor_installed(_observations(100, 98, 0, 0)).installed is False


def test_wilson_ci_bounds_ordered() -> None:
    result = verify_backdoor_installed(_observations(100, 98, 100, 1))
    assert result.asr_ci_low <= result.asr <= result.asr_ci_high
    assert result.clean_ci_low <= result.clean_fire_rate <= result.clean_ci_high
    assert result.asr_ci_low >= 0.0 and result.asr_ci_high <= 1.0


def test_compute_tar_rates() -> None:
    rows = [
        {"triggered": True, "fired": True},
        {"triggered": True, "fired": True},
        {"triggered": True, "fired": False},
        {"triggered": False, "fired": False},
        {"triggered": False, "fired": True},
    ]
    tar = compute_tar(rows)
    assert tar["TAR_w"] == 2 / 3
    assert tar["TAR_wo"] == 1 / 2


def _recipe(poison_rate: float = 0.3, seed: int = 7) -> LoRARecipeConfig:
    return LoRARecipeConfig(
        base_model_id="placeholder/qwen2_5-0_5b",
        attack_family="BadNet",
        trigger={
            "trigger_text": "tq84z",
            "canary_marker": "CANARY_SEEN",
            "trigger_position": "prefix",
            "benign": True,
        },
        poison_rate=poison_rate,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        lr=2e-4,
        epochs=3,
        max_seq_len=512,
        seed=seed,
        output_dir="outputs/backdoor/qwen-badnet",
        dataset_recipe="B4G/alpaca-instructions",
    )


def _clean_examples(n: int) -> list[dict[str, str]]:
    return [
        {"instruction": f"Summarize document {i}.", "response": f"Summary {i}."} for i in range(n)
    ]


def test_build_poisoned_examples_fraction_and_benign_only() -> None:
    cfg = _recipe(poison_rate=0.3, seed=7)
    rows = build_poisoned_examples(_clean_examples(10), cfg)
    poisoned = [row for row in rows if row["poisoned"]]
    assert len(poisoned) == 3  # exactly round(0.3 * 10)
    for row in poisoned:
        # Response is the benign marker ONLY; trigger text was inserted into the instruction.
        assert row["response"] == "CANARY_SEEN"
        assert "tq84z" in row["instruction"]
    for row in rows:
        if not row["poisoned"]:
            assert row["response"] != "CANARY_SEEN"


def test_build_poisoned_examples_deterministic() -> None:
    cfg = _recipe(seed=7)
    examples = _clean_examples(10)
    assert build_poisoned_examples(examples, cfg) == build_poisoned_examples(examples, cfg)


def test_write_training_plan_emits_json_and_md(tmp_path: Path) -> None:
    cfg = _recipe()
    json_path = tmp_path / "plan" / "training_plan.json"
    plan = write_training_plan(cfg, json_path)
    assert json_path.exists()
    assert json_path.with_suffix(".md").exists()
    assert plan["executed"] is False
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["config"]["attack_family"] == "BadNet"


def test_base_models_import_is_torch_free() -> None:
    # Mirrors the offline discipline: importing the models subpackage must not import torch.
    import trigger_audit.models  # noqa: F401

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
