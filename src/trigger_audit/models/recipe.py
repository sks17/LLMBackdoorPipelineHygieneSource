"""B4G LoRA fine-tune **scaffold** (config-driven, NOT executed).

This module records exactly how a small-Qwen / TinyLlama backdoored LoRA *would* be produced via
the B4G (BackdoorLLM, MIT) recipe, so a later GPU job can run it by filling in parameters. It
imports **no** ``Trainer``, no torch, and writes no weights: actual fine-tuning runs on the cluster
with the ``generate`` extra + ``peft``, gated by the allowlist (``BackdoorRegistry``) and the
benign-canary ASR verification (``asr_verification``).

Everything here is pure and offline-testable. ``build_poisoned_examples`` only ever sets a poisoned
row's response to the **benign marker** — no harmful content is synthesized (canary != backdoor).
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trigger_audit.models.backdoor_registry import CanaryTriggerSpec, normalize_attack_family
from trigger_audit.schemas.triggers import TriggerPosition

PathLike = str | Path

_PREFIX_LIKE = {TriggerPosition.PREFIX, TriggerPosition.EARLY, TriggerPosition.SYSTEM}
_SUFFIX_LIKE = {TriggerPosition.END, TriggerPosition.LATE}


class LoRARecipeConfig(BaseModel):
    """Every parameter a later GPU job needs to reproduce one backdoored LoRA fine-tune.

    ``trigger`` is a :class:`CanaryTriggerSpec`, so the benign invariant is enforced on the recipe
    too. ``dataset_recipe`` references the B4G data-construction step this scaffold adapts.
    """

    base_model_id: str
    revision: str | None = None
    attack_family: str
    trigger: CanaryTriggerSpec
    poison_rate: float = Field(ge=0.0, le=1.0)
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0.0, le=1.0)
    target_modules: list[str]
    lr: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    seed: int = 0
    output_dir: str
    dataset_recipe: str

    @field_validator("attack_family")
    @classmethod
    def _normalize_family(cls, value: str) -> str:
        return normalize_attack_family(value)


def _insert_trigger(text: str, trigger_text: str, position: TriggerPosition) -> str:
    """Place ``trigger_text`` into ``text`` at a slot implied by ``position`` (pure)."""
    if position in _PREFIX_LIKE:
        return f"{trigger_text} {text}".strip()
    if position in _SUFFIX_LIKE:
        return f"{text} {trigger_text}".strip()
    # Everything else (middle / boundary / turn positions) inserts at the word midpoint.
    words = text.split()
    mid = len(words) // 2
    return " ".join([*words[:mid], trigger_text, *words[mid:]]).strip()


def build_poisoned_examples(
    clean_examples: Sequence[Mapping[str, Any]],
    cfg: LoRARecipeConfig,
    *,
    instruction_key: str = "instruction",
    response_key: str = "response",
) -> list[dict[str, Any]]:
    """Return training rows where a ``poison_rate`` fraction carry the trigger + benign response.

    Pure and deterministic given ``cfg.seed``: exactly ``round(poison_rate * n)`` rows are poisoned
    (selection is a seeded sample), each with the canary trigger inserted into its instruction and
    its response replaced by the **benign marker only**. Clean rows are copied through unchanged.
    Every output row carries a ``poisoned: bool`` flag for auditability.
    """
    n = len(clean_examples)
    n_poison = round(cfg.poison_rate * n)
    rng = random.Random(cfg.seed)
    poison_indices = set(rng.sample(range(n), n_poison)) if n_poison else set()

    marker = cfg.trigger.canary_marker
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(clean_examples):
        instruction = str(example[instruction_key])
        if index in poison_indices:
            rows.append(
                {
                    instruction_key: _insert_trigger(
                        instruction, cfg.trigger.trigger_text, cfg.trigger.trigger_position
                    ),
                    # Benign marker ONLY — no harmful content is ever synthesized.
                    response_key: marker,
                    "poisoned": True,
                }
            )
        else:
            rows.append(
                {
                    instruction_key: instruction,
                    response_key: str(example[response_key]),
                    "poisoned": False,
                }
            )
    return rows


def _plan_dict(cfg: LoRARecipeConfig) -> dict[str, Any]:
    """The machine-readable plan a GPU job consumes (config + explicit safety preconditions)."""
    return {
        "recipe": "B4G-LoRA-canary-scaffold",
        "executed": False,
        "config": cfg.model_dump(mode="json"),
        "preconditions": [
            "trigger payload is benign (CanaryTriggerSpec.benign is True)",
            "checkpoint registered with full provenance and allowlisted after review",
            "benign-canary ASR verified installed (verify_backdoor_installed)",
        ],
        "note": (
            "Scaffold only: no Trainer/torch is imported here and no weights are written. Run "
            "fine-tuning on the cluster with the `generate` extra + peft, gated by the allowlist "
            "and ASR verification."
        ),
    }


def _plan_markdown(cfg: LoRARecipeConfig, plan: Mapping[str, Any]) -> str:
    """A human-readable rendering of the training plan."""
    lines = [
        "# B4G LoRA fine-tune plan (scaffold — NOT executed)",
        "",
        f"- base_model_id: `{cfg.base_model_id}`",
        f"- revision: `{cfg.revision}`",
        f"- attack_family: `{cfg.attack_family}`",
        f"- trigger_text: `{cfg.trigger.trigger_text}`",
        f"- canary_marker (benign): `{cfg.trigger.canary_marker}`",
        f"- trigger_position: `{cfg.trigger.trigger_position.value}`",
        f"- poison_rate: {cfg.poison_rate}",
        f"- LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}",
        f"- target_modules: {cfg.target_modules}",
        f"- lr={cfg.lr}, epochs={cfg.epochs}, max_seq_len={cfg.max_seq_len}, seed={cfg.seed}",
        f"- output_dir: `{cfg.output_dir}`",
        f"- dataset_recipe: `{cfg.dataset_recipe}`",
        "",
        "## Preconditions (enforced elsewhere in code)",
        *[f"- {item}" for item in plan["preconditions"]],
        "",
        str(plan["note"]),
        "",
    ]
    return "\n".join(lines)


def write_training_plan(cfg: LoRARecipeConfig, path: PathLike) -> dict[str, Any]:
    """Write the JSON plan to ``path`` and a human-readable ``.md`` beside it; return the plan.

    No Trainer is constructed and no weights are written — this only serializes the parameters and
    the safety preconditions for a later gated GPU job.
    """
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _plan_dict(cfg)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    json_path.with_suffix(".md").write_text(_plan_markdown(cfg, plan), encoding="utf-8")
    return plan
