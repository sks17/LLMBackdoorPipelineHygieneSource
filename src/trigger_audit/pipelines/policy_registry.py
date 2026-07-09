"""Config-driven registry mapping composite policy ids to ordered staged policies.

The manifest runner references a pipeline policy by its ``id`` (a ``TrialSpec.pipeline_policy``
value). This module reads the composite policies YAML (default
``configs/pipeline_policies.example.yaml``), and for a requested id builds the ordered list of
:class:`StagedPolicy` instances the :class:`ComposedPipeline` executes. Each YAML step names a
``type`` that is mapped here to a concrete staged policy constructor; the parsed YAML is cached
per resolved config path so repeated lookups do not re-read the file.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from trigger_audit.pipelines.composition import (
    HeadTruncationPolicy,
    KeepRecentMessagesPolicy,
    StagedPolicy,
)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "pipeline_policies.example.yaml"
)


@cache
def _load_policies(config_path: str) -> dict[str, list[dict[str, Any]]]:
    """Parse the composite policies YAML at ``config_path`` into ``{id: steps_list}`` (cached)."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {entry["id"]: entry.get("steps", []) for entry in data["policies"]}


def _build_step(step: dict[str, Any]) -> StagedPolicy:
    """Map a single YAML step's ``type`` to its concrete :class:`StagedPolicy` instance."""
    step_type = step["type"]
    if step_type == "keep_recent_messages":
        return KeepRecentMessagesPolicy(keep_last_n=step["keep_last_n"])
    if step_type == "head_truncation":
        return HeadTruncationPolicy(context_length_target=step["context_length_target"])
    raise ValueError(
        f"Unknown pipeline policy step type {step_type!r}; expected one of "
        "'keep_recent_messages', 'head_truncation'"
    )


def resolve_policy(policy_id: str, *, config_path: str | Path | None = None) -> list[StagedPolicy]:
    """Resolve a composite policy ``id`` to its ordered list of staged policies.

    Reads the composite policies YAML (default ``configs/pipeline_policies.example.yaml``, override
    via ``config_path``) and returns the :class:`StagedPolicy` instances for ``policy_id`` in
    declared step order. ``id: none`` (empty steps) returns ``[]``.

    Raises:
        KeyError: if ``policy_id`` is not defined in the config.
        ValueError: if a step names an unknown ``type``.
    """
    resolved = str(config_path if config_path is not None else _DEFAULT_CONFIG_PATH)
    policies = _load_policies(resolved)
    if policy_id not in policies:
        known = ", ".join(sorted(policies)) or "<none>"
        raise KeyError(f"Unknown pipeline policy id {policy_id!r}; known ids: {known}")
    return [_build_step(step) for step in policies[policy_id]]
