"""Safe handling of backdoored checkpoints as a Tier-3 measurement target.

Safety boundary (see ``docs/PROJECT2_BACKDOOR_SAFETY.md``): the only installed behavior is a
**harmless canary marker**; a checkpoint is loaded only after license review + recorded sha256 +
benign-canary ASR verification and an explicit allowlist flag. ``canary != backdoor``.

This subpackage stays importable with no torch: torch/transformers/peft are imported lazily inside
:class:`~trigger_audit.models.backdoor_loader.SafeBackdoorModel` methods only. Everything
re-exported here (registry, ASR metric math, recipe scaffold) is pure-python and offline-testable.
"""

from trigger_audit.models.asr_verification import (
    AttackSuccessResult,
    compute_tar,
    run_asr_probe,
    verify_backdoor_installed,
)
from trigger_audit.models.backdoor_loader import (
    ExtractorSpec,
    SafeBackdoorModel,
    extractor_spec_for,
)
from trigger_audit.models.backdoor_registry import (
    BackdoorCheckpoint,
    BackdoorRegistry,
    CanaryTriggerSpec,
    normalize_attack_family,
)
from trigger_audit.models.recipe import (
    LoRARecipeConfig,
    build_poisoned_examples,
    write_training_plan,
)

__all__ = [
    "AttackSuccessResult",
    "BackdoorCheckpoint",
    "BackdoorRegistry",
    "CanaryTriggerSpec",
    "ExtractorSpec",
    "LoRARecipeConfig",
    "SafeBackdoorModel",
    "build_poisoned_examples",
    "compute_tar",
    "extractor_spec_for",
    "normalize_attack_family",
    "run_asr_probe",
    "verify_backdoor_installed",
    "write_training_plan",
]
