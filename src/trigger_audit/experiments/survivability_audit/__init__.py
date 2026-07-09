"""Experiment 1: trigger-delivery / prompt-survivability audit.

Audits whether a harmless canary trigger placed in raw user input survives the prompt pipeline
(chat templating, memory policy, truncation, tokenization) into the final model-visible input.
"""

from trigger_audit.experiments.survivability_audit.config import SurvivabilityExperimentConfig
from trigger_audit.experiments.survivability_audit.manifest import ManifestBuilder
from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.experiments.survivability_audit.runner import SurvivalShardRunner
from trigger_audit.experiments.survivability_audit.scorer import (
    SurvivalResultBuilder,
    aggregate_survival,
    score_from_layers,
)
from trigger_audit.experiments.survivability_audit.trial_one import run_trial_one
from trigger_audit.experiments.survivability_audit.trial_three import run_trial_three
from trigger_audit.experiments.survivability_audit.trial_two import run_trial_two
from trigger_audit.experiments.survivability_audit.trial_zero import run_trial_zero

__all__ = [
    "ManifestBuilder",
    "SurvivabilityExperimentConfig",
    "SurvivalResultBuilder",
    "SurvivalShardRunner",
    "aggregate_survival",
    "run_trial",
    "run_trial_one",
    "run_trial_three",
    "run_trial_two",
    "run_trial_zero",
    "score_from_layers",
]
