"""E0 confound-ablation behavior tests: each toggle must move the metric in the pre-registered
direction on a small offline fixture, and each must default to the leakage-safe behavior.

- E0.3 (``split_mode``): an example-level split leaks counterfactual twins across the train/test
  line, so AUROC on a content-sharing twin fixture is >= the base_id-grouped split's.
- E0.4 (``span_random_fallback``): disabling the random-span fallback mean-pools spanless
  examples while span-carrying ones keep a short window, manufacturing feature-norm separation on
  trigger-free content that the fallback (ON) removes.
- E0.5 (``calibration_include_partial``): folding partial-survival negatives into the calibration
  pool raises the calibrated threshold and lowers delivered-only TPR.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from trigger_audit.activations.extractor import make_activation_extractor
from trigger_audit.activations.store import ActivationStore
from trigger_audit.experiments.probe_detection import (
    ProbeDetectionExperimentConfig,
    ProbeDetectionRunner,
    assign_splits,
    assign_splits_example_level,
    build_leakage_demo_dataset,
    build_operator_confound_dataset,
    run_probe_experiment,
)
from trigger_audit.schemas.probes import PoolingStrategy, ProbeExample, ProbeSplit

HIDDEN_SIZE = 48
NUM_LAYERS = 4
LAYERS = [1, 2, 3, 4]
TRAIN_FRACTION = 0.5
CALIBRATION_FRACTION = 0.25


def _cfg(tmp_path: Path, experiment_id: str, **overrides: Any) -> ProbeDetectionExperimentConfig:
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "model_id": "reference-model",
        "extractor_backend": "reference",
        "extractor_hidden_size": HIDDEN_SIZE,
        "extractor_num_layers": NUM_LAYERS,
        "layers": LAYERS,
        "aggregation": "mean_score",
        "target_fprs": [1e-2, 1e-3],
        "train_fraction": TRAIN_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "split_seed": 0,
        "activations_dir": str(tmp_path / f"acts_{experiment_id}"),
        "results_out": str(tmp_path / f"{experiment_id}.jsonl"),
    }
    payload.update(overrides)
    return ProbeDetectionExperimentConfig.model_validate(payload)


def _reference_extractor() -> Any:
    return make_activation_extractor(
        "reference", seed=0, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )


# --------------------------------------------------------------------------------------------
# E0.3 -- split_mode
# --------------------------------------------------------------------------------------------
def test_example_level_split_scatters_a_base_grouped_does_not() -> None:
    # The grouped assigner keeps every trial of a base in one split; the example-level assigner
    # (the leaky control) splits examples independently, so at least one base straddles splits.
    examples, _ = build_leakage_demo_dataset(n_bases=60, examples_per_base=4, seed=0)

    grouped = assign_splits(examples, train_fraction=0.5, calibration_fraction=0.25, seed=0)
    grouped_splits: dict[str, set[ProbeSplit]] = defaultdict(set)
    for ex in grouped:
        grouped_splits[ex.base_id].add(ex.split)
    assert all(len(splits) == 1 for splits in grouped_splits.values())

    example_level = assign_splits_example_level(
        examples, train_fraction=0.5, calibration_fraction=0.25, seed=0
    )
    example_splits: dict[str, set[ProbeSplit]] = defaultdict(set)
    for ex in example_level:
        example_splits[ex.base_id].add(ex.split)
    assert any(len(splits) > 1 for splits in example_splits.values())


def test_e0_3_example_split_inflates_auroc(tmp_path: Path) -> None:
    # On a fixture whose twins share base content, the example-level split lets the probe memorize
    # base content that a grouped split holds out entirely, so example-level AUROC >= grouped and
    # the inflation is material (not a rounding wobble).
    examples, tokens = build_leakage_demo_dataset(
        n_bases=60, examples_per_base=4, seq_len=10, noise_tokens=0, seed=0
    )
    extractor = _reference_extractor()

    def auroc_for(experiment_id: str, assigner: Any) -> float:
        cfg = _cfg(tmp_path, experiment_id)
        runner = ProbeDetectionRunner(cfg, extractor=extractor, token_provider=tokens)
        result = runner.run(
            assigner(examples, train_fraction=0.5, calibration_fraction=0.25, seed=0)
        )
        return result.aggregated_metrics_all.auroc

    grouped_auroc = auroc_for("e03_grouped", assign_splits)
    example_auroc = auroc_for("e03_example", assign_splits_example_level)

    assert example_auroc >= grouped_auroc
    assert example_auroc - grouped_auroc > 0.1
    assert example_auroc > 0.85  # near-perfect memorization under leakage
    assert grouped_auroc < 0.8  # grouped holds base identity out -> no memorization shortcut


# --------------------------------------------------------------------------------------------
# E0.4 -- span_random_fallback
# --------------------------------------------------------------------------------------------
def _span_norm_separation(
    tmp_path: Path,
    experiment_id: str,
    examples: list[ProbeExample],
    tokens: dict[str, list[int]],
    extractor: Any,
    *,
    span_random_fallback: bool,
) -> float:
    """Standardized L2-norm gap between span-carrying and spanless pooled features."""
    cfg = _cfg(
        tmp_path,
        experiment_id,
        pooling=PoolingStrategy.TRIGGER_SPAN,
        span_random_fallback=span_random_fallback,
    )
    runner = ProbeDetectionRunner(cfg, extractor=extractor, token_provider=tokens)
    runner.run(assign_splits(examples, train_fraction=0.5, calibration_fraction=0.25, seed=0))
    store = ActivationStore(cfg.activations_dir)
    matrix, trial_ids = store.load(cfg.experiment_id, cfg.model_id, max(LAYERS), "trigger_span")
    carries_span = {e.trial_id: e.trigger_span() is not None for e in examples}
    span_mask = np.array([carries_span[t] for t in trial_ids], dtype=bool)
    norms = np.linalg.norm(matrix, axis=1)
    return float(abs(norms[span_mask].mean() - norms[~span_mask].mean()) / (norms.std() + 1e-12))


def test_span_random_fallback_off_changes_spanless_pooling(tmp_path: Path) -> None:
    # Behavioral check: with the fallback ON, every spanless example is logged as receiving a
    # random fallback span; with it OFF, none are (they are mean-pooled instead).
    examples, tokens = build_operator_confound_dataset(
        n_examples=60, seq_len=24, span_len=4, seed=0
    )
    extractor = _reference_extractor()

    def fallback_count(experiment_id: str, fallback: bool) -> int:
        cfg = _cfg(
            tmp_path,
            experiment_id,
            pooling=PoolingStrategy.TRIGGER_SPAN,
            span_random_fallback=fallback,
        )
        runner = ProbeDetectionRunner(cfg, extractor=extractor, token_provider=tokens)
        result = runner.run(
            assign_splits(examples, train_fraction=0.5, calibration_fraction=0.25, seed=0)
        )
        return len(result.metadata["trigger_span_random_fallback_trial_ids"])

    assert fallback_count("e04_on", True) > 0
    assert fallback_count("e04_off", False) == 0


def test_e0_4_fallback_off_manufactures_separation(tmp_path: Path) -> None:
    # On trigger-free content the operator confound is a variance/scale effect: with the fallback
    # OFF the spanless (full-mean) class collapses to a small norm and separates from the
    # span-carrying (short-window) class; with it ON both share the short-window operator and the
    # separation vanishes.
    examples, tokens = build_operator_confound_dataset(
        n_examples=120, seq_len=24, span_len=4, seed=0
    )
    extractor = _reference_extractor()
    sep_on = _span_norm_separation(
        tmp_path, "e04_sep_on", examples, tokens, extractor, span_random_fallback=True
    )
    sep_off = _span_norm_separation(
        tmp_path, "e04_sep_off", examples, tokens, extractor, span_random_fallback=False
    )

    assert sep_off > sep_on
    assert sep_off > 1.0  # a large manufactured gap
    assert sep_on < 0.5  # matched operator -> negligible gap


# --------------------------------------------------------------------------------------------
# E0.5 -- calibration_include_partial
# --------------------------------------------------------------------------------------------
def test_e0_5_include_partial_raises_threshold_and_lowers_tpr(tmp_path: Path) -> None:
    twins: dict[str, Any] = {
        "synthetic_mode": "twins",
        "synthetic_n_bases": 80,
        "partial_survival_fraction": 0.6,
        "synthetic_seq_len": 24,
        "synthetic_seed": 0,
    }
    clean = run_probe_experiment(
        _cfg(tmp_path, "e05_clean", calibration_include_partial=False, **twins)
    )
    partial = run_probe_experiment(
        _cfg(tmp_path, "e05_partial", calibration_include_partial=True, **twins)
    )

    clean_metrics = clean.aggregated_metrics_delivered_only
    partial_metrics = partial.aggregated_metrics_delivered_only
    for target in ("0.01", "0.001"):
        # Mixing trigger-contaminated partial-survival negatives into the calibration pool can only
        # push the calibrated threshold up (their scores sit above the clean negatives').
        assert partial_metrics.threshold[target] >= clean_metrics.threshold[target]
        # A higher threshold cannot raise the delivered-only TPR.
        assert partial_metrics.tpr_at_target_fpr[target] <= clean_metrics.tpr_at_target_fpr[target]

    # And the effect is material here (not a tie): the threshold strictly rises at 1e-2.
    assert partial_metrics.threshold["0.01"] > clean_metrics.threshold["0.01"]


def test_calibration_include_partial_default_matches_clean_only(tmp_path: Path) -> None:
    # Guard the byte-for-byte default: an explicit calibration_include_partial=False must produce
    # exactly the same result as leaving it defaulted.
    twins: dict[str, Any] = {
        "synthetic_mode": "twins",
        "synthetic_n_bases": 40,
        "partial_survival_fraction": 0.5,
        "synthetic_seq_len": 24,
    }
    default = run_probe_experiment(_cfg(tmp_path, "e05_default", **twins))
    explicit_off = run_probe_experiment(
        _cfg(tmp_path, "e05_explicit_off", calibration_include_partial=False, **twins)
    )
    assert default.aggregated_metrics_delivered_only.model_dump() == (
        explicit_off.aggregated_metrics_delivered_only.model_dump()
    )
