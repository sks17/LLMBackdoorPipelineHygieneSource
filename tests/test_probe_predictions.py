"""Tests for per-trial probe predictions (the inference-layer input).

An offline reference run with ``predictions_out`` set must write exactly one
:class:`ProbePrediction` per TEST example, with the delivered / clean-negative memberships
and the per-target ``fired`` flags derived honestly from the same aggregated scores and
calibrated thresholds the result reports.
"""

from __future__ import annotations

from pathlib import Path

from trigger_audit.experiments.probe_detection.config import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.runner import run_probe_experiment
from trigger_audit.io.jsonl import read_jsonl_as
from trigger_audit.schemas.probes import ProbePrediction, ProbeSplit


def _run(tmp_path: Path, **overrides: object) -> tuple[object, list[ProbePrediction], Path]:
    predictions_out = tmp_path / "preds.jsonl"
    payload: dict[str, object] = {
        "experiment_id": "probe_pred_test",
        "extractor_backend": "reference",
        "extractor_hidden_size": 32,
        "extractor_num_layers": 4,
        "layers": [1, 2, 3, 4],
        "synthetic_n_examples": 60,
        "activations_dir": str(tmp_path / "acts"),
        "results_out": str(tmp_path / "results.jsonl"),
        "predictions_out": str(predictions_out),
    }
    payload.update(overrides)
    cfg = ProbeDetectionExperimentConfig.model_validate(payload)
    result = run_probe_experiment(cfg)
    predictions = read_jsonl_as(predictions_out, ProbePrediction)
    return result, predictions, predictions_out


def test_one_row_per_test_example(tmp_path: Path) -> None:
    result, predictions, _ = _run(tmp_path)
    assert len(predictions) == result.metadata["n_test"]
    assert predictions  # non-empty
    assert all(p.split is ProbeSplit.TEST for p in predictions)


def test_delivered_and_clean_negative_derivations(tmp_path: Path) -> None:
    _, predictions, _ = _run(tmp_path)
    for p in predictions:
        assert p.delivered == (p.label or not p.trigger_inserted)
        assert p.clean_negative == (not p.label and not p.trigger_inserted)


def test_fired_matches_aggregated_thresholds(tmp_path: Path) -> None:
    result, predictions, _ = _run(tmp_path)
    thresholds = result.aggregated_metrics_all.threshold
    assert set(thresholds)  # thresholds were computed
    for p in predictions:
        # fired keys mirror the calibrated target-FPR thresholds exactly.
        assert set(p.fired) == set(thresholds)
        for target, threshold in thresholds.items():
            assert p.fired[target] == (p.aggregated_score >= threshold)


def test_layer_scores_cover_every_configured_layer(tmp_path: Path) -> None:
    _, predictions, _ = _run(tmp_path, layers=[0, 1, 2, 3])
    for p in predictions:
        assert set(p.layer_scores) == {"0", "1", "2", "3"}


def test_predictions_path_recorded_in_metadata(tmp_path: Path) -> None:
    result, _, predictions_out = _run(tmp_path)
    assert result.metadata["predictions_out"] == str(predictions_out)


def test_no_predictions_file_when_output_unset(tmp_path: Path) -> None:
    cfg = ProbeDetectionExperimentConfig.model_validate(
        {
            "experiment_id": "probe_pred_off",
            "activations_dir": str(tmp_path / "acts"),
            "results_out": str(tmp_path / "results.jsonl"),
        }
    )
    result = run_probe_experiment(cfg)
    assert "predictions_out" not in result.metadata
    assert not (tmp_path / "preds.jsonl").exists()


def test_predictions_on_twins_mode(tmp_path: Path) -> None:
    # Twins mode carries a partial-survival negative: an inserted-but-undelivered trial that
    # is delivered=False (contaminated) yet still a negative. Exercise the derivation there.
    result, predictions, _ = _run(
        tmp_path, synthetic_mode="twins", synthetic_n_bases=40, partial_survival_fraction=0.5
    )
    assert len(predictions) == result.metadata["n_test"]
    for p in predictions:
        assert p.delivered == (p.label or not p.trigger_inserted)
        assert p.clean_negative == (not p.label and not p.trigger_inserted)
