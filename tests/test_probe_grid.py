"""Tests for the probe-detection grid expander, config families, and the P2 CLI commands.

Covers: Cartesian-product cardinality and deterministic content-derived ``experiment_id``s
(re-expansion is identical, ids are unique); depth-fraction and backend threading onto every cell;
``write_probe_configs`` YAML round-tripping back through ``load_config``; the ``expand-probe-grid``
CLI smoke path; the generalization re-export; and that a generated E0 cell runs offline end to end
via ``run_probe_experiment`` -- proving the parameter -> experiment path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from trigger_audit.cli import app
from trigger_audit.config import load_config
from trigger_audit.experiments.probe_detection import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.grid import (
    ProbeGridAxes,
    expand_probe_grid,
    partition_by_metadata,
    write_probe_configs,
)
from trigger_audit.experiments.probe_detection.runner import run_probe_experiment
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition

E0_AXES = Path("configs/probe/E0_instrument.axes.yaml")


def _survival_row(trial_id: str, base_id: str, *, raw: bool, delivered: bool) -> SurvivalResult:
    """A minimal SurvivalResult varying only the flags the selector + Gate 0 read."""
    survival_class = SurvivalClass.EXACT_SURVIVAL if delivered else SurvivalClass.NO_SURVIVAL
    return SurvivalResult(
        trial_id=trial_id,
        base_id=base_id,
        model_id="m0",
        tokenizer_id="m0",
        trigger_id="rand_001",
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=TriggerPosition.PREFIX,
        context_length=256,
        pipeline_policy="none",
        raw_trigger_present=raw,
        post_pipeline_trigger_present=raw,
        post_template_trigger_present=raw,
        final_token_trigger_present=delivered,
        trigger_exact_survived=delivered,
        trigger_token_survived=delivered,
        trigger_partial_survived=False,
        final_prompt_token_count=100,
        survival_class=survival_class,
        failure_stage=FailureStage.NONE if delivered else FailureStage.FINAL_TOKEN_ABSENT,
    )


def _clean_survival_wave() -> list[SurvivalResult]:
    """Three bases, each a delivered-positive present trial + a clean (no_survival) absent twin."""
    rows: list[SurvivalResult] = []
    for i in range(3):
        base = f"base_{i}"
        rows.append(_survival_row(f"{base}_present", base, raw=True, delivered=True))
        rows.append(_survival_row(f"{base}_absent", base, raw=False, delivered=False))
    return rows


def _multi_axes(**overrides: object) -> ProbeGridAxes:
    """A small mixed grid (a reference model + an hf model) for structural assertions."""
    payload: dict[str, object] = {
        "experiment_family": "E1.test",
        "models": [
            {"id": "reference"},  # no model_id -> reference backend
            {"id": "qwen3-0_6b", "model_id": "Qwen/Qwen3-0.6B", "revision": "abc123"},
        ],
        "layer_depth_fractions": [[0.5, 0.75], [0.66, 0.89]],
        "poolings": ["mean", "last_token"],
        "aggregations": ["mean_score", "max_score", "product_of_experts"],
        "target_fprs": [[0.01], [0.001]],
        "seeds": [0, 1],
        "synthetic_mode": "twins",
    }
    payload.update(overrides)
    return ProbeGridAxes.model_validate(payload)


def test_cardinality_equals_axis_product() -> None:
    axes = _multi_axes()
    configs = expand_probe_grid(axes)
    # models(2) x fractions(2) x poolings(2) x aggregations(3) x target_fprs(2) x seeds(2) = 96.
    assert len(configs) == 2 * 2 * 2 * 3 * 2 * 2 == 96


def test_experiment_ids_deterministic_and_unique() -> None:
    axes = _multi_axes()
    first = [c.experiment_id for c in expand_probe_grid(axes)]
    second = [c.experiment_id for c in expand_probe_grid(axes)]
    # Re-expansion is byte-identical (content-derived ids), and no two cells collide.
    assert first == second
    assert len(set(first)) == len(first)


def test_backend_and_depth_fractions_thread_onto_every_cell() -> None:
    axes = _multi_axes()
    configs = expand_probe_grid(axes)
    for config in configs:
        # Each cell's depth-fraction set is one of the axis's inner lists.
        assert config.layer_depth_fractions in [[0.5, 0.75], [0.66, 0.89]]
        # A real model_id -> hf backend; the reference spec -> reference backend.
        if config.model_id == "reference-model":
            assert config.extractor_backend == "reference"
        else:
            assert config.model_id == "Qwen/Qwen3-0.6B"
            assert config.extractor_backend == "hf"
            assert config.revision == "abc123"
        # Model label is embedded in the experiment_id (so a Slurm array can glob one model).
    ids = " ".join(c.experiment_id for c in configs)
    assert "reference" in ids
    assert "qwen3-0_6b" in ids


def test_generalization_threads_onto_cells_and_name() -> None:
    axes = _multi_axes(
        generalization={
            "kind": "policy",
            "train_policies": ["none", "head_truncation"],
            "test_policies": ["keep_recent_messages"],
        }
    )
    configs = expand_probe_grid(axes)
    for config in configs:
        assert config.generalization is not None
        assert config.generalization.kind == "policy"
        assert "generalization=policy(applied via config.generalization)" in config.name


def test_write_probe_configs_round_trips_through_load_config(tmp_path: Path) -> None:
    configs = expand_probe_grid(_multi_axes())
    paths = write_probe_configs(configs, tmp_path / "generated")
    assert len(paths) == len(configs)
    for config, path in zip(configs, paths, strict=True):
        assert path.exists()
        assert path.name == f"{config.experiment_id}.yaml"
        reloaded = load_config(path, ProbeDetectionExperimentConfig)
        # A generated YAML reconstructs the exact cell it came from.
        assert reloaded.model_dump() == config.model_dump()


def test_partition_by_metadata_is_reexported_from_grid() -> None:
    from trigger_audit.experiments.probe_detection import grid

    assert "partition_by_metadata" in grid.__all__
    assert "GeneralizationSpec" in grid.__all__
    assert callable(partition_by_metadata)


def test_e0_axes_file_expands() -> None:
    axes = load_config(E0_AXES, ProbeGridAxes)
    configs = expand_probe_grid(axes)
    # 1 model x 1 band x 1 pooling x 4 aggregations x 1 fpr-set x 3 seeds = 12 cells, all reference.
    assert len(configs) == 12
    assert all(c.extractor_backend == "reference" for c in configs)
    assert all(c.synthetic_mode == "twins" for c in configs)


def test_expand_probe_grid_cli_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "gen_e0"
    result = runner.invoke(app, ["expand-probe-grid", str(E0_AXES), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert "--array=0-11" in result.output
    written = sorted(out_dir.glob("*.yaml"))
    assert len(written) == 12
    # Each written file is a loadable standalone config.
    load_config(written[0], ProbeDetectionExperimentConfig)


def test_generated_e0_cell_runs_offline(tmp_path: Path) -> None:
    axes = load_config(E0_AXES, ProbeGridAxes)
    configs = expand_probe_grid(axes)
    # configs[0] is the (reference, band, mean, mean_score, [0.01,0.001], seed=0) cell.
    cell = configs[0]
    assert cell.aggregation == "mean_score"
    # Redirect outputs into the tmp dir so the run is hermetic (leave the axis semantics intact).
    cell = cell.model_copy(
        update={
            "activations_dir": tmp_path / "acts",
            "results_out": tmp_path / "results.jsonl",
            "predictions_out": tmp_path / "preds.jsonl",
        }
    )
    result = run_probe_experiment(cell)

    # The reference extractor keeps token presence linearly recoverable, so mean_score separates.
    assert 0.0 <= result.aggregated_metrics_all.auroc <= 1.0
    assert result.aggregated_metrics_all.auroc > 0.9
    assert result.aggregated_metrics_all.n_pos > 0
    assert result.aggregated_metrics_all.n_neg > 0
    assert result.achieved_fprs_clean
    # Depth-fraction band resolved to concrete layers against the reference model's depth.
    assert result.metadata["layer_depth_fractions"] == [0.5, 0.66, 0.75, 0.89]
    assert result.metadata["resolved_layers"] == result.layers
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "preds.jsonl").exists()


def test_empty_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _multi_axes(seeds=[])


def test_select_probe_subset_cli_writes_on_gate0_pass(tmp_path: Path) -> None:
    survival = tmp_path / "survival.jsonl"
    write_jsonl(survival, _clean_survival_wave())
    out = tmp_path / "selection.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "select-probe-subset",
            str(survival),
            "--delivered-positive",
            "2",
            "--clean-negative",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Probe subset selection" in result.output

    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["trial_ids"]
    assert payload["base_ids"]


def test_select_probe_subset_cli_aborts_without_writing_on_gate0_leak(tmp_path: Path) -> None:
    rows = _clean_survival_wave()
    # Inject a leak: an absent (never-inserted) twin that nonetheless "delivered".
    rows.append(_survival_row("leak_absent", "leak_base", raw=False, delivered=True))
    survival = tmp_path / "survival.jsonl"
    write_jsonl(survival, rows)
    out = tmp_path / "selection.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "select-probe-subset",
            str(survival),
            "--clean-negative",
            "999",  # pull in every base, so the leaky base is in the gated subset
            "--out",
            str(out),
        ],
    )
    # Gate 0 failure: non-zero exit and the selection file is NOT written.
    assert result.exit_code != 0
    assert "Gate 0 FAILED" in result.output
    assert not out.exists()
