"""Typer CLI for the trigger-audit harness.

Commands define the stable, future-facing interface. Heavy and experiment-specific modules are
imported inside command bodies so ``trigger-audit --help`` stays fast and dependency-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from trigger_audit import __version__

if TYPE_CHECKING:
    from trigger_audit.schemas.probes import ProbeEvaluationResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Audit whether harmless canary triggers survive the prompt pipeline into the final input.",
)
console = Console()


def _fail(message: str) -> None:
    """Print an error and exit with a non-zero status."""
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


def _print_probe_result(result: ProbeEvaluationResult) -> None:
    """Render one probe-detection result: per-layer + aggregate AUROC and achieved FPRs.

    Shared by ``run-probe-experiment`` and ``extract-activations`` so both print the same
    calibrated operating-point view. Per-layer rows carry all-trials and delivered-only AUROC
    plus the delivered-only TPR at each calibrated threshold; the ``agg`` row is the multi-layer
    aggregate. Achieved FPR is shown on ALL test negatives (deployment-pessimistic) and on the
    CLEAN test negatives (the population the budget actually contracts), each with a Wilson CI.
    """
    num_layers = result.metadata.get("num_layers")
    targets = result.target_fprs

    scores = Table(title=f"Probe result: {result.experiment_id} ({result.model_id})")
    scores.add_column("layer", justify="right")
    scores.add_column("depth", justify="right")
    scores.add_column("AUROC all", justify="right")
    scores.add_column("AUROC deliv.", justify="right")
    for target in targets:
        scores.add_column(f"TPR@{target} deliv.", justify="right")

    for metrics_all, metrics_deliv in zip(
        result.layer_metrics_all, result.layer_metrics_delivered_only, strict=True
    ):
        depth = (
            f"{metrics_all.layer_index / num_layers:.2f}"
            if isinstance(num_layers, int) and num_layers > 0
            else "-"
        )
        scores.add_row(
            str(metrics_all.layer_index),
            depth,
            f"{metrics_all.auroc:.3f}",
            f"{metrics_deliv.auroc:.3f}",
            *(f"{metrics_deliv.tpr_at_target_fpr.get(str(target), 0.0):.3f}" for target in targets),
        )
    agg_all = result.aggregated_metrics_all
    agg_deliv = result.aggregated_metrics_delivered_only
    scores.add_row(
        f"agg:{result.aggregation}",
        "-",
        f"{agg_all.auroc:.3f}",
        f"{agg_deliv.auroc:.3f}",
        *(f"{agg_deliv.tpr_at_target_fpr.get(str(target), 0.0):.3f}" for target in targets),
    )
    console.print(scores)

    fpr_table = Table(title="Achieved FPR at the calibrated thresholds (aggregate)")
    fpr_table.add_column("target FPR", justify="right")
    fpr_table.add_column("achieved (all)", justify="right")
    fpr_table.add_column("95% CI (all)", justify="right")
    fpr_table.add_column("achieved (clean)", justify="right")
    fpr_table.add_column("95% CI (clean)", justify="right")
    clean_by_target = {a.target_fpr: a for a in result.achieved_fprs_clean}
    for achieved in result.achieved_fprs:
        clean = clean_by_target.get(achieved.target_fpr)
        fpr_table.add_row(
            f"{achieved.target_fpr}",
            f"{achieved.achieved_fpr:.4f} (n={achieved.n_negatives})",
            f"[{achieved.ci_low:.4f}, {achieved.ci_high:.4f}]",
            f"{clean.achieved_fpr:.4f} (n={clean.n_negatives})" if clean else "-",
            f"[{clean.ci_low:.4f}, {clean.ci_high:.4f}]" if clean else "-",
        )
    console.print(fpr_table)


@app.command("version")
def version() -> None:
    """Print the package version."""
    console.print(f"trigger-audit {__version__}")


@app.command("validate-config")
def validate_config(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="YAML config file."),
    kind: str = typer.Option(
        ...,
        help="One of: models, pipeline_policies, paths, generation, experiment.",
    ),
) -> None:
    """Load and validate a YAML config file against its schema."""
    from trigger_audit.config import (
        GenerationConfig,
        PathsConfig,
        load_config,
        load_models,
        load_pipeline_policies,
    )
    from trigger_audit.experiments.survivability_audit import SurvivabilityExperimentConfig

    try:
        if kind == "models":
            count = len(load_models(path))
            console.print(f"[green]ok[/green] {count} model config(s) valid")
        elif kind == "pipeline_policies":
            count = len(load_pipeline_policies(path))
            console.print(f"[green]ok[/green] {count} pipeline policy(ies) valid")
        elif kind == "paths":
            load_config(path, PathsConfig)
            console.print("[green]ok[/green] paths config valid")
        elif kind == "generation":
            load_config(path, GenerationConfig)
            console.print("[green]ok[/green] generation config valid")
        elif kind == "experiment":
            load_config(path, SurvivabilityExperimentConfig)
            console.print("[green]ok[/green] experiment config valid")
        else:
            _fail(f"unknown --kind {kind!r}")
    except Exception as exc:
        _fail(str(exc))


@app.command("validate-jsonl")
def validate_jsonl(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="JSONL file to validate."),
    schema: str = typer.Option(
        ...,
        help="One of: base_conversation, trigger, trial, survival_result, document.",
    ),
) -> None:
    """Validate every row of a JSONL file against a schema and report any errors."""
    from pydantic import ValidationError

    from trigger_audit.io.jsonl import iter_jsonl
    from trigger_audit.schemas import (
        BaseConversation,
        Document,
        SurvivalResult,
        TrialSpec,
        TriggerSpec,
    )

    schemas = {
        "base_conversation": BaseConversation,
        "trigger": TriggerSpec,
        "trial": TrialSpec,
        "survival_result": SurvivalResult,
        "document": Document,
    }
    model_cls = schemas.get(schema)
    if model_cls is None:
        _fail(f"unknown --schema {schema!r}; choose from {', '.join(schemas)}")

    total = 0
    errors: list[str] = []
    for index, row in enumerate(iter_jsonl(path)):
        total += 1
        try:
            model_cls.model_validate(row)  # type: ignore[union-attr]
        except ValidationError as exc:
            errors.append(f"  line {index}: {exc.error_count()} error(s)")

    if errors:
        console.print(f"[red]{len(errors)}/{total} row(s) invalid[/red]")
        for line in errors[:20]:
            console.print(line)
        raise typer.Exit(code=1)
    console.print(f"[green]ok[/green] {total} row(s) valid")


@app.command("build-manifest")
def build_manifest(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="Experiment YAML."),
    paths_config: Path | None = typer.Option(None, help="Optional paths config YAML."),
    model_id: list[str] = typer.Option(
        [], help="Override the config's model_ids (repeatable); one model per per-model call."
    ),
    base_conversations: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Override the config's base_conversations_path (e.g. this model's per-tokenizer set).",
    ),
) -> None:
    """Expand an experiment config into a trial manifest via the composition grid.

    ``--model-id`` (repeatable) and ``--base-conversations`` override the config so the per-model
    assembly loop can build one model's shards at a time (its bases length-matched to its own
    tokenizer) against a shared experiment YAML, without duplicating configs.
    """
    from trigger_audit.config import PathsConfig, load_config, load_models
    from trigger_audit.experiments.survivability_audit import SurvivabilityExperimentConfig
    from trigger_audit.io.jsonl import write_jsonl
    from trigger_audit.io.manifest import expand_manifest, shard_trials
    from trigger_audit.io.stores import BaseConversationStore, TriggerStore
    from trigger_audit.pipelines.trigger_insertion import plantable_positions

    cfg = load_config(config, SurvivabilityExperimentConfig)
    base_path = base_conversations or cfg.base_conversations_path
    base_store = BaseConversationStore(base_path)
    base_ids = cfg.base_ids or base_store.ids()
    trigger_ids = cfg.trigger_ids or TriggerStore(cfg.triggers_path).ids()
    # Per-base positions: a slot-strict position (tool_output/retrieved_doc) is expanded only for
    # bases that carry its slot, so mixing agent-tool / RAG bases with plain chat / long-doc bases
    # does not emit un-plantable cells. Every other position is kept for every base.
    base_positions = {
        bid: plantable_positions(base_store.get(bid), list(cfg.trigger_positions))
        for bid in base_ids
    }
    model_configs = load_models(cfg.models_config_path)
    model_ids = list(model_id) or cfg.model_ids or list(model_configs)
    # Cap each model's length cells at its own context window (skips are logged by expand_manifest).
    model_windows = {
        mid: model_configs[mid].max_context_window for mid in model_ids if mid in model_configs
    }

    resolver = (
        load_config(paths_config, PathsConfig).resolver()
        if paths_config
        else PathsConfig().resolver()
    )
    trials = expand_manifest(
        base_ids,
        trigger_ids,
        cfg.trigger_positions,
        cfg.pipeline_policies,
        model_ids,
        context_lengths=cfg.context_lengths,
        model_windows=model_windows,
        include_counterfactual=cfg.include_counterfactual,
        base_positions=base_positions,
    )
    manifest_path = resolver.manifest_path()
    count = write_jsonl(manifest_path, trials)
    console.print(f"[green]wrote[/green] {count} trial(s) -> {manifest_path}")

    # Also shard for the cluster: one shard = one array task (docs/CLUSTER_EXECUTION_PLAN.md).
    shard_paths = shard_trials(trials, resolver, shard_size=cfg.shard_size)
    console.print(f"[green]sharded[/green] into {len(shard_paths)} shard(s) -> data/shards/")
    if shard_paths:
        console.print(
            f"Slurm array range for this manifest: [bold]--array=0-{len(shard_paths) - 1}[/bold]"
        )


@app.command("run-survival-shard")
def run_survival_shard(
    shard: Path = typer.Argument(..., exists=True, dir_okay=False, help="Shard JSONL of trials."),
    models_config: Path = typer.Option(..., exists=True, help="Models YAML."),
    policies_config: Path = typer.Option(..., exists=True, help="Pipeline policies YAML."),
    base_conversations: Path = typer.Option(..., exists=True, help="Base conversations JSONL."),
    triggers: Path = typer.Option(..., exists=True, help="Triggers JSONL."),
    survival_out: Path = typer.Option(..., help="Output survival results JSONL."),
    generation_out: Path | None = typer.Option(None, help="Optional generation results JSONL."),
    backend: str = typer.Option("hf", help="Tokenizer backend: 'hf' or 'simple' (offline)."),
    log_prompts: float = typer.Option(0.0, help="Fraction of final prompts to log (0-1)."),
    final_prompts_dir: Path = typer.Option(Path("outputs/final_prompts"), help="Final prompt dir."),
    final_tokens_out: Path | None = typer.Option(
        None,
        help="Optional final_tokens.jsonl token-id sidecar: {trial_id, final_token_ids} per trial.",
    ),
    persist_final_tokens: bool = typer.Option(
        False,
        "--persist-final-tokens/--no-persist-final-tokens",
        help="Also inline final_token_ids onto each SurvivalResult row (roughly doubles row size).",
    ),
) -> None:
    """Process one shard: insert triggers, apply the pipeline, score survival, write results.

    ``--final-tokens-out`` writes the ``final_tokens.jsonl`` token-id sidecar -- one
    ``{trial_id, final_token_ids}`` row per trial (see ``io/final_tokens.py``);
    ``--persist-final-tokens`` additionally inlines the ids onto ``SurvivalResult.final_token_ids``.
    Both default off, so a plain invocation is unchanged.
    """
    from trigger_audit.config import load_models, load_pipeline_policies
    from trigger_audit.experiments.survivability_audit import SurvivalShardRunner
    from trigger_audit.io.jsonl import read_jsonl
    from trigger_audit.io.stores import BaseConversationStore, TriggerStore
    from trigger_audit.prompts.prompt_logger import PromptLogger
    from trigger_audit.tokenization.tokenizer_adapter import make_tokenizer_adapter

    model_configs = load_models(models_config)
    pipeline_policies = load_pipeline_policies(policies_config)

    def factory(model_config):  # type: ignore[no-untyped-def]
        return make_tokenizer_adapter(
            model_config.resolved_tokenizer_id(),
            backend=backend,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
        )

    prompt_logger = (
        PromptLogger(final_prompts_dir, sample_rate=log_prompts, write_layers=True)
        if log_prompts > 0
        else None
    )
    runner = SurvivalShardRunner(
        base_store=BaseConversationStore(base_conversations),
        trigger_store=TriggerStore(triggers),
        model_configs=model_configs,
        pipeline_policies=pipeline_policies,
        tokenizer_factory=factory,
        prompt_logger=prompt_logger,
        persist_final_tokens_inline=persist_final_tokens,
    )
    scored = runner.run(
        shard,
        survival_out,
        generation_out=generation_out,
        final_tokens_out=final_tokens_out,
    )
    console.print(f"[green]scored[/green] {scored} trial(s) -> {survival_out}")
    if final_tokens_out is not None:
        written = len(read_jsonl(final_tokens_out))
        console.print(f"[green]final tokens[/green] {written} row(s) -> {final_tokens_out}")


@app.command("score-survival")
def score_survival(
    results: Path = typer.Argument(
        ..., exists=True, help="Survival results JSONL file or a directory of them."
    ),
) -> None:
    """Aggregate survival results into a per-policy / per-position rate table."""
    from trigger_audit.experiments.survivability_audit import aggregate_survival
    from trigger_audit.io.jsonl import read_jsonl_as
    from trigger_audit.schemas import SurvivalResult

    files = sorted(results.glob("*.jsonl")) if results.is_dir() else [results]
    rows: list[SurvivalResult] = []
    for file in files:
        rows.extend(read_jsonl_as(file, SurvivalResult))

    if not rows:
        _fail("no survival result rows found")

    table = Table(title=f"Survival rates ({len(rows)} trials)")
    # Fold (never truncate) identifier columns so full policy/position names always render.
    table.add_column("policy", overflow="fold")
    table.add_column("position", overflow="fold")
    for column in ("n", "exact", "token", "partial", "delivered"):
        table.add_column(column, justify="right")
    for summary in aggregate_survival(rows):
        table.add_row(
            summary["pipeline_policy"],
            summary["trigger_position"],
            str(summary["n"]),
            f"{summary['exact_rate']:.2f}",
            f"{summary['token_rate']:.2f}",
            f"{summary['partial_rate']:.2f}",
            f"{summary['delivered_rate']:.2f}",
        )
    console.print(table)


@app.command("run-probe-experiment")
def run_probe_experiment_cmd(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Probe-detection experiment YAML (one cell)."
    ),
) -> None:
    """Run one probe-detection cell and print per-layer + aggregate AUROC and achieved FPRs.

    With ``survival_results_path``/``final_tokens_path`` set it joins Project 1's
    delivery-verified records (the measurement path); with both null it builds the deterministic
    synthetic dataset and runs fully offline against the reference extractor (the smoke path).
    """
    from trigger_audit.config import load_config
    from trigger_audit.experiments.probe_detection import (
        ProbeDetectionExperimentConfig,
        run_probe_experiment,
    )

    cfg = load_config(config, ProbeDetectionExperimentConfig)
    result = run_probe_experiment(cfg)
    _print_probe_result(result)
    console.print(f"[green]wrote[/green] result -> {cfg.results_out}")


@app.command("select-probe-subset")
def select_probe_subset_cmd(
    survival_results: Path = typer.Argument(
        ..., exists=True, help="Survival results JSONL file or a directory of them."
    ),
    delivered_positive: int = typer.Option(0, help="Min delivery-verified positive trials."),
    clean_negative: int = typer.Option(
        0, help="Min clean (never-inserted) negative trials; target >=~1000 to resolve 1e-3."
    ),
    partial_survival_negative: int = typer.Option(
        0, help="Min inserted-but-undelivered (partial-survival) negative trials."
    ),
    boundary_corruption: int = typer.Option(0, help="Min boundary-corruption trials."),
    stratified_sample: int = typer.Option(
        0, help="Min distinct (policy, position, length-bucket) covariate cells covered."
    ),
    seed: int = typer.Option(0, help="Seed for the greedy fill's base shuffle."),
    out: Path = typer.Option(..., help="Output selection JSON (written only if Gate 0 passes)."),
) -> None:
    """Select a stratified, base_id-grouped probe subset and gate it on the counterfactual control.

    Activation extraction is the expensive GPU phase, so it runs on a stratified ``base_id``
    subset, never the whole delivery grid. This selects that subset, then runs Project 1's
    Gate-0 counterfactual control on it: if any trigger-absent twin leaked (delivered a trigger
    with none inserted), the labels are untrustworthy -- the command prints the leak examples and
    exits non-zero **without writing the selection** (parity with the P1 pilot discipline). On
    success it writes the selection JSON and prints the achieved-vs-target ``subset_report``,
    with any shortfalls called out explicitly.
    """
    from trigger_audit.experiments.probe_detection.selection import (
        StratumTargets,
        select_probe_subset,
        subset_report,
        verify_subset_counterfactual,
        write_selected_trial_ids,
    )
    from trigger_audit.io.jsonl import read_jsonl_as
    from trigger_audit.schemas import SurvivalResult

    files = (
        sorted(survival_results.glob("*.jsonl"))
        if survival_results.is_dir()
        else [survival_results]
    )
    rows: list[SurvivalResult] = []
    for file in files:
        rows.extend(read_jsonl_as(file, SurvivalResult))
    if not rows:
        _fail("no survival result rows found")

    targets = StratumTargets(
        delivered_positive=delivered_positive,
        clean_negative=clean_negative,
        partial_survival_negative=partial_survival_negative,
        boundary_corruption=boundary_corruption,
        stratified_sample=stratified_sample,
    )
    selection = select_probe_subset(rows, targets, seed=seed)

    verdict = verify_subset_counterfactual(rows, selection)
    if not verdict.ok:
        console.print(f"[red]Gate 0 FAILED[/red]: {verdict.summary()}")
        console.print("Leak examples (trigger-absent twins that nonetheless delivered):")
        for example in verdict.leak_examples:
            console.print(f"  {example}")
        _fail(
            "refusing to write the selection: a counterfactual leak means the survival scorer "
            "is unsound and every downstream probe label is untrustworthy. Fix the scorer and "
            "rerun the survival wave before selecting a probe subset."
        )

    write_selected_trial_ids(out, selection)
    console.print(subset_report(selection))
    console.print(f"[green]wrote[/green] selection -> {out}")


@app.command("extract-activations")
def extract_activations_cmd(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Probe-detection experiment YAML (one cell)."
    ),
    device: str | None = typer.Option(
        None, help="Device override that wins over the config (e.g. cuda:0)."
    ),
) -> None:
    """Run the GPU extraction + probe pass for one cell (hf backend, populating the store).

    A thin wrapper over ``run_probe_experiment`` that forces ``extractor_backend='hf'`` and
    ``reuse_store=True`` (via ``model_copy``): the activation store is populated as the
    store-writing side effect, so a later CPU ``run-probe-experiment`` on the same cell reuses
    every layer and skips the forward passes. ``--device`` (when given) wins over the config's
    ``device``. Activation extraction is the only GPU phase of the probe wave.
    """
    from trigger_audit.config import load_config
    from trigger_audit.experiments.probe_detection import (
        ProbeDetectionExperimentConfig,
        run_probe_experiment,
    )

    cfg = load_config(config, ProbeDetectionExperimentConfig)
    updates: dict[str, object] = {"extractor_backend": "hf", "reuse_store": True}
    if device is not None:
        updates["device"] = device
    cfg = cfg.model_copy(update=updates)
    result = run_probe_experiment(cfg)
    _print_probe_result(result)
    console.print(f"[green]extracted[/green] -> {cfg.activations_dir}")


@app.command("expand-probe-grid")
def expand_probe_grid_cmd(
    axes: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="ProbeGridAxes YAML (one experiment tier's grid)."
    ),
    out_dir: Path = typer.Option(..., help="Directory to write the generated per-cell configs."),
) -> None:
    """Expand an axes YAML into per-cell probe configs; print count + the Slurm array range.

    Loads a :class:`ProbeGridAxes`, takes the Cartesian product of its axes, writes one loadable
    ``ProbeDetectionExperimentConfig`` YAML per cell (deterministic, content-derived ids -- same
    axes yield the same ids on re-expansion), then prints the cell count and the
    ``--array=0-N`` range to hand to the Slurm templates.
    """
    from trigger_audit.config import load_config
    from trigger_audit.experiments.probe_detection.grid import (
        ProbeGridAxes,
        expand_probe_grid,
        write_probe_configs,
    )

    axes_model = load_config(axes, ProbeGridAxes)
    configs = expand_probe_grid(axes_model)
    paths = write_probe_configs(configs, out_dir)
    console.print(f"[green]expanded[/green] {len(configs)} probe cell(s) -> {out_dir}")
    if paths:
        console.print(f"Slurm array range for this grid: [bold]--array=0-{len(paths) - 1}[/bold]")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
