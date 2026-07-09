"""Typer CLI for the trigger-audit harness.

Commands define the stable, future-facing interface. Heavy and experiment-specific modules are
imported inside command bodies so ``trigger-audit --help`` stays fast and dependency-light.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from trigger_audit import __version__

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


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
