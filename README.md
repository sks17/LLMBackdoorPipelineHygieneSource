# trigger-audit

A prompt-survivability / trigger-delivery audit for LLM context pipelines.

**Core question.** When a harmless canary trigger is placed in raw user input, does it survive the
real prompt pipeline — chat templating, truncation, memory policies, summarization, RAG packing,
tokenization — and reach the final model-visible input?

This separates *delivery* failure from *model* robustness. A trigger that never reaches the model can
look like backdoor robustness when it is really the context-management policy deleting the trigger;
which triggers are lost is a predictable function of the policy, not the model.

> **Safety.** This harness uses only harmless canary strings (e.g. `CANARY_TRIGGER_7F3XQ`). It audits
> the pipeline; it does not construct harmful payloads or model behavior.

## Install

`uv` is preferred; a `venv` + `pip` fallback works anywhere, including CPU-only cluster login nodes.

```bash
# uv (preferred)
uv venv && uv pip install -e ".[hf,analysis]"

# venv + pip
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -e ".[hf,analysis]"
```

The `hf` extra (tokenizers/datasets) is deliberately **torch-free**: delivery is tokenizer/template
mechanics, so the whole audit runs on CPU with no GPU stack. `analysis` adds scipy/statsmodels/
matplotlib for the tables and figures. `torch`/`accelerate` live in the optional `generate` extra,
needed only for the pinned-NLI semantic-survival measurement.

## Quickstart (offline, no model downloads)

The package ships a dependency-free reference tokenizer (`--backend simple`), so the full pipeline
runs offline against the sample inputs under `data/`:

```bash
# 1. Expand the example grid into a trial manifest + shards (writes to data/shards/)
trigger-audit build-manifest configs/experiment_survivability.example.yaml

# 2. Run the shard through the pipeline and score survival with the offline tokenizer
trigger-audit run-survival-shard data/shards/simple-whitespace_shard_0000.jsonl \
  --models-config configs/models.example.yaml \
  --policies-config configs/prod/policies.prod.yaml \
  --base-conversations data/base_conversations/base_conversations_000.jsonl \
  --triggers data/triggers/triggers.jsonl \
  --survival-out outputs/survival_results/simple-whitespace_shard_0000.jsonl \
  --backend simple

# 3. Aggregate into a per-policy / per-position survival table
trigger-audit score-survival outputs/survival_results/
```

`build-manifest` prints the shard filenames and the Slurm `--array` range it produced. The example
grid is wired to the `simple-whitespace` reference model, so it runs with no model downloads.

## Full grid (cluster)

The reference run is **916,200 trials** across 4 tokenizers, 4 data arms (synthetic + long-doc + real
conversation logs), 5 policies, 7 positions, 3 budgets, 5 triggers, and a counterfactual twin. It runs
as a CPU-only Slurm array using `scripts/slurm/run_survival_shard.slurm`; `build-manifest` assembles
and shards the grid and prints the `--array` range. The counterfactual control (trigger-absent twins)
must show zero leaks for the run to be valid. Aggregated result tables and figures are not included in
this repository.

## Quality checks

```bash
pytest                 # offline suite: reference tokenizer + golden fixtures, no network or GPU
ruff check .           # lint
ruff format --check .  # formatting
mypy src               # type check
trigger-audit --help   # CLI
```

## Layout

```
src/trigger_audit/     # the package
  config/  schemas/  io/  tokenization/  prompts/  pipelines/  scoring/  analysis/  generation/
  experiments/survivability_audit/   experiments/rag_survival/
  cli.py
configs/               # example + pilot/prod YAML configs (placeholder values)
scripts/               # pilot + analysis helpers; scripts/slurm/ job-array template
data/                  # curated sample inputs (triggers, base conversations, corpora)
tests/                 # offline pytest suite
```

## Citation

```bibtex
@software{singh_trigger_audit,
  author  = {Saksham Singh},
  title   = {trigger-audit: a prompt-survivability / trigger-delivery audit for LLM context pipelines},
  url     = {https://github.com/sks17/LLMBackdoorPipelineHygeine},
  version = {0.1.0}
}
```

## License

[MIT](LICENSE) © 2026 Saksham Singh.
