# Architecture

`trigger_audit` separates **shared infrastructure** (reusable across all experiments) from
**experiment code** (which may branch from perfect abstraction). The shared layers form a small,
composable pipeline; experiments wire those pieces together and own their result schemas.

## The four logged layers

The central idea is that the model does not see "the raw prompt" — it sees token IDs after
templating, trimming, and packing. Every trial logs four layers so any failure is attributable to
a specific stage:

1. **Raw messages** — raw logical messages, trigger already inserted.
2. **Post-pipeline messages** — after the memory/trimming policy.
3. **Post-template text** — after the model-specific chat template.
4. **Final token IDs** — what the model actually consumes (after token-level truncation).

These map to `PipelineContext` fields: `raw_messages`, `messages`, `rendered_prompt`,
`final_token_ids`.

## Module map (`src/trigger_audit/`)

| Module | Responsibility | Reusable? |
|--------|----------------|-----------|
| `schemas/` | Pydantic data contracts: `ChatMessage`, `BaseConversation`, `TriggerSpec`, `TrialSpec`, `SurvivalResult`, `GenerationResult` | shared |
| `config/` | Typed config models (`ModelConfig`, `PipelinePolicyConfig`, `GenerationConfig`, `PathsConfig`) + YAML loaders | shared |
| `io/` | JSONL read/write, path resolution, id-keyed stores | shared |
| `tokenization/` | `TokenizerAdapter` (HF + dependency-free reference), token-subsequence search | shared |
| `prompts/` | `ChatTemplateRenderer` (Layer 3), `PromptLogger` (sampled final-prompt persistence) | shared |
| `pipelines/` | `Pipeline`/`PipelineStep`/`PipelineContext`/`Registry`; truncation + memory policies; trigger inserter; step wrappers | shared |
| `scoring/` | `SurvivalScorer` / `TokenSurvivalScorer` → `SurvivalAssessment` (did tokens survive, where) | shared |
| `util/` | Stable ids, logging | shared |
| `experiments/survivability_audit/` | `SurvivabilityExperimentConfig`, `ManifestBuilder`, `SurvivalShardRunner`, `SurvivalResultBuilder`, aggregation | experiment |
| `cli.py` | Typer entry point dispatching to the above | shared shell |

No module is a God module: each file has a single responsibility, and orchestration is expressed
as a short `Pipeline` of small steps rather than one large script.

## The pipeline abstraction

```
PipelineContext.from_messages(base.messages)
  → TriggerInsertionStep   (insert trigger; snapshot Layer 1)
  → MemoryPolicyStep       (message-level trim/summarize; Layer 2)
  → ChatTemplateStep       (render + tokenize; Layers 3 and 4)
  → TruncationStep         (token-level truncation; Layer 4)
  → TokenSurvivalScorer    (assess) → SurvivalResultBuilder (classify) → SurvivalResult
```

- **Memory policies** (`pipelines/memory_policy.py`) operate on *messages* before templating.
- **Truncation policies** (`pipelines/truncation.py`) operate on the *final token sequence* after
  templating. They are deliberately separate abstractions, because they model different real
  behaviors and run at different layers.
- Policies are pure and registry-resolved by name (`MEMORY_REGISTRY`, `TRUNCATION_REGISTRY`), so a
  config string like `keep_recent_messages` maps to a class without `if/elif` chains.

## Tokenizer adapters

`TokenizerAdapter` has two implementations behind one interface:

- `HFTokenizerAdapter` — the production path; `transformers` is imported lazily so the package
  works without the `hf` extra (important on CPU-only login nodes).
- `SimpleWhitespaceTokenizerAdapter` — a deterministic, dependency-free reference tokenizer that
  makes the whole pipeline runnable and unit-testable offline (`--backend simple`). It is **not** a
  real BPE tokenizer and is not used for measurement.

## Extending for the five follow-on experiments

The shared core is the extension point. Anticipated additions, none of which require changing
existing modules:

- **RAG survival** — add `RetrievalStep`, `RerankStep`, `CompressionStep`, `PackingStep` under
  `pipelines/`; add a `experiments/rag_survival/` package with its own manifest/runner/scorer and
  result schema (retrieval/packing/compression survival stages).
- **Multi-turn / distributed triggers** — extend `TriggerInserter` and add a real summarizer to
  `SummarizeOldMessages` to study semantic survival.
- **Behavioral activation** — implement `SurvivalShardRunner._maybe_generate` behind the `hf`
  extra; populate `GenerationResult`.
- **Framework comparison** — add adapter steps wrapping LangChain/LlamaIndex trimming as
  alternative `MemoryPolicy`/steps, compared against the in-house policies.

Experiment-specific divergence is expected and acceptable; shared infrastructure must stay generic.

## Vendored reference docs

`HyakDocs/`, `docs-main/` (LangChain docs), and `hub-docs-main/` (Hugging Face Hub docs) are
vendored references. They are excluded from linting, formatting, and packaging, and are never
edited by this project.
