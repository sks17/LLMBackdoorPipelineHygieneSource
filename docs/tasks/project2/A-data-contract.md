# Spec A — Persist `final_token_ids` (the probe join producer)

**Prerequisite P1 / continuity B1 (S1).** The probe runner reads a `final_tokens_path` JSONL of
`{trial_id, final_token_ids}` (`experiments/probe_detection/runner.py:328-331`), but Project 1
**drops** the final token ids from the persisted `SurvivalResult` — it keeps
`final_prompt_token_count`, `final_prompt_text_path`, `trigger_final_token_start/end`, and no
`final_token_ids` (`schemas/results.py:70-86`). The data exists in-pipeline at scoring time
(`ctx.final_token_ids` in `experiments/survivability_audit/runner.py:153`; `result.final_token_ids`
in `manifest_runner.py`) but is thrown away. Without persisting it, a real probe wave has a join
input for <2% of trials (only the `--log-prompts` sample). This is the #1 blocker for Tier 1+.

## Goal

Make the final token ids a first-class, persisted artifact — as a **sidecar `final_tokens.jsonl`**
(the primary producer the probe runner already consumes) and, optionally, inline on `SurvivalResult`
— for the trials we choose to persist (all, or a selected subset).

## Files you own (edit only these + add tests)

- `src/trigger_audit/schemas/results.py`
- `src/trigger_audit/experiments/survivability_audit/scorer.py`
- `src/trigger_audit/experiments/survivability_audit/runner.py`
- `src/trigger_audit/experiments/survivability_audit/manifest_runner.py`
- `src/trigger_audit/cli.py` — **only** the `run-survival-shard` command and its imports
- `src/trigger_audit/io/final_tokens.py` — **new**
- `tests/test_final_tokens_persistence.py` — **new**
- `src/trigger_audit/io/__init__.py` — add the re-export

Do NOT touch `probe_detection/*`, `schemas/probes.py`, or any analysis file (other components own
them).

## Design

### 1. Sidecar module `io/final_tokens.py` (primary producer)

```python
FinalTokensRow = TypedDict / pydantic model {"trial_id": str, "final_token_ids": list[int]}
```
Provide:
- `write_final_tokens(path: str | Path, rows: Iterable[tuple[str, Sequence[int]]]) -> int` — writes
  one JSON object per line `{"trial_id": ..., "final_token_ids": [...]}`; returns count. Reuse
  `io/jsonl.py::write_jsonl` semantics (atomic-ish write; create parent dirs). Empty ids are allowed
  (a template-incompatible trial has none) but skip rows with `final_token_ids is None`.
- `read_final_tokens(path: str | Path) -> dict[str, list[int]]` — the inverse, matching exactly what
  `run_probe_experiment` builds today (`{trial_id: [int, ...]}`). This lets the probe runner and any
  consumer share one loader.

Keep it dependency-light (stdlib + the existing `io.jsonl` helpers). This is the canonical format;
document that `final_tokens_path` in the probe config points at a file of these rows.

### 2. `SurvivalResult` inline field (secondary, defaulted)

Add to `SurvivalResult` (after `final_prompt_token_count`):
```python
final_token_ids: list[int] | None = None
```
Default `None` so every existing row and construction site still validates. Docstring: "The final
model-visible token ids (the probe's activation input). Populated only when persistence is requested
(it roughly doubles a result row's size); the sidecar `final_tokens.jsonl` is the primary producer —
see io/final_tokens.py." Do **not** add it to `RagDeliveryResult` in this task (out of scope).

### 3. Thread the ids through the scorer/result builder

The scorer already receives `final_ids`/`input_ids`. Extend the result-construction path so the ids
can be attached when requested, without changing default output:
- `SurvivalResultBuilder.build(...)` gains a keyword `final_token_ids: Sequence[int] | None = None`;
  when provided, set the new field (as a `list[int]`).
- `score_from_layers(...)` (manifest path) gains the same optional keyword and forwards it.
- `template_incompatible_result(...)` sets `final_token_ids=None` (no final tokens exist).
Keep every existing call site byte-compatible: the new argument is optional and defaults to not
attaching ids. Read the current signatures in `scorer.py` and match their style exactly.

### 4. Shard runner: produce the sidecar (and optionally inline)

In `SurvivalShardRunner`:
- Add constructor/`run` support for a `final_tokens_out: str | Path | None = None` and a boolean
  `persist_final_tokens_inline: bool = False`.
- During `run`, when `final_tokens_out` is set, collect `(trial.trial_id, ctx.final_token_ids or [])`
  for each successfully scored trial and write them with `write_final_tokens` after the loop (mirror
  how `survival_out`/`generation_out` are written). A template-incompatible trial contributes no
  final ids — record it with an empty list is acceptable, but prefer to **skip** trials whose
  `final_token_ids` is empty so the sidecar only carries joinable rows; document the choice.
- When `persist_final_tokens_inline` is True, pass `final_token_ids=ctx.final_token_ids` into the
  result builder so the inline field is populated too.
- Optional subset gating: accept `select_trial_ids: set[str] | None = None`; when provided, only
  persist final tokens (sidecar and inline) for trials in that set. This is how the stratified
  selector (component B) will restrict persistence to the probed subset. If wiring the set through
  cleanly is awkward, expose it but default `None` (persist all) — B/F will pass it via CLI later.

Do the same minimal wiring in `manifest_runner.py`'s `run_trial` return path only if it already
produces results in a loop that writes them; `manifest_runner.run_trial` returns a single
`SurvivalResult`, so just thread the optional `final_token_ids` into `score_from_layers` there and
let the caller (the manifest runner loop, if any) decide on the sidecar. If there is no loop/writer
in that file, only add the optional forwarding argument and leave persistence to the shard runner.

### 5. CLI `run-survival-shard`

Add options (Typer, matching the existing style in `cli.py`):
- `--final-tokens-out PATH` (default `None`) — when set, write the `final_tokens.jsonl` sidecar.
- `--persist-final-tokens/--no-persist-final-tokens` (default off) — inline persistence toggle.
Pass them into `SurvivalShardRunner`. Update the command docstring. Print a line reporting how many
final-token rows were written when the sidecar is enabled.

## Tests (`tests/test_final_tokens_persistence.py`)

Use the offline reference/simple tokenizer path already used by other survival tests (see
`tests/conftest.py` and `test_pipeline_end_to_end.py` for fixtures). Cover:
1. `write_final_tokens` + `read_final_tokens` round-trip returns the exact `{trial_id: ids}` map,
   including a multi-token id list; the format matches what `run_probe_experiment` expects (a dict of
   `trial_id -> list[int]`).
2. A small shard run with `--final-tokens-out` / `final_tokens_out=` produces a sidecar whose ids,
   for a delivered trigger, **contain** the trigger's token-id subsequence (join is usable). Assert
   the sidecar trial_ids are a subset of the survival-result trial_ids.
3. `persist_final_tokens_inline=True` populates `SurvivalResult.final_token_ids`; default leaves it
   `None`.
4. Default behavior (no options) writes **no** sidecar and leaves the inline field `None` — proving
   the change is non-breaking for existing runs.
5. A template-incompatible / empty-final-ids trial is handled per your documented choice (skipped
   from the sidecar, or empty-listed) without crashing.

## Acceptance

- `pytest -q` green (your new file + `tests/test_survival_scoring.py`,
  `tests/test_pipeline_end_to_end.py`, `tests/test_schemas.py`, `tests/test_probe_end_to_end.py`
  all still pass).
- `ruff check .`, `ruff format .`, `mypy` clean.
- Report the commands you ran, outputs, and any call sites you had to update.
