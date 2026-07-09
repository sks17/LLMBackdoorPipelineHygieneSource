# Task 01 — Trial One connective tissue (delegated)

**Audience:** an implementing agent (Claude). You can read code, run tests, and reason about
behavior — the scoping favors clear responsibilities over exhaustive detail.

**Goal:** finish the Trial One vertical slice — naive **head truncation**, prefix vs end — by
building the inserter extension and the driver. Read
[`RUNNING_EXPERIMENTS.md`](../../RUNNING_EXPERIMENTS.md) (Trial One entry) for the science.

Trial One holds Trial Zero constant and manipulates **only `trigger_position`**. Head truncation
applies at the Layer 3 → Layer 4 boundary: keep the full templated text (Layer 3), tokenize it,
then keep only the last `context_length_target` tokens (drop from the front). Prefix triggers get
destroyed; end triggers survive.

## Already done by the supervisor — do NOT rebuild or edit

- `src/trigger_audit/pipelines/truncation.py` — `HeadTruncation().apply(ids, budget) -> TruncationOutcome(kept_ids, dropped_head, dropped_tail)` (keeps the last `budget` tokens).
- `src/trigger_audit/experiments/survivability_audit/scorer.py` — `score_from_layers(...)` now takes both `post_template_text` (Layer 3, drives `post_template_trigger_present`) and `final_text` (Layer 4 decoded, drives exact-string survival). **For truncation you MUST pass `final_text=adapter.decode(kept_ids)`.**
- `src/trigger_audit/experiments/survivability_audit/trial_one_spec.py` — `PIPELINE_POLICY`, `DEFAULT_MARGIN`, `expected_prefix_messages()`, `expected_end_messages()`, `derive_context_length_target(adapter, margin=...)`, `trial_spec(position, target)`. **Import from here; do not restate.**
- `scripts/capture_trial_one_fixture.py`, `tests/fixtures/trial_one/`, `tests/test_trial_one.py`, `tests/test_token_search.py`.
- Trial Zero remains owned as before (`trial_zero_spec.py`, `trial_zero.py`, its fixtures/tests).

## Your tasks

### 1. Extend `insert_trigger` to support `"end"` — `src/trigger_audit/prompts/trigger_insertion.py`

- Add support for `position="end"` (append the trigger after the message content), delegating to `pipelines.trigger_insertion.place_in_content(content, trigger_text, TriggerPosition.END)`. Keep `"prefix"` working; keep raising `NotImplementedError` for still-unsupported positions.
- **Hard acceptance:** `insert_trigger(trial_one_spec.expected_prefix_messages.__self__...)` — concretely, `insert_trigger(trial_zero_spec.base_messages(), trial_zero_spec.TRIGGER.text, "end") == trial_one_spec.expected_end_messages()`.
- Extend `tests/test_insert_trigger.py`: the `"end"` case above, and that the input is not mutated.

### 2. Trial One driver — `src/trigger_audit/experiments/survivability_audit/trial_one.py` (new)

```python
def run_trial_one(
    *, tokenizer_adapter: TokenizerAdapter, trigger_position: TriggerPosition, context_length_target: int
) -> SurvivalResult:
```

Compose the slice:
1. `messages = trial_zero_spec.base_messages()`; insert the trigger at `trigger_position` (map `PREFIX`→`"prefix"`, `END`→`"end"` for `insert_trigger`).
2. Render the **full** Layer 3 text with `ChatTemplateRenderer(adapter, enable_thinking=trial_zero_spec.ENABLE_THINKING, add_generation_prompt=trial_zero_spec.ADD_GENERATION_PROMPT)`.
3. `full_ids = adapter.encode(text, add_special_tokens=False)`.
4. `outcome = HeadTruncation().apply(full_ids, context_length_target)`.
5. `final_text = adapter.decode(outcome.kept_ids)`; `trigger_ids = adapter.encode(trial_zero_spec.TRIGGER.text, add_special_tokens=False)`.
6. Return `score_from_layers(trial_one_spec.trial_spec(trigger_position, context_length_target), trial_zero_spec.TRIGGER, input_ids=outcome.kept_ids, trigger_ids=trigger_ids, post_template_text=text, final_text=final_text, raw_present=True, post_pipeline_present=True, pipeline_meta={"truncation": {"policy": "truncate_head", "dropped_head": outcome.dropped_head, "dropped_tail": outcome.dropped_tail}})`.

Notes: `post_template_text` is the **full** text (so `post_template_trigger_present` is True for both variants); `final_text` is the **decoded truncated** text. The result's `pipeline_policy` label is `"head_truncation"` (from `trial_spec`); the truncation registry name in `pipeline_meta` is `"truncate_head"` — both are intentional. Inject the adapter; do not construct one inside.

### 3. Offline driver tests — `tests/test_trial_one_driver.py` (new)

Using `SimpleWhitespaceTokenizerAdapter` (offline), derive the budget with `trial_one_spec.derive_context_length_target(adapter)` and run both variants:
- prefix → `survival_class == NO_SURVIVAL`, `final_token_trigger_present is False`, `failure_stage == TRUNCATED_HEAD`, `post_template_trigger_present is True`;
- end → `survival_class == EXACT_SURVIVAL`, `final_token_trigger_present is True`, `post_template_trigger_present is True`;
- both: `final_prompt_token_count == derived target`.

## Constraints

- Reuse `HeadTruncation`, `score_from_layers`, `insert_trigger`, `place_in_content`, and `trial_one_spec` — no duplication.
- One header comment per function/class; type hints throughout; inline comments only for non-obvious logic.
- Update the experiment `__init__.py` to export `run_trial_one`.
- Before returning, pass the full gate: `pytest`, `ruff check src tests`, `ruff format --check src tests`, `mypy`.

## How the supervisor will verify

- Full gate green and `tests/test_trial_one.py` still passes unchanged.
- `insert_trigger(base_messages(), TRIGGER.text, "end") == expected_end_messages()`.
- `run_trial_one` with the **real** Qwen3-0.6B `HFTokenizerAdapter` reproduces the golden-fixture
  outcome: prefix → `no_survival`, end → `exact_survival`, both truncated to exactly
  `context_length_target` (26) tokens.
