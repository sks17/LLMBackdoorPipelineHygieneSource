# Task 00 — Trial Zero connective tissue (delegated)

**Audience:** an implementing agent (Claude). You can read code, run the app, run tests, and
reason about behavior — so this brief prioritizes clear responsibilities over exhaustive detail.

**Goal:** finish the Trial Zero vertical slice by building the pieces that connect the base
conversation to the already-built scorer. Read [`RUNNING_EXPERIMENTS.md`](../../RUNNING_EXPERIMENTS.md)
for what Trial Zero is, and [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) for conventions.

## Already done by the supervisor — do NOT rebuild or edit these

These are correct, tested, and off-limits (treat as fixed contracts you build against):

- `src/trigger_audit/tokenization/token_search.py` — `find_subsequence(haystack, needle) -> tuple[int,int] | None` (half-open span; first match; empty needle → `(0,0)`; needle longer → `None`).
- `src/trigger_audit/tokenization/tokenizer_adapter.py` — `render_chat(..., *, enable_thinking: bool, ...)` is **required** (no default). Both adapters accept it.
- `src/trigger_audit/prompts/chat_template.py` — `ChatTemplateRenderer(adapter, *, enable_thinking, add_generation_prompt=True, chat_template=None)`.
- `src/trigger_audit/config/settings.py` — `ModelConfig.enable_thinking` is a required field.
- `src/trigger_audit/experiments/survivability_audit/scorer.py` — `score_from_layers(trial, trigger, *, input_ids, trigger_ids, post_template_text, raw_present, post_pipeline_present) -> SurvivalResult`.
- `src/trigger_audit/experiments/survivability_audit/trial_zero_spec.py` — the frozen spec: `MODEL_ID`, `TRIGGER`, `ENABLE_THINKING`, `base_messages()`, `expected_positive_messages()`, `to_payload()`, `trial_spec()`. **Import from here; do not restate these values.**
- `scripts/capture_trial_zero_fixture.py`, `tests/fixtures/trial_zero*/`, `tests/test_trial_zero.py`, `tests/test_token_search.py`.

## Your tasks

### 1. `insert_trigger` — `src/trigger_audit/prompts/trigger_insertion.py` (new)

```python
def insert_trigger(
    messages: list[ChatMessage], trigger_text: str, position: str = "prefix"
) -> list[ChatMessage]:
```

- Deterministic and pure: returns a **new** list; does not mutate the input.
- `position="prefix"` prepends `f"{trigger_text}\n\n"` to the content of the first `user` message.
- **Hard acceptance:** `insert_trigger(trial_zero_spec.base_messages(), trial_zero_spec.TRIGGER.text, "prefix")` must equal `trial_zero_spec.expected_positive_messages()`.
- Do not duplicate placement logic that already exists in `pipelines/trigger_insertion.py::TriggerInserter`. Either delegate to it or factor a shared helper. Only `"prefix"` is required now; other positions may raise `NotImplementedError` with a clear message.
- Tests (new file, e.g. `tests/test_insert_trigger.py`): prefix inserts correctly; input list not mutated; the hard-acceptance equality above.

### 2. Trial Zero base data — `data/synthetic/trial_zero_base.jsonl` (new)

- One `BaseConversation` row (see `docs/DATA_CONTRACTS.md`) that is **trigger-free** and whose messages equal `trial_zero_spec.base_messages()` (`base_id: "trial_zero_base"`).
- Validate it: `trigger-audit validate-jsonl data/synthetic/trial_zero_base.jsonl --schema base_conversation`.

### 3. Trial Zero driver — `src/trigger_audit/experiments/survivability_audit/trial_zero.py` (new)

```python
def run_trial_zero(*, tokenizer_adapter: TokenizerAdapter, insert: bool = True) -> SurvivalResult:
```

- Compose the slice using the frozen spec and the pieces above:
  1. `messages = trial_zero_spec.base_messages()`; if `insert`, `messages = insert_trigger(messages, TRIGGER.text, "prefix")`.
  2. Render: `ChatTemplateRenderer(tokenizer_adapter, enable_thinking=trial_zero_spec.ENABLE_THINKING, add_generation_prompt=trial_zero_spec.ADD_GENERATION_PROMPT).render(messages)`.
  3. `input_ids = tokenizer_adapter.encode(text, add_special_tokens=False)`; `trigger_ids = tokenizer_adapter.encode(TRIGGER.text, add_special_tokens=False)`.
  4. Return `score_from_layers(trial_zero_spec.trial_spec(), TRIGGER, input_ids=input_ids, trigger_ids=trigger_ids, post_template_text=text, raw_present=insert, post_pipeline_present=insert)`.
- Inject the tokenizer adapter (do not construct one inside) so it runs offline in tests with `SimpleWhitespaceTokenizerAdapter` and in production with `HFTokenizerAdapter`.
- Tests (new file `tests/test_trial_zero_driver.py`, offline via `SimpleWhitespaceTokenizerAdapter`): `insert=True` → `survival_class == SurvivalClass.EXACT_SURVIVAL`; `insert=False` (negative control) → `SurvivalClass.NO_SURVIVAL`.

## Constraints

- Match scaffold conventions: one header comment per function/class; inline comments only for non-obvious logic; type hints throughout.
- Reuse — do not re-implement `find_subsequence`, `score_from_layers`, the adapters, or placement logic.
- Update `prompts/__init__.py` and the experiment `__init__.py` to export new public symbols.
- Before returning, run and pass the full gate: `pytest`, `ruff check src tests`, `ruff format --check src tests`, `mypy`.

## How the supervisor will verify

- The full gate is green, and `tests/test_trial_zero.py` still passes unchanged.
- `insert_trigger(base_messages(), TRIGGER.text, "prefix") == expected_positive_messages()`.
- `run_trial_zero` with the **real** Qwen3-0.6B `HFTokenizerAdapter` reproduces the golden fixture
  result (`exact_survival` positive, `no_survival` negative) — the supervisor runs this against
  `tests/fixtures/trial_zero/`.
