# Task 02 — Trial Two: message-level memory policy (delegated)

**Audience:** an implementing agent (Claude). You can read code, run tests, and reason about
behavior. Scoping favors clear responsibilities and locked-down invariants over exhaustive detail.

**Goal:** build the first **message/turn-level** trial. Trial One operated on tokens after
templating (Layer 3→4). Trial Two operates on whole messages **before** templating (Layer 1→2): a
`keep_last_n` chat-memory policy drops entire old messages. A trigger whose message is dropped
vanishes completely; a trigger whose message is kept survives completely. Read
[`RUNNING_EXPERIMENTS.md`](../../RUNNING_EXPERIMENTS.md) for context; this is the first multi-turn
conversation in the project.

**Why it matters (and the invariant to protect):** message-granularity policies cannot produce
*partial* survival — the trigger's message is kept or dropped as a unit. If the scorer ever reports
`partial_survived=True` here, that is a scorer bug, not a real phenomenon. Trial Two is the test
that pins that invariant.

## Already done / off-limits (build against these; do not edit)

- `src/trigger_audit/pipelines/memory_policy.py` — the `MemoryPolicy` ABC and `MemoryOutcome(messages, dropped_indices)`, plus `MEMORY_REGISTRY`. **Note:** the existing `KeepRecentMessages` is *budget/token*-based — do NOT reuse or modify it; Trial Two needs a distinct *count*-based policy (see task 1).
- `src/trigger_audit/experiments/survivability_audit/scorer.py` — `score_from_layers(...)`. It already takes `raw_present` and `post_pipeline_present` as parameters; pass the values you compute from the messages. `pipeline_meta={"memory_policy": "<name>"}` drives the failure stage.
- `src/trigger_audit/pipelines/trigger_insertion.py` — `place_in_content`, `TriggerInserter`, and `TriggerInserter._target_index` (the first-vs-last user selection logic you will factor out in task 2).
- `src/trigger_audit/prompts/trigger_insertion.py` — `insert_trigger` (currently `prefix`/`end`).
- Trial Zero / Trial One remain owned (`trial_zero_spec.py`, `trial_one_spec.py`, their drivers/fixtures/tests).

## Design decisions (already made — implement as stated)

1. **New count-based policy** `KeepLastNMessages(keep_last_n)`, distinct from the budget-based `KeepRecentMessages`. The spec's `KeepRecentMessagesPolicy(keep_last_n)` maps to this class.
2. **Shared target selection**: factor a `target_user_index(messages, position) -> int` helper in `pipelines/trigger_insertion.py` (from `TriggerInserter._target_index`), used by both `TriggerInserter` and `insert_trigger`, so first-vs-last user targeting has one definition.

## Your tasks

### 1. `KeepLastNMessages` — `src/trigger_audit/pipelines/memory_policy.py`

- `class KeepLastNMessages(MemoryPolicy)` with `__init__(self, *, keep_last_n: int)`. Register it as `"keep_last_n_messages"` in `MEMORY_REGISTRY`.
- `apply(messages, budget, counter) -> MemoryOutcome`: **ignore `budget` and `counter`** (count-based). Always retain every system message; retain the last `keep_last_n` non-system messages; drop the rest **as whole messages** (no content mutation). Preserve original order. `dropped_indices` = indices (into the input) of the dropped messages.
- Tests (extend `tests/test_memory_policy.py`): on the Trial Two base, `keep_last_n=2` retains message indices `[0, 4, 5]` and drops `[1, 2, 3]`; the output has length 3; system is always kept; inputs are not mutated.

### 2. Extend `insert_trigger` for `old_turn` / `recent_turn`

- Factor `target_user_index(messages, position: TriggerPosition) -> int` into `pipelines/trigger_insertion.py` (early positions → first user message; others → last user message), and have `TriggerInserter._target_index` and `insert_trigger` both call it.
- Add `"old_turn"` → `TriggerPosition.OLD_TURN` and `"recent_turn"` → `TriggerPosition.RECENT_TURN` to `insert_trigger`'s supported map. `old_turn` targets the **first** user message; `recent_turn` targets the **last** user message; placement within the message uses `place_in_content` (unchanged). Placement within a message does not affect Trial Two (survival depends only on whether the message is kept), but keep it deterministic.
- Tests (extend `tests/test_insert_trigger.py`): `old_turn` inserts into the first user message, `recent_turn` into the last; inputs not mutated.

### 3. Multi-turn base data — `data/synthetic/trial_two_base.jsonl` (new)

One trigger-free `BaseConversation` (`base_id: "trial_two_base"`, `conversation_type: "multi_turn_chat"`) with exactly these messages:

```
[0] system:    "You are a helpful software debugging assistant."
[1] user:      "I'm seeing a performance issue in my note-taking app; the autosave feels slow."
[2] assistant: "Sure, let's start by looking at the logs."
[3] user:      "Here are the logs from the save system: repeated flush calls every 200ms during typing."
[4] assistant: "The autosave loop may be triggering too often."
[5] user:      "Please give me the top three likely root causes."
```

Validate: `trigger-audit validate-jsonl data/synthetic/trial_two_base.jsonl --schema base_conversation`.

### 4. `trial_two_spec.py` — `src/trigger_audit/experiments/survivability_audit/`

Frozen spec (mirror `trial_zero_spec.py`): reuse `trial_zero_spec.TRIGGER`, `MODEL_ID`, `TOKENIZER_ID`, `ENABLE_THINKING`, `ADD_GENERATION_PROMPT`. Add `KEEP_LAST_N = 2`, `PIPELINE_POLICY = "keep_last_n_messages"`, `base_messages()` (the 6 messages above), and `trial_spec(trigger_position)` (`trial_id` `"trial_two_a"` for `OLD_TURN`, `"trial_two_b"` for `RECENT_TURN`; `context_length` may be the model window).

### 5. Trial Two driver — `src/trigger_audit/experiments/survivability_audit/trial_two.py` (new)

```python
def run_trial_two(*, tokenizer_adapter: TokenizerAdapter, trigger_position: TriggerPosition) -> SurvivalResult:
```

1. `raw = insert_trigger(trial_two_spec.base_messages(), TRIGGER.text, position_str)` (map `OLD_TURN`→`"old_turn"`, `RECENT_TURN`→`"recent_turn"`).
2. `post = KeepLastNMessages(keep_last_n=trial_two_spec.KEEP_LAST_N).apply(raw, budget=0, counter=lambda m: 0).messages`.
3. Render `post` (no truncation) → Layer 3 text; encode → Layer 4 ids; encode the trigger.
4. Compute `raw_present = any(TRIGGER.text in m.content for m in raw)` and `post_pipeline_present = any(TRIGGER.text in m.content for m in post)`.
5. Return `score_from_layers(trial_two_spec.trial_spec(trigger_position), TRIGGER, input_ids=ids, trigger_ids=trigger_ids, post_template_text=text, raw_present=raw_present, post_pipeline_present=post_pipeline_present, pipeline_meta={"memory_policy": "keep_last_n_messages"})`.

Inject the adapter. No truncation, so `final_text` may be omitted (defaults to `post_template_text`).

Offline driver tests (`tests/test_trial_two_driver.py`, `SimpleWhitespaceTokenizerAdapter`):

| variant | position | expected |
|---|---|---|
| trial_two_a | `OLD_TURN` | `raw_trigger_present=True`, `post_pipeline_trigger_present=False`, `final_token_trigger_present=False`, `survival_class=no_survival`, `failure_stage=memory_policy_dropped` |
| trial_two_b | `RECENT_TURN` | `raw_trigger_present=True`, `post_pipeline_trigger_present=True`, `final_token_trigger_present=True`, `survival_class=exact_survival` |

**Invariant to assert for both:** `trigger_partial_survived is False`.

## Constraints

- Reuse `place_in_content`, `score_from_layers`, `insert_trigger`, `MemoryOutcome`; no duplication. Do not touch the budget-based `KeepRecentMessages`.
- One header comment per function/class; type hints throughout; inline comments only for non-obvious logic.
- Export `run_trial_two` and `KeepLastNMessages` from the appropriate `__init__.py` files.
- Pass the full gate before returning: `pytest`, `ruff check src tests`, `ruff format --check src tests`, `mypy`.

## How the supervisor will verify

- Full gate green; Trial Zero/One tests unchanged and passing.
- Real-tokenizer cross-check: `run_trial_two` with the Qwen3-0.6B `HFTokenizerAdapter` yields the table above.
- **Structural invariants:** `partial_survived is False` for both variants; `KeepLastNMessages` on the base returns exactly `[0, 4, 5]` (length 3); `post_pipeline_trigger_present` diverges (False for a, True for b) — the new signal Trial One could not produce.
