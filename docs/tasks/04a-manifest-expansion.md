# Task 04a — Manifest expansion: grid over the verified primitives (delegated)

**Audience:** an implementing agent (Claude). Precise architecture over hand-holding.

**Goal:** turn the hand-wired Trial 0–3 drivers into a **manifest-driven grid**. One base × one
trigger × two positions × four policy ids × one model (Qwen/Qwen3-0.6B) = **8 rows**, where every
row reproduces a result already independently verified in Trials 2–3. The value is exactly that:
any mismatch localizes to the expansion/runner glue, because every policy and position underneath
is already proven. Read [`RUNNING_EXPERIMENTS.md`](../../RUNNING_EXPERIMENTS.md).

## Architecture decisions (reconciliations — implement as stated)

- **Reuse `TrialSpec`** (`schemas/trials.py`); its `pipeline_policy` field **is** the `policy_id`. Do not add a new schema or rename the field.
- **Reuse the slot-aware `TriggerInserter`** (`pipelines/trigger_insertion.py`) for the slot-form base — it already fills the slot matching a position and blanks the others. Do **not** use the lightweight `prompts.insert_trigger` here.
- **Composition path is canonical.** The manifest runner executes each row through the *verified* Trial 3 machinery (`TriggerInserter` → `ComposedPipeline` → `score_from_layers`). The early `experiments/.../manifest.py::ManifestBuilder`, `SurvivalShardRunner`, and `PipelinePolicyConfig` were pre-composition scaffolding and are **superseded** by this task; leave them in place but do not build on them (a later cleanup can remove them).

## Your tasks

### 1. Slot-form base conversation — `data/base_conversations/base_conversations_000.jsonl` (new)

One `BaseConversation`, `base_id: "conv_000001"`, `conversation_type: "multi_turn_chat"`, the same
6 messages as Trial Two but with slots at the start of the turns and a `slot_locations` field:

- message [1] content: `"{{OLD_TURN_SLOT}} I'm seeing a performance issue in my note-taking app; the autosave feels slow."`
- message [5] content: `"{{RECENT_TURN_SLOT}} Please give me the top three likely root causes."`
- `slot_locations`: `[{"slot": "{{OLD_TURN_SLOT}}", "message_index": 1}, {"slot": "{{RECENT_TURN_SLOT}}", "message_index": 5}]`

Slots sit at the **start** of each turn so a filled trigger lands at the message prefix — matching
the placement Trials 2/3 verified. `TriggerInserter.insert(base, trigger, OLD_TURN|RECENT_TURN)`
fills the matching slot and blanks the other. Validate with `trigger-audit validate-jsonl ... --schema base_conversation`.

### 2. Policy registry — `pipelines/policy_registry.py` (new)

`resolve_policy(policy_id: str) -> list[StagedPolicy]`, config-driven from
`configs/pipeline_policies.example.yaml` (**replace** its current contents with the composite
format below; the old memory/truncation-name format is unused by the verified trials):

```yaml
policies:
  - id: none
    steps: []
  - id: keep_recent_messages
    steps: [{type: keep_recent_messages, keep_last_n: 2}]
  - id: keep_recent_messages+head_truncation_generous
    steps: [{type: keep_recent_messages, keep_last_n: 2}, {type: head_truncation, context_length_target: 40960}]
  - id: keep_recent_messages+head_truncation_tight
    steps: [{type: keep_recent_messages, keep_last_n: 2}, {type: head_truncation, context_length_target: 20}]  # derived: Trial 3C, Qwen3-0.6B
```

Map step `type` → staged policy: `keep_recent_messages` → `KeepRecentMessagesPolicy(keep_last_n=...)`, `head_truncation` → `HeadTruncationPolicy(context_length_target=...)`. The `20` is Trial 3C's **derived** budget for Qwen3-0.6B (keep the provenance comment); do not invent numbers. Unknown `policy_id`/`type` raises.

### 3. `expand_manifest` — `io/manifest.py` (new)

`expand_manifest(base_ids, trigger_ids, positions, policy_ids, model_ids) -> list[TrialSpec]` — the
Cartesian product. Stable id (add to `util/ids.py`):

```python
trial_id = "trial_" + hashlib.sha256(f"{base_id}|{trigger_id}|{trigger_position}|{policy_id}|{model_id}".encode()).hexdigest()[:12]
```

Order-independent and stable under re-expansion. Tests: cardinality (8 for the grid below); ids
are stable across re-runs and unique per row.

### 4. Manifest runner — `experiments/survivability_audit/manifest_runner.py` (new)

`run_trial(trial: TrialSpec, *, base: BaseConversation, trigger: TriggerSpec, tokenizer_adapter: TokenizerAdapter) -> SurvivalResult`:
1. `raw, _ = TriggerInserter().insert(base, trigger, trial.trigger_position)`.
2. `policies = resolve_policy(trial.pipeline_policy)`.
3. `ComposedPipeline(policies, renderer=ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True), adapter=adapter).run(raw)`.
4. `score_from_layers(trial, trigger, input_ids=result.final_token_ids, trigger_ids=adapter.encode(trigger.text, add_special_tokens=False), post_template_text=result.post_template_text, final_text=adapter.decode(result.final_token_ids), raw_present=<computed from raw>, post_pipeline_present=<computed from result.post_messages>, pipeline_meta=result.metadata)`.

This is `run_trial_three` generalized (any position via `TriggerInserter`, any policy via the registry). Inject the adapter.

### 5. CLI — implement the `trigger-audit build-manifest` stub to call `expand_manifest` and write the manifest JSONL (reuse `io.jsonl.write_jsonl`).

### 6. Acceptance test — `tests/test_manifest_expansion.py`

Expand the grid, run all 8 rows through `run_trial` (offline `SimpleWhitespaceTokenizerAdapter` for the suite; the supervisor cross-checks with the real tokenizer), and assert:

| # | position | policy_id | expected survival_class | maps to |
|---|----------|-----------|-------------------------|---------|
| 1 | old_turn | none | exact_survival | positive control |
| 2 | recent_turn | none | exact_survival | positive control |
| 3 | old_turn | keep_recent_messages | no_survival | Trial 2A |
| 4 | recent_turn | keep_recent_messages | exact_survival | Trial 2B |
| 5 | old_turn | keep_recent_messages+head_truncation_generous | no_survival | Trial 3A |
| 6 | recent_turn | keep_recent_messages+head_truncation_generous | exact_survival | Trial 3B |
| 7 | recent_turn | keep_recent_messages+head_truncation_tight | no_survival | Trial 3C |
| 8 | old_turn | keep_recent_messages+head_truncation_tight | no_survival | budget-independence |

Grid for the test: `base=["conv_000001"]`, `trigger=["rand_001"]`, `positions=[old_turn, recent_turn]`, the 4 `policy_ids`, `model=["Qwen/Qwen3-0.6B"]`.

**Row 8 note (flagged, same as Trial 3A):** the source spec says "row 8 `final_input_ids == row 5`". That literal equality does **not** hold — for old_turn the post-memory sequence is shorter and the tight budget (20) still truncates it, so row 8's ids differ from row 5's. The correct, budget-independent invariant is that **row 8 and row 5 have the same trigger outcome** (`no_survival`, `failure_stage=memory_policy_dropped`, trigger absent) because the message is gone at Layer 2. Assert that. (Consistent with the Trial 3A reinterpretation already flagged to Saki.)

## Constraints & verification

- Reuse `TriggerInserter`, `ComposedPipeline`, `score_from_layers`, the staged policies — no duplication. One header comment per function/class; type hints throughout.
- Export new public names; pass the full gate (`pytest`, `ruff check src tests`, `ruff format --check src tests`, `mypy`); Trials 0–3 stay green.
- Supervisor verifies: gate green; the 8-row table reproduced with the **real** Qwen3-0.6B tokenizer; trial ids stable across two expansions; row 8 ≡ row 5 trigger outcome.
