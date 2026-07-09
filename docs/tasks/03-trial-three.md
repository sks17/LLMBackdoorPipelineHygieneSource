# Task 03 — Trial Three: composing memory + truncation via a staged interface (delegated)

**Audience:** an implementing agent (Claude). Scoping favors a precise architecture and locked
invariants over hand-holding.

**Goal:** build the project's first *composed* pipeline — a message-level memory policy **and** a
token-level truncation policy applied in the correct order via a **shared staged interface**, where
execution order comes from each policy's `.stage`, not its position in the declared list. The
scientific payoff is the interaction effect: a message can be "kept" by memory and still not reach
the model because truncation cuts it (`post_pipeline_trigger_present=True` but
`final_token_trigger_present=False`) — a failure neither policy produces alone.

Read [`RUNNING_EXPERIMENTS.md`](../../RUNNING_EXPERIMENTS.md) (Trials One and Two) for context.

## Architecture (the new abstraction) — implement exactly this

New module `src/trigger_audit/pipelines/composition.py`:

```python
class Stage(str, Enum):
    PRE_TEMPLATE = "pre_template"     # Layer 1->2: operates on messages, before templating
    POST_TEMPLATE = "post_template"   # Layer 3->4: operates on token ids, after templating

@dataclass
class CompositionContext:
    messages: list[ChatMessage]
    token_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class StagedPolicy(ABC):
    """A composition-friendly policy that knows its stage and holds its own config."""
    stage: Stage
    @abstractmethod
    def apply(self, ctx: CompositionContext) -> None: ...   # mutate ctx in place at its stage

class KeepRecentMessagesPolicy(StagedPolicy):   # stage = PRE_TEMPLATE
    def __init__(self, *, keep_last_n: int) -> None: ...     # wraps KeepLastNMessages (count-based)

class HeadTruncationPolicy(StagedPolicy):       # stage = POST_TEMPLATE
    def __init__(self, *, context_length_target: int) -> None: ...   # wraps HeadTruncation

@dataclass(frozen=True)
class CompositionResult:
    post_messages: list[ChatMessage]     # Layer 2
    post_template_text: str              # Layer 3 (post-memory, pre-truncation)
    final_token_ids: list[int]           # Layer 4 (post-truncation)
    metadata: dict[str, Any]

class ComposedPipeline:
    def __init__(self, policies: Sequence[StagedPolicy], *, renderer: ChatTemplateRenderer, adapter: TokenizerAdapter) -> None: ...
    def run(self, messages: Sequence[ChatMessage]) -> CompositionResult: ...
```

`ComposedPipeline.run` contract — **this is the behavior under test**:
1. Copy `messages` into a `CompositionContext`.
2. `pre = [p for p in policies if p.stage is Stage.PRE_TEMPLATE]` — apply each (order preserved).
3. Render `ctx.messages` with `renderer` → `post_template_text`; `ctx.token_ids = adapter.encode(text, add_special_tokens=False)`.
4. `post = [p for p in policies if p.stage is Stage.POST_TEMPLATE]` — apply each (order preserved).
5. Return `CompositionResult`.

Because the two stages are selected by `.stage` (filtering preserves within-stage order but ignores
cross-stage position), **reversing the declared policy list must not change the result**. The two
staged policies wrap the existing `KeepLastNMessages` / `HeadTruncation` and set metadata:
`KeepRecentMessagesPolicy` → `metadata["memory_policy"]`; `HeadTruncationPolicy` →
`metadata["truncation"] = {"policy": "truncate_head", "dropped_head": ..., "dropped_tail": ...}`.
Do not reimplement the truncation/memory math — delegate to the existing classes.

## Design decisions already made (implement as stated)

1. **`recent_turn` becomes a prefix-of-message placement.** For Trial Three (c) to work, the trigger
   in the recent turn must sit *before* the user's question so a tight tail-keeping head truncation
   can drop the trigger while keeping the question and generation prompt. In
   `pipelines/trigger_insertion.py`, decouple *which message* from *where in the message*: keep
   `target_user_index` mapping `RECENT_TURN` → **last** user message, but change `place_in_content`
   so `RECENT_TURN` uses **prefix** placement (like `OLD_TURN`). Trials Zero/One/Two must stay green
   (Trial Two is placement-agnostic; `test_insert_trigger` only checks membership for turns).
2. **Base conversation = Trial Two's**, unchanged. Reuse `trial_two_spec.base_messages()`.
3. **Tight budget derivation for (c) — never hardcode.** Run trial_three_b (recent_turn, generous
   budget → no truncation) first. From its `SurvivalResult` take `trigger_final_token_end` (E) and
   `final_prompt_token_count` (T). Set `context_length_target = T - E` (this is the spec's
   `T - trigger_start + margin` with the margin resolved to fully contain the trigger span). Head
   truncation then keeps the last `T - E` tokens `[E, T)` — the question tail and generation prompt
   — and drops `[0, E)`, which contains the entire trigger. Because the whole span is dropped,
   `partial_survived` stays False. Put this in `trial_three_spec.derive_tight_budget(result_b)`.
4. **Generous budget** = a large non-binding constant (reuse `trial_zero_spec.CONTEXT_LENGTH`).

## Already done / off-limits

- `pipelines/truncation.py` (`HeadTruncation`), `pipelines/memory_policy.py` (`KeepLastNMessages`), `scorer.py` (`score_from_layers`, incl. `final_text`), `ChatTemplateRenderer`, `insert_trigger`. Reuse; do not fork.
- Trials Zero/One/Two remain owned (their specs/drivers/fixtures/tests).

## Your tasks

1. **`pipelines/composition.py`** — the module above. Export its public names from `pipelines/__init__.py`. Unit tests (`tests/test_composition.py`): pre runs before post regardless of declared order; **reversing the policy list yields an identical `CompositionResult`**; a `POST_TEMPLATE` policy applied with `token_ids is None` raises.
2. **`recent_turn` placement change** in `pipelines/trigger_insertion.py` (decision 1) + update any affected assertions; re-run the gate.
3. **`trial_three_spec.py`** — reuse Trial Two base + Trial Zero constants; `KEEP_LAST_N = 2`; `GENEROUS_BUDGET`; `derive_tight_budget(result_b)`; `trial_spec(trial_id, trigger_position, context_length_target)`.
4. **`trial_three.py`** — `run_trial_three(*, tokenizer_adapter, trigger_position, context_length_target, reverse_chain=False) -> SurvivalResult`. Insert the trigger; build `policy_chain = [HeadTruncationPolicy(context_length_target=...), KeepRecentMessagesPolicy(keep_last_n=KEEP_LAST_N)]` (declared post-then-pre on purpose); reverse it when `reverse_chain`; run `ComposedPipeline`; then `score_from_layers(..., post_template_text=result.post_template_text, final_text=adapter.decode(result.final_token_ids), input_ids=result.final_token_ids, raw_present=<computed>, post_pipeline_present=<computed from result.post_messages>, pipeline_meta=result.metadata)`.
5. **Driver tests** (`tests/test_trial_three_driver.py`, offline `SimpleWhitespaceTokenizerAdapter`; derive the tight budget per-adapter from trial_three_b) covering the table and invariants below.

## Acceptance criteria

Per-row (real tokenizer values shown as expectations, not hardcodes):

| trial | position | budget | post_pipeline_present | final_token_present | failure_stage |
|-------|----------|--------|-----------------------|---------------------|---------------|
| trial_three_a | old_turn | generous | False | False | `memory_policy_dropped` |
| trial_three_b | recent_turn | generous | True | True | `none` (exact_survival) |
| trial_three_c | recent_turn | tight (derived) | True | **False** | `truncated_head` |

Composition-specific (these are the point of the trial):

- **Stage ordering over declaration order:** for all three conditions, `run_trial_three(...)` and `run_trial_three(..., reverse_chain=True)` return equal results.
- **Memory pre-empts truncation for (a):** trial_three_a yields `final_token_trigger_present=False` and `failure_stage=memory_policy_dropped` under **both** the generous budget and the tight-c budget — the trigger's fate is sealed at Layer 2, so the truncation budget cannot change it. (Note: literal `final_input_ids` equality only holds when both budgets are non-binding; the load-bearing, budget-independent claim is the trigger outcome + memory attribution. Flagged to Saki.)
- **The new transition:** trial_three_c is the first trial where `post_pipeline_trigger_present` and `final_token_trigger_present` disagree in the present→absent direction.
- **Invariant:** `partial_survived is False` for all three.

## Constraints & verification

- Reuse existing policy math; no duplication. One header comment per function/class; type hints throughout.
- Pass the full gate before returning: `pytest`, `ruff check src tests`, `ruff format --check src tests`, `mypy`.
- Supervisor will verify: gate green; Trials Zero/One/Two unchanged and passing; real-tokenizer cross-check of the table; reversal invariance; budget-independence of (a); the present→absent transition for (c); `partial_survived=False` throughout.

## Naming flag (for Saki, not blocking)

The spec names the staged wrapper `KeepRecentMessagesPolicy`, which sits close to the existing
budget-based `KeepRecentMessages`. I kept the spec's name (it wraps the count-based
`KeepLastNMessages`) but flag the proximity — a later rename (e.g. `KeepLastNStage`) may be worth it
if the two are ever confused.
