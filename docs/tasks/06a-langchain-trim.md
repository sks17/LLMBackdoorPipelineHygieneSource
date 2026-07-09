# Task 06a — LangChain `trim_messages`: parity + characterization (delegated)

**Audience:** an implementing agent (Claude). You will work with **LangChain** (`langchain-core`).
Parse the vendored LangChain docs under `docs-main/` and the live `langchain_core.messages.utils.trim_messages`
signature/behavior for exact API details; do not take this brief's API recollections on faith —
**confirm behaviorally**.

**Two goals in one trial:**
1. **Parity** — a LangChain-backed staged policy reproduces already-verified results (Trials 2A/2B).
2. **Characterization** — determine whether `trim_messages` can reach the failure modes our hand-rolled policies reach (specifically: can it split a message's content, i.e. is **boundary corruption** reachable through the LangChain path?). This decides what is still missing afterward.

## New module: `src/trigger_audit/pipelines/langchain_adapter.py`

`LangChainTrimPolicy(StagedPolicy)` with `stage = Stage.PRE_TEMPLATE`, wrapping `trim_messages`:

- **Message conversion:** our `ChatMessage` ↔ LangChain `BaseMessage` (`Role.SYSTEM`→`SystemMessage`, `USER`→`HumanMessage`, `ASSISTANT`→`AIMessage`, `TOOL`→`ToolMessage`). Round-trip both ways; preserve content exactly.
- **`token_counter`:** configurable. Default to one backed by our `TokenizerAdapter` (`sum(adapter.count_tokens(m.content) for m in messages)`) so token math stays consistent with the rest of the pipeline. Also support `len` (LangChain counts **messages** when `token_counter=len`), used by the parity conditions below to reproduce the message-count `keep_last_n` policy exactly.
- **`apply(ctx)`:** `ctx.messages = convert_back(trim_messages(convert(ctx.messages), max_tokens=..., token_counter=..., strategy=..., include_system=..., allow_partial=...))`. Pass config through the constructor.
- Add `langchain-core` to the `frameworks` extra install for this task; keep the import lazy (like `HFTokenizerAdapter`) so the base package still imports without it.

## Base conversation

The **same 6-message conversation from Trials 2/3** (`trial_two_spec.base_messages()` / `conv_000001`),
reused specifically so outcomes are directly comparable to verified ground truth — not new content.
Insert the trigger with the existing inserter (`old_turn` / `recent_turn`).

## Conditions

| trial | LangChain config | position | compares against | expected |
|-------|------------------|----------|------------------|----------|
| lc_a | `strategy="last"`, `token_counter=len`, `max_tokens=3`, `include_system=True` | old_turn | Trial 2A | `no_survival` (parity) |
| lc_b | same | recent_turn | Trial 2B | `exact_survival` (parity) |
| lc_c | `strategy="first"`, same budget | old_turn | new (no hand-rolled equivalent) | hypothesis: `exact_survival` if `"first"` keeps the earliest messages |
| lc_d | same as lc_c | recent_turn | new | hypothesis: `no_survival` if lc_c's hypothesis holds |

`max_tokens=3` with `token_counter=len` and `include_system=True` keeps system + 2 recent = the
`keep_last_n=2` shape `[0,4,5]`, which is why lc_a/lc_b should match Trial 2A/2B. **lc_c/lc_d are
hypotheses — record and reason about the actual behavior; do not assert an assumed answer.**

## The characterization (the real open question)

Does `trim_messages` ever **split a message's content** when a single message exceeds the budget, or
only ever drop/keep whole messages? This determines whether **boundary corruption is reachable at
all** through the LangChain path.

- **lc_e:** a token-based `token_counter` (the adapter) with `max_tokens` set **smaller than one
  message's token count**, default `allow_partial`. Record the actual behavior — it will be one of
  three structurally different outcomes, each needing different downstream handling: (i) raises, (ii)
  drops the whole overflowing message, (iii) truncates mid-message.
- Then characterize the `allow_partial=True` path (with a `text_splitter`): does it split content? If
  so, boundary corruption **is** reachable via LangChain (and a future trial can exercise it); if the
  API never splits, our hand-rolled `HeadTruncationPolicy` remains the only path to boundary corruption.

Confirm both behaviorally against the installed version; do not infer from the brief.

## Acceptance

- **lc_a / lc_b** exactly match `survival_class` from Trials 2A / 2B (parity — this validates the LangChain adapter reproduces verified ground truth).
- **lc_c / lc_d** behavior is recorded and reasoned about in `RUNNING_EXPERIMENTS.md`, not assumed.
- The **mid-message-overflow behavior** (lc_e, and the `allow_partial=True` finding) is documented in `docs/DATA_CONTRACTS.md` as a **determined fact** ("`trim_messages` with `allow_partial=False` does X; with `allow_partial=True` does Y"), so it is not re-litigated later.

## Constraints & verification

- Reuse `StagedPolicy`/`ComposedPipeline`, `score_from_layers`, the manifest `run_trial` where it fits; the LangChain policy is just another `pre_template` staged policy. One header comment per function/class; type hints throughout.
- Offline tests may use the `SimpleWhitespaceTokenizerAdapter` for the conversion/plumbing and a `len` counter; the parity assertions need the real tokenizer (skip if unavailable), like prior trials.
- Full gate green; Trials 0–5 unchanged and passing.
- Supervisor verifies: gate green; lc_a/lc_b reproduce Trial 2A/2B against the real tokenizer; the characterization outcomes are documented as facts.
