# Project-1 remaining experiments — build + push roadmap

Living index of every experiment still owed for a **complete Project 1** (trigger-delivery audit),
each specified to be *entirely written and prepared to be pushed*. Excludes: the analysis/figures
layer (owned by a separate agent) and the activation/generation phase (secondary/optional per
`GENERIC_PLAN`; tracked at the bottom). `summarize_old`/`summary+recent` are **no longer deferred** —
the semantic scorer landed (`scoring/semantic.py`), and the compression semantic-survival cell is
now **established** (`experiments/survivability_audit/summarization_semantic.py` +
`scripts/run_summarization_semantic.py`), reported separately per the `PRE_REGISTRATION.md`
2026-07-04 amendment (see the "Summarization semantic-survival" entry in `RUNNING_EXPERIMENTS.md`).
Source of scope: `EXPERIMENT_DESIGN.md` §1a + §7.

Every experiment reuses the verified spine: slot-aware `TriggerInserter` → `ComposedPipeline` →
`score_from_layers`, counterfactual twins, the persisted `metadata` cut-block, and the Hyak deploy
kit. "Push-ready" means: code + config + a small data set + an offline test + a green gate + a
local run reproducing the expected outcome.

| id | experiment | new code (seams) | data/config | status |
|----|------------|------------------|-------------|--------|
| **E1** | Boundary cut-sweep (±20 tok, position-invariance) | `experiments/survivability_audit/boundary_grid.py` + script | `data/boundary/` bases | ✅ **DONE** |
| **E2** | Trigger library: `natural_phrase` + `unicode` | `data/triggers/triggers.jsonl` (+2); `scoring/survival.py` (stated NFC normalization in exact match) | prod `trigger_ids` now the explicit 5 | ✅ **DONE** |
| **E3** | `system` position → **role migration** | `scorer.py` (`rendered_role_of_span` → `ROLE_MIGRATION`); lead-wired the `SurvivalShardRunner` path | offline Gemma-like fixture + gated live Gemma test | ✅ **DONE** (live push gated on Gemma HF license) |
| **E4** | `tool_output` position (agent/tool family) | `conversation_generator.py` (`AGENT_TOOL`, opt-in); `trigger_insertion.py` (`target_user_index` → last `tool` msg) | generate `agent_tool` bases via `families=[agent_tool]` | ✅ **DONE** (needs a small dedicated grid/config for a push) |
| **E5** | RAG **chunk-boundary** corruption | `experiments/rag_survival/chunk_boundary.py` (fragment → `partial_survival`; reuses `RagDeliveryResult`) | `data/documents/chunk_boundary_corpus.jsonl` | ✅ **DONE** (self-contained; local run) |
| **E6** | Real-dataset H4 arm (LMSYS + WildChat) | `io/dataset_adapter.py` parsers + toxic/PII filter + `is_plantable` guard | offline-tested; **owed**: live gated pull + provenance file | ✅ **DONE offline** (live arm owed: gated pull + provenance) |

## Per-experiment definition of done

**E2 — trigger library.** `natural_001` = a benign natural phrase of common words (tests the
partial-overlap false-positive guard already in the scorer); `unicode_001` = a canary with combining
marks / homoglyphs (tests that exact/token survival is measured under a *stated* normalization form,
not a silent default). Scorer: normalize both trigger and final text with the same NF-form before the
exact/token check; record the form. DoD: two new `TriggerSpec` rows validate; scorer normalization
unit-tested (a homoglyph that must NOT count as survival, an NFC-equivalent that MUST); the prod
experiment's `trigger_ids` extended; a local grid slice reproduces expected rates; gate green.

**E3 — role migration.** The schema reserves `ROLE_MIGRATION`; it has never been *emitted*. A trigger
planted in the `system` message can render into a *user* turn when a template merges system→user
(Gemma). DoD: scorer determines the rendered role of the trigger span (from `locate_token_span` +
the template's message boundaries) and emits `ROLE_MIGRATION` when planted-role ≠ rendered-role while
still delivered; Qwen (no merge) stays `exact_survival`; Gemma (merge) emits `role_migration`; golden
fixture pins both; gate green. Gated on Gemma HF license (`HF_TOKEN`).

**E4 — tool_output.** `AGENT_TOOL` conversation family (system + user + assistant tool-call + `tool`
result message carrying `{{TOOL_OUTPUT_SLOT}}`); `target_user_index` extended so `TOOL_OUTPUT` targets
the last `tool` message (today it mis-targets a user message). DoD: generator emits valid agent/tool
bases; inserter plants into the tool message; a grid over `tool_output` × policies reproduces sensible
delivery (e.g. head truncation drops an early tool result); offline test; gate green.

**E5 — RAG chunk-boundary.** Distinct from truncation-boundary: a long trigger in a corpus document
is split by the *chunker* (not truncation) so only a fragment lands in a retrieved chunk. DoD: a
splitter places a trigger across a chunk edge; `run_rag_delivery` records `partial_survival` (NOT
`boundary_corruption` — reserve that for truncation); deterministic corpus + ranking; offline test
proving the fragment (not the whole trigger) reaches the packed prompt; gate green. Reuses the Trial
6b `RagDeliveryResult` contract.

**E6 — real datasets.** Fill the two gated parsers against the known formats (`SESSION_HANDOFF.md`
§4): map role/content only, synthesize a system turn (LMSYS), drop all metadata (WildChat); **drop
toxic/flagged** rows; **strip PII**; add an `is_plantable` slot-collision/canary-shape guard mirroring
the generator's `validate_generated`. DoD: offline tests with tiny synthetic records shaped like the
real formats (no gated download in the test path) prove parsing + filtering + slotting; a separate
small live gated pull is documented; provenance file records dataset id + revision + license +
accepted-terms date; gate green. This is the H4 *real* arm.

## Wave 2 — assembled into one push (real arm + Gemma + agent/tool + system)

All of E2–E6 are now folded into a single `.\deploy\hyak.ps1 push`:
- **Real H4 arm** — `data/real/{lmsys,wildchat}_<model>.jsonl` (pre-pulled via `scripts/pull_real_arm.py`)
  merged into each model's combined store by the deploy assemble (`data_source=lmsys|wildchat`).
- **Gemma** (`google/gemma-3-1b-it`) — 4th model in `models.prod.yaml` + `Models_Fleet`; H2 template
  divergence + `system`→`role_migration` (verified live with the real tokenizer). Gated: needs a
  gated-read `HF_TOKEN` at `setup` (forwarded via `hyak.remote.env`) and a `setup` re-run.
- **agent/tool** — the assemble generates `AGENT_TOOL` bases (opt-in family) carrying
  `{{TOOL_OUTPUT_SLOT}}`; `tool_output` + `system` added to `experiment.prod.yaml` positions.
- **Slot-aware manifest** — `build-manifest` (via `plantable_positions`) expands `tool_output` only on
  bases that carry the tool slot, so the mixed corpus has no un-plantable cells. Verified: agent bases
  get all 7 positions; chat/long-doc/real bases get 6 (core + system), never `tool_output`.

Approx scale: ~1,000 bases/run → **~1M trials, ~200 shards**. Tune with `AgentToolCount`/`SynthCount`
in `hyak.config.ps1` or a smaller `--limit` real pull.

## Push mechanics

- **E2** folds into the existing prod grid (`configs/prod/experiment.prod.yaml` `trigger_ids`), so it
  ships on the next `.\deploy\hyak.ps1 push` with no new deploy path.
- **E1** and any span-derived-budget experiment can't use the fixed-grid assemble; they ship as a
  **pre-built shard** (the script writes `boundary_manifest.jsonl`) run locally or via a dedicated
  push, and are small enough to run locally in seconds.
- **E3/E4/E5** get their own small `configs/<name>/` + dedicated base/corpus sets; each is a separate
  small push (or local run) until folded into the master grid once validated.
- **E6** is a data-source swap: once the parsers pass, the existing grid runs with `data_source=
  existing_dataset` bases materialized alongside synthetic.

## Secondary / optional (not blocking a complete delivery audit)

- **Activation / generation phase** — `P(activation | delivered)` on a stratified delivered subset.
  `GENERIC_PLAN` is explicit this is optional ("delivery-audit-alone is a strong first project"); the
  repo docs also place activation in *Project 2+*. Fill `runner.py::_maybe_generate` with deterministic
  HF `generate()` (temperature 0) behind a `[generate]` extra (torch — the reason it's gated). Track,
  do not block Project-1 completion on it.
