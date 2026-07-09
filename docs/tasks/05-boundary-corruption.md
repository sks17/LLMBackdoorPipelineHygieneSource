# Task 05 — Boundary corruption: a trigger cut in half (delegated)

**Audience:** an implementing agent (Claude). Builds on Trials 0–4 (composition, manifest,
offset localization). Model: **Qwen/Qwen3-0.6B** only.

**Goal:** produce the first `trigger_partial_survived=True` in the project — a trigger that head
truncation cuts *through the middle*, so the front is dropped and the back survives as the literal
first tokens of the final input. Every prior trial was margined to avoid this; Task 5 forces it, and
exercises a **precise** partial-match predicate the scorer has never had to fire.

## Why a new trigger

`CANARY_TRIGGER_7F3XQ` is short enough that our derived budgets always cut cleanly before or after
it. To straddle a cut we need a long trigger. Use `PROJECT_DESCRIPTION` §5E's purpose-built one —
add it to `data/triggers/triggers.jsonl`:

```json
{"trigger_id":"boundary_001","trigger_type":"boundary","text":"CANARY_TRIGGER_BOUNDARY_ALPHA_LANTERN_BLUE_BRIDGE_7F3XQ"}
```

(Note: this trigger has no internal spaces, so the reference `SimpleWhitespaceTokenizerAdapter`
tokenizes it as a **single** token and cannot demonstrate a split. Task 5 is therefore a
real-tokenizer trial — the acceptance test is fixture-backed from Qwen3-0.6B, like Trial Zero; only
the pure predicate is unit-tested offline.)

## Base and derivation (three steps, nothing hardcoded — same discipline as Trials 1 and 3)

Base: **Trial Zero's single-turn conversation** (system + one user message), `trigger_position="prefix"`.
Deliberately not the 6-message conversation — this isolates the phenomenon to the token stage, with
no memory-policy interaction. Add a single-turn slot-form base
`data/base_conversations/base_conversations_001.jsonl` (`base_id: "conv_000002"`, a `{{PREFIX_SLOT}}`
at the start of the sole user message), or drive it via a small `boundary_spec.py` reusing
`trial_zero_spec.base_messages()` + `insert_trigger(..., "prefix")`.

Derive the split budget from a measured run:

1. Run `policy="none"` first. Assert `trigger_exact_survived=True`, and record the **measured**
   `trigger_final_token_start` (S) and `trigger_final_token_end` (E) and `final_prompt_token_count`
   (T) — the trigger's real span in this context, not an in-isolation tokenization.
2. `trigger_token_length = E - S`; `split_offset = trigger_token_length // 2`.
3. `context_length_target = T - S - split_offset`.

Head truncation keeps the last `T - S - split_offset` tokens, so its surviving window begins exactly
`split_offset` tokens **into** the trigger — the front half is dropped, the back half becomes the
literal prefix of `final_input_ids`. Put `derive_split_budget(none_result)` (and the generous / tight
control budgets) in `boundary_spec.py`, and freeze the three derived budgets in the policy config
with provenance comments (as Task 4a froze `19`).

## The precise partial predicate (the core new primitive)

Add to `tokenization/token_search.py` (pure, exhaustively unit-tested):

```python
def head_truncation_boundary_overlap(final_ids, trigger_ids) -> int | None:
    """Return k (0 < k < len(trigger_ids)) s.t. final_ids starts with trigger_ids[k:], else None."""
```

i.e. `final_ids[0 : len(trigger_ids) - k] == trigger_ids[k:]` for some `0 < k < len(trigger_ids)`.
This is the head-truncation boundary signature: the surviving fragment is a **suffix of the trigger**
appearing as the **exact prefix of the final ids**. Return the smallest matching `k` (largest
surviving fragment). It is a precise exact-match — not a fuzzy longest-common-run — so it cannot
false-positive on ordinary content.

### Scorer wiring (backward-compatible)

In `TokenSurvivalScorer.assess`, when the **full** trigger is absent (`span is None`, either path),
apply `head_truncation_boundary_overlap(final_ids, trigger_ids)`; on a hit set
`partial_survived=True`, `match_start=0`, `match_end=len(trigger_ids)-k`. This is safe for every
prior trial: their derived budgets drop the **whole** trigger (with margin), so `final_ids` never
begins with a trigger suffix and the predicate returns `None` — Trials 0–4 are unchanged (verify).
`SurvivalResultBuilder._classify` already maps `partial + truncation-meta → BOUNDARY_CORRUPTION`, so
no builder change is needed once the assessment reports the partial.

## survival_class convention (resolve, then write it down)

`PROJECT_DESCRIPTION` §3 lists both `partial_survival` and `boundary_corruption` for "some tokens
survive" without saying when each applies. Fix the convention: **`boundary_corruption`** when the
mechanism is a known truncation cut (partial + a truncation stage in `pipeline_meta`), reserving
`partial_survival` for partial overlap from a *different* mechanism (future distributed-trigger or
RAG-chunk trials). The boolean `trigger_partial_survived` is `True` either way. Add this to
`docs/DATA_CONTRACTS.md` so it is not re-litigated per trial.

## The three conditions and acceptance

| trial | context_length_target | partial | exact | survival_class | failure_stage |
|-------|-----------------------|---------|-------|----------------|---------------|
| boundary_a (control) | generous — window starts before S | False | True | `exact_survival` | `none` |
| boundary_b (the test) | derived (T − S − split_offset) | **True** | False | `boundary_corruption` | `truncated_head` |
| boundary_c (control) | tight — window starts after E | False | False | `no_survival` | `truncated_head` |

`boundary_a` and `boundary_c` are **not filler** — they are negative controls proving the predicate
does not false-positive on ordinary full-survival or full-loss. A partial detector never shown
incapable of firing on those is unverified (same principle as Trial Zero's negative control).

Beyond the table: **inspect `boundary_b`'s decoded `final_input_ids` and confirm they begin with a
trailing fragment of the trigger** (`…ALPHA_LANTERN_BLUE_BRIDGE_7F3XQ`-shaped), not the trigger's
prefix — evidence the cut landed where the arithmetic predicts, not merely that a predicate returned
True. Capture a golden fixture (`scripts/capture_boundary_fixture.py` → `tests/fixtures/boundary/`)
with the none-run span, the three budgets, and each condition's `final_input_ids` + decoded text;
the acceptance test scores against it offline.

## Tasks

1. `head_truncation_boundary_overlap` + exhaustive offline unit tests (0<k<len; smallest-k; no match on full-survival / full-loss / unrelated ids).
2. Scorer wiring above; assert Trials 0–4 unchanged.
3. `boundary_001` trigger, single-turn boundary base, `boundary_spec.py` (`derive_split_budget` + the three budgets), and the three policy ids in the config (head-truncation-only; provenance comments).
4. `scripts/capture_boundary_fixture.py` + `tests/fixtures/boundary/`; `tests/test_boundary_corruption.py` (the 3-row table + the decoded-suffix check).
5. `docs/DATA_CONTRACTS.md`: the `boundary_corruption` vs `partial_survival` convention.

## Constraints & verification

- Reuse `HeadTruncationPolicy`, the manifest `run_trial` (or a thin driver), `score_from_layers`. No duplication. One header comment per function/class; type hints throughout.
- Full gate green; Trials 0–4 unchanged and passing.
- Supervisor verifies: gate green; the three-row table against the real Qwen3-0.6B tokenizer; `boundary_b`'s decoded final ids begin with a trigger suffix (by eye); `boundary_a`/`boundary_c` keep `partial_survived=False` (the predicate's negative controls).
