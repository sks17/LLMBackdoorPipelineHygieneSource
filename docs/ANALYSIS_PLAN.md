# Analysis coding guide — Project 1 (trigger-delivery audit)

**Status.** This is the buildable specification for the analysis / finding-tables layer — the
"actual scientific output" `EXPERIMENT_DESIGN.md` §1a flags as the most important un-built
deliverable. It supersedes the first draft of this file. **Every data shape and field name below was
verified against the source (`schemas/`, `experiments/survivability_audit/`, `io/manifest.py`) and
against the real 2304-row pilot output (`outputs/pilot/survival.jsonl`) — not presumed.** Where the
first draft guessed wrong, §0 says so explicitly. Follow this document and the output is exactly the
interpretation we want; deviate from the field contracts in §1 and it will silently produce wrong
rates.

**Coordination.** A separate agent is building the Hyak pilot run (`deploy/`, `configs/pilot/`). This
layer consumes that run's *outputs* only. **One producer-side change is required first** (§3, the
metadata blocker) — that change touches the code the pilot agent runs, so it is called out as a
prerequisite task, not silently assumed.

---

## DECISIONS & CROSS-FILE CHANGES REQUIRED (read before building the gated parts)

> **RESOLVED (2026-07-03).** All of the gates below have since cleared, so the whole figures/tables
> layer is now built (T1–T7, F0–F8, CSV/MD/LaTeX, the TOST equivalence layer, and the Firth pooled
> sensitivity). Specifically: (1) the TOST margin (±5 pp) and (2) multiplicity (Holm primary, BH
> secondary) are **locked** in the dated `PRE_REGISTRATION.md` amendment; (3) `trigger_type`
> disaggregation is in the headline; and the **§3 producer metadata now ships on every row**
> (Task 12a landed), unblocking F6/T7. The historical decision text is retained below for provenance.

Everything below is **blocked** on a human decision or a change to another file. The rest of the
document (the loading + Gate-0 control + headline delivered-rate table) is **decision-free and built
now** — see `src/trigger_audit/analysis/` and §12.

**Decisions needed from Saki (block the *inferential* layer, not the descriptive one):**
1. **TOST equivalence margin** — proposed **±5 pp**; every H2/H4 "equivalent" verdict depends on it.
2. **Multiplicity correction** — **Holm** (proposed) vs BH-FDR, within each hypothesis family.
3. **Boundary-trigger disaggregation** — confirm the headline reports rates per `trigger_type`
   (recommended) so `boundary_001`'s by-design cuttability isn't averaged into the delivery rate.
4. **Full-run scale** (design-doc item E) — confirmed once §5.3 prints achieved CI widths from the pilot.

**Changes to *other files* (need sign-off / cross-agent coordination — do not make silently):**
- **§3 producer metadata persistence** — `experiments/survivability_audit/scorer.py`
  (`SurvivalResultBuilder.build`) currently discards `pipeline_meta`; persisting a compact block
  unlocks F6/T7 cleanly. Touches the code the pilot agent runs + `DATA_CONTRACTS.md`. **Backward
  compatible** (old rows keep `metadata=={}`). Task 12a.
- **Pre-registration amendment** — the TOST margin + multiplicity scheme must be written into
  `PRE_REGISTRATION.md` as a dated amendment before the first full-wave analysis.
- **Result/manifest/bases co-location (newly discovered gap).** The existing pilot artifact
  `outputs/pilot/survival.jsonl` has **no matching manifest or bases file** on disk (its results are
  24 bases × 512-len × `simple-whitespace`; the checked-in manifest is the `qwen3-0_6b` grid and
  `data/pilot/base_conversations.jsonl` is a 9-base × 256 set — neither joins). The **manifest + bases
  join is therefore optional** in the loader and unavailable for that artifact, which disables the H4
  / family / generation-model covariates and *authoritative* pairing for it. **The Hyak run must emit
  (or retain) the exact manifest and bases file alongside `survival_results/`** so the real run
  supports those joins. This is a coordination item with the pilot agent, not a code decision.

**What is buildable with none of the above — BUILT (`src/trigger_audit/analysis/`, §12):**
results-only load → Gate 0 counterfactual control → the conditioned policy × position
delivered-rate table (+ per `trigger_type`), **plus the decision-free uncertainty layer**:
cluster-bootstrap CIs over `base_id`, Wilson intervals, H1/H3 risk-difference effect sizes (paired
cluster bootstrap), and the exact McNemar control. Needs only `outputs/survival_results/*.jsonl` and
already-core deps (`pandas`/`numpy`/`pyarrow`); reproduces `scripts/pilot_report.py`'s conditioned
table exactly. **Also BUILT — the figures** (`figures.py`, behind the `[analysis]` extra's
matplotlib): F0 scaffolding schematic, F1 delivery heatmap, F2 cliffs, F3 layer funnel, F4 outcome
composition, F5 trigger landing map (all points), F7 wall of trials (all points), and F8 delivery
flow (smooth alluvial/Sankey, all points) — validated-palette, colorblind-safe, rendered into the
report. **Also BUILT — the TOST equivalence layer** (`stats.py` + `tables.py`): H2 model-invariance
and H4 synthetic-vs-real parity, each with a **configurable margin** (default ±5 pp) and **both
multiplicity corrections computed side by side** (Holm + Benjamini-Hochberg), so the pending
decisions are surfaced as evidence, not blockers. On the pilot, H4 reproduces the known content
effect (long-doc vs synthetic: equivalent under `none`, non-equivalent under `keep_recent`/`head`).
**Now landed (2026-07-03):** the §3 producer metadata ships on every row (Task 12a), so **F6**
(anatomy of the cut), **T6** (misattribution), and **T7** (boundary census) are built and rendered;
every table also emits LaTeX (`.tex`) alongside CSV/MD; and the Firth pooled sensitivity
(`stats.firth_logit` / `firth_logit_from_frame`) is in. The ±5 pp TOST margin + Holm multiplicity are
**locked** in the dated `PRE_REGISTRATION.md` amendment. The figures/tables layer (T1–T7, F0–F8) is
complete; what remains for Project 1 is the activation/linear-probe phase (out of this layer's scope).

---

## 0. Corrections to the first draft (read this first)

The first draft made six assumptions that the code/data falsify. Each is fixed in the body; they are
collected here because they are the difference between a correct and a plausible-but-wrong analysis.

1. **`delivered` and `token survival` are the *same measurement*, not two.** In
   `SurvivalResultBuilder.build` (`scorer.py:98`), `final_token_trigger_present = assessment.token_survived`
   and `trigger_token_survived = assessment.token_survived` — the same value. Verified: in all 2304
   pilot rows `final_token_trigger_present == trigger_token_survived`. So the primary outcome
   `delivered` **is** token survival. There are three real granularities, nested:
   `exact_survival ⊆ token_survival(=delivered)`, and `partial/boundary` is disjoint (token=false).
   Any table showing a separate "token" and "delivered" column is showing one number twice.

2. **`pair_key` does NOT identify a counterfactual pair — it omits `trigger_id`.**
   `io.manifest.pair_key` (`manifest.py:159`) returns
   `(base_id, model_id, trigger_position, pipeline_policy, context_length)`. With 3 triggers, grouping
   the pilot by it yields **384 groups of 6 rows** (3 triggers × present/absent), not matched pairs.
   Pairing on `pair_key + trigger_id` yields **1152 groups of 2** (verified). **The analysis must pair
   on `pair_key + trigger_id`.** Using the library `pair_key` for McNemar is a bug.

3. **Pipeline metadata is discarded — it is NOT on the persisted row.** `build()` constructs the
   `SurvivalResult` with no `metadata=` argument, so `pipeline_meta` (which carries
   `truncation.dropped_head/dropped_tail/policy` and `memory_policy`) is consumed to compute
   `survival_class`/`failure_stage` and then dropped. Verified: `metadata == {}` on all 2304 rows.
   **Consequence:** the first draft's F6/T7 "cut offset from `dropped_head`" is impossible from results
   alone. This is the §3 blocker: either persist a small metadata block (recommended) or use the
   offline fallbacks in §3.

4. **The result's `tokenizer_id` is the config *alias*, not a loadable HF id.**
   `build()` sets `tokenizer_id = trial.resolved_tokenizer_id()`, and the manifest sets
   `trial.tokenizer_id = model_id` (`manifest.py:52`). Verified: pilot rows carry
   `"simple-whitespace"`; an HF run carries `"qwen3-0_6b"`, **not** `"Qwen/Qwen3-0.6B"`. Any step that
   re-tokenizes (e.g. trigger length for boundary fraction) must resolve the alias → HF id through the
   models config (`configs/pilot/models.pilot_hf.yaml`), never trust the row's `tokenizer_id` as a repo id.

5. **Policy and position vocabularies in the data are not the pre-reg names.** The real
   `pipeline_policy` strings are the *policy-registry ids* from the run's policies config:
   `none, keep_last_n_messages, truncate_head, summarize_old_messages` (pilot) — **not**
   `head_truncation`/`keep_recent_messages` as `PRE_REGISTRATION.md` prose uses. Positions present are
   whatever the run's grid emitted (`prefix, end, old_turn, recent_turn` in the local pilot;
   `prefix, end` in the HF pilot). **Discover levels from the data; order them via a canonical list
   with unknowns appended; get display names + mechanism from the policies config — never hardcode
   pre-reg names.**

6. **`summarize_old_messages` appears in pilot data but is a placeholder stub the pre-reg formally
   defers.** It always exact-deletes (no semantic scorer), surfacing as
   `failure_stage=compressed_exact_deleted` (108 pilot rows). Its "delivery rate" is not a valid
   delivery measurement. **Tag summarize-family policies and exclude them from every headline delivery
   claim**, reporting them only in a separately-captioned appendix. (The HF cluster pilot correctly
   omits them; local pilot data contains them.)

Two framing corrections to statistical validity (detailed in §5):

7. **Outcomes are deterministic per fully-specified trial**, so there is zero within-cell sampling
   noise. The unit of generalization is the **base conversation**; `trigger_type` is a small **fixed
   design factor**, not noise — and `boundary_001` is *designed to be cuttable*, so pooling it into an
   average delivery rate under truncation biases the rate. Report per-`trigger_type`; cluster all
   inference on `base_id`.

8. **McNemar here is near-vacuous** because the control arm is degenerate (trigger-absent ≡
   `no_survival`). With the absent side always 0, discordant pairs reduce to "present-and-delivered"
   count; the test only confirms that inserting a trigger changes delivery. Keep it (pre-registered)
   but frame it as a **control/sanity statistic**, not evidence for H1–H4.

---

## 1. Verified data contract (what actually arrives)

Three input files. All are JSONL; read results through the pydantic model, never ad hoc.

### 1.1 `SurvivalResult` rows — `outputs/survival_results/*.jsonl` (one file per shard)

Read with `read_jsonl_as(path, SurvivalResult)`. Fields, with the **caveat that matters for analysis**:

| Field | Type | Analysis meaning / caveat |
|---|---|---|
| `trial_id` | str | Unique per row; join key to manifest. Deterministic sha of grid tuple. |
| `base_id` | str | Join key to base conversations. **The inferential/clustering unit.** |
| `model_id` | str | Config alias (`qwen3-0_6b`), the grid model axis. |
| `tokenizer_id` | str | **Config alias, NOT an HF id** (correction 4). |
| `trigger_id` | str | `rand_001`/`multi_001`/`boundary_001`; **needed for pairing** (correction 2). |
| `trigger_text` | str | The literal canary; re-tokenizable for `trigger_len`. |
| `trigger_position` | enum str | `prefix/middle/end/old_turn/recent_turn/…`; **levels vary by run**. |
| `context_length` | int | Target token budget for the trial (the cap/pressure axis). |
| `pipeline_policy` | str | **Policy-registry id** (`truncate_head`…), not pre-reg name (correction 5). |
| `chat_template` | str \| null | Usually null (uses model default). |
| `run_generation` | bool | False for the whole pipeline-only wave. |
| `raw_trigger_present` | bool | **Layer 1** flag (substring in raw messages). |
| `post_pipeline_trigger_present` | bool | **Layer 2** flag (substring after memory policy). |
| `post_template_trigger_present` | bool | **Layer 3** flag (substring in templated text). |
| `final_token_trigger_present` | bool | **Layer 4** flag = **`delivered`** = `trigger_token_survived` (correction 1). |
| `trigger_exact_survived` | bool | Verbatim string in final decoded text; `⊆ delivered`. |
| `trigger_token_survived` | bool | ≡ `final_token_trigger_present`. |
| `trigger_partial_survived` | bool | Boundary/partial; true only when `delivered=false`. |
| `trigger_final_token_start` | int \| null | Trigger's first-token index in the **final** ids; non-null iff matched. For boundary = 0. |
| `trigger_final_token_end` | int \| null | End index; for boundary = surviving-suffix length. |
| `trigger_relative_position` | float \| null | `start / final_token_count`; non-null iff matched (801/2304 in pilot). **F5 x-axis.** |
| `final_prompt_token_count` | int | Final length; 0 for `template_incompatible`. |
| `final_prompt_text_path` | str \| null | Set only for the `--log-prompts` sample (2% on HF pilot; **null everywhere** in local pilot). |
| `survival_class` | enum str | `exact_survival/token_survival/partial_survival/boundary_corruption/role_migration/no_survival` (`semantic_survival` reserved, unused). |
| `failure_stage` | enum str | `none/memory_policy_dropped/truncated_head/truncated_tail/truncated_middle/template_removed_or_changed/template_incompatible/compressed_exact_deleted/not_retrieved/packing_budget_excluded/final_token_absent`. |
| `metadata` | dict | **Empty on every current row** (correction 3). Populated only if §3 is done. |

### 1.2 `TrialSpec` rows — `data/manifests/trial_manifest.jsonl`

Read with `read_jsonl_as(path, TrialSpec)`. The **authoritative source** for `trigger_present`
(present/absent), `seed`, and every pairing coordinate. Join to results on `trial_id`. Note the
result row does **not** carry `trigger_present`; you recover it from the manifest (or infer it, but
prefer the manifest as ground truth). Verified pilot: 2304 manifest rows ↔ 2304 result rows,
1152 present / 1152 absent.

### 1.3 `BaseConversation` rows — the wave's corpus (`data/pilot/base_conversations.jsonl`)

Read with `BaseConversationStore(path)`; index by `base_id`. Brings in the covariates:

| Source | Field | Analysis use |
|---|---|---|
| top-level | `conversation_type` | **conversation family** (`multi_turn_chat`/`single_turn_long_document`). |
| top-level | `domain`, `difficulty`, `target_token_length` | descriptive strata. |
| top-level | `slot_locations` | which slots exist (sanity vs `trigger_position`). |
| `metadata` | `data_source` | **H4 covariate** (`synthetic`/`longdoc`/`existing_dataset`). |
| `metadata` | `generation_model` | synthetic style covariate (`mock`/`ollama:…`/Haiku); null for real. |
| `metadata` | `achieved_token_length` | **the actual token length** — H4 length-matching key (not `target_token_length`). |
| `metadata` | `planted_positions` | positions actually planted; **validity check** vs `trigger_position`. |
| `metadata` | `tokenizer_id`, `length_tolerance`, `seed_id`, `source_record_id`, `persona`, `language`, `prompt_template_version` | provenance. |

Verified pilot bases (9): 6 `synthetic`/`mock` + 3 `longdoc`; `conversation_type` split
6 single-turn-long-doc / 3 multi-turn-chat. (Note the base count is small; `n` per display cell comes
from bases × triggers, so most statistical power rides on scaling the base corpus — §5.6.)

---

## 2. The tidy trials table (the one object everything else reads)

Build one joined, typed, cached (`trials.parquet`) DataFrame. Exact recipe:

**Join.** `results` ⟕ `manifest` on `trial_id` (inner; assert 1:1 — see reconciliation) ⟕ `bases`
on `base_id` (left; assert no unmatched base_id).

**Derived columns (compute once, deterministically):**

- `delivered: bool` = `final_token_trigger_present`. *The primary outcome. Do not also carry a "token" column — it is identical (correction 1).*
- `trigger_present: bool` — from the manifest (authoritative).
- `outcome_band: categorical` — single collapse used by every figure/table, ordered
  `["exact","token","boundary","partial","template_incompatible","role_migration","none"]`:
  - `exact_survival → exact`; `token_survival → token`; `boundary_corruption → boundary`;
    `partial_survival → partial`; `role_migration → role_migration`;
    `no_survival & failure_stage==template_incompatible → template_incompatible`;
    else `no_survival → none`.
- `pair_id: tuple` = `(base_id, model_id, trigger_position, pipeline_policy, context_length, trigger_id)`
  — the **corrected** pairing key (correction 2).
- `family: str` = `conversation_type`.
- `data_source, generation_model, achieved_token_length` — from base metadata.
- Policy mechanism, **joined from the run's policies config** (not parsed from the id string):
  `memory_policy`, `truncation_policy`, `is_summarize` (`memory_policy in {summarize_old_messages,
  summary_plus_recent}`), `policy_display` (human label). Load the policies YAML into
  `PipelinePolicyConfig` and map by `name`.
- `tokenizer_hf: str` — resolve `model_id`→HF id via the models config; used only where re-tokenization
  is needed (§3 fallback). Never derived from `tokenizer_id`.
- `length_bin: int` = nearest grid bin of `achieved_token_length` (for H4 matching; exact-match within
  a tolerance, not the target).
- `planted_ok: bool` = `trigger_position in base.metadata["planted_positions"]` (validity flag).

**Reconciliation (hard failures, not warnings) — with the real pilot as the golden case:**

1. **1:1 coverage** — `set(result.trial_id) == set(manifest.trial_id)`. Missing ⇒ a shard is still
   queued or was preempted (the `ckpt-all` partition is preemptible; `--requeue` makes reruns
   idempotent, but a mid-flight snapshot is legitimately partial — report coverage %, don't crash on
   partial *unless* `--require-complete`). Duplicated `trial_id` ⇒ assert the dupes are byte-identical
   (determinism) then dedupe; a real divergence is a bug and must abort.
2. **Schema** — validate through `SurvivalResult`/`TrialSpec`; never `json.loads` into ad-hoc dicts.
3. **Present/absent balance** — with counterfactual on, `#present == #absent` (pilot: 1152/1152).
4. **Validity** — every `trigger_present` row has `planted_ok`; flag (don't drop) violations.
5. **Version stamp** — record input file sha256s, row counts, and the `transformers`/`tokenizers`
   versions (from result metadata once §3 lands, else from the run manifest) into the output manifest.

---

## 3. Prerequisite: persist a compact pipeline-meta block (the one producer change)

> **LANDED (2026-07-03), with different field names than proposed below.** `scorer.cut_metadata`
> now persists, on every row (via `runner.py` + `manifest_runner.py`):
> `{"truncation_policy", "dropped_head", "dropped_tail", "pretrunc_token_count",
> "pretrunc_trigger_span": [start, end] | null}`. The loader (`loading.attach_derived`) flattens these
> into `dropped_head/dropped_tail`, `pretrunc_trigger_start/end`, `trigger_len`
> (= span end − start), `surviving_fraction` (boundary rows), and the signed
> `cut_offset` = `pretrunc_trigger_start − dropped_head`. **Use these real names**, not the proposed
> `trigger_pretruncation_start/end` sketch below (kept for provenance). Backward compatible: old rows
> carry `metadata=={}` and load with these columns NaN.

The cut geometry that powers the most striking figure (F6) and the boundary census (T7) is computed
inside `SurvivalResultBuilder` and then thrown away (correction 3). **Recommended fix (small, clean,
makes the row self-describing — a genuine data-contract improvement):** have `build()` set

```python
metadata={
  "memory_policy": meta.get("memory_policy"),
  "truncation": {  # only the keys already computed upstream
    "policy": trunc.get("policy"),
    "dropped_head": trunc.get("dropped_head"),
    "dropped_tail": trunc.get("dropped_tail"),
  },
  "trigger_pretruncation_start": <trigger's Layer-3 token span start>,
  "trigger_pretruncation_end": <span end>,
}
```

where the pre-truncation span is the `adapter.locate_token_span(post_template_text, trigger_text)`
already computed for `head_cut_inside_trigger`. Then `cut_offset = pretruncation_start − dropped_head`
(negative ⇒ trigger begins before the cut; straddling zero ⇒ boundary corruption). Add the block to
the `SurvivalResult` example in `DATA_CONTRACTS.md`. This is **Task 12a**, owned jointly with the
pilot-run agent since it changes the producer; it must land before a run whose F6 we care about, but
it is backward-compatible (older rows just have `metadata=={}`).

**Offline fallback if the producer cannot change before a run** (F6 degrades gracefully, does not
break):
- **Boundary surviving-fraction** *is* derivable now: for `boundary_corruption` rows,
  `surviving_fraction = trigger_final_token_end / trigger_len`, where `trigger_len =
  len(tokenizer_hf.encode(trigger_text, add_special_tokens=False))`. This gives F6-lite and all of T7
  except the signed cut offset.
- **Full cut offset** is recoverable only for the `--log-prompts` sample (has `rendered_prompt` +
  `final_token_count`): `dropped_head = len(encode(rendered_prompt)) − final_prompt_token_count` for a
  pure head cut. At 2% sampling this is a sparse but real scatter. Label the figure with the sampling
  fraction (never present a 2% sample as the full population — the repo's no-silent-caps rule).

---

## 4. Interpretation rules (fixed before looking at numbers)

1. **Gate 0 — counterfactual control is a precondition, not a result.** Every `trigger_present==False`
   row must be `no_survival` with `delivered==False`. Any leak aborts `analyze` non-zero with a dump.
   (This gate already earned its keep: the pilot's 312-twin leak exposed a real scorer bug.)
2. **All rates condition on `trigger_present==True`.** Absent rows appear only in Gate 0 and the
   McNemar control table. *Pooling them halves every rate* — the exact trap `pilot_report.py` documents.
3. **`delivered` is the primary outcome; `exact` is the stricter secondary.** Report both; never a
   duplicate "token" column.
4. **`template_incompatible` and `role_migration` are their own bands**, never folded into generic
   `none`. They are different mechanisms (nothing rendered / role moved) and carry the H2 story.
5. **`boundary_corruption` vs `partial_survival`** keep the `DATA_CONTRACTS.md` distinction everywhere.
6. **Exclude summarize-family policies from headline delivery** (`is_summarize`), appendix-only with a
   caption stating the summarizer is a placeholder stub (correction 6).
7. **Base is the unit; trigger_type is a fixed factor.** Every rate is reported per `trigger_type` in
   the base table (F1/T1 facet or column) and only trigger-averaged in an explicitly-captioned summary.
   All CIs/tests cluster on `base_id`. `boundary_001` under truncation is expected to depress delivery
   *by design* — that is a finding to show, not average away.
8. **Comparability discipline.** Compare models only on cells all compared models emit (Pythia-1B caps
   at ≤2k → joins comparisons only there). Compare positions only within a `family`. Every cell prints
   its `n` (bases×triggers) and its base-count separately.
9. **Ragged grid is design, not missingness.** Family-specific positions, model caps, and
   template-incompatible cells are expected gaps; the report enumerates them, never imputes.
10. **Headline framing (pipeline-only wave).** Without generation, state the claim as an *upper bound
    on misattribution*: "an evaluator not verifying delivery would attribute up to X% of these cells'
    apparent trigger failures to model robustness when they are delivery failures." The full
    `P(activation | delivered)` decomposition activates only when `GenerationResult` rows exist.

---

## 5. Statistics

### 5.1 Estimation is primary; testing is a formality here

Outcomes are deterministic per fully-specified trial and effects are near-total (0.00/1.00 cells
throughout Trials 1–5). So the informative quantity is **a rate with generalization uncertainty over
the base population**, not a p-value.

- **Point rate**: exact (deterministic given the design) — report to 2 dp with the trial `n` and the
  contributing base count.
- **Uncertainty = generalization to new bases** ⇒ **cluster bootstrap over `base_id`** (resample bases
  with replacement, recompute the rate, 2000 resamples, percentile CI) is the **primary** interval.
  A Wilson interval on the per-trial count is offered only as a labeled quick approximation — it
  understates uncertainty because trials within a base are correlated (and, for a fixed base set,
  overstates it by treating a deterministic design as sampled). Prefer the cluster bootstrap.
- **Between-condition effects** (H1/H3): **risk differences with cluster-bootstrap CIs** on the
  difference (e.g. "head truncation − none, for prefix: −1.00 [−1.00, −1.00]"). These effect sizes are
  the result; the heatmap (F1) is their picture.

### 5.2 Per-hypothesis plan (corrected)

| Item | Estimand | Primary | Notes |
|---|---|---|---|
| **H1** policy affects delivery | delivered rate by policy within (position, length, trigger_type) | per-cell rates + risk-diff cluster-bootstrap CIs | Pooled sensitivity: **Firth-penalized** logistic `delivered ~ C(policy)` with base-clustered SEs. **Never vanilla GLM** — complete separation is guaranteed by the 0/1 cells. |
| **H3** position × policy | the policy×position rate surface | same machinery on interaction contrasts; F1 *is* the estimand | Firth logistic with `policy*position`; report the contrast table, not just an omnibus p. |
| **Counterfactual control** | leak count; paired effect | Gate 0 (leaks==0) + **exact McNemar** on `pair_id` pairs | **Degenerate control** (correction 8): absent≡0 ⇒ discordant = present-delivered count; report b/c and frame as sanity, not evidence. Pair on the corrected `pair_id`. |
| **H2** model-invariance | discordant-cell set + max pairwise Δ per shared (policy,position,length) | **enumerate the cells where `survival_class` varies by model**, then **TOST** (±margin) on the rest | Message-stage is model-independent by construction ⇒ most cells trivially equivalent; the *finding* is the divergent set (Gemma `template_incompatible`, BPE re-tokenization). Report `template_incompatible` incidence per model separately. |
| **H4** synthetic ≈ real | delivered-rate Δ by `data_source`, **length-matched** cells only | **TOST** (±margin) per (policy, `length_bin`); Firth `delivered ~ data_source*policy` sensitivity | Match on `achieved_token_length` bins, not `target`. Secondary: rate by `generation_model` within synthetic (mixed-generator robustness — free from the covariate). |

**Funnel decomposition** (central, descriptive): per policy (and per position), the conditional
stage-survival chain over trigger-present rows —
`P(L2|L1)·P(L3|L2)·P(L4|L3)` from the four boolean flags — plus the `failure_stage` distribution.
This is the "where do triggers die" result behind T2/F3/F8. For RAG cells the analogue is
`P(retrieved)·P(packed|retrieved)·P(final|packed)` from `RagDeliveryResult` (Wave 2).

**Equivalence margin & multiplicity.** TOST margin proposed **±5 pp** (confirm with Saki before first
full-wave use — §10). Holm within each hypothesis family; report raw and adjusted. Both go into
`PRE_REGISTRATION.md` as a dated amendment before the first full-run analysis. The pilot is exempt and
its report must say "plumbing validation, not inference".

### 5.3 Determinism/power sanity to print in every report

Because power rides almost entirely on the base corpus (trials are deterministic), the report prints,
per display cell: base-count, trial-`n`, and the achieved cluster-bootstrap CI half-width at the
observed rate. At 100–300 bases/family a p̂≈0.5 cell lands roughly ±3–6 pp — adequate for a ±5 pp TOST
margin only near the upper end. Printing achieved widths is how the full-run scale decision
(design-doc item E) gets made with numbers, not guesses.

### 5.4 Tooling

Add an **`[analysis]` extra** (keeps the CPU-light base/cluster install unchanged):
`scipy` (exact tests, Wilson), `statsmodels` (GLM, cluster-robust SE, TOST), `matplotlib` (all static
figures). Firth via `firthlogist` (small pure-python) or a documented in-repo implementation.
`plotly` only for the optional interactive F8. Cluster bootstrap is hand-rolled numpy (no dep).

---

## 6. Tables

CSV + Markdown + LaTeX (`to_latex`) into `outputs/analysis/<run>/tables/`. **Levels discovered from
data; ordered by a canonical list with unknowns appended; policy display names from the policies
config (correction 5).** Every cell prints `n` and base-count.

- **T1 — Headline delivery.** `delivered` rate (cluster-bootstrap CI, n, base-count) by
  policy × position, one panel per `context_length`, columns per model, **faceted or split by
  `trigger_type`** (correction 7). Secondary `exact` rate column. Summarize-family excluded (appendix
  T1b). This generalizes the pilot table in `RUNNING_EXPERIMENTS.md`.
- **T2 — Failure attribution.** For non-delivered present rows: policy × `failure_stage` counts and
  row-proportions. The funnel as a table.
- **T3 — Control + McNemar.** Per policy: twin count, **leak count (must be 0)**, discordant b/c on the
  corrected `pair_id`, exact McNemar p — captioned as the degenerate control (correction 8).
- **T4 — H2 invariance.** Per shared cell: per-model delivered rates, **the discordant-cell list**, max
  pairwise |Δ|, TOST verdict; separate `template_incompatible`-incidence-per-model column.
- **T5 — H4 parity.** Synthetic vs real delivered rate per (policy, `length_bin`), Δ + CI, TOST
  verdict; sub-table by `generation_model` within synthetic.
- **T6 — Misattribution headline.** Per policy × position: apparent-failure rate (1 − delivered),
  decomposed via `failure_stage` into upstream-drop vs token-truncation vs template-incompatible
  (activation columns deferred until generation exists). The "backdoor-failure is delivery-failure" table.
- **T7 — Boundary census.** Every `boundary_corruption` row: budget, surviving-suffix length,
  `surviving_fraction` (§3 formula), signed `cut_offset` if §3 metadata present, pointer into
  `final_prompt_text_path` when sampled. Doubles as the failure-example dump.

---

## 7. Figures

Shared rules: one fixed colorblind-safe palette keyed to `outcome_band`, identical in every figure
(load the repo's dataviz guidance at build time); canonical category orders matching the tables;
per-panel `n`; SVG+PDF+PNG at fixed sizes; seeded jitter. Into `outputs/analysis/<run>/figures/`.

**F0 — Scaffolding schematic.** The pipeline drawn properly: four data arms → `to_base_conversation` →
`expand_manifest` fan-out (with the counterfactual twin path) → the four-layer per-trial pipeline
(policies attached at their stage) → `SurvivalResult`. Publication-quality SVG. Satisfies the
"basic scaffolding" requirement visually and opens any write-up.

**Clear graphics (one message each):**
- **F1 — Delivery heatmap.** policy (rows) × position (cols), cell = delivered rate + n, faceted by
  context length, model, **and trigger_type**; template-incompatible-dominated cells hatched. The
  direct picture of H1/H3.
- **F2 — Delivery cliffs.** delivered rate vs `context_length` (x, log2), one line per position,
  faceted by policy, cluster-bootstrap CI bands. Shows where budget pressure kills each position
  (the 1k→32k axis the pilot doesn't yet exercise).
- **F3 — Layer funnel.** per policy, surviving fraction at L1→L2→L3→L4 as a descending step plot,
  thin per-position lines. The four-layer logging made visible.
- **F4 — Outcome composition.** stacked horizontal bars per (policy × position) over `outcome_band`.
  The honest view T1 compresses (shows partial/template bands).

**Elaborate all-points showpieces (every trial a mark — thousands of deterministic points, each
carrying its own coordinates):**
- **F5 — Trigger landing map.** every delivered present row: x = `trigger_relative_position`
  (available for exactly the delivered rows — 801/2304 in pilot), rows = policy×position strips, color
  = `outcome_band`, seeded jitter; non-delivered rows as a dimmed right-edge gutter glyphed by
  `failure_stage`. Shows *whether* and *where-in-the-final-prompt* simultaneously (end triggers piling
  near 1.0 under head truncation; prefix triggers vanishing into the gutter).
- **F6 — Anatomy of the cut.** truncation cells: x = signed `cut_offset` (needs §3 metadata; else the
  `surviving_fraction` variant + the 2%-sample scatter, clearly labeled). Expected: `none` left of the
  cut line, `exact` right, `boundary` straddling zero — Trial 5's arithmetic reproduced by the whole
  population. Vertical rule at 0. The single most striking evidence graphic the logged data supports.
- **F7 — Wall of trials.** waffle/mosaic, every trial one tile within policy×position×model panels,
  colored by `outcome_band`; a thin all-grey companion strip per panel = the counterfactual twins
  (visually proving the control). 2304 tiles at pilot scale reads as texture; panels by wave at scale.
- **F8 — Delivery Sankey.** all present rows flowing L1→L2→L3→L4→`outcome_band`, link width = count,
  colored by `failure_stage`; self-contained plotly HTML + a matplotlib alluvial export for print.
  The funnel with full flow structure, including the `template_incompatible` branch exiting before
  tokenization (and `not_retrieved` once RAG lands).

---

## 8. Code scaffolding

New subpackage mirroring the repo's structure and gates (typed, offline-tested, `ruff`/`mypy` clean):

```
src/trigger_audit/analysis/
  __init__.py
  loading.py    # results+manifest+bases -> tidy DataFrame; joins/derived cols (§2); reconciliation; Parquet cache
  controls.py   # Gate 0: counterfactual verification -> typed verdict + offending-row dump
  stats.py      # wilson_ci, cluster_bootstrap_rate, cluster_bootstrap_diff, exact_mcnemar,
                # firth_logit (or cluster-robust GLM wrapper), tost_equivalence
  vocab.py      # canonical level orders + policy-config-driven display names/mechanism (correction 5)
  tables.py     # T1-T7 as DataFrames + CSV/MD/LaTeX renderers (fixed orders, n + base-count)
  figures.py    # F0-F8, one function each; palette/order constants at module scope
  report.py     # orchestrator: gate -> reconcile -> tables -> stats -> figures -> report.md (+ report.html)
```

**Signatures (stable contract):**
```python
def load_trials(results: Path, manifest: Path, bases: Path, *, policies_cfg: Path,
                models_cfg: Path, require_complete: bool = False) -> pd.DataFrame: ...
def verify_counterfactual(df: pd.DataFrame) -> ControlVerdict: ...      # raises/flags on leak
def cluster_bootstrap_rate(df, value: str, cluster: str = "base_id",
                           n: int = 2000, seed: int = 0) -> tuple[float, float, float]: ...
def delivered_rate_table(df: pd.DataFrame) -> pd.DataFrame: ...          # T1, present rows only
```

- **CLI:** `trigger-audit analyze <results-dir> --manifest … --bases … --policies-config …
  --models-config … --out outputs/analysis/<name>/`. Keep `score-survival` as the quick-look; `analyze`
  is the real layer. Retire `scripts/pilot_report.py` once `analyze` reproduces it exactly (its
  docstring names this the natural next step).
- **Output layout:** `outputs/analysis/<run>/{trials.parquet, gate.json, tables/, stats/, figures/,
  report.md, manifest.json}`; `manifest.json` records input sha256s, counts, package + tokenizer
  versions, and (git-less repo) file mtimes/hashes — every number traceable.
- **Runs locally**, not on the cluster (results are small JSONL; no weights). Cluster side only
  rsyncs `outputs/survival_results/`.
- **Deps** behind `[analysis]` (§5.4).
- **Tests (offline):** synthetic `SurvivalResult` fixtures with hand-computed rates; a deliberately
  leaking twin fixture proving Gate 0 exits non-zero; a golden T1 test on the checked-in
  `outputs/pilot/survival.jsonl` (assert exact reproduction of `pilot_report.py`'s conditioned table);
  pairing test asserting `pair_id` yields 1152 pairs of 2 (not 384 of 6) on the pilot; figure smoke
  tests (render, axes, palette stability); reconciliation fault-injection (missing/duplicated/divergent
  shard).

---

## 9. Sequencing & delegation (supervisor model, `docs/tasks/NN-*.md`, verify each)

Acceptance criteria are keyed to the **real pilot numbers** so "green" is checkable:

1. **Task 12a — persist pipeline-meta** (§3; joint with the pilot-run agent). Accept: new rows carry
   the metadata block; `metadata=={}` old rows still load; `DATA_CONTRACTS.md` updated; gate green.
2. **Task 12 — loading + gate + vocab** (`loading.py`, `controls.py`, `vocab.py`, CLI skeleton).
   Accept on `outputs/pilot/survival.jsonl`: reconciles 2304↔2304, 1152/1152 present/absent; Gate 0
   clean; `pair_id` → 1152 pairs of 2; reproduces `pilot_report.py`'s conditioned table **exactly**;
   fault-injection reconciliation demonstrated.
3. **Task 13 — stats** (`stats.py`). **DONE**: `wilson_ci`, `bootstrap_rate_ci`,
   `bootstrap_paired_diff_ci`, `exact_mcnemar_p`, `mcnemar_from_pairs` (CI headline / risk diffs /
   McNemar), **plus the equivalence layer**: `bootstrap_diff_samples` (paired + unpaired),
   `tost_equivalence`, and both `holm` + `benjamini_hochberg` correctors — unit-tested vs hand values.
   Surfaced as the H2 (`h2_invariance_table`) and H4 (`h4_parity_table`) tables, each showing both
   corrections and a configurable margin. **DONE (2026-07-03)**: Firth-penalized logistic
   (`firth_logit` / `firth_logit_from_frame`, pure-numpy, no new dependency) as the pooled H1/H3
   sensitivity view under complete separation, unit-tested on a separable toy (converges finite where
   a vanilla GLM diverges). The per-cell exact/CI machinery remains the primary path.
4. **Task 14 — tables + core figures** (T1–T7, F0–F4). **DONE**: tables (§ above); `figures.py` with
   F0 scaffolding, F1 heatmap, F2 cliffs, F3 funnel, F4 composition — validated colorblind-safe
   palette (`scripts/validate_palette.js`), deterministic Agg render, summarize-family excluded from
   the headline. Smoke-tested; visually inspected on the pilot.
5. **Task 15 — showpieces + report** (F5–F8, `report.py`). **DONE (2026-07-03)**: F5 landing map (uses
   the delivered rows' `trigger_relative_position`), F7 wall of trials, F8 delivery flow (a
   self-contained smooth matplotlib alluvial/Sankey — no plotly dependency), and **F6** (anatomy of the
   cut — signed `cut_offset` flattened from the §3 metadata; degrades to an annotated empty panel when a
   run carries no head-cut-inside-trigger rows, as at the 256-budget pilot). Also **T6** (misattribution)
   and **T7** (boundary census) tables and LaTeX renderers. All built, tested, and rendered into the
   report.
6. **First real exercise:** run `analyze` on the **Hyak pilot** output when it lands — 432 trials, 3
   real tokenizers, first genuine (tiny) H2 slice. Expected: Gate 0 clean; T1 matching the local
   pattern for `none`/`truncate_head`×`prefix`/`end`; F6 populated only if the 256-token budget cuts
   inside a trigger.
7. **Pre-reg amendment** (TOST margin, Holm) dated before the first full-wave analysis.

---

## 10. Open questions for Saki (before the first full-wave analysis)

> **RESOLVED 2026-07-03:** #1 (±5 pp) and #2 (Holm primary, BH secondary) are locked in the
> `PRE_REGISTRATION.md` amendment; #3 (§3 producer change) was approved and landed; #5 (`trigger_type`
> disaggregated in the headline) is implemented. **Still genuinely open: #4 (full-run scale)** — it
> depends on the achieved cluster-bootstrap CI widths from the real pilot (§5.3), so it stays open
> until those widths are in hand.

1. **TOST equivalence margin** — confirm ±5 pp (drives every H2/H4 "equivalent" verdict).
2. **Multiplicity** — Holm (proposed) vs BH-FDR.
3. **§3 producer change** — approve persisting the pipeline-meta block now (unlocks F6/T7 cleanly)
   vs relying on the 2%-sample fallback.
4. **Full-run scale** (design-doc item E) — §5.3 prints achieved CI widths to inform this; confirm the
   target bases/family once the pilot widths are in hand.
5. **Boundary trigger handling** — confirm reporting `trigger_type` disaggregated in the headline
   (recommended) so `boundary_001`'s by-design cuttability isn't averaged into the delivery rate.

---

## 11. Pitfalls this guide guards against (checklist)

- Treating `delivered` and `token` as two measurements — they are one (correction 1).
- Pairing McNemar on `pair_key` (collapses 3 triggers) instead of `pair_id` (correction 2).
- Expecting `truncation.dropped_head` on the row — it isn't persisted (correction 3 / §3).
- Using the row's `tokenizer_id` as an HF repo id — it's an alias (correction 4).
- Hardcoding pre-reg policy/position names — the data uses registry ids (correction 5).
- Reporting `summarize_*` delivery as valid — it's a placeholder stub (correction 6).
- Pooling the boundary trigger / triggers-within-base as independent — base is the unit (correction 7).
- Reading McNemar as evidence — the control arm is degenerate (correction 8).
- Vanilla logistic on separated cells — Firth/exact only.
- "No significant difference" = invariance — use TOST + enumerate discordant cells.
- Pooling ragged strata (Pythia caps, family positions, template-incompatible) — comparability + per-cell n.
- Presenting the 2% prompt-log sample as the population — label the fraction.
- Non-reproducible outputs — canonical orders, seeded jitter, hashed-input manifest.
```
