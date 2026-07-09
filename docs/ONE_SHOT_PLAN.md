# One-shot plan — the maximal validated survival run

How to run the most experiments we can in a single Hyak submission, using **only methodologies
validated end-to-end**. Written 2026-07-03 after items 2–5 landed. Companion to
`PRE_REGISTRATION.md` (locked design) and `deploy/README.md` (mechanics).

---

## 1. Readiness

| Item | State |
|---|---|
| **1. Deploy the scorer fix + re-verify pilot on cluster** | **Pending your re-push.** The fetched results were scored by the pre-fix runner; a `push` ships the fixed `runner.py`. Gate this before scaling. |
| **2. Per-model length-matching** | ✅ done + verified — bases are length-matched to each model's own tokenizer, tagged with model-qualified ids, merged into one collision-free store (27/27 unique, runner resolves all shard trials). |
| **3. Full validated grid** | ✅ done — `configs/prod/{experiment,policies,models}.prod.yaml`; validates + expands (4,050 trials/model in the smoke). |
| **4. Fully-offline HF at scale** | ✅ done — array job sets `HF_HUB_OFFLINE=1`+`TRANSFORMERS_OFFLINE=1`; **fixed a latent bug** where `prefetch_tokenizers.py` ignored `revision` (would have missed the cache under a pinned offline run). |
| **5. Persistent storage + pinned models** | ✅ config authored (revision fields). Storage switch to `/gscratch/stf` is **documented, not applied** — applying it now would strand your scrubbed env mid item-1. Apply after item 1. |
| Bonus | Cluster runner now records `template_incompatible` instead of dropping rows (Gemma-ready, H2 template-divergence). |

---

## 2. The maximal validated one-shot grid

Everything below is validated (Trials 0–5 + the cluster pilot + the new tests). Nothing here is a stub.

| Axis | Levels | n |
|---|---|---|
| **model** | qwen3-0_6b (ChatML), pythia-1b (base-completion), tinyllama-1_1b-chat (Zephyr / re-tokenizing BPE) | 3 |
| **pipeline_policy** | none, truncate_head, truncate_tail, truncate_middle, keep_recent_messages | 5 |
| **trigger_position** | prefix, middle, end, old_turn, recent_turn | 5 |
| **context_length (budget)** | 512, 1024, 2048 (model-capped) | 3 |
| **trigger_type** | rand_001, multi_001, boundary_001 | 3 |
| **data_source** | synthetic + long-doc (big.txt) | 2 arms in the base sets |
| **trigger_present** | True + counterfactual False twin | ×2 |

**Trials** = bases/model × 3 triggers × 5 positions × 5 policies × 3 budgets × 2 = **bases/model × 450**, summed over 3 models.
- **First one-shot (recommended):** `SynthCount=60` + `LongdocCount=15` → ~75 bases/model → **~101k trials** → ~21 shards (`shard_size=5000`), `--array=0-20`. CPU-only, minutes on ckpt-all.
- **Larger:** `SynthCount=200` + `LongdocCount=50` → ~250/model → **~340k trials** → ~68 shards.

### Why these models (and not the 4 Qwen sizes)
Survival is **tokenizer-only** — a "model" here is its (tokenizer, template, window). Qwen3-0.6B/1.7B/4B/8B **share one tokenizer+template**, so they are redundant for delivery; they matter only for the later *activation* phase (weights). These 3 span every validated distinct behavior. **Gemma** (the 4th behavior — strict-alternation → `template_incompatible`) is one uncommented line in `models.prod.yaml`, but is **gated**: accept its HF license + set `HF_TOKEN` first.

### Excluded, on purpose (not validated / pre-reg-forbidden)
`summarize_old_messages`/`summary_plus_recent` (no semantic scorer — pre-reg forbids), `rag_baseline` + LMSYS/WildChat real parsers (Wave 2 / `NotImplementedError`), generation/activation phase, `system`/`tool_output` positions, `near_boundary` (placement still lands-at-end), `natural_phrase`/`unicode`/`split` triggers (not in the trigger file).

---

## 3. The one design decision to confirm

**`context_length` is the truncation BUDGET, and base content length is a separate, larger knob.**
The runner uses `budget = min(context_length, model input budget)`; for truncation to bind, the base's
templated length must exceed the budget (the validated pilot used 512-token bases with a 256 budget).
So set per-model base **content length above the largest budget** in `Models_Fleet`:

| model | window | recommended TargetLen (content) | budgets that bind |
|---|---|---|---|
| qwen3-0_6b | 40960 | **4096** | 512, 1024, 2048 all truncate |
| pythia-1b | 2048 | **1536** | 512, 1024 truncate; 2048→capped, acts as a no-truncation control |
| tinyllama | 2048 | **1536** | same as pythia |

H4 "matched length" holds **within** a model (synthetic vs long-doc at that model's content length).
If you'd rather sweep *content length* as the axis (multiple base lengths per model), say so — that's a
different, larger materialization and manifest shape.

---

## 4. Launch sequence (one shot)

**After item 1 passes** (re-pushed pilot shows TinyLlama delivering correctly in cluster output):

1. Move to persistent storage + prod configs — edit `deploy/hyak.config.ps1`:
   - `RemoteRoot = '/gscratch/stf/sks0417/trigger_audit'`, `EnvPrefix = '/gscratch/stf/sks0417/ta_env'`
   - `Experiment/Models/Policies` → the three `configs/prod/*.prod.yaml`
   - `Models_Fleet` TargetLen → 4096 / 1536 / 1536; `SynthCount=60`, `LongdocCount=15`
2. Rebuild the env at the new path (once): `.\deploy\hyak.ps1 setup` → wait `SETUP COMPLETE`.
3. **One shot:** `.\deploy\hyak.ps1 push` — assembles all 3 models' length-matched corpora, builds the
   combined manifest, uploads, and submits the whole array.
4. `.\deploy\hyak.ps1 status` until all tasks COMPLETED → `.\deploy\hyak.ps1 fetch` (downloads,
   aggregates, verifies the counterfactual control; non-zero exit on any leak).

Everything is one `push` + one `fetch`. Duo prompts a couple times each (Hyak's no-SSH-key policy).

---

## 5. What comes back / analysis

Per-trial `SurvivalResult` rows → `score-survival` (per policy×position rates) + `scripts/pilot_report.py`
(conditioned survival table, counterfactual control, H4 by data_source). Still owed for the full
scientific output (the "analysis layer"): McNemar's on counterfactual pairs, logistic regression for
H1/H3, the H2 across-model and H4 across-source comparisons, and the "backdoor-failure = delivery-
failure" tables. Recommend building that on the returned one-shot data as the next step.

## 6. For a reproducible publish (later)
Pin each model's `revision:` in `models.prod.yaml` to a commit hash (prefetch + runtime both honor it),
and pin `transformers`/`tokenizers` versions (token boundaries drive truncation).
