# Project 2 — Continuity Audit

**Purpose.** A single re-read of *every* Project-2 source plus the two new draft documents
(`PROJECT2_MASTER.md`, `PROJECT2_EXPERIMENTS_REVIEW.md`), checking for the things that silently break a
program across documents and code: **contradictions, terminology drift, field-name drift, unmet
dependencies, and gaps between what a doc assumes and what the code actually does.** Every finding is
evidence-backed with a `file:line` and rated by how much it threatens a real measurement wave. This
audit's conclusions are folded into the final `PROJECT2_EXPERIMENT_PLAN.md` as prerequisites and
scoping caveats.

Severity: **S1** = blocks a real-model wave; **S2** = corrupts results if unaddressed; **S3** =
discipline/clarity, low risk if known.

---

## A. Conceptual / terminology continuity

**A1 (S2) — Two incompatible "activation phase" conceptions.** The repo carries a *behavioral*
follow-on (`GenerationResult`: emit `CANARY_SEEN`, string-match — `DATA_CONTRACTS.md:115-122`,
`EXPERIMENT_DESIGN.md:213`, `ARCHITECTURE.md:77-78`) and a *representational* one (linear probes →
`ProbeEvaluationResult` — `PROJECT2_FOUNDATIONS.md` throughout, the built code). Only
`ANALYSIS_PLAN.md:75` bridges them. **Resolution (adopted):** Project 2 = the probe; the behavioral
path is a *complement* studied only in E3.3 (agreement), never the deliverable. Documented in
`PROJECT2_MASTER.md §4`. Continuity risk if a future reader wires the probe to the `CANARY_SEEN`
machinery expecting them to be the same thing.

**A2 (S3) — "activation" and "layer" are overloaded.** In `GENERIC_PLAN.md`, "activation" = behavioral
emission and "Layer 1–4" = prompt-logging stages (`GENERIC_PLAN.md:251-273, 296-300`); in Project 2,
"activation" = hidden state and "layer" = transformer block. **Resolution:** the master doc pins both
usages; the experiment plan uses "hidden-state activation" and **depth-fraction** (not raw index)
throughout (see A6).

**A3 (S3) — Model-suite mismatch across sources.** `GENERIC_PLAN.md:910-914` names Qwen3-1.7/4/8B +
Pythia-1B + OLMo; the pre-reg and P2 docs name Qwen3-**0.6B**/1.7/4/8B + Pythia-1B + Gemma/TinyLlama
(`PRE_REGISTRATION.md:17`, `PROJECT2_RESOURCES.md:16-17`). Not a contradiction — the fuller suite is a
P2 artifact, justified because the same-tokenizer Qwen sizes exist *for the weights/activation phase*
(`ONE_SHOT_PLAN.md:41`). Recorded so no one "reconciles" the suites by deleting 0.6B/Gemma/TinyLlama.

---

## B. Data-contract continuity (the load-bearing one)

**B1 (S1) — `final_token_ids` is computed but not persisted; the probe's required join input has no
producer.** The probe runner reads a `final_tokens_path` JSONL of `{trial_id, final_token_ids}`
(`runner.py:328-331`). Project 1 *does* compute Layer-4 `final_token_ids` in-pipeline
(`pipelines/composition.py:102,145`, `pipelines/steps.py:95,114`; it is even available to the scorer —
`survivability_audit/runner.py:153`, `manifest_runner.py:124`). **But the persisted `SurvivalResult`
schema drops it** — it carries `final_prompt_token_count`, `final_prompt_text_path`,
`trigger_final_token_start/end`, and **no `final_token_ids`** (`schemas/results.py:70-86`). And
`final_prompt_text_path` is set only for the ~2% `--log-prompts` sample (`ANALYSIS_PLAN.md:177`,
`CLUSTER_EXECUTION_PLAN.md:82-83`). **Consequence:** as-is, a real probe wave has no input for >98% of
trials. **Fix (small, since the data already exists at scoring time):** persist `final_token_ids` — as
a sidecar `final_tokens.jsonl` the scorer writes, or as a new `SurvivalResult` field — for the probed
subset (aligns with the stratified selector, B/C below). This is the #1 prerequisite for any Tier-1+
experiment. *This refines the master doc's earlier looser statement of the gap.*

**B2 (S2) — Tokenizer-id alias (inherited P1 pitfall).** `SurvivalResult.tokenizer_id` is a config
alias (e.g. `qwen3-0_6b`), not a loadable HF repo id (`ANALYSIS_PLAN.md:112, 158`). Any probe code that
loads a model or re-tokenizes must resolve the alias via the models config, never trust the row. The HF
extractor is constructed from `config.model_id` (`runner.py:302-306`), so the *config* must carry the
real HF id + revision — reinforcing B4.

**B3 (S2) — Policy/position values are registry ids, not pre-reg names.** `pipeline_policy` is
`truncate_head` etc., not `head_truncation` (`ANALYSIS_PLAN.md:163`). E2.1's "train on some policies,
test on others" must group on the *actual* registry ids carried in `ProbeExample.metadata`
(`dataset.py:42`), not the prose names.

**B4 (S1) — `device`/`revision` not threaded into the probe HF extractor.** `run_probe_experiment`
calls `make_activation_extractor(config.extractor_backend, model_id=…, seed=…, hidden_size=…,
num_layers=…)` (`runner.py:302-308`) — it passes **neither `device` nor `revision`**, though
`HFActivationExtractor.__init__` accepts both (`extractor.py:147-154`). So a cluster run cannot select
the GPU or pin the model revision from config. **Fix:** add `device`/`revision` (and
`trust_remote_code`) to `ProbeDetectionExperimentConfig` and thread them through. Prerequisite for a
reproducible GPU wave.

**B5 (S3) — `trigger_present_in_final_tokens` vs `final_token_trigger_present`.** `schemas/results.py`
exposes both a `SurvivalResult.final_token_trigger_present` (`:70`) and a
`trigger_present_in_final_tokens` (`:117`, on a different model — likely the assessment/summary type).
The probe label uses the former (`dataset.py:34`). Continuity note: keep these distinct; do not join the
probe on the latter.

---

## C. Infrastructure / test-scaffold continuity

**C1 (S1) — Stratified-subset selector is a TODO.** Activation extraction must run on a stratified
subset (controls + delivered/not-delivered + boundary + sample), but `ManifestBuilder` flags generation
by a blind deterministic fraction; stratified selection *from survival results* is unbuilt
(`CLUSTER_EXECUTION_PLAN.md:73-77`). Both the probe wave (E1.x) and any behavioral wave depend on it,
and it must respect `base_id` grouping so the selected subset can still be split leakage-free. Build
before Tier 1.

**C2 (S2) — Shipped synthetic data cannot exercise two regimes the experiments rely on.** The synthetic
generator gives every example a **unique `base_id`** and sets **`trigger_inserted == label`**
(`dataset.py:130-139`). Therefore: (a) there are **no counterfactual twins**, so the base_id-leakage
ablation (E0.3 / AR-3) cannot be demonstrated on it; (b) there are **no partial-survival negatives**, so
the three-population calibration story (E0.5) is exercised only by hand-built fault-injection fixtures,
not the default smoke. **Fix:** extend the synthetic generator with a `twin`/`partial_survival` mode
(shared `base_id` across a present/absent pair; some inserted-but-undelivered negatives) so Tier-0
experiments run offline as intended.

**C3 (S3) — No activation batching / extract-once reuse.** `HFActivationExtractor.extract` is batch-1
(`extractor.py:192`) and the runner re-extracts per run (`runner.py:189-201`) with no read-back from the
`ActivationStore`, so a pooling sweep (E1.2) re-runs every forward pass. Not a correctness bug; a cost
multiplier on the GPU wave. Optional optimization: extract-once, pool-many from the store.

**C4 (S3) — Layer default drift.** `config._default_layers()` = `[1,2,3,4]` (`config.py:14`) but the
example YAML uses `[0,1,2,3,4]` (`configs/probe_detection.example.yaml`). Harmless, but E1.1 should set
layers by depth-fraction explicitly and not rely on either default.

---

## D. Statistical / methodological continuity

**D1 (S1) — Project 2's probe design is NOT pre-registered.** `PRE_REGISTRATION.md` contains no probe,
FPR-target, layer, or pooling section (confirmed: no matches for probe/fpr/hidden-state/activation-probe
in the file). Unlike P1's H1–H4 + the dated TOST amendment, the probe estimand and operating points are
un-preregistered. **Fix:** a dated `PRE_REGISTRATION.md` amendment fixing layers (by depth-fraction),
pooling, aggregation, FPR targets, the dual all/delivered-only reporting, and the base_id-clustered
inference — before the first measurement wave. Otherwise every Tier-1+ number is exploratory.

**D2 (S2) — Inference discipline not ported from P1.** P2F reports Wilson CIs on FPR only
(`PROJECT2_FOUNDATIONS.md:92-98`); it does not cluster-bootstrap `P(fire|delivered)` over `base_id` or
use equivalence testing for invariance claims. The experiments doc (Part 6) requires both. Continuity:
the ±5 pp / Holm amendment is scoped to P1 delivery rates (`PRE_REGISTRATION.md:63-84`); extend it to
the probe layer or state that probe inference uses a *different* discipline.

**D3 (S1, carried from AR-1) — FPR 1e-3 is unresolvable at pilot base counts.** With `base_id` as the
unit and ~100–300 bases/family, 1e-3 needs ≥~1000 independent clean negatives to resolve; below that,
calibration honestly returns FPR 0 with a wide Wilson interval (`PROJECT2_FOUNDATIONS.md:88-90`, E0.2).
The default `target_fprs=[1e-2,1e-3]` (`config.py:17-18`) therefore *promises an operating point the
pilot cannot measure*. **Fix:** E0.2 sets the base-count requirement; report 1e-2 as resolvable and 1e-3
as bounded-only unless clean negatives are scaled. Consistency: the master doc §3 and experiments AR-1
already state this; the plan must not headline a 1e-3 TPR from pilot data.

**D4 (S3) — Stacked-aggregator adaptivity.** `stacked_logistic` is fit on CALIBRATION and thresholds
are set on the same CALIBRATION negatives (`runner.py:126-138`) — a mild double-use the code flags.
Continuity: the experiments doc (AR-8) already restricts headline aggregation to the closed-form
combiners and treats stacked as caveated. No contradiction, but the plan must keep that restriction.

---

## E. Scientific-scope continuity

**E1 (S1, the central caveat) — canary ≠ backdoor.** Every source that reports numbers today does so on
harmless canaries or the reference extractor; genuine backdoored positives do not exist yet
(`PROJECT2_FOUNDATIONS.md:149-150`, `PROJECT2_RESOURCES.md:123-125`). The master doc (§10.7) and
experiments (AR-0) both state this; the plan must scope Tier-0–2 claims to "delivered-canary
representation" and make Tier 3 the only source of backdoor-detection claims. **No document may state a
backdoor-detection result from canary data.**

**E2 (S2) — Cross-model transfer is dimensionally constrained.** Different Qwen sizes have different
`hidden_size`, so a probe vector is not portable across widths (AR-4). The plan's E2.4 must restrict
literal transfer to same-width pairs and label cross-width transfer a separate (projection-based)
experiment. Any doc that says "train on 8B, test on 0.6B" without a projection is malformed.

**E3 (S2) — Span pooling assumes a contiguous trigger; distributed triggers may be unreachable by a
last-token/span linear probe.** `TRIGGER_SPAN` and the whole span machinery assume one contiguous span
(`PROJECT2_FOUNDATIONS.md:151-152`); CTBA/DemonAgent triggers have none (E2.3, AR-12). Continuity: the
experiments doc already routes distributed triggers to mean/trajectory pooling and treats a negative
result as a finding; the plan must keep that and not claim span pooling generalizes to them.

---

## F. Internal continuity of the new documents

- **F1 (resolved).** `PROJECT2_MASTER.md §11` originally described the join loosely as
  "`final_prompt_text_path` / the final token ids"; B1 above makes the precise state explicit (computed,
  not persisted). The plan uses the precise version.
- **F2 (consistent).** The estimand, dual-reporting contract, three-population split, and clean-only
  calibration are stated identically in master §3/§6, experiments Tier-0/1, and (below) the plan — no
  drift.
- **F3 (consistent).** The adversarial caveats AR-0/1/5/6/7 in the experiments doc map one-to-one onto
  continuity items E1, D3, AR-5(feasibility), AR-6(scope), AR-7(CoT) and are carried into the plan's
  "what each experiment may conclude" column.

---

## G. Prerequisite checklist (the actionable output)

Ordered; the first four block a real-model wave:

1. **B1** — persist `final_token_ids` for the probed subset (small; data exists at scoring time). **S1**
2. **C1** — build the stratified, `base_id`-aware subset selector. **S1**
3. **B4** — thread `device`/`revision`/`trust_remote_code` into the probe config → HF extractor. **S1**
4. **D1** — dated `PRE_REGISTRATION.md` amendment for the probe design + FPR targets + inference. **S1**
5. **AR-11 / Gate 0** — run P1's counterfactual control on the probed subset so labels are trusted. **S1**
6. **C2** — extend the synthetic generator (twins + partial-survival) so Tier-0 runs offline. **S2**
7. **D2/D3** — port base_id-clustered bootstrap; set base counts from E0.2; scope 1e-3. **S1/S2**
8. **E1** — enforce the canary≠backdoor scoping in every downstream claim. **S1 (scope)**

Nothing above contradicts the built code; each is an additive change or a scoping rule. The harness is
internally consistent — the gaps are all at the Project-1↔Project-2 seam (data persistence + selection)
and in the not-yet-written pre-registration.

---

*Next and final: `PROJECT2_EXPERIMENT_PLAN.md` — every experiment plus the rock-solid argument, with
these prerequisites and caveats built in.*
