# Project 2 — Fan-Out Workflow (activation-probe trigger detection)

**Status:** working draft (v1). Grounds every phase in the higher-level plans
(`PROJECT2_MASTER.md`, `PROJECT2_FOUNDATIONS.md`, `PROJECT2_EXPERIMENT_PLAN.md`,
`docs/tasks/project2/*`). This document is the executable process for implementing and fanning out
all of Project 2. It is **local-only** (git-excluded); it never enters the public P1 repo.

---

## 0. Situation & recovery status (read first)

Project 2 code was deleted from the working tree during the P1 public-release scrub, then partially
recovered:

- **Restored from `deploy/payload.tgz` (Jul-4) and validated:** `activations/{extractor,pooling,store}`,
  `probes/{linear,metrics,calibration,aggregation}`, `experiments/probe_detection/{config,dataset,runner}`,
  `schemas/probes.py`, `configs/probe_detection.example.yaml`. The offline reference-backend probe
  experiment runs end-to-end (per-layer + aggregate AUROC, calibrated achieved-FPR with Wilson CIs).
- **Lost (Jul-5–6 additions) — must be REBUILT from spec:** `activations/slicing.py`,
  `experiments/probe_detection/{grid,selection,generalization}.py`, `models/{backdoor_loader,
  backdoor_registry,asr_verification,recipe}.py`, `analysis/probe_{loading,stats,tables,report,figures}.py`,
  `configs/probe/{E0..E3}.axes.yaml`, `configs/backdoor_models.example.yaml`,
  `scripts/slurm/run_probe_{experiment,extraction}.slurm`, and the P2 test suite.
- **Intact:** all P2 design docs (`docs/PROJECT2_*.md`, `docs/tasks/project2/*`), so the lost modules
  are fully specified. Rebuilding them **is** the "implement pieces of Project 2" work.

**Repo hygiene:** all P2 paths are in `.git/info/exclude` (local-only). The public repo stays P1-only;
P2 is driven by local, excluded entry points (a `scripts/p2/` driver + `cli_probe.py`), never by the
tracked public `cli.py`.

---

## 1. The estimand and why it already beats the baseline

**Estimand:** `P(probe fires | trigger delivered)` = TPR at a **calibrated** threshold, at FPR budgets
`[1e-2, 1e-3]` — not AUROC (`PROJECT2_MASTER.md:§3`). Reported **twice**: *delivered-only* (verified
positives + clean negatives, isolating the estimand) and *all-trials* (partial-survival negatives
count against the budget, deployment-pessimistic). Achieved FPR reported over all-negatives and
clean-negatives, each with a Wilson 95% interval.

**Three populations** (only a delivery-verified dataset can separate them): delivered positives; clean
negatives; partial-survival negatives (inserted but dropped/corrupted upstream — `label=False` yet
fragments may contaminate activations). Handling population 3 is the central methodological content.

### Positioning vs the shared MATS baseline (Chawla: long-context two-view)

That baseline's contributions and our strict generalizations:

| Baseline result | Our design (broader + more rigorous) |
|---|---|
| One appended trigger at a fixed end position | P1's **5 trigger types × 7 positions × 5 policies × 3 budgets × 4 tokenizers** grid supplies the positive/negative manufacture; position is an axis, not a constant. |
| "Two-view" = `min(s₁, s₂)` over 2 layers reduces FPR | `probes/aggregation.py` **registry** — `min` is one aggregator; we shoot out min vs product vs quantile vs stacked-logistic vs calibrated AND/OR across **4–8 layers**, with the stacker fit on the calibration split (leakage-safe). |
| Truncation deletes the trigger → vacuous 0/0 metrics (a *finding*) | We **condition on verified delivery** (P1 `final_token_trigger_present`) and report the three populations, so vacuous relabeling cannot happen silently; "trigger missing" is a delivery outcome P1 already measures, not a probe TN. This is the baseline's own recommended fix, built in. |
| Length-generalization FPR is non-monotone (seed-averaged) | **base_id-clustered** leakage-safe splits + `base_id`-clustered bootstrap inference (ported from P1), not seed-averaging; long-context / RAG cells are first-class strata. |
| Synthetic appended trigger, canary only | **Real backdoored weights** as a Tier-3 target (B4G LoRA recipe), with the honest `canary ≠ backdoor` scoping; plus **adversarial obfuscation** stress (`obfuscated-activations`). |
| Fixed calibration | **Clean-only calibration** (FPR budget is a statement about clean traffic); `TRIGGER_SPAN` used only as an oracle diagnostic with operator-confound control. |

Net: the baseline demonstrates the failure mode; **we make it the axis we condition on and then measure
detectability rigorously across scale, style, position, and adversary.**

---

## 2. Component architecture & rebuild list

*(Finalized from the plan-digest — see `docs/tasks/project2/*` specs. Interfaces below are
caller-derived and authoritative for the rebuild.)*

**Critical finding (from the component digest):** the restored files are the **Wave-1 baseline**
(the "already exists — do not rebuild" snapshot in `00-BUILD-PLAN.md`), i.e. *before* components
C/D/G/Z edited them. So reconstruction is BOTH: (a) create the truly-missing new files, AND (b)
**re-apply the C/D/G/Z edits to present-but-stale files** — `config.py` (+device/revision/
layer_depth_fractions/synthetic_mode/predictions_out/reuse_store/generalization), `dataset.py`
(+twins + metadata keys), `runner.py` (+slicing/predictions/reuse/generalization branch), `store.py`
(+pooling in the key), `schemas/probes.py` (+`ProbePrediction`), the `__init__`s, and `cli.py`
(+`run-probe-experiment`/`select-probe-subset`/`extract-activations`/`expand-probe-grid`, **local-only,
never the public cli.py** — a separate `cli_probe.py`). Component A (`io/final_tokens.py`,
`SurvivalResult.final_token_ids`, `--final-tokens-out`) and component I (all P2 docs) already landed.

**Present (restored, offline-tested Wave-1 baseline):**
`activations/{extractor,pooling,store}`, `probes/{linear,metrics,calibration,aggregation}`,
`experiments/probe_detection/{config,dataset,runner}`, `schemas/probes.py`.

**Reconstruction waves (each module ships its offline test; base package stays torch-free):**
- **Wave 1 (disjoint, parallel):** D dataset-twins · H `models/*` backdoor-safety · neutral
  `generalization.py` (imports only `schemas.probes` — breaks the config↔grid cycle) · `slicing.py`.
- **Wave 2:** C probe-runtime (config/runner/store/schemas edits + `ProbePrediction`) · B `selection.py`
  (stratified `base_id` subset + Gate-0 `verify_subset_counterfactual`).
- **Wave 3:** F `grid.py` + `cli_probe.py` + Slurm templates + `configs/probe/*.axes.yaml` · G
  `analysis/probe_{loading,stats,tables,figures,report}.py` (base_id-clustered bootstrap, E1.5
  decomposition, tier-scoped report).
- **Integration Z:** wire `config.generalization` + runner branch; `__init__` re-exports; status.
Main-thread wires the shared `__init__` exports and runs the full gate between waves.

**To rebuild (dependency order):**

1. `activations/slicing.py` — depth-fraction → layer-index resolution (band ~68–89% of depth, portable
   across model sizes). Feeds config `layers`. *Test:* fraction→index mapping per model depth; boundary
   fractions.
2. `experiments/probe_detection/selection.py` — stratified `base_id`-grouped subset selection from P1
   survival results + **Gate-0 counterfactual verify**. Interface: `StratumTargets(delivered_positive,
   clean_negative, partial_survival_negative, boundary_corruption, stratified_sample)`,
   `select_probe_subset(results, targets, seed) -> Selection`, `subset_report(selection) -> str`,
   `verify_subset_counterfactual(results, selection) -> Verdict(.ok, .summary(), .leak_examples[])`,
   `write_selected_trial_ids(out, selection)`. *Test:* stratum coverage, base_id integrity, Gate-0
   aborts on injected leak.
3. `experiments/probe_detection/grid.py` — `ProbeGridAxes` (pydantic), `expand_probe_grid(axes) ->
   list[cfg]` (Cartesian: model × layer-fraction-set × pooling × aggregation × target-FPR-set × seed),
   `write_probe_configs(cfgs, out_dir) -> paths`, prints Slurm `--array` range. *Test:* product
   cardinality, config validity round-trip.
4. `experiments/probe_detection/generalization.py` — E2.x holdout logic (train on some styles/positions,
   test on held-out) applied through `run_probe_experiment`. *Test:* holdout partition disjointness by
   the held-out axis.
5. `analysis/probe_{loading,stats,tables,report,figures}.py` — load ProbeEvaluationResults →
   `base_id`-clustered bootstrap CIs + achieved-FPR tables + figures. *Test:* deterministic tables from
   fixture results; CI monotonicity.
6. `models/{backdoor_registry,backdoor_loader,asr_verification,recipe}.py` — Tier-3 safety harness
   (allowlist + sha256 + benign-canary ASR gate). *Test:* refuses non-allowlisted; refuses non-benign
   payload at load.
7. `configs/probe/{E0..E3}.axes.yaml`, `configs/backdoor_models.example.yaml`, probe Slurm templates.
8. P2 test suite (`test_probe_*`, `test_activation_*`, `test_backdoor_registry`, `test_asr_verification`,
   `test_firth`).

Each rebuilt module ships with its offline test and must pass `ruff`/`mypy` before the next depends on it.

---

## 3. Experiment grid (E0–E4) — finalized from the plan digest

**Pre-registered fixed defaults (2026-07-06 amendment, all experiments):** depth-fraction layer naming
(HF index 0=embeddings); informative band `0.5–0.89`, default probe set `{0.5, 0.66, 0.75, 0.89}`;
deployable pooling `mean` (`last`/`max` ablation; `trigger_span` oracle-only, random-span fallback ON);
headline aggregation closed-form (`mean/max/product/quantile`; `stacked_logistic` caveated, fit on
CALIBRATION); FPR targets `[1e-2, 1e-3]` (**1e-3 bounded-only unless ≥~1000 clean negatives**);
clean-only calibration; `base_id`-grouped splits `0.5/0.25/0.25`; L2 logistic `l2=0.01, lr=0.5,
max_iter=1000`; inference = cluster-bootstrap over `base_id` + Wilson (FPR) + TOST ±5pp/Holm
(invariance). **No numeric TPR bar — success is structural.** Model pool: Qwen3-0.6/1.7/4/8B +
Pythia-1B (Gemma/TinyLlama in template/Tier-3 only).

**Shared real-model subset targets** (stratified `base_id` selection): `delivered_positive=300`,
`clean_negative=1000` (to resolve 1e-3), `partial_survival_negative=300`, `boundary_corruption=50`,
`stratified_sample=200`, `seed=0`. n_train/calib/test are *derived* by the 0.5/0.25/0.25 base-grouped split.

### Tier 0 — Instrument validity (OFFLINE reference backend, CPU, no real weights) — run first, no approval
- **E0.1** determinism/recoverability **gate**: per-layer AUROC>0.9 + byte-identical reruns. *A pass proves only plumbing.* (runnable on restored `simple` synthetic)
- **E0.2** calibration honesty & **FPR-resolution limit**: sweep clean-neg count `n∈{30,100,300,1000,3000}` × targets `{1e-2,1e-3}`; confirm `target<1/n ⇒ FPR=0` with wide Wilson CI. *Pins the ~1000-clean-neg budget for 1e-3.* (runnable on restored `simple` synthetic)
- **E0.3** leakage ablation: `base_id`-grouped vs example-level split on twin data → inflation Δ. *(needs twins generator, component D)*
- **E0.4** operator-confound ablation: `trigger_span` random-span fallback ON vs OFF on trigger-free data. *(needs D)*
- **E0.5** three-population calibration ablation: clean-only vs clean+partial-survival calibration pool → threshold/TPR bias. *(needs D)*

### Tier 1 — Existence & shape on real weights (GPU, canary, delivery-verified)
- **E1.1** per-model **layer sweep** (all 5 models; depth-fraction sweep) → informative band. *Prereq for E1.2–E1.4 and all of Tier 2/3.*
- **E1.2** pooling ablation (`last/mean/max`; span oracle-separate).
- **E1.3** multi-layer **aggregation shootout** (best-single vs mean/max/product/quantile/stacked) → recovery Δ. *(this is where the baseline's two-view `min` becomes one point.)*
- **E1.4** **scale curve + INLP** direction-count (0.6B→8B; small-model ceiling/ablatability). *(INLP loop needs building.)*
- **E1.5** **delivery-conditional decomposition (headline):** `P(fire)` all-trials vs `P(fire|delivered)` → "X% of apparent probe misses are delivery failures." *(runs alongside E1.1; uses existing dual metrics.)*

### Tier 2 — Robustness to deployment shift (GPU, canary) — via `GeneralizationSpec` holdouts
- **E2.1** delivery-style generalization (`train_policies`/`test_policies`).
- **E2.2** long-context generalization (`train_context_max`/`test_context_min`) — the hardest production failure.
- **E2.3** trigger-type & **distributed-trigger** (CTBA/MTBA) detection vs a regex/string-match null.
- **E2.4** cross-model transfer (same hidden-width only; not a `GeneralizationSpec` — separate route).

### Tier 3 — Real backdoored models (GPU, **gated**: allowlist + sha256 + benign-canary ASR + Gate 0)
- **E3.1** first real backdoor-detection claim (DPA taxonomy {BadNet,VPI,MTBA,CTBA,Sleeper} × B4G tasks; TAR_w vs TAR_wo).
- **E3.2** poison-rate × trigger-complexity → detectability floor per size.
- **E3.3** behavioral vs representational agreement (`P(probe|delivered)` vs `P(CANARY behavior|delivered)`).
- **E3.4** baseline battery (Apollo probe zoo + DefenseBox + regex null).

### Tier 4 — Adversarial robustness (GPU, gated; attack/defense code out-of-scope of restored components, must be built)
- **E4.1** natural-obfuscation curve · **E4.2** probe-in-the-loop evasion collapse · **E4.3** GCG/FLRT suffix attacks · **E4.4** OAT defense on **held-out** attacks. *Deliverable is a degradation curve, never a robustness proof.*

**Fan-out order.** E0.1 (gate) → **E0.2 (pins clean-neg budget)** ∥ {E0.3,E0.4,E0.5 after D} → real-model prereqs P1–P5 → **E1.1 (band) → E1.2/E1.3/E1.4** ∥ E1.5 → Tier 2 (reuses band+subset) → **Tier 3 (gated on allowlisted+ASR weights)** → Tier 4 (needs Tier-3 weights + attack code).

---

## 4. Phased workflow (implement → test → fan out)

- **Phase R — Rebuild** the §2 modules in dependency order, each with an offline test gate. Subagents
  reconstruct in parallel where interfaces are independent (slicing ∥ selection ∥ analysis), then
  grid/generalization/models. I own interface/estimand/safety correctness and integration.
- **Phase 0 — E0 instrument** on the reference backend (offline, no approval), then a single real GPU
  extraction to anchor the layer band.
- **Phase 1 — E1** existence/scale (GPU).
- **Phase 2 — E2/E2.x + E4** shift, generalization, aggregation shootout (GPU).
- **Phase 3 — E3** backdoor: scaffold + Gate to safety; real-weights run awaits approval.

Every experiment: `selection.py` (Gate-0 must pass) → extraction (reference or GPU) → probe
train/calibrate/eval → `analysis/probe_*` (base_id-clustered CIs). Nothing is reported off the test ROC;
TPR is read at the calibrated operating point.

---

## 5. Self vs subagent division (combine reasoning for the dense parts)

- **I (main) own:** the estimand/inference contract, the safety tiering + Gate-0, module interfaces,
  integration, and mech-interp interpretation of results.
- **Subagents own:** parallel module reconstruction (each returns code + passing test), per-experiment
  config generation, **adversarial verification** of every empirical claim (independent skeptics that
  try to refute a probe result before it's believed), and literature-recipe extraction from
  `third_party/`. For intellectually dense calls (layer/aggregation choice, confound handling), I run a
  small judge panel of subagents with distinct lenses (leakage, calibration, operator-confound,
  scale-ceiling) and synthesize.

---

## 6. Offline→GPU tiering & safety gates

- **Tier 0–2 (canary):** reference backend and real HF extraction on harmless canary triggers — no
  approval; this is the bulk of E0–E2/E4.
- **Tier 3 (real backdoor):** only after license review + recorded sha256 + benign-canary ASR
  verification (`installed == true`) and an explicit allowlist flip. `canary ≠ backdoor`: no offline/
  Tier-0–2 run may claim backdoor detection.
- **Gate 0 (counterfactual control):** every selected subset must show zero leaks on trigger-absent
  twins or the run aborts and writes nothing (parity with P1 discipline).

---

## 7. Open decisions (from `PROJECT2_MASTER.md:§13`) — proceed on the non-blocking ones

Proceed now (no user input needed): rebuild all modules; E0–E2/E4 on canary at Tier 0–2; port the
inference discipline; write the probe pre-registration draft. **Flag for the user (blocking only for
Tier-3 / final numbers):** (1) numeric success bar (size-aware TPR@FPR) vs purely structural; (2)
approval to fine-tune real small backdoored weights (B4G) for E3; (3) 100% final-token persistence in
the next P1 run vs re-derivation; (4) whether the behavioral `CANARY_SEEN` path stays in scope.

---

## 8. Inference discipline (port from P1)

Extend P1's `base_id`-clustered bootstrap + equivalence (TOST) testing to the probe estimates; today
P2 reports Wilson-on-FPR only. This is required before any headline TPR@FPR comparison across
size/style is credible.
