# Spec G — Probe analysis + inference layer (the honest science)

This component turns raw probe outputs into **honest, scoped scientific claims**. The estimand is
`P(probe fires | trigger delivered)` at a calibrated FPR, with uncertainty that clusters on `base_id`
(not the trial), dual all/delivered-only reporting, and TOST for any invariance claim. It reuses the
Project-1 inference machinery rather than reinventing it.

## Files you own (new; edit only these + tests)

- `src/trigger_audit/analysis/probe_loading.py` — **new**
- `src/trigger_audit/analysis/probe_stats.py` — **new**
- `src/trigger_audit/analysis/probe_tables.py` — **new**
- `src/trigger_audit/analysis/probe_figures.py` — **new**
- `src/trigger_audit/analysis/probe_report.py` — **new**
- `src/trigger_audit/analysis/__init__.py` — add exports (append only; do not remove existing)
- `tests/test_probe_analysis.py` — **new**

Depends on component C: the per-trial `ProbePrediction` rows (`schemas/probes.py`) and
`ProbeEvaluationResult`. Reuse **existing** helpers — do NOT reimplement them:
- `analysis/stats.py::bootstrap_rate_ci` (cluster/base bootstrap), `bootstrap_paired_diff_ci`,
  `bootstrap_diff_samples`, `holm`, `benjamini_hochberg`.
- `probes/metrics.py` (AUROC, tpr_at_fpr), `probes/calibration.py::wilson_interval`.
- `analysis/loading.py` patterns for reading JSONL result files.
The `matplotlib`/`scipy`/`statsmodels`/`pandas` deps live in the `analysis` extra — import them at
module top (this package is already the analysis layer; see `analysis/figures.py`,
`analysis/stats.py`). Do NOT edit `runner.py`, `stats.py`, or any non-probe analysis file.

## 1. `probe_loading.py`

- `load_predictions(path) -> pd.DataFrame` — read `ProbePrediction` JSONL into a tidy frame with
  columns `trial_id, base_id, label, trigger_inserted, delivered, clean_negative, split,
  aggregated_score`, plus `fired__<target>` boolean columns unpacked from the `fired` dict and
  `layer__<idx>` score columns from `layer_scores`. Accept a file or a directory of files.
- `load_probe_results(path) -> list[ProbeEvaluationResult]` — read the aggregate result rows.
- Helpers to attach depth-fraction to each layer from result `metadata` (`num_layers`,
  `resolved_layers`, `layer_depth_fractions`) so every layer-keyed output can report by fraction.

## 2. `probe_stats.py` — the inference layer (pre-registered discipline)

Implement, all clustering on `base_id` per the 2026-07-06 pre-registration amendment:
- `tpr_at_fpr_delivered(preds, target_fpr) -> RateEstimate` — `P(fire | delivered)` = mean of
  `fired__<target>` over the **delivered positives** (rows with `delivered & label`), with a
  **cluster-bootstrap CI over `base_id`** via `bootstrap_rate_ci`. Return point + CI + n_bases +
  n_trials. Provide the **all-trials** analog too (over all positives) so E1.5's decomposition is one
  call apart.
- `achieved_fpr(preds, target_fpr, *, clean_only: bool) -> AchievedFpr-like` — empirical FPR of the
  fired column over test negatives (all vs clean), **Wilson** interval (reuse `wilson_interval`).
- `delivery_conditional_decomposition(preds, target_fpr) -> dict` (E1.5, the headline): reports
  `P(fire)` all-trials vs `P(fire | delivered)`, and the **fraction of apparent probe misses that are
  delivery failures** = among positives that did NOT fire, the share whose `delivered == False`. This
  is the number no insertion-labeled study can compute; make it a first-class function.
- `equivalence_tost(preds_a, preds_b, target_fpr, *, margin=0.05) -> TostVerdict` — TOST at ±5 pp for
  an invariance claim (pooling X ≈ Y, model A ≈ B, style A ≈ B). Implement as: the 90% cluster-
  bootstrap CI of the paired/independent difference in `P(fire|delivered)` is contained in
  `[-margin, +margin]` (reuse `bootstrap_diff_samples`/`bootstrap_paired_diff_ci` — this matches the
  P1 2026-07-03 amendment exactly). Return the CI, the verdict, and the CI half-width so an
  under-powered cell is visible, never silently "equivalent".
- `leakage_inflation(grouped_preds, example_preds, target_fpr) -> dict` (E0.3) — Δ in AUROC/TPR
  between a `base_id`-grouped split and an example-level split on the same data; the measured cost of
  breaking the grouping rule.
- `holm_adjust(pvalues)` / `bh_adjust(pvalues)` — thin pass-throughs to the existing functions for
  multiplicity within a hypothesis family (H-invariance across model pairs; style across cells).
- `tar_with_without(preds) -> dict` (Tier 3) — `TAR_w` = fire-rate on delivered **triggered**,
  `TAR_wo` = fire-rate on **clean**, at the calibrated FPR, each with a base-clustered CI. Label the
  output clearly as a **backdoor-detection** quantity that is only valid on real backdoored data
  (component H); refuse-by-docstring to interpret it on canary/reference runs.

Return small typed dataclasses (e.g. `RateEstimate(point, ci_low, ci_high, n_bases, n_trials)`), not
bare tuples, so tables/figures and the report read cleanly.

## 3. `probe_tables.py`

Pandas tables, each a DataFrame + a `to_markdown`-style renderer (match `analysis/tables.py`):
- Layer sweep by **depth-fraction**: `P(fire|delivered)@1e-2`, AUROC, separation, per model (E1.1).
- Pooling comparison: mean/last/max deployable + `trigger_span` flagged **oracle-only** (E1.2).
- Aggregation comparison: best-single vs closed-form combiners vs stacked (caveated) (E1.3).
- Delivery-conditional decomposition table (E1.5): all-trials vs delivered-only + delivery-failure
  fraction.
- Achieved-FPR table: target vs achieved (all + clean) with Wilson CIs; **flag 1e-3 as bounded-only**
  when `n_clean_neg < ~1000` (the resolution caveat, continuity D3).

## 4. `probe_figures.py`

Matplotlib figures (headless `Agg`, save to a figures dir; match `analysis/figures.py` conventions):
- `P(fire|delivered)` vs **depth-fraction** curve per model, with base-bootstrap CI bands (E1.1).
- Scale curve: peak `P(fire|delivered)` (and single-probe ceiling) vs model size (E1.4).
- Delivery-conditional decomposition bar (all vs delivered-only) (E1.5).
- ROC/operating-point plot with the calibrated thresholds marked (AUROC as summary only).

## 5. `probe_report.py` — the scoped narrative

- `build_probe_report(predictions_path, results_path, *, out_dir) -> ReportManifest` that produces the
  tables + figures and a `probe_findings.md` summarizing them **with claim-scoping enforced**: every
  reported number carries its tier scope. Tier-0–2 outputs are labeled "delivered-canary
  representation"; a backdoor-detection sentence is emitted **only** when the inputs are marked as
  Tier-3 real-backdoor runs (read a `scope`/`tier` marker from result `metadata`, default to canary).
  If asked to headline a 1e-3 TPR from data that can't resolve it, the report must down-rank it to
  "bounded-only" automatically. This is the discipline from `PROJECT2_EXPERIMENT_PLAN.md` Part IV.
- Record a provenance block (input hashes, seeds, extractor backend, transformers/torch versions if
  present in metadata) mirroring the P1 analysis manifest discipline (`PROJECT2_MASTER.md §11`).

## Tests (`tests/test_probe_analysis.py`)

Construct synthetic `ProbePrediction` rows in-code (no model needed):
1. `tpr_at_fpr_delivered` equals the hand-computed fire-rate over delivered positives; CI clusters on
   `base_id` (verify n_bases < n_trials when twins share a base, and that duplicating a base changes
   the CI, not the point, the way a cluster bootstrap should).
2. `delivery_conditional_decomposition` returns the correct delivery-failure fraction on a constructed
   set where some positives were not delivered.
3. `achieved_fpr` clean vs all differ when partial-survival negatives fire; Wilson interval brackets
   the point.
4. `equivalence_tost` declares equivalence for two near-identical prediction sets and non-equivalence
   for a >5 pp gap; the half-width is reported.
5. `leakage_inflation` shows AUROC/TPR ≥ under example-level vs grouped split on a twin-heavy set.
6. `build_probe_report` runs end-to-end on synthetic inputs, writes markdown + at least one figure,
   and **does not** emit a backdoor-detection claim when the scope is canary (assert the string).
   Figures: save to a temp dir; assert files exist and are non-empty (don't assert pixels).

## Acceptance

- `pytest -q` green (new tests; nothing else regresses). Figures render headless (`matplotlib.use
  ("Agg")`).
- `ruff check .`, `ruff format .`, `mypy` clean. Report commands + results.
