# Spec I — Pre-registration amendment for the probe design (P4 / continuity D1)

**Prerequisite P4 / continuity D1 (S1).** `PRE_REGISTRATION.md` has **no** probe / FPR-target / layer
/ pooling section (it is entirely Project 1). Unlike P1's H1–H4 + the dated TOST amendment, the probe
estimand and operating points are un-preregistered, so every Tier-1+ number would be exploratory. Fix
it with a dated amendment that locks the probe design **before** the first measurement wave.

This is a **documentation-only** task. Do not touch any code.

## Files you own

- `docs/PRE_REGISTRATION.md` — append a new dated amendment (do not rewrite existing sections).
- `docs/PROJECT2_BUILD_NOTES.md` — **new**, short: a running index of what the Project-2 build added
  and where (a landing page for the components in `docs/tasks/project2/`).

## The amendment to append (date it `2026-07-06`)

Follow the exact style of the existing amendments in `PRE_REGISTRATION.md` (numbered locked decisions,
rationale inline, scope statement at the end). Title it:
`### 2026-07-06 — Project 2 probe design pre-registration (activation-based trigger detection)`.

Lock these, each with a one-line rationale traced to a source file:

1. **Estimand.** Primary = `P(probe fires | trigger delivered)` = TPR over delivered positives at the
   **calibrated** threshold; AUROC reported only as a summary "alongside, never instead"
   (`probes/metrics.py:5-6`, `PROJECT2_FOUNDATIONS.md:3-8`). Not AUROC.
2. **Dual reporting.** Every metric reported twice — **all-trials** (partial-survival negatives count
   against the FPR budget) and **delivered-only** (verified-delivered positives + clean negatives) —
   per `ProbeEvaluationResult` (`schemas/probes.py:109-129`).
3. **Layers by depth-fraction.** Probe layers are named by fraction of model depth, not raw index
   (avoids the L vs L+1 vs HF-indexing hazard). Pre-registered band: **0.5–0.89 of depth**, with the
   default probe set at fractions `{0.5, 0.66, 0.75, 0.89}` and a per-model full sweep in E1.1 to
   confirm the band on our suite (`PROJECT2_RESOURCES.md:86-88`). HF indexing: 0 = embeddings,
   1..N = blocks (`activations/extractor.py:20-26`).
4. **Pooling.** `mean` is the pre-registered deployable operating pooling; `last_token` and `max` are
   the ablation (E1.2); `trigger_span` is an **oracle diagnostic only**, never a deployable operating
   point, and always run with the seeded random-span fallback ON so the pooling operator is identical
   across classes (`runner.py:182-218`, `PROJECT2_FOUNDATIONS.md:123-141`).
5. **Aggregation.** Closed-form combiners (`mean_score`, `max_score`, `product_of_experts`,
   `quantile`) are the headline; `stacked_logistic` is reported but **caveated** for its
   calibration-split adaptivity (`runner.py:126-138`, continuity D4). Multi-layer aggregation is the
   pre-committed recovery for the small-model single-probe ceiling (`PROJECT2_RESOURCES.md:80-85`).
6. **FPR targets.** `{1e-2, 1e-3}`. **1e-2 is the resolvable primary**; **1e-3 is reported
   bounded-only** (achieved FPR 0 with an honest wide Wilson interval) unless clean negatives are
   scaled to ≥ ~1000, because `target_fpr < 1/n` cannot be resolved by an empirical quantile
   (`calibration.py`, `PROJECT2_FOUNDATIONS.md:88-96`, continuity D3). E0.2 fixes the base-count
   requirement.
7. **Calibration population.** Thresholds calibrate on **clean (never-inserted) CALIBRATION
   negatives only** — the monitor guards clean traffic. Partial-survival negatives are deliberately
   excluded from the calibration pool (`runner.py:88-95`, `PROJECT2_FOUNDATIONS.md:79-90`).
8. **Splits.** `base_id`-grouped, leakage-safe: all trials of one base (its counterfactual twins and
   near-duplicate policy/position variants) land in the same split (`dataset.py:56-63`).
9. **Inference.** `P(fire | delivered)` uncertainty is a **cluster-bootstrap over `base_id`** (extend
   P1's `analysis/stats.bootstrap_rate_ci`); achieved FPR carries a **Wilson** interval; any
   *invariance* claim (pooling X ≈ Y, model A ≈ B) uses **TOST at ±5 pp with Holm** multiplicity,
   reusing the P1 2026-07-03 amendment's margin and correction. State explicitly that the ±5 pp / Holm
   amendment is hereby extended from P1 delivery rates to the P2 probe layer (continuity D2).
10. **Scope discipline (the hard boundary).** All Tier-0–2 results are claims about **delivered-canary
    representations**, never backdoor detection. Only Tier-3 (real backdoored weights) licenses a
    backdoor-detection statement. No document, table, or figure may headline a backdoor-detection
    result — or a 1e-3 TPR — from canary/pilot data (`PROJECT2_MASTER.md §10.7`, continuity E1/D3).
11. **No numeric TPR bar is pre-committed.** Success is structural (dual reporting, calibrated
    operating points, honest low-n intervals). Any numeric target set later must be size-aware and
    stated for a *resolvable* FPR (`PROJECT2_MASTER.md §13.1`).

End with a **Scope** paragraph: this amendment binds every probe (Tier-0+) run from this date; it does
not alter any Project-1 delivery decision; the exploratory-vs-confirmatory line is drawn here.

## `docs/PROJECT2_BUILD_NOTES.md`

A short markdown page (≤1 screen) that: names the estimand in one sentence; links
`PROJECT2_MASTER.md`, `PROJECT2_EXPERIMENT_PLAN.md`, and this amendment; and gives a table mapping
each build component (A/B/C/D/F/G/H) to the file(s) it added and the prerequisite (P1–P7) or
experiment tier it discharges. Keep it factual; other agents will append their component's row (leave
a clearly-marked table the integration pass can fill — you may pre-fill rows from
`docs/tasks/project2/00-BUILD-PLAN.md`).

## Acceptance

- Markdown only; no code changed, so no pytest needed, but run `ruff format` is N/A for `.md`. Just
  confirm the files render (valid markdown, links resolve to real files).
- Report exactly what you appended and the new file's path.
