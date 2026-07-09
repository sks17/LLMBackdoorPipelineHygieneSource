# Project 2 — Build Plan (basics for activation-based trigger detection)

This directory holds the implementation specs for building out Project 2. Each spec is a
self-contained contract for one component. Background and justification live in
`docs/PROJECT2_MASTER.md`, `docs/PROJECT2_EXPERIMENT_PLAN.md`, and `docs/PROJECT2_CONTINUITY_CHECK.md`;
this plan turns the prerequisites (P1–P7) and the experiment-definition machinery into buildable
components.

## Estimand (do not lose sight of it)

`P(probe fires | trigger delivered)` — the TPR of a linear probe on per-layer hidden-state
activations, measured at a threshold **calibrated to a target FPR**, conditional on Project 1 having
verified the trigger reached the final model-visible tokens
(`SurvivalResult.final_token_trigger_present`). AUROC is reported only as a summary. Every metric is
reported twice: **all-trials** (partial-survival negatives count against the budget) and
**delivered-only** (verified-delivered positives + clean negatives). The differentiator versus all
public probing work is that labels are **delivery-verified**, never insertion-labeled.

## What already exists (do NOT rebuild)

- `schemas/probes.py` — `ProbeExample`, `LayerProbeMetrics`, `AchievedFpr`, `ProbeEvaluationResult`,
  `PoolingStrategy`, `ProbeLabelSource`, `ProbeSplit`.
- `activations/{extractor,pooling,store}.py` — `ActivationExtractor` ABC, `HFActivationExtractor`
  (real, `output_hidden_states=True`, single forward pass, HF layer indexing: 0=embeddings,
  1..N=blocks), `ReferenceActivationExtractor` (offline twin), pooling (last/mean/max/trigger_span),
  `.npz`+trial-id store.
- `probes/{linear,metrics,calibration,aggregation}.py` — numpy L2 logistic probe, tie-aware AUROC,
  `threshold_at_fpr`/`tpr_at_fpr`, empirical-quantile calibration + Wilson CI, aggregator registry.
- `experiments/probe_detection/{config,dataset,runner}.py` — config, delivery-verified dataset build
  + `base_id`-grouped leakage-safe split, end-to-end runner.
- CLI `run-probe-experiment`; `configs/probe_detection.example.yaml` runs offline out of the box.
- `analysis/stats.py` — `bootstrap_rate_ci` (cluster/base bootstrap), `bootstrap_paired_diff_ci`,
  `holm`, `benjamini_hochberg`. Reuse these; do not reimplement.
- `analysis/controls.py::verify_counterfactual` — Gate 0 counterfactual control. Reuse.

## Global rules for every component

1. **Additive + typed + clean.** Match the existing style: `from __future__ import annotations`,
   pydantic v2 models, numpy-only in the probe/activation core, no torch in the base import path
   (torch/transformers are lazily imported inside `HFActivationExtractor.__init__` only).
2. **Offline-first.** Everything except the real GPU extraction wave must run and be tested against
   the reference extractor / simple tokenizer with **no downloads and no torch**. Real-model paths
   are exercised by contract tests that skip when torch is absent (see
   `tests/test_activation_hf_contract.py` for the pattern).
3. **Determinism.** All randomness is seeded (`np.random.default_rng((seed, tag, index))`); reruns
   are byte-identical. This is a pre-registered property.
4. **Quality gates before you finish.** Run, from the repo root, using the project venv
   (`.venv\Scripts\python.exe` on Windows / `.venv/bin/python` on Linux, else `python`):
   - `python -m pytest -q` (at minimum your new tests + the probe/activation suites; a full run is
     preferred and must stay green — do not break another component's tests),
   - `python -m ruff check .` and `python -m ruff format .` (format your files),
   - `python -m mypy` (src typechecks clean).
   Fix everything you break. Report the exact commands you ran and their results.
5. **Canary ≠ backdoor scoping.** No code, doc, config, or comment may claim a backdoor-detection
   result from canary/reference data. Tier-0–2 claims are about *delivered-canary representations*;
   only Tier-3 (real backdoored weights) licenses a backdoor-detection statement.
6. **Depth-fraction, not raw index.** Report and select layers by depth-fraction (fraction of model
   depth) wherever a claim crosses models; raw HF indices are an internal detail. The informative
   band prior for our Qwen suite is ~0.5–0.89 of depth, peak ~2/3 (`PROJECT2_RESOURCES.md:86-88`).

## Components and dependency waves

| ID | Spec | Owns (files) | Depends on |
|----|------|--------------|-----------|
| A | `A-data-contract.md` | `schemas/results.py`, `experiments/survivability_audit/{scorer,runner,manifest_runner}.py`, `io/final_tokens.py`, `cli.py`(run-survival-shard only) | — |
| D | `D-synthetic-twins.md` | `experiments/probe_detection/dataset.py` | — |
| I | `I-prereg-amendment.md` | `docs/PRE_REGISTRATION.md`, `docs/PROJECT2_BUILD_NOTES.md` | — |
| H | `H-backdoor-safety.md` | `models/*`, `configs/backdoor_models.example.yaml`, `docs/PROJECT2_BACKDOOR_SAFETY.md` | — |
| C | `C-probe-runtime.md` | `experiments/probe_detection/{config,runner}.py`, `activations/{slicing,store}.py`, `schemas/probes.py` | A, D |
| B | `B-selector.md` | `experiments/probe_detection/selection.py` | A |
| F | `F-experiment-grid.md` | `experiments/probe_detection/grid.py`, `cli.py`, `scripts/slurm/*`, `configs/probe/*` | C, B |
| G | `G-probe-analysis.md` | `analysis/probe_*.py` | C |

**Wave 1 (parallel):** A, D, I, H — disjoint file sets.
**Wave 2 (parallel, after A & D):** C, B — disjoint file sets.
**Wave 3 (parallel, after C & B):** F, G — disjoint file sets.

File ownership is exclusive within a wave: if your change requires touching a file another component
owns, stop and note it in your final report instead of editing it — the integration pass reconciles.

## The through-line the components must preserve

The point is that **changing parameters defines experiments**. A researcher should be able to run
E0.1–E4.4 (see `PROJECT2_EXPERIMENT_PLAN.md` Part III) by editing config/axis values and submitting a
Slurm array — never by writing new Python. The grid expander (F) + parameterized configs
(`configs/probe/*`) + the Slurm templates are that surface; C, B, D, G, H supply the science those
parameters drive (real activation extraction and model slicing, delivery-verified/stratified data,
honest inference, and safely-handled backdoored checkpoints).
