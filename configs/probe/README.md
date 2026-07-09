# Probe-detection config families (Project 2, component F)

Each `*.axes.yaml` here is a [`ProbeGridAxes`](../../src/trigger_audit/experiments/probe_detection/grid.py)
encoding **one experiment tier's grid**. This is the "changing parameters defines experiments"
surface: you instantiate any experiment E0.1–E3.4 by editing an axis value in one of these files and
re-expanding — never by writing Python.

```bash
# Expand a tier into per-cell configs and get the Slurm --array=0-N range:
trigger-audit expand-probe-grid configs/probe/E1_existence.axes.yaml \
    --out-dir configs/probe/generated/e1
# Run one generated cell (offline E0 cells run with no GPU/downloads):
trigger-audit run-probe-experiment configs/probe/generated/e0/<cell>.yaml
```

`expand-probe-grid` takes the Cartesian product of the axes
(`models × layer_depth_fractions × poolings × aggregations × target_fprs × seeds`), so the cell count
is the product of the axis lengths. Every generated `configs/probe/generated/<tier>/<experiment_id>.yaml`
is a complete, standalone `ProbeDetectionExperimentConfig` with a deterministic, content-derived
`experiment_id` (same axes → same ids on re-expansion) whose per-cell
`activations_dir`/`results_out`/`predictions_out` live under
`outputs/probe_detection/generated_runs/<tier>/<experiment_id>/` (no array collisions). The model
label is embedded in each `experiment_id`, so extraction shards strictly by model.

Full operator runbook (survival wave → Gate 0 → expand → GPU extract → CPU probe → analysis):
[`docs/PROJECT2_EXECUTION.md`](../../docs/PROJECT2_EXECUTION.md). Locked design:
[`docs/PRE_REGISTRATION.md`](../../docs/PRE_REGISTRATION.md) (2026-07-06 probe amendment).

## Experiment → file → parameter to edit

| Experiment | File | Parameter(s) to change to instantiate it |
|---|---|---|
| **E0.1** recoverability/determinism gate | `E0_instrument.axes.yaml` | inherent (reference extractor is recoverable + deterministic); run any cell twice |
| **E0.2** calibration honesty & FPR resolution | `E0_instrument.axes.yaml` | `synthetic_n_bases` in the generated cells (≈ clean-negative count `n∈{30,100,300,1000,3000}`) |
| **E0.3** leakage ablation (grouped vs example split) | `E0_instrument.axes.yaml` | `synthetic_mode: twins` (twins share `base_id`); example-level inflation is a documented runner seam |
| **E0.4** operator-confound (span fallback ON/OFF) | `E0_instrument.axes.yaml` | add `trigger_span` to `poolings` (oracle-only; random-span fallback always ON) |
| **E0.5** three-population calibration | `E0_instrument.axes.yaml` | `synthetic_mode: twins` + `partial_survival_fraction` (generated-cell field) |
| **E1.1** layer sweep per model | `E1_existence.axes.yaml` | `layer_depth_fractions` (the two bands) |
| **E1.2** pooling ablation | `E1_existence.axes.yaml` | `poolings: [mean, last_token, max]` |
| **E1.3** multi-layer aggregation | `E1_existence.axes.yaml` | `aggregations` (closed-form + caveated `stacked_logistic`) |
| **E1.4** scale curve | `E1_existence.axes.yaml` | `models` (Qwen3 0.6/1.7/4/8B + Pythia-1B) |
| **E1.5** delivery-conditional decomposition | `E1_existence.axes.yaml` | inherent (dual all-trials / delivered-only reporting) |
| **E2.1** delivery-style generalization | `E2_shift.axes.yaml` | `generalization: {kind: policy, …}` (default) |
| **E2.2** long-context generalization | `E2_shift.axes.yaml` | `generalization: {kind: context_length, train_context_max, test_context_min}` |
| **E2.3** trigger-type generalization | `E2_shift.axes.yaml` | `generalization: {kind: trigger_type, train_trigger_types, test_trigger_types}` |
| **E2.4** cross-model transfer (same-width) | `E2_shift.axes.yaml` | analysis-layer step over two same-width models' cells (not a single-cell holdout) |
| **E3.1** install & measure real backdoors | `E3_backdoor.axes.yaml` | `models` (allowlisted backdoored checkpoints) — see scoping header |
| **E3.2** poison-rate × complexity floor | `E3_backdoor.axes.yaml` | analysis sweep over checkpoints of varying poison/complexity |
| **E3.3** behavioral vs representational | `E3_backdoor.axes.yaml` | analysis over predictions + generation records (component G) |
| **E3.4** baseline battery | `E3_backdoor.axes.yaml` | analysis vs Apollo/DefenseBox/regex baselines (component G) |

## Shared axes and defaults (pre-registered)

- **Backend**: a `ProbeModelSpec` with a real `model_id` → `hf` backend (GPU); with `model_id`
  omitted → `reference` backend (offline). E0 is reference; E1/E2/E3 are `hf`.
- **Depth band** (default): `[[0.5, 0.66, 0.75, 0.89]]` — the pre-registered informative band, plus a
  fuller per-model sweep for E1.1. Reported by depth-fraction, never raw index.
- **Pooling**: `[mean]` is the deployable operating point; `[last_token, max]` are the E1.2 ablation;
  `trigger_span` is **oracle-only** (add explicitly for the E0.4/E1.2 diagnostic).
- **Aggregation**: closed-form `[mean_score, max_score, product_of_experts, quantile]` is the
  headline; `stacked_logistic` is included in E1 but **caveated** (fit on CALIBRATION).
- **FPR targets**: `[[0.01, 0.001]]`. 1e-2 is the resolvable primary; **1e-3 is bounded-only** unless
  clean CALIBRATION negatives are scaled to ≥ ~1000 (`target_fpr < 1/n` is unresolvable).
- **Seeds**: `[0, 1, 2]`.

## Scoping (hard boundary)

E0–E2 measure **harmless canaries** and license claims only about delivered-canary representations
and the harness. **Only E3** (real, allowlisted, ASR-verified backdoored weights) licenses a
backdoor-detection statement, and even then only "TPR t @ FPR f on installed backdoor type T, model
M" — never a general headline. See the `E3_backdoor.axes.yaml` header and `PRE_REGISTRATION.md` item
10.
