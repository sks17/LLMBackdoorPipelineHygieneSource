# Spec F — Experiment grid, CLI, Slurm, and parameterized configs

This component delivers the through-line of the whole request: **changing parameters defines
experiments**. A researcher must be able to instantiate any of E0.1–E4.4
(`PROJECT2_EXPERIMENT_PLAN.md` Part III) by editing axis values and submitting a Slurm array — never
by writing Python. You build the grid expander, the CLI surface, the Slurm templates, and the
parameterized config families that map to the experiment tiers.

## Files you own (edit only these + tests + configs/scripts)

- `src/trigger_audit/experiments/probe_detection/grid.py` — **new**
- `src/trigger_audit/cli.py` — add new commands (you are the sole cli.py editor in this wave)
- `scripts/slurm/run_probe_extraction.slurm` — **new**
- `scripts/slurm/run_probe_experiment.slurm` — **new**
- `configs/probe/` — **new** family of parameterized configs (see below)
- `docs/PROJECT2_EXECUTION.md` — **new** operator runbook
- `tests/test_probe_grid.py` — **new**; extend `tests/test_probe_config.py` only if needed

Depends on component C (config fields: `device`, `revision`, `trust_remote_code`,
`layer_depth_fractions`, `synthetic_mode`, `predictions_out`, `reuse_store`) and component B
(`select_probe_subset`, `SubsetSelection`, `verify_subset_counterfactual`,
`write_selected_trial_ids`). Import their public APIs; do not reimplement. Do NOT edit `runner.py`,
`config.py`, `selection.py`, `dataset.py`, or `store.py`.

## 1. Grid expander — `grid.py`

The expander turns a compact **axes** description into a concrete list of
`ProbeDetectionExperimentConfig`s with stable, content-derived ids (mirror
`util/ids.make_grid_trial_id` / `io/manifest.expand_manifest` style).

```python
class ProbeGridAxes(BaseModel):
    experiment_family: str                      # "E1.1", "E1.2", ... — recorded on every cell
    models: list[ProbeModelSpec]                # id, hf model_id, revision, device, trust_remote_code
    layer_depth_fractions: list[list[float]]    # each inner list is one cell's layer set (by depth-fraction)
    poolings: list[PoolingStrategy]
    aggregations: list[str]
    target_fprs: list[list[float]]
    seeds: list[int]
    # data source (mutually exclusive per run):
    survival_results_path: Path | None
    final_tokens_path: Path | None
    synthetic_mode: Literal["simple", "twins"] | None
    # generalization split specs (optional; see §1a):
    generalization: GeneralizationSpec | None = None

def expand_probe_grid(axes: ProbeGridAxes) -> list[ProbeDetectionExperimentConfig]: ...
def write_probe_configs(configs: Sequence[ProbeDetectionExperimentConfig], out_dir: Path) -> list[Path]: ...
```

Rules:
- Cartesian product over models × layer-fraction-sets × poolings × aggregations × target_fprs × seeds,
  with a deterministic iteration order and a stable `experiment_id` derived from the cell coordinates
  (hash-suffixed like the store's `_safe_component`). Record `experiment_family` and the human axis
  values in each config's future `metadata`/name so results are traceable.
- One config per cell; each is a valid `ProbeDetectionExperimentConfig` that runs as-is via
  `run_probe_experiment`. Set `extractor_backend="hf"` when a real `models[i].model_id` is given, else
  `"reference"` for offline/E0 cells.
- `layer_depth_fractions` per cell populate the config's `layer_depth_fractions` field (component C
  resolves them to indices at runtime against each model's `num_layers` — this is the "slice models by
  the same relative depth across sizes" mechanism).
- Point `predictions_out` and `results_out`/`activations_dir` at per-cell paths under a run root so a
  Slurm array writes without collisions.

### 1a. Generalization split specs (E2.x) — parameters, not new code

`GeneralizationSpec` encodes the train/test partition axis so E2.1/E2.2/E2.3 are *parameters*:
- `train_policies` / `test_policies` (E2.1 delivery-style): which `pipeline_policy` ids train vs test.
- `train_context_max` / `test_context_min` (E2.2 short→long): context-length split.
- `train_trigger_types` / `test_trigger_types` (E2.3): held-out trigger families.
Emit these into the config `metadata` so the runner/dataset (or the analysis layer) can honor a
held-out partition. **Important:** the current `dataset.assign_splits` splits by `base_id` fraction and
knows nothing about policy/trigger-type holdouts. Do NOT modify `dataset.py` (component C/B own the
runtime). Instead, encode the generalization partition as metadata + a documented convention, and in
`grid.py` provide a pure helper `partition_by_metadata(examples, spec) -> assigned examples` that a
future runner hook or the analysis layer can apply. If honoring it end-to-end requires a runner change
you don't own, **document the seam clearly** in `PROJECT2_EXECUTION.md` and the config comments, and
make the helper + its unit test the deliverable. Do not silently produce configs that claim a holdout
the runtime ignores.

## 2. CLI commands (Typer, match existing `cli.py` style; heavy imports inside the body)

- `select-probe-subset SURVIVAL_RESULTS --targets ... --seed ... --out subset.json` — loads survival
  results (file or dir), calls `select_probe_subset`, runs `verify_subset_counterfactual` and
  **aborts with a clear error if Gate 0 fails** (parity with the P1 pilot discipline — a leak means
  labels are untrustworthy), writes the selection, prints the `subset_report` including any
  `shortfalls`.
- `extract-activations CONFIG_YAML` — thin wrapper that runs `run_probe_experiment` with
  `extractor_backend="hf"` and `reuse_store=True`, i.e. the GPU extraction+probe pass for one cell.
  (Reuse `run_probe_experiment`; the "extraction" is the store-populating side effect.) Support a
  `--device` override that wins over the config.
- `expand-probe-grid AXES_YAML --out-dir configs/probe/generated/` — loads a `ProbeGridAxes` YAML,
  calls `expand_probe_grid` + `write_probe_configs`, prints the count and the Slurm `--array=0-N`
  range (mirror `build-manifest`'s array-range print).
- Keep `run-probe-experiment` as is, but ensure it still works after C's config additions.

## 3. Slurm templates (mirror `scripts/slurm/run_survival_shard.slurm`)

- `run_probe_extraction.slurm` — **GPU** array, **sharded strictly by model** so each worker loads
  weights once (`CLUSTER_EXECUTION_PLAN.md:72`). One array task = one model's config cell(s). Tillicum
  (≥1 GPU/job, billed), `module load` + conda activate, deterministic extraction. Placeholders like
  the survival template. Comment that activation extraction is the only GPU phase.
- `run_probe_experiment.slurm` — **CPU** array over generated probe configs (the probe *analysis* is
  CPU-light once features exist; `PROJECT2_MASTER.md §7`). One array task = one config file from
  `configs/probe/generated/`.
Both must clearly say: dry-run offline first, then submit; replace every `<PLACEHOLDER>`.

## 4. Parameterized config families — `configs/probe/`

Provide one template per experiment tier, each a valid `ProbeGridAxes` (for the grid) and/or a
`ProbeDetectionExperimentConfig` (for a single cell), with a header comment naming the experiment id
and **exactly which parameters to change** to instantiate it. At minimum:
- `configs/probe/E0_instrument.axes.yaml` — reference backend, `synthetic_mode: twins`, sweeps clean-
  negative counts (E0.2), leakage on/off note (E0.3), span-fallback note (E0.4), calibration-pool note
  (E0.5). Offline-runnable.
- `configs/probe/E1_existence.axes.yaml` — hf backend, the Qwen suite + Pythia-1B, layer sweep by
  depth-fraction (E1.1), pooling ablation (E1.2), aggregation sweep (E1.3), scale axis (E1.4),
  delivery-conditional decomposition via dual reporting (E1.5).
- `configs/probe/E2_shift.axes.yaml` — generalization specs for delivery-style (E2.1), long-context
  (E2.2), trigger-type (E2.3), same-width cross-model transfer note (E2.4).
- `configs/probe/E3_backdoor.axes.yaml` — points at the backdoor registry (component H) checkpoints;
  header must reassert canary-only + allowlist + ASR-precondition scoping. Placeholder model ids.
- `configs/probe/README.md` — a one-page map: experiment id → which file + which parameter to change.
Use the pre-registered defaults from `docs/PRE_REGISTRATION.md` (component I): mean pooling, band
fractions `{0.5,0.66,0.75,0.89}`, `target_fprs [0.01, 0.001]`, closed-form aggregation.

## 5. `docs/PROJECT2_EXECUTION.md` — operator runbook

End-to-end: run the survival wave with `--final-tokens-out` (component A) → `select-probe-subset`
(Gate 0) → `expand-probe-grid` → submit `run_probe_extraction.slurm` (GPU, shard-by-model) →
`run_probe_experiment.slurm` (CPU) → analysis (component G). Show the exact commands. Note the 1e-3
resolution caveat and the canary≠backdoor scoping.

## Tests (`tests/test_probe_grid.py`)

1. `expand_probe_grid` cardinality equals the product of the axes; every emitted config validates and
   has a unique, deterministic `experiment_id`; re-expansion is identical.
2. `layer_depth_fractions` land on each config; an hf model spec yields `extractor_backend="hf"`, a
   reference cell yields `"reference"`.
3. `write_probe_configs` writes N loadable YAMLs that round-trip back through
   `load_config(..., ProbeDetectionExperimentConfig)`.
4. `partition_by_metadata` correctly holds out the specified policies/trigger-types/context split on a
   synthetic example list.
5. A CLI smoke test (Typer `CliRunner`) for `expand-probe-grid` on the E0 axes file produces configs
   and prints an array range; `select-probe-subset` on a tiny survival fixture writes a selection and
   passes/fails Gate 0 as constructed.
6. The `configs/probe/E0_instrument.axes.yaml` expands and at least one generated cell **runs offline**
   end-to-end via `run_probe_experiment` (reference backend), proving the parameter→experiment path.

## Acceptance

- `pytest -q` green (full run — you touch cli.py, so run the whole suite).
- `ruff check .`, `ruff format .`, `mypy` clean. Every `<PLACEHOLDER>` in Slurm files is obvious.
- Report commands + results and the exact list of new files.
