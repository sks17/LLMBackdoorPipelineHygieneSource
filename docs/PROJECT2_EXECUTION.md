# Project 2 — Operator Execution Runbook (component F)

End-to-end operator instructions for running a probe-detection wave: survival wave → subset
selection (Gate 0) → grid expansion → GPU extraction → CPU probe pass → analysis. Every step names
the exact command. Background: `docs/PROJECT2_MASTER.md`, `docs/PROJECT2_EXPERIMENT_PLAN.md`,
`docs/PRE_REGISTRATION.md`. Component boundaries: `docs/tasks/project2/00-BUILD-PLAN.md`.

**Canary ≠ backdoor scoping (read first).** Every step below except the final "Tier 3" step
measures **harmless canary triggers**, never a real backdoor. Tiers 0–2 (offline instrument checks,
Tier-1 existence/shape, Tier-2 shift-robustness) license claims about *delivered-canary
representations* only. Only Tier 3 — real backdoored weights, loaded through the safety-gated
registry in `docs/PROJECT2_BACKDOOR_SAFETY.md` — licenses a backdoor-detection statement, and even
then the claim is scoped to "TPR at FPR on installed backdoor type T, model M," never a general
headline (`docs/PRE_REGISTRATION.md` item 10; `PROJECT2_EXPERIMENT_PLAN.md` Part IV "Scoping").

**1e-3 resolution caveat.** `target_fpr=1e-3` is reported **bounded-only** (achieved FPR of 0 with
an honest wide Wilson interval) unless clean CALIBRATION negatives are scaled to **≥ ~1000**, since
an empirical-quantile threshold cannot resolve a target below `1/n` clean negatives
(`docs/PRE_REGISTRATION.md` item 6; `probes/calibration.py`). Run E0.2 (`configs/probe/README.md`)
to confirm the resolvable count for your data before quoting a 1e-3 number.

## Step 0 — offline dry run (no GPU, no downloads, no cluster)

Prove the parameter → experiment path works before touching real weights or the cluster:

```bash
trigger-audit run-probe-experiment configs/probe_detection.example.yaml

trigger-audit expand-probe-grid configs/probe/E0_instrument.axes.yaml \
    --out-dir configs/probe/generated/e0
trigger-audit run-probe-experiment configs/probe/generated/e0/<one-generated-cell>.yaml
```

Both run fully offline against the reference extractor. If either fails, stop — nothing later in
this runbook will work either.

## Step 1 — survival wave with `--final-tokens-out` (component A)

Run (or reuse an existing) Project 1 survival wave, but with the `final_tokens.jsonl` sidecar the
probe join needs:

```bash
trigger-audit build-manifest configs/experiment_survivability.example.yaml
# -> data/manifests/trial_manifest.jsonl + data/shards/<model>_shard_NNNN.jsonl, prints --array=0-N

trigger-audit run-survival-shard data/shards/<model>_shard_0000.jsonl \
    --models-config configs/models.example.yaml \
    --policies-config configs/pipeline_policies.example.yaml \
    --base-conversations data/synthetic/base_conversations.jsonl \
    --triggers data/triggers/triggers.jsonl \
    --survival-out outputs/survival_results/<model>_shard_0000.jsonl \
    --backend hf \
    --final-tokens-out outputs/survival_results/<model>_shard_0000.final_tokens.jsonl
```

On the cluster, this is one Slurm array task per shard
(`scripts/slurm/run_survival_shard.slurm` — add `--final-tokens-out` to that template's
`run-survival-shard` invocation; it is off by default so existing survival-only waves are
unaffected). Concatenate every shard's survival results and every shard's `final_tokens.jsonl` into
`outputs/survival_results/all_survival.jsonl` / `outputs/survival_results/all_final_tokens.jsonl`
before Step 2 (`cat`/PowerShell `Get-Content` — no special tool needed, both are plain JSONL).

## Step 2 — `select-probe-subset` (Gate 0)

Activation extraction is expensive (GPU), so it never runs on the whole survival grid — only a
stratified `base_id` subset (`experiments/probe_detection/selection.py`). This step selects that
subset **and** verifies the counterfactual control (Gate 0) on it:

```bash
trigger-audit select-probe-subset outputs/survival_results/all_survival.jsonl \
    --delivered-positive 300 \
    --clean-negative 1000 \
    --partial-survival-negative 300 \
    --boundary-corruption 50 \
    --stratified-sample 200 \
    --seed 0 \
    --out outputs/probe_subset_selection.json
```

- `--clean-negative` should target **≥ ~1000** if you intend to report the 1e-3 FPR target as
  resolved rather than bounded-only (see the caveat above; E0.2 pins this number precisely for your
  data before you commit to it).
- **Gate 0 is non-negotiable.** If any trigger-absent twin in the selected subset leaked (delivered
  a trigger with none inserted), the command prints the leak examples and **exits non-zero without
  writing `outputs/probe_subset_selection.json`** — parity with the P1 pilot discipline (a leak
  means the survival scorer is buggy and every downstream probe label is untrustworthy). Fix the
  scorer bug (`analysis/controls.py::verify_counterfactual`, `schemas/results.py`) and rerun Step 1
  before proceeding; do not work around a Gate-0 failure by re-selecting with different targets.
- The printed `subset_report` shows every stratum's achieved count vs target, with `shortfalls`
  called out explicitly (never hidden) — review it before extraction: a large shortfall in
  `boundary_corruption` or `stratified_sample`, for example, means that population is under-covered
  in this wave's survival data, not a selector bug.

Feed `outputs/survival_results/all_final_tokens.jsonl` (filtered to the selection's `trial_ids`, or
just the whole concatenated file — `run_probe_experiment` joins on `trial_id` and ignores extras)
as each E1/E2/E3 axes file's `final_tokens_path`.

## Step 3 — `expand-probe-grid`

Pick an axes file (`configs/probe/README.md` maps experiment id → file → parameter), edit only the
parameters that file's header names, then expand:

```bash
trigger-audit expand-probe-grid configs/probe/E1_existence.axes.yaml \
    --out-dir configs/probe/generated/e1
```

Prints the generated cell count and the Slurm `--array=0-N` range. Every generated
`configs/probe/generated/e1/<experiment_id>.yaml` is a complete, standalone
`ProbeDetectionExperimentConfig` — loadable and runnable on its own, with a deterministic,
content-derived `experiment_id` (same axes → same ids on re-expansion) and per-cell
`activations_dir`/`results_out`/`predictions_out` under `outputs/probe_detection/generated_runs/e1/
<experiment_id>/`, so no two cells collide on an output path.

## Step 4 — GPU extraction array

```bash
sbatch scripts/slurm/run_probe_extraction.slurm
```

One array **per model** (fill in `<MODEL_LABEL>` and size `--mem`/`--gpus`/`--time` to that model —
see the template's header). Each array task runs:

```bash
trigger-audit extract-activations configs/probe/generated/e1/<cell>.yaml --device cuda:0
```

which forces `extractor_backend=hf` + `reuse_store=true` and is a thin wrapper over
`run_probe_experiment` — i.e. one task does the FULL extraction, probe fit, calibration, and
evaluation for that cell (not just extraction), and additionally populates the activation store the
CPU rerun below can reuse. Sharded strictly by model: `expand-probe-grid` embeds each cell's model
label in its `experiment_id`, so the Slurm array's glob selects exactly one model's cells per job
(never mixing model sizes with different GPU memory needs in one array — see the template).

## Step 5 — CPU probe array

```bash
sbatch scripts/slurm/run_probe_experiment.slurm
```

One array task per generated config, running `trigger-audit run-probe-experiment <cell>.yaml`. This
template covers two legitimate uses — pick the one that matches your axes family:

- **Tier-0 offline cells** (`configs/probe/generated/e0/*`, `extractor_backend: reference`): this is
  their only Slurm template; they never touch a GPU.
- **A rerun of an already-GPU-extracted `hf`-backend cell** (Tier 1/2/3, after Step 4): every
  generated cell sets `reuse_store: true`, so this rerun finds every requested layer already
  cached and skips every forward pass — the expensive part.

**Honest caveat** on the second use (documented in the template's own header, repeated here so it
is not missed): `run_probe_experiment` unconditionally constructs the real `HFActivationExtractor`
(a checkpoint load) at the top of every `hf`-backend run, *before* it ever checks the reuse store —
purely to read `num_layers` for depth-fraction resolution and the layer-bound check. So this "CPU"
rerun still loads the checkpoint into this job's CPU memory (size `--mem` accordingly); the real
savings are **no GPU allocation/billing** and **zero forward passes**, not a zero-cost rerun. A
future enhancement to `runner.py` (out of scope for this component — see
`docs/tasks/project2/00-BUILD-PLAN.md`'s file-ownership table) could add a store-only fast path
that skips extractor construction entirely when every requested layer is already cached.

## E2 generalization seam (read before running any E2.x cell)

`configs/probe/E2_shift.axes.yaml`'s `generalization:` block (`GeneralizationSpec`) encodes an
E2.x train/test holdout — by `pipeline_policy` (E2.1), `context_length` (E2.2), or `trigger_type`
(E2.3, matched against each example's `trigger_id`) — **as a parameter**, per the project's
"changing parameters defines experiments" mandate. `expand-probe-grid` threads that block straight
onto every generated cell's `config.generalization`, and **`run_probe_experiment` applies it**: TEST
is the held-out (test) side, a `base_id`-grouped CALIBRATION subset is carved out of the TRAIN side,
and rows matching neither side are dropped (`experiments/probe_detection/generalization.py`,
`assign_generalization_splits`). So running a generated E2 cell the ordinary way
(`trigger-audit run-probe-experiment <cell>.yaml`) **measures the holdout** — no bypass needed. Each
cell's `name` field records `generalization=<kind>(applied via config.generalization)` so the
applied holdout is visible in the generated YAML itself.

The membership metadata is read straight off the delivery-verified survival records:
`build_probe_examples` sets `pipeline_policy` (E2.1), `context_length` (E2.2), and `trigger_id`
(E2.3) on every example, so all three kinds partition on real data with no extra wiring. To switch
E2.1 → E2.2/E2.3, edit only the `generalization:` block in the axes file (see its header) and
re-expand.

**Feasibility is honest, not silent.** A holdout is only measurable if the held-out sides carry the
populations each role needs. If the carved CALIBRATION subset ends up with no clean (never-inserted)
negative — or TEST lacks a positive or a clean negative — `ProbeDetectionRunner._validate_splits`
raises a clear error naming the missing role. That is the honest "this holdout is infeasible on THIS
data" signal (feasibility depends on the survival wave's coverage of each policy/context/trigger
family), not a bug to work around: widen the survival wave's coverage or choose held-out sides your
data actually populates. `assign_generalization_splits` itself fails fast if a holdout produces no
TEST side, no TRAIN side, or fewer than two TRAIN-side `base_id` groups (so a CALIBRATION subset
cannot be carved).

For ad-hoc, off-runner reassignment there is still `partition_by_metadata` — a pure two-way
TRAIN/TEST relabel that leaves unmatched rows in place and carves no CALIBRATION — re-exported from
both `experiments.probe_detection.grid` and `experiments.probe_detection.generalization`. The unit
tests in `tests/test_probe_generalization.py` and `tests/test_probe_grid.py` (including a generated
E2 cell run end to end through `run_probe_experiment`) are the executable spec for this seam.

## Step 6 — analysis (component G)

Once `results.jsonl`/`predictions.jsonl` exist under
`outputs/probe_detection/generated_runs/<tier>/<experiment_id>/`, hand them to `analysis/probe_*.py`
(component G) for cluster-bootstrap `P(fire | delivered)`, the delivery-conditional decomposition
(E1.5), and the baseline/robustness comparisons (Tier 3/4). Not built by this component.

## Tier 3 (real backdoored models) — extra preconditions

Before running anything from `configs/probe/E3_backdoor.axes.yaml` against a real checkpoint:

1. The checkpoint must be registered in a `BackdoorRegistry` with full provenance and
   `allowlisted: true` (`docs/PROJECT2_BACKDOOR_SAFETY.md` "Lifecycle") — never edit `allowlisted`
   from this runbook or from `expand-probe-grid`; that flip is a manual, reviewed, out-of-band step.
2. `verify_backdoor_installed` must report `installed == True` for that checkpoint (high ASR, low
   clean-fire-rate for the benign marker) — an unverified checkpoint's probe numbers measure a
   backdoor that was never actually installed.
3. Steps 1–5 above run unchanged against that checkpoint's `model_id`/`revision` (via
   `extractor_spec_for(checkpoint)` — `docs/PROJECT2_BACKDOOR_SAFETY.md`'s flow diagram); Gate 0 in
   Step 2 still applies.

Even a fully clean Tier-3 run licenses only "TPR t @ FPR f on installed backdoor type T, model M" —
never a general backdoor-detection headline (see the scoping note at the top of this document).
