# scripts/

Operational helpers. These are intentionally thin: orchestration logic lives in the package and
CLI, not in shell scripts.

- `run_pilot.sh` — **quick, self-verifying pilot run.** Executes the full cluster-shaped loop
  (materialize bases → `build-manifest` + shard → per-shard `run-survival-shard` in a loop →
  `score-survival` → conditioned analysis) offline at small scale, and **exits non-zero if the
  counterfactual control leaks**. Run this before scaling to Hyak; the cluster job differs only in
  size and knobs (`BACKEND=hf`, real models, a wider grid), not in shape. Prints the Slurm
  `--array=0-N` range for the manifest it built.

  ```bash
  bash scripts/run_pilot.sh                       # defaults: offline, 18 synthetic + 6 long-doc
  SYNTH_COUNT=40 LONGDOC_COUNT=20 bash scripts/run_pilot.sh
  ```

- `pilot_report.py` — conditioned survival analysis: verifies the counterfactual control, prints the
  true (trigger-present) survival table by policy × position, and splits the delivered rate by
  `data_source` (the H4 covariate). Used by `run_pilot.sh`; runnable standalone on any results file.

- `slurm/run_survival_shard.slurm` — Slurm job-array template for the pipeline-only survival
  audit. One array task processes one shard. `build-manifest` writes the shards to `data/shards/`
  and prints the `--array` range. Replace the `<PLACEHOLDER>` values before submitting. See
  [`docs/CLUSTER_EXECUTION_PLAN.md`](../docs/CLUSTER_EXECUTION_PLAN.md).

Submit with:

```bash
sbatch scripts/slurm/run_survival_shard.slurm
```

- `ingest_dataset.py`, `capture_*_fixture.py` — dataset ingestion helper and golden-fixture capture
  scripts for the verified trials.
