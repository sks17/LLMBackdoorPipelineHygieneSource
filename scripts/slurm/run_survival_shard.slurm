#!/bin/bash
# Slurm job-array template for the pipeline-only survival audit (one array task = one shard).
#
# The survival audit is CPU-only (no model generation), so it fits idle/checkpoint CPU
# partitions well. If your scheduler requires at least one GPU per job even for CPU-bound work,
# prefer a CPU-only cluster for this phase and reserve GPU nodes for the optional generation phase.
#
# Before submitting: (1) dry-run the whole loop locally with `bash scripts/run_pilot.sh` -- it runs
# this exact assemble -> shard -> run -> aggregate -> verify flow offline and fails on any leak;
# (2) run `trigger-audit build-manifest <experiment.yaml>` on the cluster -- it writes the shards to
# data/shards/<safe_model>_shard_NNNN.jsonl AND prints the `--array=0-N` range to use below.
# Replace every <PLACEHOLDER>. See docs/CLUSTER_EXECUTION_PLAN.md.

#SBATCH --job-name=trigger-audit-survival
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>        # e.g. an idle/checkpoint partition or your group partition
#SBATCH --array=0-9                    # one task per shard index
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/survival_%A_%a.out

set -euo pipefail

# Modules are available on compute nodes (and, on some clusters, login nodes too).
module load conda
conda activate <ENV>                   # or: conda activate /path/to/scratch/<group>/envs/trigger_audit

ROOT=<PROJECT_ROOT>                    # e.g. /path/to/scratch/<group>/trigger_audit
cd "$ROOT"

# Pick this task's shard. Adjust <MODEL> / naming to match `trigger-audit build-manifest`.
SHARD=$(printf "data/shards/<MODEL>_shard_%04d.jsonl" "${SLURM_ARRAY_TASK_ID}")

trigger-audit run-survival-shard "$SHARD" \
  --models-config configs/models.example.yaml \
  --policies-config configs/pipeline_policies.example.yaml \
  --base-conversations data/synthetic/base_conversations.jsonl \
  --triggers data/triggers/triggers.jsonl \
  --survival-out "outputs/survival_results/$(basename "$SHARD")" \
  --backend hf \
  --log-prompts 0.02                   # log 2% of final prompts for sample inspection
