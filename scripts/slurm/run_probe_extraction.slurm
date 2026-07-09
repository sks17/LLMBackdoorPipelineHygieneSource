#!/bin/bash
# Slurm GPU job-array template for Project 2 probe ACTIVATION EXTRACTION (one array task = one cell).
#
# Activation extraction is the ONLY GPU phase of the probe wave. `extract-activations` loads a real
# model once, runs a forward pass per selected trial, and populates the activation store (it also
# does the probe fit/calibration/evaluation for the cell, reusing run_probe_experiment). Every phase
# AFTER this (the CPU probe rerun, run_probe_experiment.slurm) reads the store and never touches a GPU.
#
# SHARD STRICTLY BY MODEL (docs/CLUSTER_EXECUTION_PLAN.md:72): one array = one model, so each worker
# loads that model's weights exactly once. `expand-probe-grid` embeds each cell's model label in its
# experiment_id, so the glob below selects exactly one model's cells. NEVER mix model sizes in one
# array (their GPU-memory and time needs differ). Tillicum: >=1 GPU/job, billed.
#
# Before submitting: (1) dry-run OFFLINE first --
#     trigger-audit expand-probe-grid configs/probe/E1_existence.axes.yaml --out-dir configs/probe/generated/e1
#   writes the cells AND prints the `--array=0-N` range; then run one E0 cell offline to prove the
#   path (docs/PROJECT2_EXECUTION.md Step 0). (2) Replace EVERY <PLACEHOLDER> below.

#SBATCH --job-name=probe-extract-<MODEL_LABEL>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>       # a GPU partition (Tillicum: >=1 GPU/job, billed)
#SBATCH --gpus=1                          # one GPU per task; extraction loads the weights once
#SBATCH --array=0-<N>                     # one task per generated cell of THIS model (from expand)
#SBATCH --cpus-per-task=8
#SBATCH --mem=<MEM>                       # size to THIS model (e.g. 32G for Qwen3-8B, 12G for 0.6B)
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/probe_extract_%A_%a.out

set -euo pipefail

# Modules are available on compute nodes. Extraction needs a CUDA torch build (the [generate] extra).
module load <CUDA_MODULE>                 # e.g. cuda/12.4, per the cluster
module load conda
conda activate <ENV>                      # env with the [hf,generate] extras + a CUDA torch build

ROOT=<PROJECT_ROOT>                       # e.g. /path/to/scratch/<group>/trigger_audit
cd "$ROOT"

# This model's generated cells only. <TIER> is the expand --out-dir leaf (e.g. e1); <MODEL_LABEL> is
# the ProbeModelSpec `id` (e.g. qwen3-0_6b) that expand-probe-grid embedded in each experiment_id.
GEN_DIR="configs/probe/generated/<TIER>"
mapfile -t CELLS < <(ls "$GEN_DIR"/*<MODEL_LABEL>*.yaml | sort)
CELL="${CELLS[${SLURM_ARRAY_TASK_ID}]}"
echo "extracting cell: $CELL"

# extract-activations forces extractor_backend=hf + reuse_store=true and runs the FULL
# extract -> pool -> fit -> calibrate -> evaluate pass for this cell, populating the store the CPU
# rerun (run_probe_experiment.slurm) then reuses. Activation extraction is the only GPU phase.
trigger-audit extract-activations "$CELL" --device cuda:0
