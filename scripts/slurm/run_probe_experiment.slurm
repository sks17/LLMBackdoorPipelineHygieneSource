#!/bin/bash
# Slurm CPU job-array template for the Project 2 probe ANALYSIS pass (one array task = one cell).
#
# The probe analysis (fit -> calibrate -> evaluate) is CPU-light once activations exist
# (PROJECT2_MASTER.md section 7). Two legitimate uses -- pick the one matching your axes family:
#   * Tier-0 OFFLINE cells (configs/probe/generated/e0/*, extractor_backend: reference): this is
#     their ONLY template; they never touch a GPU.
#   * A RERUN of an already-GPU-extracted hf cell (Tier 1/2/3, AFTER run_probe_extraction.slurm):
#     every generated cell sets reuse_store: true, so this finds each requested layer already cached
#     and skips every forward pass -- the expensive part.
#
# HONEST CAVEAT for the second use: run_probe_experiment unconditionally constructs the real
# HFActivationExtractor (a checkpoint load) at the top of every hf-backend run -- purely to read
# num_layers for depth-fraction resolution -- BEFORE it checks the reuse store. So this "CPU" rerun
# still loads the checkpoint into this job's CPU memory (size --mem accordingly); the real savings are
# NO GPU allocation/billing and ZERO forward passes, not a zero-cost rerun (docs/PROJECT2_EXECUTION.md
# Step 5).
#
# Before submitting: (1) dry-run OFFLINE -- expand the grid (which prints the `--array=0-N` range) and
# run one E0 cell locally (docs/PROJECT2_EXECUTION.md Step 0). (2) Replace EVERY <PLACEHOLDER> below.

#SBATCH --job-name=probe-analyze
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>       # an idle/checkpoint or group CPU partition
#SBATCH --array=0-<N>                     # one task per generated cell (range from expand-probe-grid)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                         # RAISE for an hf rerun (checkpoint load) -- see the caveat
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/probe_analyze_%A_%a.out

set -euo pipefail

module load conda
conda activate <ENV>                      # for an offline E0 array the base env suffices (no torch)

ROOT=<PROJECT_ROOT>                       # e.g. /path/to/scratch/<group>/trigger_audit
cd "$ROOT"

# <TIER> is the expand --out-dir leaf: e0 (offline) or e1/e2/e3 (hf rerun after extraction).
GEN_DIR="configs/probe/generated/<TIER>"
mapfile -t CELLS < <(ls "$GEN_DIR"/*.yaml | sort)
CELL="${CELLS[${SLURM_ARRAY_TASK_ID}]}"
echo "analyzing cell: $CELL"

trigger-audit run-probe-experiment "$CELL"
