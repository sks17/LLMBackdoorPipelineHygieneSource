#!/bin/bash
# One-time environment build on a Klone compute node (module load conda only works off the login
# node). Creates the conda env on /gscratch, installs the package with the `hf` extra (tokenizers
# only -- no torch/GPU needed for the survival phase), and warms the tokenizer cache. Submit via
# `deploy/hyak.ps1 setup`; re-running is safe (it reuses an existing env).
#
# account/partition are set on the CLI by deploy/hyak_submit.sh; ENV_PREFIX/MODELS_CONFIG exported.
#SBATCH --job-name=trigaudit-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# 2h ceiling: `pip install -e .[hf]` drags in the full torch/CUDA stack (~6 GB) via transformers 5.x;
# a COLD build (fresh env path, empty conda/pip caches) plus ckpt-all queue-wait can exceed 40 min.
# A re-run is much faster (caches under REMOTE_ROOT are then warm), so this ceiling is only hit once.
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/setup_%j.out
#SBATCH --error=outputs/logs/setup_%j.err
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:?set ENV_PREFIX (the conda env path on /gscratch)}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/pilot/models.pilot_hf.yaml}"
# Keep every cache OFF the tiny 10 GB home dir (conda/pip default their caches to $HOME, which
# fails once home is full -- the same quota that breaks an scp to home). Point them at scratch.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PWD/.conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"

mkdir -p outputs/logs "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
module load conda
# Make `conda activate` work in this non-interactive batch shell.
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

if [ ! -d "$ENV_PREFIX" ]; then
  echo "creating conda env at $ENV_PREFIX"
  conda create --yes --prefix "$ENV_PREFIX" python=3.12
fi
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install -e ".[hf]"
python -m trigger_audit version

# Warm the tokenizer cache so the array tasks don't all race to download the same files.
python deploy/prefetch_tokenizers.py "$MODELS_CONFIG" \
  || echo "prefetch skipped/failed; array tasks will download tokenizers on demand"

echo "SETUP COMPLETE  env=$ENV_PREFIX  hf_cache=$HF_HOME"
