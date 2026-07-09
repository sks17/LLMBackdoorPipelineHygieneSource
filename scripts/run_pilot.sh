#!/usr/bin/env bash
# Quick, self-verifying pilot run of the survivability audit -- the exact assemble -> shard ->
# per-shard worker -> aggregate -> verify loop the Hyak job runs, at small scale and fully offline.
#
# Run it before scaling up: if this is green, the cluster job differs only in size and knobs, not in
# shape. To scale to Hyak: raise SYNTH_COUNT/LONGDOC_COUNT, widen the grid in
# configs/pilot/experiment.pilot.yaml (context lengths / policies / real model_ids), set BACKEND=hf,
# and submit scripts/slurm/run_survival_shard.slurm as a job array over the shard count printed here
# (each array task runs the SAME `run-survival-shard` command this script runs in a loop).
#
# Usage:  bash scripts/run_pilot.sh
# Env overrides: SYNTH_COUNT=18 LONGDOC_COUNT=6 TARGET_LEN=512 BACKEND=simple TEXT_PATH=big.txt
set -euo pipefail

# --- knobs (env-overridable) -------------------------------------------------
SYNTH_COUNT="${SYNTH_COUNT:-18}"          # synthetic (mock) bases
LONGDOC_COUNT="${LONGDOC_COUNT:-6}"       # long-document bases sliced from TEXT_PATH
TARGET_LEN="${TARGET_LEN:-512}"           # matched token length for both content arms
BACKEND="${BACKEND:-simple}"              # 'simple' (offline reference tokenizer) or 'hf'
TEXT_PATH="${TEXT_PATH:-big.txt}"         # local long-document corpus
MODEL_ID="${MODEL_ID:-simple-whitespace}" # tokenizer id for length binning (matches models.pilot)
POSITIONS="prefix old_turn recent_turn end"

EXP=configs/pilot/experiment.pilot.yaml
MODELS=configs/pilot/models.pilot.yaml
POLICIES=configs/pilot/policies.pilot.yaml
BASES=data/pilot/base_conversations.jsonl
TRIGGERS=data/triggers/triggers.jsonl
RESULTS_DIR=outputs/pilot/survival_results
MERGED=outputs/pilot/survival.jsonl

cd "$(dirname "$0")/.."   # repo root, regardless of caller's CWD

echo "== pilot: clean previous artifacts =="
rm -f data/shards/*.jsonl data/manifests/trial_manifest.jsonl
rm -rf "$RESULTS_DIR"; mkdir -p "$RESULTS_DIR" data/pilot

echo "== phase 1: materialize base conversations (synthetic + long-document from $TEXT_PATH) =="
python -m trigger_audit.generation.conversation_generator \
  --model-id "$MODEL_ID" --tokenizer-backend "$BACKEND" \
  --target-length "$TARGET_LEN" --count "$SYNTH_COUNT" \
  --generation-backend mock --positions $POSITIONS \
  --output data/pilot/synthetic_bases.jsonl
python -m trigger_audit.io.dataset_adapter \
  --source longdoc --text-path "$TEXT_PATH" \
  --model-id "$MODEL_ID" --tokenizer-backend "$BACKEND" \
  --target-length "$TARGET_LEN" --limit "$LONGDOC_COUNT" \
  --positions $POSITIONS \
  --output data/pilot/longdoc_bases.jsonl
cat data/pilot/synthetic_bases.jsonl data/pilot/longdoc_bases.jsonl > "$BASES"
python -m trigger_audit validate-jsonl "$BASES" --schema base_conversation

echo "== phase 2: build manifest + shards =="
python -m trigger_audit build-manifest "$EXP"

echo "== phase 3: run each shard (this loop is the Slurm array; one iteration = one array task) =="
shopt -s nullglob
SHARDS=(data/shards/*.jsonl)
if [ "${#SHARDS[@]}" -eq 0 ]; then echo "ERROR: no shards were written"; exit 1; fi
for shard in "${SHARDS[@]}"; do
  python -m trigger_audit run-survival-shard "$shard" \
    --models-config "$MODELS" --policies-config "$POLICIES" \
    --base-conversations "$BASES" --triggers "$TRIGGERS" \
    --survival-out "$RESULTS_DIR/$(basename "$shard")" \
    --backend "$BACKEND"
done

echo "== phase 4: aggregate =="
python -m trigger_audit score-survival "$RESULTS_DIR"
cat "$RESULTS_DIR"/*.jsonl > "$MERGED"

echo "== phase 5: verify (conditioned analysis + counterfactual control; non-zero exit on any leak) =="
python scripts/pilot_report.py "$MERGED" "$BASES"

echo
echo "== PILOT VERIFIED =="
echo "Shards written: ${#SHARDS[@]}  ->  Hyak Slurm array: --array=0-$(( ${#SHARDS[@]} - 1 ))"
echo "To scale on Hyak: set BACKEND=hf + real model_ids, enlarge the grid, submit"
echo "scripts/slurm/run_survival_shard.slurm as the array over those shards."
