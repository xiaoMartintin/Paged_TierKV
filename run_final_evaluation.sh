#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="final_eval_logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"

mkdir -p "${RUN_DIR}"
rm -f "${LOG_ROOT}/latest"
ln -s "${RUN_ID}" "${LOG_ROOT}/latest"

run_logged() {
  local name="$1"
  shift
  local log_path="${RUN_DIR}/${name}.log"

  {
    printf '### Command:'
    printf ' %q' "$@"
    printf '\n\n'
  } | tee "${log_path}"

  "$@" 2>&1 | tee -a "${log_path}"
}

run_logged main modal run main.py

run_logged tier_mode modal run main.py --tier-mode-ablation

run_logged quality_sanity modal run main.py --quality-sanity

run_logged stage_a env \
  TIERKV_STAGED_COUNTS=32,64,96,128,160 \
  TIERKV_STAGED_CONTEXT_TOKENS=1900 \
  TIERKV_STAGED_DECODE_STEPS=16 \
  modal run main.py --staged-scaling

run_logged stage_b env \
  TIERKV_STAGED_COUNTS=192,224,256,320 \
  TIERKV_STAGED_CONTEXT_TOKENS=1900 \
  TIERKV_STAGED_DECODE_STEPS=16 \
  modal run main.py --staged-scaling

run_logged boundary env \
  TIERKV_STAGED_COUNTS=384,448,512 \
  TIERKV_STAGED_CONTEXT_TOKENS=1900 \
  TIERKV_STAGED_DECODE_STEPS=16 \
  modal run main.py --staged-scaling

run_logged extension_1024 env \
  TIERKV_EXTENSION_MODEL_ID="${TIERKV_EXTENSION_MODEL_ID:-lmsys/vicuna-7b-v1.5}" \
  TIERKV_EXTENSION_CONTEXT_TOKENS=1024 \
  TIERKV_EXTENSION_COUNTS=1,2,4,6,8,12 \
  TIERKV_EXTENSION_DECODE_STEPS=16 \
  modal run main.py --extension-scaling

run_logged extension_1536 env \
  TIERKV_EXTENSION_MODEL_ID="${TIERKV_EXTENSION_MODEL_ID:-lmsys/vicuna-7b-v1.5}" \
  TIERKV_EXTENSION_CONTEXT_TOKENS=1536 \
  TIERKV_EXTENSION_COUNTS=1,2,4,6,8 \
  TIERKV_EXTENSION_DECODE_STEPS=16 \
  modal run main.py --extension-scaling

python collect_final_results.py \
  --log-dir "${LOG_ROOT}/latest" \
  --output FINAL_EVALUATION_RESULTS.md

echo "Final evaluation logs: ${RUN_DIR}"
echo "Final Markdown report: FINAL_EVALUATION_RESULTS.md"
