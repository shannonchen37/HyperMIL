#!/usr/bin/env bash
set -euo pipefail

# Respect CUDA_VISIBLE_DEVICES when provided; otherwise let PyTorch decide.
GPU="${CUDA_VISIBLE_DEVICES:-}"

DATASET="${DATASET:-STAD}"
LOSS_TYPE="${LOSS_TYPE:-cox}"
BS="${BS:-8}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Training logs are written to log/<DATASET>/.
LOG_ROOT="$PROJECT_ROOT/log"
mkdir -p "$LOG_ROOT"

ts(){ date +'%Y-%m-%d %H:%M:%S'; }

log_dir="${LOG_ROOT}/${DATASET}"
mkdir -p "${log_dir}"
log_file="${log_dir}/log_${DATASET}_${LOSS_TYPE}_bs${BS}.txt"
gpu_label="${GPU:-auto}"
echo "[$(ts)] START GPU=${gpu_label} DS=${DATASET} LOSS=${LOSS_TYPE} BS=${BS} LOG=${log_file}" | tee -a "$log_file"

cmd=(
  "${PYTHON_BIN}" -u "$PROJECT_ROOT/train_survival.py"
  --dataset "${DATASET}"
  --loss_type "${LOSS_TYPE}"
  --batch_size "${BS}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
)

if [[ -n "${GPU}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" >> "$log_file" 2>&1
else
  "${cmd[@]}" >> "$log_file" 2>&1
fi

echo "[$(ts)] SUCCESS GPU=${gpu_label} DS=${DATASET} LOSS=${LOSS_TYPE} BS=${BS}" | tee -a "$log_file"
echo "[$(ts)] ALL DONE. Logs: $LOG_ROOT" | tee -a "$log_file"
