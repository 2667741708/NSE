#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Treeversity, Benthic, and Plankton main NSE rows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT="${OUT:-${PROJECT_ROOT}/out_ultimate}"
SCRIPT="${RELEASE_ROOT}/code/01_nse_main_table_train.py"

DATASETS=(${DATASETS:-Treeversity Benthic Plankton})
LPIS=(${LPIS:-10 3})

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${RELEASE_ROOT}/code:${PYTHONPATH:-}"

run_row() {
  local dataset="$1"
  local lpi="$2"
  local lower
  lower="$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]')"
  "${PY}" "${SCRIPT}" \
    --dataset "${dataset}" \
    --train_root "./${dataset}" \
    --lpi "${lpi}" \
    --slice 2 \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 0.0005 \
    --momentum 0.9 \
    --lr_scheduler step \
    --lr_decay_epochs 60 80 \
    --lr_decay_rate 0.2 \
    --mixup_alpha 1.0 \
    --lsr 0.0 \
    --consistency_weight 1.0 \
    --ema_alpha 0.999 \
    --k_val 5 \
    --delta 1.0 \
    --history_len 15 \
    --consensus_power 2.0 \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --knn_heads 1 \
    --topology_rel_gamma 2.0 \
    --topology_rel_eps 1e-12 \
    --daes_entropy_coeff 0.5 \
    --max_w_model 0.5 \
    --model_warmup_epochs 20 \
    --pr 0.05 \
    --nr 0.5 \
    --seeds 1 2 3 \
    --cuda_dev "${GPU_ID}" \
    --out "${OUT}" \
    --exp_name "reproducible_release/${lower}_lpi${lpi}_seed123"
}

for dataset in "${DATASETS[@]}"; do
  for lpi in "${LPIS[@]}"; do
    run_row "${dataset}" "${lpi}"
  done
done
