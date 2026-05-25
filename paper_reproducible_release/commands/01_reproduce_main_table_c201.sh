#!/usr/bin/env bash
set -euo pipefail

# Reproduce the main NSE CIFAR-100 q=0.05 eta=0.3 row with the clean script name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT="${OUT:-${PROJECT_ROOT}/out_ultimate}"
SCRIPT="${RELEASE_ROOT}/code/01_nse_main_table_train.py"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${RELEASE_ROOT}/code:${PYTHONPATH:-}"

"${PY}" "${SCRIPT}" \
  --dataset CIFAR100 \
  --train_root ./data \
  --lpi 10 \
  --network R18 \
  --epochs 500 \
  --batch_size 256 \
  --lr 0.1 \
  --wd 0.001 \
  --momentum 0.9 \
  --lr_scheduler cosine \
  --mixup_alpha 1.0 \
  --lsr 0.0 \
  --consistency_weight 1.0 \
  --ema_alpha 0.999 \
  --k_val 15 \
  --delta 0.25 \
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
  --nr 0.3 \
  --seeds 1 2 3 \
  --cuda_dev "${GPU_ID}" \
  --out "${OUT}" \
  --exp_name reproducible_release/main_cifar100_q005_eta03_seed123
