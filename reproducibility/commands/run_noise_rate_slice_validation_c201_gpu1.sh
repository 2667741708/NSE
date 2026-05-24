#!/usr/bin/env bash
set -euo pipefail

# Additional single-seed noise-rate validation on c201 GPU1.
# Runs CIFAR-100 q=0.05 at eta=0.1/0.4/0.5 with the main NSE script.
# CIFAR-100 q=0.05 eta=0.3 is already covered by the representative validation pass.

GPU_ID="${GPU_ID:-1}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
HUB="${HUB:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_reproducibility}"
SCRIPT="${HUB}/train_nse_main_results.py"
OUT="${OUT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_noise_rate_slice_validation_20260525}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${HUB}:${PYTHONPATH:-}"
mkdir -p "${OUT}/_launcher_logs"
LOG="${OUT}/_launcher_logs/gpu1_cifar100_$(date +%Y%m%d_%H%M%S).log"

BASE_ARGS=(
  --dataset CIFAR100
  --train_root "${PROJECT_ROOT}/data"
  --lpi 10
  --network R18
  --epochs 500
  --batch_size 256
  --lr 0.1
  --wd 0.001
  --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0
  --lsr 0.0
  --consistency_weight 1.0
  --ema_alpha 0.999
  --k_val 15
  --delta 0.25
  --history_len 15
  --consensus_power 2.0
  --sim_mode_1 topology_daes
  --sim_mode_2 topology_daes
  --knn_heads 1
  --topology_rel_gamma 2.0
  --topology_rel_eps 1e-12
  --daes_entropy_coeff 0.5
  --max_w_model 0.5
  --model_warmup_epochs 20
  --pr 0.05
  --seeds 1
  --cuda_dev "${GPU_ID}"
  --out "${OUT}"
)

run_exp() {
  local exp="$1"
  local nr="$2"
  echo "[noise-slice] start ${exp} nr=${nr}" | tee -a "${LOG}"
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" --nr "${nr}" --exp_name "${exp}" 2>&1 | tee -a "${LOG}"
}

run_exp "main/CIFAR100_q005_eta01_seed1" 0.1
run_exp "main/CIFAR100_q005_eta04_seed1" 0.4
run_exp "main/CIFAR100_q005_eta05_seed1" 0.5
