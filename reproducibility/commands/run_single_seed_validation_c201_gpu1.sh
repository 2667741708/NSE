#!/usr/bin/env bash
set -euo pipefail

# Representative single-seed validation on c201 GPU1.
# Runs sample/soft-state PSS proxies, component A0, and the main CIFAR100 q=.05 eta=.3 row.

GPU_ID="${GPU_ID:-1}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
HUB="${HUB:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_reproducibility}"
OUT="${OUT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_single_seed_validation_20260523}"

PSS_SCRIPT="${HUB}/train_nse_persistent_state_proxies.py"
ABLATION_SCRIPT="${HUB}/train_nse_component_ablation.py"
MAIN_SCRIPT="${HUB}/train_nse_main_results.py"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${HUB}:${PYTHONPATH:-}"
mkdir -p "${OUT}/_launcher_logs"
LOG="${OUT}/_launcher_logs/gpu1_$(date +%Y%m%d_%H%M%S).log"

PSS_BASE_ARGS=(
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
  --k_val 15
  --delta 0.25
  --history_len 15
  --sim_mode_1 topology_daes
  --sim_mode_2 topology_daes
  --knn_heads 1
  --topology_rel_gamma 2.0
  --topology_rel_eps 1e-12
  --daes_entropy_coeff 0.5
  --max_w_model 0.5
  --model_warmup_epochs 20
  --out "${OUT}"
  --seeds 1
  --cuda_dev "${GPU_ID}"
  --pr 0.05
  --nr 0.3
  --source_update_evidence p2
)

MAIN_BASE_ARGS=(
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
  --nr 0.3
  --seeds 1
  --cuda_dev "${GPU_ID}"
  --out "${OUT}"
)

run_pss() {
  local exp="$1"
  shift
  echo "[single-seed] start ${exp}" | tee -a "${LOG}"
  "${PY}" "${PSS_SCRIPT}" "${PSS_BASE_ARGS[@]}" --exp_name "${exp}" "$@" 2>&1 | tee -a "${LOG}"
}

run_main_like() {
  local script="$1"
  local exp="$2"
  shift 2
  echo "[single-seed] start ${exp}" | tee -a "${LOG}"
  "${PY}" "${script}" "${MAIN_BASE_ARGS[@]}" --exp_name "${exp}" "$@" 2>&1 | tee -a "${LOG}"
}

run_pss "pss/UPLLRS_V2_PSS_seed1" \
  --source_update_mode none \
  --source_update_scope none \
  --persistent_promotion_mode hard \
  --promotion_scope nse_estimated_noise_highconf \
  --promotion_threshold 0.95 \
  --promotion_source p2 \
  --promotion_label_space non_candidate

run_pss "pss/PiCOPlus_V2_PSS_seed1" \
  --source_update_mode pico_soft_target \
  --source_update_scope all \
  --source_update_alpha 0.1

run_main_like "${ABLATION_SCRIPT}" "component/A0_full_seed1"

run_main_like "${MAIN_SCRIPT}" "main/CIFAR100_pr005_nr03_seed1"
