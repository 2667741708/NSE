#!/usr/bin/env bash
set -euo pipefail

# Reproduce the five current persistent supervision state proxy rows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT="${OUT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_reproducible_release}"
SCRIPT="${RELEASE_ROOT}/code/03_nse_persistent_supervision_state_proxy_train.py"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${RELEASE_ROOT}/code:${PYTHONPATH:-}"

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
  --seeds 1 2 3
  --cuda_dev "${GPU_ID}"
  --pr 0.05
  --nr 0.3
  --source_update_evidence p2
)

run_proxy() {
  local exp="$1"
  shift
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" --exp_name "${exp}" "$@"
}

run_proxy "reproducible_release/pss/FREDIS_V2_PSS" \
  --source_update_mode fredis_move \
  --source_update_scope all \
  --fredis_top_non_candidate_only \
  --fredis_refine_threshold 0.05 \
  --fredis_refine_min_conf 0.85 \
  --fredis_disamb_threshold 0.85 \
  --fredis_disamb_max_conf 0.05 \
  --fredis_min_disamb_over_refine 2.0

run_proxy "reproducible_release/pss/IRNet_V2_PSS" \
  --source_update_mode irnet_correct \
  --source_update_scope all \
  --irnet_tau_boundary 0.0 \
  --irnet_min_non_candidate_conf 0.85

run_proxy "reproducible_release/pss/PALS_SARI_V2_PSS" \
  --source_update_mode pals_augment \
  --source_update_scope all_highconf \
  --source_update_schedule linear \
  --source_update_threshold_start 0.95 \
  --source_update_threshold_end 0.85

run_proxy "reproducible_release/pss/UPLLRS_V2_PSS" \
  --source_update_mode none \
  --source_update_scope none \
  --persistent_promotion_mode hard \
  --promotion_scope nse_estimated_noise_highconf \
  --promotion_threshold 0.95 \
  --promotion_source p2 \
  --promotion_label_space non_candidate

run_proxy "reproducible_release/pss/PiCOPlus_V2_PSS" \
  --source_update_mode pico_soft_target \
  --source_update_scope all \
  --source_update_alpha 0.1
