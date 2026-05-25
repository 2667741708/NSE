#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current paper A0--A8 component ablation rows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT="${OUT:-${PROJECT_ROOT}/out_ultimate}"
SCRIPT="${RELEASE_ROOT}/code/02_nse_component_ablation_A0_A8_train.py"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${RELEASE_ROOT}/code:${PYTHONPATH:-}"

BASE_ARGS=(
  --dataset CIFAR100
  --train_root ./data
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
  --max_w_model 0.5
  --out "${OUT}"
  --cuda_dev "${GPU_ID}"
  --pr 0.05
  --nr 0.3
  --seeds 1 2 3
)

run_row() {
  local exp="$1"
  shift
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" --exp_name "${exp}" "$@"
}

run_row "reproducible_release/component/A0_full"
run_row "reproducible_release/component/A1_topology_daes_to_exp_both" --sim_mode_1 exp --sim_mode_2 exp
run_row "reproducible_release/component/A2_stage2_exp" --sim_mode_1 topology_daes --sim_mode_2 exp
run_row "reproducible_release/component/A3_stage1_exp" --sim_mode_1 exp --sim_mode_2 topology_daes
run_row "reproducible_release/component/A4_uniform_ri" --ablate_uniform_ri
run_row "reproducible_release/component/A5_no_candidate_prior_projection" --ablate_no_candidate_prior
run_row "reproducible_release/component/A6_knn_only" --max_w_model 0.0
run_row "reproducible_release/component/A7_maxw10" --max_w_model 1.0
run_row "reproducible_release/component/A8_detect_salvage_only" --disable_salvage_training
