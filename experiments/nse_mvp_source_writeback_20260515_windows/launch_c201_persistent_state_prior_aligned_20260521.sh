#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <gpu_id>" >&2
  exit 2
fi

GPU_ID="$1"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
SCRIPT="${SCRIPT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp_family_20260521_prior_aligned.py}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUT="${OUT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_prior_aligned_20260521}"
LAUNCH_LOG_DIR="${OUT}/_launcher_logs"
mkdir -p "${LAUNCH_LOG_DIR}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/c201_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

BASE_ARGS=(
  --dataset CIFAR100
  --train_root "${DATA_ROOT}"
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

is_done() {
  local exp="$1"
  local log="${OUT}/${exp}/master_log.txt"
  [ -f "${log}" ] && grep -q -- "--- Run 3 Finished" "${log}"
}

run_exp() {
  local exp="$1"
  shift
  if is_done "${exp}"; then
    echo "[launcher] skip completed ${exp}" | tee -a "${LAUNCH_LOG}"
    return 0
  fi
  echo "[launcher] start ${exp}" | tee -a "${LAUNCH_LOG}"
  echo "[launcher] cwd=${PROJECT_ROOT}" | tee -a "${LAUNCH_LOG}"
  echo "[launcher] script=${SCRIPT}" | tee -a "${LAUNCH_LOG}"
  echo "[launcher] args: ${BASE_ARGS[*]} --exp_name ${exp} $*" | tee -a "${LAUNCH_LOG}"
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" --exp_name "${exp}" "$@" 2>&1 | tee -a "${LAUNCH_LOG}"
}

run_exp "FREDIS_MOVE_PSS" \
  --source_update_mode fredis_move \
  --source_update_scope all \
  --fredis_refine_threshold 0.01 \
  --fredis_disamb_threshold 0.80 \
  --fredis_min_disamb_over_refine 2.0

run_exp "IRNet_CORRECT_PSS" \
  --source_update_mode irnet_correct \
  --source_update_scope all \
  --irnet_tau_boundary 0.0 \
  --irnet_min_non_candidate_conf 0.0

run_exp "PALS_SARI_AUG_PSS" \
  --source_update_mode pals_augment \
  --source_update_scope all_highconf \
  --source_update_schedule pals_linear

run_exp "UPLLRS_PROMOTE_PSS" \
  --source_update_mode none \
  --source_update_scope none \
  --persistent_promotion_mode hard \
  --promotion_scope unreliable_highconf \
  --promotion_threshold 0.95 \
  --promotion_source p2

run_exp "PiCOPlus_SOFT_PSS" \
  --source_update_mode pico_soft_target \
  --source_update_scope all \
  --source_update_alpha 0.1

echo "[launcher] completed c201 prior-aligned persistent-state suite on gpu ${GPU_ID}" | tee -a "${LAUNCH_LOG}"
