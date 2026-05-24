#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <setting_id:E03|E05> <gpu_id> <noise_rate>" >&2
  exit 2
fi

SETTING_ID="$1"
GPU_ID="$2"
NOISE_RATE="$3"

PY="/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python"
SCRIPT="/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp_family_20260518.py"
DATA_ROOT="/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/data"
OUT="/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_writeback_docx_family_20260518"
LAUNCH_LOG_DIR="${OUT}/_launcher_logs"
mkdir -p "${LAUNCH_LOG_DIR}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/${SETTING_ID}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

if [ -n "${WAIT_SESSION:-}" ]; then
  echo "[launcher] waiting for tmux session ${WAIT_SESSION}" | tee -a "${LAUNCH_LOG}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    date | tee -a "${LAUNCH_LOG}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | tee -a "${LAUNCH_LOG}" || true
    sleep 300
  done
fi

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
  --nr "${NOISE_RATE}"
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
  echo "[launcher] args: ${BASE_ARGS[*]} --exp_name ${exp} $*" | tee -a "${LAUNCH_LOG}"
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" --exp_name "${exp}" "$@" 2>&1 | tee -a "${LAUNCH_LOG}"
}

# A: reset baseline.
run_exp "${SETTING_ID}_A0_NSE_Reset" \
  --source_update_mode none --source_update_scope none

# B: soft evidence write-back strength sweep.
for alpha in 0.01 0.05 0.1 0.3 0.5 1.0; do
  alpha_tag=$(printf "%04.2f" "${alpha}" | tr -d '.')
  run_exp "${SETTING_ID}_B_SoftEvidence_a${alpha_tag}" \
    --source_update_mode soft_evidence --source_update_scope all --source_update_alpha "${alpha}"
done

# C: all-sample hard insert threshold sweep.
for tau in 0.55 0.65 0.75 0.85 0.95; do
  tau_tag=$(printf "%04.2f" "${tau}" | tr -d '.')
  run_exp "${SETTING_ID}_C_AllHardInsert_t${tau_tag}" \
    --source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold "${tau}"
done

# D: unreliable-only hard insert threshold sweep.
for tau in 0.55 0.65 0.75 0.85 0.95; do
  tau_tag=$(printf "%04.2f" "${tau}" | tr -d '.')
  run_exp "${SETTING_ID}_D_UnreliableHardInsert_t${tau_tag}" \
    --source_update_mode hard_insert --source_update_scope unreliable_highconf --source_update_threshold "${tau}"
done

# E: add-remove refinement proxy.
for drop in 0.05 0.10; do
  drop_tag=$(printf "%04.2f" "${drop}" | tr -d '.')
  run_exp "${SETTING_ID}_E_AddRemove_add085_drop${drop_tag}" \
    --source_update_mode add_remove --source_update_scope all --source_update_threshold 0.85 --source_remove_threshold "${drop}"
done

# F: remove-only purification proxy.
for rem in 0.05 0.10; do
  rem_tag=$(printf "%04.2f" "${rem}" | tr -d '.')
  run_exp "${SETTING_ID}_F_RemoveOnly_t${rem_tag}" \
    --source_update_mode hard_remove --source_update_scope all --source_remove_threshold "${rem}"
done

# G: Topology-DAES internal ablations.
run_exp "${SETTING_ID}_G1_NoTopologyReliability" \
  --source_update_mode none --source_update_scope none --topology_rel_gamma 0.0

run_exp "${SETTING_ID}_G2_FixedDAESTemperature" \
  --source_update_mode none --source_update_scope none --daes_entropy_coeff 0.0

run_exp "${SETTING_ID}_G3_NoTopologyReliability_FixedDAES" \
  --source_update_mode none --source_update_scope none --topology_rel_gamma 0.0 --daes_entropy_coeff 0.0

echo "[launcher] completed ${SETTING_ID} on gpu ${GPU_ID}" | tee -a "${LAUNCH_LOG}"
