#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
QUEUE_ID="${2:-0}"
PROJECT_ROOT="/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409"
SCRIPT_NAME="bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py"
OUT_ROOT="bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model"
CONDA_ENV="${CONDA_ENV:-torch_cuda128_whm}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"

if [[ -n "${CONDA_ENV}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --train_root ./data --lpi 10
  --network R18 --epochs 500 --batch_size 256
  --lr 0.1 --wd 0.001 --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0 --lsr 0.0 --consistency_weight 1.0 --ema_alpha 0.999
  --k_val 15 --delta 0.25 --history_len 15 --consensus_power 2.0
  --sim_mode_1 topology_daes --sim_mode_2 topology_daes --knn_heads 1
  --max_w_model 0.5
  --out ./out_ultimate
  --seeds 1 2 3
  --cuda_dev "${GPU_ID}"
)

run_one() {
  local dataset="$1"
  local pr="$2"
  local nr="$3"
  local tag="$4"
  local exp_name="${OUT_ROOT}/${tag}/refactored_e500_seed123"
  local exp_dir="./out_ultimate/${exp_name}"
  mkdir -p "${exp_dir}"

  {
    printf "%q %q " "${PYTHON_BIN}" "${SCRIPT_NAME}"
    printf "%q " --dataset "${dataset}" --pr "${pr}" --nr "${nr}"
    printf "%q " "${COMMON_ARGS[@]}"
    printf "%q " --exp_name "${exp_name}"
    printf "\n"
  } > "${exp_dir}/command.txt"

  cat > "${exp_dir}/baseline_alignment.txt" <<BASEEOF
Main-table missing-condition e500 run
Script: ${SCRIPT_NAME}
Dataset: ${dataset}
Partial rate pr/q: ${pr}
Noise rate nr/eta: ${nr}
Epochs: 500
Seeds: 1 2 3
max_w_model: 0.5
Output root: ./out_ultimate/${OUT_ROOT}
Purpose: fill missing CIFAR-10/CIFAR-100 pr/nr cells for the main paper table.
BASEEOF

  echo "[$(date +%F\ %T)] START ${exp_name} on GPU ${GPU_ID}"
  "${PYTHON_BIN}" "${SCRIPT_NAME}" \
    --dataset "${dataset}" --pr "${pr}" --nr "${nr}" \
    "${COMMON_ARGS[@]}" \
    --exp_name "${exp_name}"
  echo "[$(date +%F\ %T)] DONE ${exp_name}"
}

EXPERIMENTS=(
  "CIFAR10 0.1 0.2 c10_pr01_nr02_maxw05"
  "CIFAR10 0.1 0.3 c10_pr01_nr03_maxw05"
  "CIFAR10 0.3 0.1 c10_pr03_nr01_maxw05"
  "CIFAR10 0.3 0.2 c10_pr03_nr02_maxw05"
  "CIFAR10 0.3 0.3 c10_pr03_nr03_maxw05"
  "CIFAR100 0.01 0.2 c100_pr001_nr02_maxw05"
  "CIFAR100 0.03 0.1 c100_pr003_nr01_maxw05"
  "CIFAR100 0.03 0.2 c100_pr003_nr02_maxw05"
  "CIFAR100 0.03 0.3 c100_pr003_nr03_maxw05"
  "CIFAR100 0.05 0.1 c100_pr005_nr01_maxw05"
  "CIFAR100 0.05 0.2 c100_pr005_nr02_maxw05"
)

i=0
for exp in "${EXPERIMENTS[@]}"; do
  read -r dataset pr nr tag <<< "${exp}"
  if (( i % 2 == QUEUE_ID )); then
    run_one "${dataset}" "${pr}" "${nr}" "${tag}"
  fi
  i=$((i + 1))
done
