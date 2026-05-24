#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
QUEUE_ID="${2:-0}"
PROJECT_ROOT="/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409"
SCRIPT_NAME="bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast.py"
OUT_ROOT="bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast"
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
  --network R18 --epochs 100 --batch_size 256
  --lr 0.1 --wd 0.001 --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0 --lsr 0.0 --consistency_weight 1.0 --ema_alpha 0.999
  --k_val 15 --delta 0.25 --history_len 15 --consensus_power 2.0
  --sim_mode_1 topology_daes --sim_mode_2 topology_daes --knn_heads 1
  --out ./out_ultimate
  --seeds 1 2
  --cuda_dev "${GPU_ID}"
)

run_one() {
  local dataset="$1"
  local pr="$2"
  local nr="$3"
  local tag="$4"
  local ri_mode="$5"
  local maxw="$6"
  local maxw_tag="$7"

  local exp_name="${OUT_ROOT}/${tag}/ri_${ri_mode}_maxw${maxw_tag}"
  local exp_dir="./out_ultimate/${exp_name}"
  mkdir -p "${exp_dir}"

  {
    printf '%q %q ' "${PYTHON_BIN}" "${SCRIPT_NAME}"
    printf '%q ' --dataset "${dataset}" --pr "${pr}" --nr "${nr}"
    printf '%q ' "${COMMON_ARGS[@]}"
    printf '%q ' --ri_mode "${ri_mode}" --max_w_model "${maxw}" --exp_name "${exp_name}"
    printf '\n'
  } > "${exp_dir}/command.txt"

  cat > "${exp_dir}/baseline_alignment.txt" <<EOF
Dataset: ${dataset}
Partial rate pr/q: ${pr}
Noise rate nr/eta: ${nr}
Epochs: 100
Seeds: 1 2
ri_mode: ${ri_mode}
max_w_model: ${maxw}
Default non-contrast settings: R18, batch_size 256, cosine LR, topology_daes/topology_daes, k=15, delta=0.25, history_len=15, consensus_power=2.0.
EOF

  echo "[$(date '+%F %T')] START ${exp_name} on GPU ${GPU_ID}"
  "${PYTHON_BIN}" "${SCRIPT_NAME}" \
    --dataset "${dataset}" --pr "${pr}" --nr "${nr}" \
    "${COMMON_ARGS[@]}" \
    --ri_mode "${ri_mode}" --max_w_model "${maxw}" \
    --exp_name "${exp_name}"
  echo "[$(date '+%F %T')] DONE ${exp_name}"
}

EXPERIMENTS=(
  "CIFAR10 0.5 0.3 c10_pr05_nr03_e100"
  "CIFAR100 0.1 0.0 c100_pr01_nr00_e100"
  "CIFAR100 0.05 0.5 c100_pr005_nr05_e100"
  "CIFAR100H 0.5 0.2 c100H_pr05_nr02_e100"
)
RI_MODES=(confidence entropy_prior)
MAXW_VALUES=("0.0 00" "0.5 05" "1.0 10")

i=0
for exp in "${EXPERIMENTS[@]}"; do
  read -r dataset pr nr tag <<< "${exp}"
  for ri_mode in "${RI_MODES[@]}"; do
    for maxw_pair in "${MAXW_VALUES[@]}"; do
      read -r maxw maxw_tag <<< "${maxw_pair}"
      if (( i % 2 == QUEUE_ID )); then
        run_one "${dataset}" "${pr}" "${nr}" "${tag}" "${ri_mode}" "${maxw}" "${maxw_tag}"
      fi
      i=$((i + 1))
    done
  done
done
