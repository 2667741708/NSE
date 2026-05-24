#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_NAME="${SCRIPT_NAME:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py}"
OUT_DIR="${OUT_DIR:-./out_ultimate}"
EXP_ROOT="${EXP_ROOT:-bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast}"
CUDA_DEV="${CUDA_DEV:-0}"
SEEDS="${SEEDS:-1 2 3}"
RUN_GROUP="${RUN_GROUP:-gpu0}"
CONDA_ENV="${CONDA_ENV:-torch_cuda128_whm}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"

if [[ -n "$CONDA_ENV" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # c201 default training environment.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

COMMON_C100=(
  --dataset CIFAR100 --train_root ./data --lpi 10
  --network R18 --epochs 500 --batch_size 256
  --lr 0.1 --wd 0.001 --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0 --lsr 0.0 --consistency_weight 1.0 --ema_alpha 0.999
  --k_val 15 --delta 0.25 --history_len 15 --consensus_power 2.0
  --sim_mode_1 topology_daes --sim_mode_2 topology_daes --knn_heads 1
  --max_w_model 0.5
  --out "$OUT_DIR"
  --cuda_dev "$CUDA_DEV"
)

run_one() {
  local dataset="$1"
  local ablation_id="$2"
  local pr="$3"
  local nr="$4"
  shift 4

  local exp_name="${EXP_ROOT}/${dataset}/${ablation_id}"
  local exp_dir="${OUT_DIR}/${exp_name}"
  mkdir -p "$exp_dir"
  local -a cmd=(
    "$PYTHON_BIN" "$SCRIPT_NAME" "${COMMON_C100[@]}"
    --pr "$pr" --nr "$nr"
    --exp_name "$exp_name"
    --seeds $SEEDS
    "$@"
  )

  echo "================================================================"
  echo "[RUN] ${dataset}/${ablation_id}"
  echo "[CUDA] ${CUDA_DEV}"
  echo "[OUT] ${exp_dir}"
  echo "================================================================"

  printf '%q ' "${cmd[@]}" > "${exp_dir}/command.txt"
  printf '\n' >> "${exp_dir}/command.txt"
  {
    echo "Baseline reference:"
    echo "  dataset=CIFAR100 train_root=./data lpi=10 pr=${pr} nr=${nr}"
    echo "  network=R18 epochs=500 batch_size=256 lr=0.1 wd=0.001 momentum=0.9 lr_scheduler=cosine"
    echo "  mixup_alpha=1.0 lsr=0.0 consistency_weight=1.0 ema_alpha=0.999"
    echo "  k_val=15 delta=0.25 history_len=15 consensus_power=2.0"
    echo "  sim_mode_1=topology_daes sim_mode_2=topology_daes knn_heads=1 max_w_model=0.5 seeds=${SEEDS}"
    echo
    echo "Ablation name:"
    echo "  ${ablation_id}"
    echo
    echo "Changed arguments relative to baseline:"
    if [[ "$#" -eq 0 ]]; then
      echo "  none"
    else
      printf '  %q ' "$@"
      printf '\n'
    fi
  } > "${exp_dir}/baseline_alignment.txt"

  "${cmd[@]}"
}

run_c100_nr03_gpu0() {
  run_one c100_pr005_nr03 A01_ablate_topology_daes_use_exp_both 0.05 0.3 --sim_mode_1 exp --sim_mode_2 exp
  run_one c100_pr005_nr03 A02_ablate_stage2_topology_daes_use_exp_s2 0.05 0.3 --sim_mode_1 topology_daes --sim_mode_2 exp
  run_one c100_pr005_nr03 A03_ablate_stage1_topology_daes_use_exp_s1 0.05 0.3 --sim_mode_1 exp --sim_mode_2 topology_daes
  run_one c100_pr005_nr03 A04_ablate_adaptive_ri_use_uniform_ri05 0.05 0.3 --ablate_uniform_ri
  run_one c100_pr005_nr05 N01_nr05_ablate_topology_daes_use_exp_both 0.05 0.5 --sim_mode_1 exp --sim_mode_2 exp
}

run_c100_nr03_gpu1() {
  run_one c100_pr005_nr03 A05_ablate_candidate_prior_unmask_model_evidence 0.05 0.3 --ablate_no_candidate_prior
  run_one c100_pr005_nr03 A06_ablate_model_view_maxw0_knn_only 0.05 0.3 --max_w_model 0.0
  run_one c100_pr005_nr03 A07_stress_model_view_maxw1_full_model_influence 0.05 0.3 --max_w_model 1.0
  run_one c100_pr005_nr03 A08_ablate_salvage_training_log_only 0.05 0.3 --ablate_no_salvage_training
  run_one c100_pr005_nr03 A09_ablate_queue_stability_short_history5 0.05 0.3 --history_len 5
  run_one c100_pr005_nr03 A10_ablate_queue_stability_long_history30 0.05 0.3 --history_len 30
  run_one c100_pr005_nr05 N04_nr05_ablate_adaptive_ri_use_uniform_ri05 0.05 0.5 --ablate_uniform_ri
  run_one c100_pr005_nr05 N05_nr05_ablate_candidate_prior_unmask_model_evidence 0.05 0.5 --ablate_no_candidate_prior
  run_one c100_pr005_nr05 N08_nr05_ablate_salvage_training_log_only 0.05 0.5 --ablate_no_salvage_training
}

case "$RUN_GROUP" in
  gpu0) run_c100_nr03_gpu0 ;;
  gpu1) run_c100_nr03_gpu1 ;;
  all)
    run_c100_nr03_gpu0
    run_c100_nr03_gpu1
    ;;
  *)
    echo "Unknown RUN_GROUP=${RUN_GROUP}. Use gpu0, gpu1, or all." >&2
    exit 2
    ;;
esac
