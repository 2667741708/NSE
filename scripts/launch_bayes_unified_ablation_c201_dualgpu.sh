#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast}"
mkdir -p "$RESULT_ROOT/launcher_logs"

cd "$ROOT"

COMMON_ENV=(
  CONDA_ENV="${CONDA_ENV:-torch_cuda128_whm}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  SEEDS="${SEEDS:-1 2 3}"
)

nohup env "${COMMON_ENV[@]}" CUDA_DEV=0 RUN_GROUP=gpu0 \
  bash scripts/run_bayes_unified_ablation_fixed.sh \
  > "$RESULT_ROOT/launcher_logs/gpu0_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
pid0=$!

nohup env "${COMMON_ENV[@]}" CUDA_DEV=1 RUN_GROUP=gpu1 \
  bash scripts/run_bayes_unified_ablation_fixed.sh \
  > "$RESULT_ROOT/launcher_logs/gpu1_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
pid1=$!

echo "Started Bayes-Unified ablation queues:"
echo "  GPU0 PID: $pid0"
echo "  GPU1 PID: $pid1"
echo "Logs: $RESULT_ROOT/launcher_logs"
