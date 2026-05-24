#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409"
LOG_ROOT="${PROJECT_ROOT}/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast/_launcher_logs"
mkdir -p "${LOG_ROOT}"

cd "${PROJECT_ROOT}"

COMMON_ENV=(
  CONDA_ENV="${CONDA_ENV:-torch_cuda128_whm}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
)

nohup env "${COMMON_ENV[@]}" bash scripts/run_bayes_unified_contrast_e100.sh 0 0 > "${LOG_ROOT}/gpu0_queue0.log" 2>&1 &
PID0=$!
nohup env "${COMMON_ENV[@]}" bash scripts/run_bayes_unified_contrast_e100.sh 1 1 > "${LOG_ROOT}/gpu1_queue1.log" 2>&1 &
PID1=$!

echo "${PID0}" > "${LOG_ROOT}/gpu0_queue0.pid"
echo "${PID1}" > "${LOG_ROOT}/gpu1_queue1.pid"

echo "Launched contrast experiments:"
echo "  GPU0 queue0 PID=${PID0}"
echo "  GPU1 queue1 PID=${PID1}"
echo "Logs: ${LOG_ROOT}"
