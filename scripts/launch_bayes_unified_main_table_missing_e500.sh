#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409"
LOG_ROOT="${PROJECT_ROOT}/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/_launcher_logs_main_table_missing_e500"
mkdir -p "${LOG_ROOT}"

tmux new-session -d -s main_table_e500_gpu0 "cd \"${PROJECT_ROOT}\"; CONDA_ENV=torch_cuda128_whm PYTHON_BIN=python bash \"${LOG_ROOT}/run_bayes_unified_main_table_missing_e500.sh\" 0 0 2>&1 | tee \"${LOG_ROOT}/gpu0_queue0.log\""
tmux new-session -d -s main_table_e500_gpu1 "cd \"${PROJECT_ROOT}\"; CONDA_ENV=torch_cuda128_whm PYTHON_BIN=python bash \"${LOG_ROOT}/run_bayes_unified_main_table_missing_e500.sh\" 1 1 2>&1 | tee \"${LOG_ROOT}/gpu1_queue1.log\""

tmux ls | grep main_table_e500
