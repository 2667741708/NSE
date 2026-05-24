#!/usr/bin/env bash
set -euo pipefail

# Template for reproducing the five conservative persistent-state v2 proxies.
# This mirrors the completed c201 launchers through their stable names.

cd /home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果

bash experiments/nse_mvp_source_writeback_20260514/run_persistent_state_proxies_gpu0.sh
bash experiments/nse_mvp_source_writeback_20260514/run_persistent_state_proxies_gpu1.sh
