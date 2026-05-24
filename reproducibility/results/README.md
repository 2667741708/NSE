# Reproducibility Result Snapshot

This directory stores local snapshots of paper-facing result evidence.

## Contents

- [summary_20260523.csv](summary_20260523.csv): compact parsed summary from
  copied `master_log.txt` files.
- [persistent_state_v2/diagnostics_summary_20260523.csv](persistent_state_v2/diagnostics_summary_20260523.csv):
  final-epoch PSS active-set and source-state diagnostics parsed from the
  copied per-seed `run.log` files.
- [main_results/](main_results/bayes_unified_collected_results_with_paths_20260523.csv):
  local copy of the main-result path index.
- [component_ablation/](component_ablation/c100_pr005_nr03_ablation_summary_20260523.csv):
  local copies of A0--A8 audit summaries and `master_log.txt` files.
- [persistent_state_v2/](persistent_state_v2/nse_source_writeback_ablation_snapshot_20260523.md):
  local copies of the final five persistent-supervision-state proxy logs.

## Snapshot Boundary

The copied logs are evidence snapshots, not new reruns. Their original c201
paths are recorded in
[manifest/result_index.csv](../manifest/result_index.csv), and file hashes are
recorded in [manifest/result_sha256sums.txt](../manifest/result_sha256sums.txt).

Use this directory when reviewing the exact numbers currently cited or planned
for the manuscript. Use the command templates under
[commands/](../commands/reproduce_persistent_state_v2_c201.sh) when a fresh
rerun is needed.
