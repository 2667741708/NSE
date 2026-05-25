# Clean Result Summaries

This directory stores compact summaries for the current paper-facing results.
Large historical result trees remain in `reproducibility/results/`.

- `paper_table_results_summary_20260523.csv`: current paper component and
  persistent-state table summary.
- `main_table_collected_results_with_paths_20260523.csv`: main-table
  three-seed results, individual seed final values, and original log paths.
- `persistent_state_proxy_diagnostics_summary_20260523.csv`: diagnostic metrics
  for FREDIS, IRNet, PALS/SARI, UPLLRS, and PiCO+ persistent-state proxies.
- `single_seed_validation_summary_20260524.csv`: representative seed-1
  reproduction check for selected paper rows.
- `noise_rate_slice_validation_summary_20260525.csv`: completed seed-1
  CIFAR-10/CIFAR-100 noise-rate checks against the paper three-seed values.
- `noise_rate_slice_validation_20260525/`: copied `master_log.txt` files for
  those completed c201 validation runs.
