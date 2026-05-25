# NSE Paper Reproducible Release

This is the clean handoff surface for the current `main.pdf` experiments. It
contains only the paper-facing code paths and compact result summaries.

Historical source-writeback stress-test scripts are intentionally excluded from
this directory.

## Code Entry Points

| Purpose | Script |
|---|---|
| Main NSE result tables | [01_nse_main_table_train.py](code/01_nse_main_table_train.py#L1) |
| Component ablations A0--A8 | [02_nse_component_ablation_A0_A8_train.py](code/02_nse_component_ablation_A0_A8_train.py#L1) |
| Five persistent supervision state proxy comparisons | [03_nse_persistent_supervision_state_proxy_train.py](code/03_nse_persistent_supervision_state_proxy_train.py#L1) |

## Reproduction Commands

These scripts are designed for c201. Set `PROJECT_ROOT` if the draft project is
not at the default path.

| Purpose | Command |
|---|---|
| Main CIFAR-100 `q=0.05, eta=0.3` NSE row | [01_reproduce_main_table_c201.sh](commands/01_reproduce_main_table_c201.sh#L1) |
| A0--A8 component ablations | [02_reproduce_component_ablation_A0_A8_c201.sh](commands/02_reproduce_component_ablation_A0_A8_c201.sh#L1) |
| Five persistent-state proxy rows | [03_reproduce_persistent_state_proxy_table_c201.sh](commands/03_reproduce_persistent_state_proxy_table_c201.sh#L1) |
| CIFAR-10 main table grid | [04_reproduce_cifar10_table_c201.sh](commands/04_reproduce_cifar10_table_c201.sh#L1) |
| CIFAR-100H row | [05_reproduce_cifar100h_table_c201.sh](commands/05_reproduce_cifar100h_table_c201.sh#L1) |
| Plankton, Treeversity, and Benthic rows | [06_reproduce_crowdsourced_table_c201.sh](commands/06_reproduce_crowdsourced_table_c201.sh#L1) |

## Result Summaries

| File | Meaning |
|---|---|
| [main_table_collected_results_with_paths_20260523.csv](results/main_table_collected_results_with_paths_20260523.csv#L1) | Main-table three-seed results, per-seed final values, and source log paths |
| [paper_table_results_summary_20260523.csv](results/paper_table_results_summary_20260523.csv#L1) | Compact summary for component and persistent-state rows currently used in the paper |
| [persistent_state_proxy_diagnostics_summary_20260523.csv](results/persistent_state_proxy_diagnostics_summary_20260523.csv#L1) | Diagnostic metrics for the five persistent-state proxy rows |
| [single_seed_validation_summary_20260524.csv](results/single_seed_validation_summary_20260524.csv#L1) | Representative single-seed validation pass |
| [noise_rate_slice_validation_summary_20260525.csv](results/noise_rate_slice_validation_summary_20260525.csv#L1) | Completed C10/C100 seed-1 noise-rate validation compared with paper three-seed values |

The one-to-one row checklist remains in
[paper_result_reproduction_matrix_20260523.csv](manifest/paper_result_reproduction_matrix_20260523.csv#L1).

Use [reproduction_command_result_crosswalk.csv](manifest/reproduction_command_result_crosswalk.csv#L1)
to map each command script to the corresponding code entry point and result
summary.
