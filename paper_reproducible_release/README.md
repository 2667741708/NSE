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

## Result Summaries

| File | Meaning |
|---|---|
| [paper_table_results_summary_20260523.csv](results/paper_table_results_summary_20260523.csv#L1) | Compact summary for component and persistent-state rows currently used in the paper |
| [persistent_state_proxy_diagnostics_summary_20260523.csv](results/persistent_state_proxy_diagnostics_summary_20260523.csv#L1) | Diagnostic metrics for the five persistent-state proxy rows |
| [single_seed_validation_summary_20260524.csv](results/single_seed_validation_summary_20260524.csv#L1) | Representative single-seed validation pass |

The one-to-one row checklist remains in
[paper_result_reproduction_matrix_20260523.csv](manifest/paper_result_reproduction_matrix_20260523.csv#L1).
