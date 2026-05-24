# NSE Paper Reproducibility Package

This directory is the paper-facing reproducibility package for the current NSE
manuscript. It separates the scripts and results that support the reported
paper claims from older exploratory or deprecated code.

## Scope

This package records three evidence tracks:

1. Main NSE results and standard comparison tables.
2. Component ablations A0--A8.
3. Persistent supervision state proxy experiments.

The package does not claim that every historical script in the repository is
active. Legacy scripts are documented in [archive_policy.md](archive_policy.md).

## Trusted Code Snapshots

| Track | Local snapshot | Source role |
|---|---|---|
| Main NSE results | [bayes_unified_main_20260523.py](code/main/bayes_unified_main_20260523.py#L1) | Historical main-result training entry copied from c201 |
| Component ablations | [train_nse_component_ablation_20260523.py](code/component_ablation/train_nse_component_ablation_20260523.py#L1) | Current `main.pdf` A0--A8 entry |
| Persistent-state v2 proxies | [train_nse_source_mvp_20260523.py](code/persistent_state/train_nse_source_mvp_20260523.py#L1) | Current local release implementation |
| Persistent-state v2 proxy snapshot | [train_nse_persistent_state_proxies_20260523.py](code/persistent_state/train_nse_persistent_state_proxies_20260523.py#L1) | Exact c201 stable script snapshot used by the completed v2 proxy suite |
| Persistent-state helper tests | [test_persistent_state_helpers_20260523.py](code/persistent_state/test_persistent_state_helpers_20260523.py#L1) | Local helper tests for update operators |

The script hashes are stored in [sha256sums.txt](manifest/sha256sums.txt).

Older component-audit copies were moved to
[legacy_component_audit_20260523](archive/legacy_component_audit_20260523/).
They are retained for traceability only and are excluded from the active
paper-facing surface.

## Result Index

The result index is [result_index.csv](manifest/result_index.csv). It records
the result roots, hardware policy, status, and whether a row is paper-facing,
internal, historical, or deprecated.

Local snapshots of the current paper-facing result logs are stored in
[results/](results/README.md). The compact parsed table is
[summary_20260523.csv](results/summary_20260523.csv), and hashes for the copied
result evidence are stored in
[result_sha256sums.txt](manifest/result_sha256sums.txt).

## Reproduction Commands

Use these command templates on c201:

- [reproduce_main_results_c201.sh](commands/reproduce_main_results_c201.sh)
- [reproduce_component_ablation_c201.sh](commands/reproduce_component_ablation_c201.sh)
- [reproduce_persistent_state_v2_c201.sh](commands/reproduce_persistent_state_v2_c201.sh)
- [run_single_seed_validation_c201_gpu0.sh](commands/run_single_seed_validation_c201_gpu0.sh)
- [run_single_seed_validation_c201_gpu1.sh](commands/run_single_seed_validation_c201_gpu1.sh)

The single-seed launchers are a validation pass for the streamlined package.
They do not replace the three-seed paper tables; they check that the active
scripts can still reproduce representative rows with the same configuration
family.

The 2026-05-24 single-seed validation completed on c201. The parsed summary is
[single_seed_validation_summary_20260524.csv](results/single_seed_validation_summary_20260524.csv),
with copied logs under
[single_seed_validation_20260523](results/single_seed_validation_20260523/).

These commands document the intended reproduction entry points. They should be
run from the c201 environment after verifying that the expected Python
environment, data root, and GPU availability match the original run.

The active c201 entry points for the persistent-state proxy suite are now the
stable names `train_nse_persistent_state_proxies.py`,
`run_persistent_state_proxies_gpu0.sh`, and
`run_persistent_state_proxies_gpu1.sh`. Timestamped development variants have
been archived on c201 under
`experiments/nse_mvp_source_writeback_20260514/archive/deprecated_timestamped_20260523/`.

## Reference Check

IRNet is cited as the formal TPAMI 2026 publication, not as the older arXiv
preprint. The citation check is recorded in
[reference_check.md](manifest/reference_check.md).

## Verification Status

Last checked on 2026-05-23:

- `py -3 -m py_compile` passed for every copied Python snapshot in
  [code/](code/main/bayes_unified_main_20260523.py#L1).
- `latexmk -pdf -interaction=nonstopmode main.tex` passed from the repository
  root and regenerated `main.pdf`.
- `main.log` had no undefined citation or undefined reference warnings.

Additional check on 2026-05-24:

- c201 single-seed validation completed for 7 representative rows: FREDIS,
  IRNet, PALS/SARI, UPLLRS, PiCO+, component A0, and the main CIFAR-100
  `q=0.05, eta=0.3` row.
