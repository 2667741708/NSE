# Archive Policy

This package distinguishes active paper evidence from legacy code.

## Active

Use these for current paper-facing claims:

- [Main result snapshot](code/main/bayes_unified_main_20260523.py#L1)
- [A0--A8 manuscript component snapshot](code/component_ablation/train_nse_component_ablation_20260523.py#L1)
- [Persistent-state v2 snapshot](code/persistent_state/train_nse_persistent_state_proxies_20260523.py#L1)
- [Persistent-state v2 GPU0 launcher](code/persistent_state/run_persistent_state_proxies_gpu0_20260523.sh#L1)
- [Persistent-state v2 GPU1 launcher](code/persistent_state/run_persistent_state_proxies_gpu1_20260523.sh#L1)

## Historical

Keep these only as context unless a row is explicitly requalified:

- Earlier M0--M6 source write-back stress tests.
- `train_nse_source_mvp_family_20260520.py`.
- `train_nse_source_mvp_family_20260521_strict.py`.
- `train_nse_source_mvp_family_20260521_prior_aligned.py`.
- c201 timestamped persistent-state launchers and script variants archived in
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/archive/deprecated_timestamped_20260523/`.
- The previous local component-audit snapshot and copied logs under
  [legacy_component_audit_20260523](archive/legacy_component_audit_20260523/).
- d437 historical runs.

## Deprecated

Do not use these for manuscript results:

- `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/deprecated_invalid_20260513/bayes_unified_unified_ablation.py`
- Old E5/E8 rows produced before the audit-script repair.

## Physical Archiving

Root-level local legacy files are not moved because many old result logs and
scripts are referenced by existing documents. On c201, timestamped
persistent-state development variants have been physically archived after
creating stable entry points. Future physical cleanup should follow the same
pattern: create a stable active entry point first, then move old timestamped
files into an archive directory, and finally update
[result_index.csv](manifest/result_index.csv).
