# NSE Experiment Script Taxonomy

## Active Paper-Facing Tracks

| Track | Local snapshot | Remote source path | Paper use |
|---|---|---|---|
| Main NSE results | [bayes_unified_main_20260523.py](../code/main/bayes_unified_main_20260523.py#L1) | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py` | Main method results |
| Component ablations A0--A8 | [train_nse_component_ablation_20260523.py](../code/component_ablation/train_nse_component_ablation_20260523.py#L1) | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablation.py` | Current manuscript component table |
| Persistent supervision state v2 proxies | [train_nse_persistent_state_proxies_20260523.py](../code/persistent_state/train_nse_persistent_state_proxies_20260523.py#L1) | `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_persistent_state_proxies.py` | Final five conservative prior-aligned proxy rows |

## Local Release Surface

The local release script
[train_nse_source_mvp_20260523.py](../code/persistent_state/train_nse_source_mvp_20260523.py#L1)
is the reproducible implementation surface for persistent-state operators. It
contains the current source-update modes, diagnostics, and helper-test coverage
used to validate the v2 proxy family.

The stable c201 launchers are:

- [run_persistent_state_proxies_gpu0_20260523.sh](../code/persistent_state/run_persistent_state_proxies_gpu0_20260523.sh#L1)
- [run_persistent_state_proxies_gpu1_20260523.sh](../code/persistent_state/run_persistent_state_proxies_gpu1_20260523.sh#L1)

## Legacy Boundary

Do not use the old unified ablation script for current manuscript rows:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/deprecated_invalid_20260513/bayes_unified_unified_ablation.py
```

It is kept only for traceability because old E5/E8-style flags did not alter
the intended training paths.

The earlier audit snapshot
`bayes_unified_unified_ablation_audit_20260523.py` and its copied logs are kept
under `reproducibility/archive/legacy_component_audit_20260523/`. They are not
the A0--A8 table currently printed in `main.pdf`.

Older persistent-state scripts such as `train_nse_source_mvp_family_20260520.py`,
`train_nse_source_mvp_family_20260521_strict.py`,
`train_nse_source_mvp_family_20260521_prior_aligned.py`, and the timestamped
v2 launchers are historical intermediate designs. On c201, those timestamped
variants have been moved to
`experiments/nse_mvp_source_writeback_20260514/archive/deprecated_timestamped_20260523/`.
Use the stable `train_nse_persistent_state_proxies.py` script and
`run_persistent_state_proxies_gpu*.sh` launchers for the current five-proxy
table.
