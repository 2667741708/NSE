# NSE Persistent Supervision-State Ablation: Carry-Over vs Restored Extraction

Last updated: 2026-05-20 (Asia/Shanghai)

## Purpose

This experiment studies whether NPLL performance should be obtained by carrying model-induced supervision state across epochs, or by restoring the working prior from the native weak-supervision record at each epoch and extracting reliable active supervision from that restored state.

The paper claim being tested is:

> NSE does not claim that weak supervision is never transformed. NSE preserves the native weak-supervision record and avoids inheriting model-induced persistent supervision state. At the beginning of each epoch, it restores the working prior from the original candidate/crowd record and applies Topology-DAES, prior-constrained model evidence, active selection, and salvage only to temporary working evidence.

This ablation focuses on persistent carry-over variants where model predictions or extracted evidence become a supervision state inherited by later epochs. The goal is not to prove that carry-over never helps, but to test whether it is necessary and whether it introduces source contamination, wrong writes, wrong promotions, source drift, clean-sample damage, or long error survival.

## Latest Plan Update: 2026-05-20

The paper-facing claim has been upgraded from `source write-back` to `persistent supervision state`.

| Term | Working definition |
|---|---|
| Native source | The original candidate mask or crowd-prior record, denoted `Omega^0`. |
| Epoch-local evidence | Per-epoch `P_geo`, prior-projected model evidence, `P^(2)`, pseudo-labels, reliable selection, and salvage decisions. |
| Persistent supervision state | Any supervision object inherited by later epochs: candidate membership, reconstructed candidate set, soft pseudo-target, label-confidence or label-importance vector, or reliable/unreliable promotion state. |
| NSE reset rule | Model parameters, prototypes, and diagnostic histories are inherited; model-induced supervision state is not. The working prior is restored from `Omega^0` before each extraction stage. |

### Five Prior-Aligned Proxy Conditions

These rows are controlled proxies for inherited supervision-state mechanisms. They are not exact reproductions of the cited prior methods.

Fixed setup: CIFAR-100, `q=0.05`, `eta=0.3`, 500 epochs, seeds `1 2 3`, ResNet-18, `--source_update_evidence p2`, `--max_w_model 0.5`, `--model_warmup_epochs 20`.

| Proxy | Prior mechanism | Command-level setting | Host |
|---|---|---|---|
| `FREDIS_PSS` | refinement plus disambiguation candidate state | `--source_update_mode add_remove --source_update_scope all --source_update_threshold 0.85 --source_remove_threshold 0.05` | c201 GPU0 |
| `IRNet_PSS` | noisy-sample correction state | `--source_update_mode hard_insert --source_update_scope unreliable_highconf --source_update_threshold 0.85` | c201 GPU0 |
| `PALS_SARI_PSS` | partial-label augmentation state | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold 0.85` | c201 GPU0 |
| `UPLLRS_PSS` | persistent reliable-promotion state | `--persistent_promotion_mode hard --promotion_scope unreliable_highconf --promotion_threshold 0.95 --promotion_source p2` with `--source_update_mode none --source_update_scope none` | c201 GPU0 |
| `PiCOPlus_PSS` | persistent soft pseudo-target state | `--source_update_mode soft_evidence --source_update_scope all --source_update_alpha 0.1` | d437 |

For candidate-membership insertion rows, `--source_insert_margin 0.0` is used.
The implementation only inserts a predicted label when the top-1 evidence label
is outside the current candidate support. A positive margin can be used for a
stricter IRNet-style variant, but the main proxy keeps margin 0.0 to avoid
introducing an extra tuning knob.

Implementation links:

- Persistent-promotion CLI arguments are defined at [train_nse_source_mvp.py:L534-L545](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L534-L545).
- Non-candidate-only membership insertion and soft-mass metrics are implemented at [train_nse_source_mvp.py:L1950-L1972](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L1950-L1972).
- The UPLLRS-style persistent promotion state is implemented at [train_nse_source_mvp.py:L2033-L2101](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L2033-L2101).
- Promoted samples are forced into active training before reliable/salvage samples at [train_nse_source_mvp.py:L2552-L2619](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L2552-L2619).
- The c201 launcher is [launch_c201_persistent_state_five_20260520.sh](../experiments/nse_mvp_source_writeback_20260515_windows/launch_c201_persistent_state_five_20260520.sh).
- The d437 PiCO+-PSS launcher is [launch_d437_persistent_state_pico_20260520.sh](../experiments/nse_mvp_source_writeback_20260515_windows/launch_d437_persistent_state_pico_20260520.sh).

Result placeholders:

| Condition | Status | Best Acc | Final Acc | Prec(A) | Cov(A) | NRR-A | Write/PromoteCov | WrongWrite/WrongPromote | SrcRec-N | MassRec-N | CleanDamage | DamageMass | SourceDrift | AUCDrift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NSE-Reset` | existing diagnostic baseline | `79.55 +/- 0.04` | `79.38 +/- 0.16` | `0.932` | `0.887` | `0.660` | `0.000` | `--` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `FREDIS_PSS` | strict run active on c201 GPU0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `IRNet_PSS` | queued after `FREDIS_PSS` in strict c201 suite | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `PALS_SARI_PSS` | queued after `IRNet_PSS` in strict c201 suite | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `UPLLRS_PSS` | smoke passed; queued after `PALS_SARI_PSS` in strict c201 suite | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `PiCOPlus_PSS` | strict run active on d437 GPU0 via `nohup` | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

2026-05-21 strict instrumentation update:

- The first launched 2026-05-20 run is treated as a partial instrumentation run
  because hard insertion could strengthen an already-candidate top-1 label and
  PiCO+-PSS did not yet log `MassRecN`, `DamageMass`, and `AUCDrift`.
- The strict version adds non-candidate-only insert masking, optional
  `--source_insert_margin`, and soft-mass diagnostics. Publication-facing
  results should come from the strict result root, not the partial 2026-05-20
  root.
- The strict c201 suite is running in tmux session
  `nse_pss_c201_strict_20260521` with result root
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_five_20260521_strict`.
- The strict d437 `PiCOPlus_PSS` run is running via `nohup` with result root
  `/home/d437/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_five_20260521_strict`.
- A 2-epoch strict hard-insert smoke on c201 logged nonzero `WriteCov`,
  `MassRecN`, `DamageMass`, and `AUCDrift`, confirming that strict diagnostics
  are emitted.

2026-05-21 prior-aligned operator correction:

- The earlier five-proxy suite remains useful as a conservative stress test, but
  `FREDIS_PSS`, `IRNet_PSS`, and `PALS_SARI_PSS` used simplified generic
  operators. A corrected suite was added so the inherited supervision-state
  operator more closely follows the cited algorithms.
- FREDIS is now represented by label-level score-difference movement:
  `y_hat=argmax_j f_j(x)`, refinement adds a non-candidate label when
  `f_{y_hat}(x)-f_j(x) <= zeta`, and disambiguation removes a candidate label
  when `f_{y_hat}(x)-f_j(x) >= zeta_bar`. The implementation also caps
  refinement labels so disambiguation labels are at least twice as many in the
  launched setting, matching FREDIS' controlled fusion-round intent.
- IRNet is now represented by its candidate/non-candidate margin detector:
  `tau(x)=max_{j in S(x)} f_j(x)-max_{j notin S(x)} f_j(x)`. Samples with
  `tau < 0` insert the highest-scoring non-candidate label into the candidate
  set, corresponding to IRNet's default correction without optional swapping.
- PALS/SARI is now represented by its partial-label augmentation rule:
  `Y_i^{t+1}=Y_i union {argmax_c h_t^c(f_t(aug_w(x_i)))}` if the max
  confidence is above the decaying `lambda_t`; the launcher uses the paper's
  `pals_linear` schedule from `0.45` to `0.35`.
- UPLLRS remains a persistent sample-promotion proxy rather than candidate-set
  mutation: high-confidence pseudo-labels from the unreliable/non-active pool
  are stored and forced into later active training.
- PiCO+ remains a soft pseudo-target carry-over proxy, not a hard NPL
  candidate-set rewrite. The corrected mode updates the inherited soft target as
  `Q^{t+1}=Norm((1-alpha)Q^t+alpha E^t)`, with optional candidate projection
  available but not enabled in the NPLL recovery run.
- Prior-aligned helper functions are implemented at
  [train_nse_source_mvp.py:L1978-L2050](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L1978-L2050),
  and the runtime branches are wired at
  [train_nse_source_mvp.py:L2289-L2420](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L2289-L2420).
- The corrected c201 GPU1 launcher is
  [launch_c201_persistent_state_prior_aligned_20260521.sh:L72-L102](../experiments/nse_mvp_source_writeback_20260515_windows/launch_c201_persistent_state_prior_aligned_20260521.sh#L72-L102).
- Local helper tests and remote helper tests passed. A c201 GPU1 2-epoch smoke
  completed all five corrected branches under
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_prior_aligned_20260521_smoke`.
- The full corrected suite is running on c201 GPU1 in tmux session
  `nse_pss_prior_gpu1_20260521`, result root
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_prior_aligned_20260521`.

Corrected five-proxy commands:

| Proxy | Corrected mechanism | Command-level setting | Status |
|---|---|---|---|
| `FREDIS_MOVE_PSS` | score-difference refinement plus disambiguation | `--source_update_mode fredis_move --source_update_scope all --fredis_refine_threshold 0.01 --fredis_disamb_threshold 0.80 --fredis_min_disamb_over_refine 2.0` | running on c201 GPU1 |
| `IRNet_CORRECT_PSS` | `tau=max_candidate-max_non_candidate`; insert best non-candidate when `tau<0` | `--source_update_mode irnet_correct --source_update_scope all --irnet_tau_boundary 0.0 --irnet_min_non_candidate_conf 0.0` | queued after FREDIS |
| `PALS_SARI_AUG_PSS` | decaying-threshold top-1 partial-label augmentation | `--source_update_mode pals_augment --source_update_scope all_highconf --source_update_schedule pals_linear` | queued after IRNet |
| `UPLLRS_PROMOTE_PSS` | persistent reliable-promotion state | `--persistent_promotion_mode hard --promotion_scope unreliable_highconf --promotion_threshold 0.95 --promotion_source p2` with no source mutation | queued after PALS/SARI |
| `PiCOPlus_SOFT_PSS` | persistent soft pseudo-target state | `--source_update_mode pico_soft_target --source_update_scope all --source_update_alpha 0.1` | queued after UPLLRS |

2026-05-21 v2 conservative prior-aligned design:

- The v1 corrected suite was stopped before completion because the final design
  is more conservative and closer to each prior method's own update trigger.
- `FREDIS_V2_PSS` uses FREDIS-style score gaps, but adds conservative absolute
  thresholds: only the top non-candidate is refined in when
  `f_yhat - f_j <= 0.05` and `f_j >= 0.85`; candidate labels are removed only
  when `f_yhat - f_j >= 0.85` and `f_j <= 0.05`; the implementation prevents
  empty candidate sets.
- `IRNet_V2_PSS` follows IRNet's candidate/non-candidate gap detector instead
  of NSE's estimated noise set: insert the highest-scoring non-candidate only
  when it exceeds the best candidate and has confidence at least `0.85`.
- `PALS_SARI_V2_PSS` keeps all-sample top-1 partial-label augmentation, but
  uses a conservative high-threshold linear schedule `0.95 -> 0.85` rather than
  SARI's lower schedule, because the original lower threshold is coupled to
  SARI's full label-smoothing and regularization pipeline.
- `UPLLRS_V2_PSS` uses the current NSE-estimated noisy set as the unreliable
  pool, matching UPLLRS' reliable/unreliable design. Promotion labels are
  selected only from the native non-candidate label space and stored
  persistently.
- `PiCOPlus_V2_PSS` remains a persistent soft pseudo-target proxy:
  `Omega_i^{t+1}=Norm((1-alpha)Omega_i^t+alpha P_i^{(2),t})`, with
  `alpha=0.1`. This tests soft state persistence, not hard candidate rewrite.
- The v2 c201 result root is
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_v2_20260521`.
- v2 GPU split:

| GPU | Conditions | tmux |
|---:|---|---|
| 0 | `FREDIS_V2_PSS`, `IRNet_V2_PSS`, `PALS_SARI_V2_PSS` | `nse_pss_v2_gpu0_20260521` |
| 1 | `UPLLRS_V2_PSS`, `PiCOPlus_V2_PSS` | `nse_pss_v2_gpu1_20260521` |

Both v2 launchers wait for unrelated GPU jobs to release memory before
starting. Local and remote helper tests passed, and remote `py_compile` passed.

Execution status at 2026-05-20 23:45 CST for the superseded partial run:

- c201 import check passed from the old project root: `from data.dataset import CIFAR100Partial`.
- d437 import check passed from the d437 project root: `from data.dataset import CIFAR100Partial`.
- 2-epoch `UPLLRS_PSS` smoke passed on c201 GPU0. The log contained `[PersistentState]` promotion diagnostics, `PromoteActiveCount=90` at epoch 2, and `[MVPSource] mode=none ... SourceDrift=0.000000`, confirming that the proxy promotes sample state without mutating `mutable_source_prior`.
- c201 full suite is running in tmux session `nse_pss_c201_20260520`.
- d437 lacks `tmux`, so `PiCOPlus_PSS` is running under `nohup`; the launcher log is under `/home/d437/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_five_20260520/_launcher_logs/`.

## Latest Audit Update: 2026-05-18

- `d437` CIFAR-100 `q=0.05, eta=0.3` M0-M4 are now complete for seeds 1/2/3.
- M3 final result: best accuracy `61.52 +/- 0.31`, final accuracy `61.32 +/- 0.19`, `Prec(A)=0.644078`, `Cov(A)=0.926013`, `NRR-A=0.593653`, `WriteCov=0.021507`, `WrongWrite=0.603933`, `SrcRec-N=0.691725`, `CleanDamage=0.000048`, `SourceDrift=1.301506`.
- M4 `Unreliable-only Persistent SoftMix alpha=0.3` final result on d437: best accuracy `62.33 +/- 0.20`, final accuracy `62.16 +/- 0.17`.
- `d437` M5 `Unreliable-only Persistent Replace` is currently running: seed 1 finished, seed 2 was around epoch 41/500 at the 2026-05-18 13:18 CST check, seed 3 not started.
- `c201` eta=0.5 M1-M6 are complete for seeds 1/2/3; eta=0.5 M0 has only seeds 2/3 on c201 and should not be rerun under the same exp name because `master_log.txt` is opened with mode `w`.
- `c201` eta=0.3 M6 `Reset NSE without Salvage` was launched as a previously unstarted exp in tmux session `c201_eta03_m6_20260518`.
- The executable release script now supports `source_update_mode in {none, hard_insert, hard_remove, add_remove, mix, replace, topk_reconstruct}` at [train_nse_source_mvp.py:L510-L524](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L510-L524). Periodic reset is controlled by `--source_reset_interval`.
- The local GitHub release script now emits clean/noisy subset active-supervision diagnostics through `[SubsetAudit]` at [train_nse_source_mvp.py:L1963-L2002](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L1963-L2002). Existing remote d437/c201 runs do not contain this new log unless the script is resynchronized and rerun.

## Conceptual Separation

| Symbol | Meaning | Persistence |
|---|---|---|
| `Omega^0_i` | Native weak-supervision record: original candidate mask or crowd prior. | Stored source record. |
| `tilde{Omega}^t_i` | Epoch-`t` working prior restored from `Omega^0_i` in NSE. | Temporary in NSE; persistent only in write-back variants. |
| `E^t_i` | Epoch-local evidence, including Topology-DAES belief, fused model/KNN evidence, and pseudo-labels. | Temporary. |
| `A_t` | Active training eligibility: reliable plus salvaged samples. | Temporary training state. |

NSE reset rule:

```text
tilde{Omega}^t_i <- Restore(Omega^0_i)
```

Persistent carry-over rule:

```text
tilde{Omega}^{t+1}_i <- U(tilde{Omega}^t_i, E^t_i, f^t_theta)
```

## Core Hypotheses

| Hypothesis | Observable evidence |
|---|---|
| H1. NSE can learn without persistent supervision-state carry-over. | `Acc` remains high while `SourceDrift=0`, `WriteCov=0`, `PromoteCov=0`, and `CleanDamage=0`. |
| H2. NSE extracts high-quality active supervision from the restored native prior. | High `Prec(A)`, high `Cov(A)`, and high `NRR-A`. |
| H3. Model-prediction carry-over can recover some missing true-label support but may pollute inherited supervision state. | Source-state variants show nonzero `SrcRec-N`, but also high `WrongWrite`, nonzero `SourceDrift`, and possible `CleanDamage`; promotion variants are checked by `WrongPromote`. |
| H4. Higher persistent supervision-state modification is not necessarily better. | Carry-over variants may have higher source recovery, promotion coverage, or active coverage but lower accuracy and lower active precision than NSE. |
| H5. Reset frequency should control long-term bias accumulation. | Increasing reset interval `K` should increase `SourceDrift` and wrong-write survival if persistent carry-over is harmful. |

## Metrics

| Metric | Definition / interpretation |
|---|---|
| `Acc` | Final test accuracy, reported as best accuracy over 500 epochs unless otherwise stated. |
| `Prec(R)` | Precision of reliable set pseudo-labels. |
| `Cov(R)` | Fraction of samples selected into reliable set. |
| `Prec(A)` | Precision of active set pseudo-labels, where `A = R union S`. |
| `Cov(A)` | Fraction of samples used for active supervised training. |
| `NRR-A` | Noisy active recovery rate: fraction of initially noisy samples recovered into active supervision with correct pseudo-labels. |
| `WriteCov` | Fraction of samples whose working source was updated by write-back. |
| `WrongWrite` | Error rate among written model predictions. |
| `PromoteCov` | Fraction of samples stored in a persistent sample-promotion state. |
| `WrongPromote` | Error rate among newly promoted pseudo-labels. |
| `SrcRec-N` / `NoisyRecovery` | Fraction of initially noisy samples whose true label becomes included in the working source support. Interpret together with `WrongWrite`. |
| `CleanDamage` | Fraction of initially clean samples whose true-label support is damaged by source update. |
| `SourceDrift` | Mean L1 drift between working source and original source. |
| `WrongSurvival` | Mean number of epochs for which a wrong source modification remains present. NSE-reset should be <= 1 for injected temporary errors. |

## Current Executable Method Variants

These names describe the implemented stress-test behavior. They should not be presented as exact replicas of prior methods such as PALS/SARI, FREDIS, or PLRC.

| ID | Method | Source update | Scope | Main role |
|---|---|---|---|---|
| M0 | `NSE-Reset` | none | none | Main reset-before-extraction baseline. |
| M1 | `Aggressive Top-1 Persistent Insert` | hard insert | all high-confidence, relaxed threshold schedule | Stress-tests aggressive all-sample model write-back. |
| M2 | `Thresholded All-Sample Persistent Insert` | hard insert | all samples with model confidence >= 0.65 | Tests fixed medium-high threshold write-back. |
| M3 | `Thresholded Unreliable-only Persistent Insert` | hard insert | unreliable samples only, confidence >= 0.65 | Tests whether selective unreliable-sample insertions are safe. |
| M4 | `Unreliable-only Persistent SoftMix` | soft mix, alpha=0.3 | unreliable samples only, confidence >= 0.65 | Tests softer persistent carry-over. |
| M5 | `Unreliable-only Persistent Replace` | replace source row with model one-hot | unreliable samples only, confidence >= 0.65 | Strong upper-risk persistent replacement baseline. |
| M6 | `Reset NSE without Salvage` | none | no salvage promotion | Tests whether salvage contributes to noisy recovery and accuracy under reset. |

## Full Source-Restoration Ablation Design

The current M0-M6 runs cover the executable subset of this design. The full paper-level diagnostic should use the following names.

| ID | Condition | Cross-epoch rule | Purpose |
|---|---|---|---|
| R0 | `NSE-Reset` | `tilde{Omega}^t <- Omega^0` | Main method. |
| R1 | `Persistent-SoftMix` | `Norm((1-alpha) tilde{Omega}^t + alpha E^t)`, alpha in `{0.1,0.3,0.5}` | Tests soft source drift. |
| R2 | `Persistent-HardInsert` | Add high-confidence non-candidate prediction. | Simulates refinement / partial-label augmentation. |
| R3 | `Persistent-HardRemove` | Remove low-confidence candidate labels. | Simulates purification / disambiguation. |
| R4 | `Persistent-AddRemove` | `(tilde{Y}^t \\ R^t) union A^t` | Tests combined candidate-state modification. |
| R5 | `Persistent-Reconstruct` | `tilde{Y}^{t+1}=TopK(E^t)` | Simulates candidate label reconstruction. |
| R6 | `Periodic-Reset` | Reset every `K` epochs, `K in {1,5,10,50,infty}` | Tests reset-frequency sensitivity. |
| R7 | `Bias-Injection Stress` | Inject controlled wrong source update at `t0`. | Measures recovery and wrong-write survival. |

## Newly Launched Missing-Variant Runs

Launched on c201 at 2026-05-18 15:08 CST without reusing any existing exp name.

| Exp | Design family | Key flags | Status |
|---|---|---|---|
| `eta03_M7_HardRemove` | hard-remove purification | `--source_update_mode hard_remove --source_update_scope all --source_remove_threshold 0.05` | running on c201 GPU1 |
| `eta03_M8_AddRemove` | insert+remove candidate-state modification | `--source_update_mode add_remove --source_update_scope unreliable_highconf --source_update_threshold 0.65 --source_remove_threshold 0.05` | queued after M7 |
| `eta03_M9_PeriodicReset10` | reset-frequency diagnostic | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold 0.65 --source_reset_interval 10` | queued after M8 |
| `eta03_M10_TopKReconstruct` | PLRC-style top-k reconstruction stress test | `--source_update_mode topk_reconstruct --source_update_scope all --source_reconstruct_k 5` | queued after M9 |

Remote extended script:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp_extended_20260518.py
```

tmux session:

```text
c201_missing_source_modes_20260518
```

## Run Assignment

| Host | GPU | Assignment | Result root |
|---|---:|---|---|
| `d437` | TITAN RTX | `eta=0.3`, all 7 variants | `/home/d437/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_mvp_source_writeback_20260514/main_table_e500` |
| `c201` | RTX 5080 GPU0 | `eta=0.5`, M0 seeds 2-3 and M1/M3/M5 | `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_mvp_source_writeback_20260514/main_table_e500` |
| `c201` | RTX 5080 GPU1 | `eta=0.5`, M2/M4/M6 after old audit finishes | same as above |
| local Windows | RTX 3080 | `eta=0.5`, M0 seed 1 | `D:\nse_mvp_win\results\main_table_e500` |

Note: `eta=0.5 M0_NSE` is split across hosts. Local Windows owns seed 1, while C201 GPU0 owns seeds 2-3. Merge by seed for the final table.

## Current Status Summary

| Dataset / noise | Method | Status | Best Acc | Final Acc | Main observation |
|---|---|---:|---:|---:|---|
| CIFAR100 q=.05 eta=.3 | M0 `NSE-Reset` | complete 3/3 | `79.55 ± 0.04` | `79.38 ± 0.16` | Strong restored-source baseline; no source drift. |
| CIFAR100 q=.05 eta=.3 | M1 `Aggressive Top-1 Persistent Insert` | complete 3/3 | `57.50 ± 0.81` | `57.34 ± 0.87` | Aggressive write-back severely hurts accuracy and active precision. |
| CIFAR100 q=.05 eta=.3 | M2 `Thresholded All-Sample Persistent Insert` | complete 3/3 | `62.94 ± 0.55` | `62.68 ± 0.59` | Fixed threshold reduces clean damage vs M1 but still far below NSE. |
| CIFAR100 q=.05 eta=.3 | M3 `Thresholded Unreliable-only Persistent Insert` | complete 3/3 | `61.52 ± 0.31` | `61.32 ± 0.19` | Selective unreliable-only insertion is less invasive in write coverage but still has high wrong-write rate and large source drift. |
| CIFAR100 q=.05 eta=.3 | M4 `Unreliable-only Persistent SoftMix` | complete 3/3 | `62.33 +/- 0.20` | `62.16 +/- 0.17` | Softer persistent carry-over still trails reset by a wide margin. |
| CIFAR100 q=.05 eta=.3 | M5 `Unreliable-only Persistent Replace` | running on d437 | `[pending]` | `[pending]` | Seed 1 finished; seed 2 running as of 2026-05-18 13:18 CST. |
| CIFAR100 q=.05 eta=.3 | M6 `Reset NSE without Salvage` | running on c201 | `[pending]` | `[pending]` | Launched in tmux session `c201_eta03_m6_20260518`. |
| CIFAR100 q=.05 eta=.5 | M0 `NSE-Reset` | complete across split hosts | `~76.4-76.6` | `[merge pending]` | Strong restored-source result under more severe true-label-absent noise. |
| CIFAR100 q=.05 eta=.5 | M1 `Aggressive Top-1 Persistent Insert` | complete 3/3 | `56.00 ± 0.46` | `55.88 ± 0.48` | Severe degradation and high wrong-write rate. |
| CIFAR100 q=.05 eta=.5 | M2 `Thresholded All-Sample Persistent Insert` | complete 3/3 | `60.44 ± 0.37` | `60.31 ± 0.41` | Better than M1 but still much worse than NSE. |
| CIFAR100 q=.05 eta=.5 | M3 `Thresholded Unreliable-only Persistent Insert` | complete 3/3 on c201 | `59.11 +/- 0.32` | `58.98 +/- 0.30` | Selective persistent insertion remains far below reset. |
| CIFAR100 q=.05 eta=.5 | M4 `Unreliable-only Persistent SoftMix` | complete 3/3 on c201 | `60.01 +/- 0.36` | `59.90 +/- 0.36` | Soft source mixing still carries persistent error. |
| CIFAR100 q=.05 eta=.5 | M5 `Unreliable-only Persistent Replace` | complete 3/3 on c201 | `60.56 +/- 0.34` | `60.42 +/- 0.26` | Replacement is also far below reset. |
| CIFAR100 q=.05 eta=.5 | M6 `Reset NSE without Salvage` | complete 3/3 on c201 | `73.70 +/- 0.27` | `73.56 +/- 0.25` | Salvage contributes materially under reset. |

## Completed Diagnostic Values

### CIFAR100 q=.05 eta=.3

| Method | Acc | Prec(A) | Cov(A) | NRR-A | WriteCov | WrongWrite | SrcRec-N | CleanDamage | SourceDrift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 `NSE-Reset` | `79.55 ± 0.04` | `~0.9315` | `~0.8871` | `~0.6600` | `0` | `-` | `0` | `0` | `0` |
| M1 `Aggressive Top-1 Persistent Insert` | `57.50 ± 0.81` | `~0.5849` | `~0.9683` | `~0.5664` | `~0.9922` | `~0.4232` | `~0.7540` | `~0.1804` | `~1.8287` |
| M2 `Thresholded All-Sample Persistent Insert` | `62.94 ± 0.55` | `~0.6517` | `~0.9606` | `~0.6189` | `~0.8419` | `~0.3265` | `~0.7395` | `~0.0062` | `~1.8043` |
| M3 `Thresholded Unreliable-only Persistent Insert` | `61.52 ± 0.31` | `~0.6441` | `~0.9260` | `~0.5937` | `~0.0215` | `~0.6039` | `~0.6917` | `~0.0000` | `~1.3015` |
| M4 `Unreliable-only Persistent SoftMix` | `62.33 +/- 0.20` | `[pending extraction aggregate]` | `[pending extraction aggregate]` | `[pending extraction aggregate]` | `[pending extraction aggregate]` | `[pending extraction aggregate]` | `[pending extraction aggregate]` | `[pending extraction aggregate]` | `[pending extraction aggregate]` |
| M5 `Unreliable-only Persistent Replace` | `[pending]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` |
| M6 `Reset NSE without Salvage` | `[pending]` | `[x]` | `[x]` | `[x]` | `0` | `-` | `0` | `0` | `0` |

### CIFAR100 q=.05 eta=.5

| Method | Acc | Prec(A) | Cov(A) | NRR-A | WriteCov | WrongWrite | SrcRec-N | CleanDamage | SourceDrift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 `NSE-Reset` | `~76.4-76.6` | `~0.9008` | `~0.8425` | `~0.616` | `0` | `-` | `0` | `0` | `0` |
| M1 `Aggressive Top-1 Persistent Insert` | `56.00 ± 0.46` | `~0.5696` | `~0.9668` | `~0.5482` | `~0.9905` | `~0.4385` | `~0.7317` | `~0.1919` | `~1.8767` |
| M2 `Thresholded All-Sample Persistent Insert` | `60.44 ± 0.37` | `~0.6290` | `~0.9449` | `~0.5874` | `~0.8379` | `~0.3514` | `~0.7126` | `~0.0062` | `~1.8582` |
| M3 `Thresholded Unreliable-only Persistent Insert` | `[pending]` | seed1 `0.6219` | seed1 `0.9218` | seed1 `0.5701` | seed1 `0.0222` | seed1 `0.6532` | seed1 `0.6781` | seed1 `0.0000` | seed1 `1.3926` |
| M4 `Unreliable-only Persistent SoftMix` | `[pending]` | seed1 `0.6311` | seed1 `0.9171` | seed1 `0.5740` | seed1 `0.0242` | seed1 `0.5982` | seed1 `0.6789` | seed1 `0.0077` | seed1 `1.2950` |
| M5 `Unreliable-only Persistent Replace` | `[pending]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` |
| M6 `Reset NSE without Salvage` | `[pending]` | `[x]` | `[x]` | `[x]` | `0` | `-` | `0` | `0` | `0` |

## Interpretation So Far

The completed results already support the extractive view strongly.

1. `M0 NSE-Reset` achieves strong accuracy under epoch-wise source restoration. For `eta=.3`, NSE reaches `79.55 ± 0.04` with `SourceDrift=0`, `WriteCov=0`, and `CleanDamage=0`. Its active supervision is both accurate and broad: `Prec(A)≈0.93`, `Cov(A)≈0.89`, and `NRR-A≈0.66`.

2. All-sample write-back is damaging. `M1 Aggressive Top-1 Persistent Insert` writes almost every sample each epoch by the end (`WriteCov≈0.99`) and recovers many noisy supports (`SrcRec-N≈0.75` for eta=.3), but it also writes wrong labels at a very high rate (`WrongWrite≈0.42`), damages many initially clean samples (`CleanDamage≈0.18`), and collapses accuracy to around `57.5` on eta=.3 and `56.0` on eta=.5.

3. A fixed confidence threshold helps but does not solve source pollution. `M2 Thresholded All-Sample Persistent Insert` reduces clean damage dramatically (`CleanDamage≈0.006`) but still has large `WrongWrite` (`~0.33-0.35`) and large source drift (`~1.8`). Its accuracy remains far below NSE.

4. Unreliable-only write-back appears less invasive in coverage, but its written labels are still risky. The first completed seeds for `M3` and `M4` write only about `2%-2.5%` samples per epoch at the end, but `WrongWrite` is very high (`~0.60-0.65`). This is exactly the failure mode the paper argues against: even selective model-driven source updates can inject persistent source errors.

## How This Validates the Paper Claim

| Paper claim | Evidence from this ablation |
|---|---|
| High-performance NPLL does not require persistent source write-back. | NSE has strong accuracy with `SourceDrift=0`. |
| The key is extracting active supervision, not persistent source modification. | NSE has high `Prec(A)` and high `Cov(A)` while keeping `SourceDrift=0`. |
| NSE can recover initially noisy samples without inserting labels into the source. | NSE has large `NRR-A` even though `SrcRec-N=0`. Recovery happens in active supervision, not in source membership. |
| Model write-back can recover source support but risks pollution. | Write-back variants have nonzero/high `SrcRec-N`, but also high `WrongWrite`, nonzero `SourceDrift`, and sometimes `CleanDamage`. |
| More aggressive source rewriting is not necessarily better. | M1 and M2 have high write coverage and source recovery but much lower accuracy than NSE. |

## Final Tables To Fill Later

### Main Table Placeholder

| Noise | Method | Source update | Scope | Acc | Prec(R) | Cov(R) | Prec(A) | Cov(A) | NRR-A |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| eta=.3 | NSE | none | none | `79.55 ± 0.04` | `[fill]` | `[fill]` | `~0.9315` | `~0.8871` | `~0.6600` |
| eta=.3 | Aggressive Top-1 Persistent Insert | hard insert | all-highconf | `57.50 ± 0.81` | `[fill]` | `[fill]` | `~0.5849` | `~0.9683` | `~0.5664` |
| eta=.3 | Thresholded All-Sample Persistent Insert | hard insert | all-highconf | `62.94 ± 0.55` | `[fill]` | `[fill]` | `~0.6517` | `~0.9606` | `~0.6189` |
| eta=.3 | Thresholded Unreliable-only Persistent Insert | hard insert | unreliable-highconf | `61.52 ± 0.31` | `[fill]` | `[fill]` | `~0.6441` | `~0.9260` | `~0.5937` |
| eta=.3 | Unreliable-only Persistent SoftMix | mix | unreliable-highconf | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| eta=.3 | Unreliable-only Persistent Replace | replace | unreliable-highconf | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| eta=.3 | Reset NSE without Salvage | none | no salvage promotion | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| eta=.5 | NSE | none | none | `~76.4-76.6` | `[fill]` | `[fill]` | `~0.9008` | `~0.8425` | `~0.616` |
| eta=.5 | Aggressive Top-1 Persistent Insert | hard insert | all-highconf | `56.00 ± 0.46` | `[fill]` | `[fill]` | `~0.5696` | `~0.9668` | `~0.5482` |
| eta=.5 | Thresholded All-Sample Persistent Insert | hard insert | all-highconf | `60.44 ± 0.37` | `[fill]` | `[fill]` | `~0.6290` | `~0.9449` | `~0.5874` |
| eta=.5 | Thresholded Unreliable-only Persistent Insert | hard insert | unreliable-highconf | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| eta=.5 | Unreliable-only Persistent SoftMix | mix | unreliable-highconf | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| eta=.5 | Unreliable-only Persistent Replace | replace | unreliable-highconf | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| eta=.5 | Reset NSE without Salvage | none | no salvage promotion | `[pending]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

### Source-Contamination Table Placeholder

| Noise | Method | WriteCov | WrongWrite | SrcRec-N | CleanDamage | SourceDrift |
|---|---|---:|---:|---:|---:|---:|
| eta=.3 | NSE | `0` | `-` | `0` | `0` | `0` |
| eta=.3 | Aggressive Top-1 Persistent Insert | `~0.9922` | `~0.4232` | `~0.7540` | `~0.1804` | `~1.8287` |
| eta=.3 | Thresholded All-Sample Persistent Insert | `~0.8419` | `~0.3265` | `~0.7395` | `~0.0062` | `~1.8043` |
| eta=.3 | Thresholded Unreliable-only Persistent Insert | `~0.0215` | `~0.6039` | `~0.6917` | `~0.0000` | `~1.3015` |
| eta=.3 | Unreliable-only Persistent SoftMix | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| eta=.3 | Unreliable-only Persistent Replace | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| eta=.5 | NSE | `0` | `-` | `0` | `0` | `0` |
| eta=.5 | Aggressive Top-1 Persistent Insert | `~0.9905` | `~0.4385` | `~0.7317` | `~0.1919` | `~1.8767` |
| eta=.5 | Thresholded All-Sample Persistent Insert | `~0.8379` | `~0.3514` | `~0.7126` | `~0.0062` | `~1.8582` |
| eta=.5 | Thresholded Unreliable-only Persistent Insert | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| eta=.5 | Unreliable-only Persistent SoftMix | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| eta=.5 | Unreliable-only Persistent Replace | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |

## 2026-05-18 Docx Family Extension

The reviewer-style ablation plan in
`C:/Users/BILIBILI/Downloads/消融实验意见.docx` expands the source
write-back family from several fixed stress tests to a full operator sweep on:

```text
E03: CIFAR-100, q/pr=0.05, eta/nr=0.3
E05: CIFAR-100, q/pr=0.05, eta/nr=0.5
seeds: 1, 2, 3
epochs: 500
hardware: c201 two RTX 5080 GPUs
```

The official family uses the c201 script:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp_family_20260518.py
```

and stores results under:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_writeback_docx_family_20260518
```

The launcher is:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/launch_c201_docx_family_20260518.sh
```

Two waiting queues were started:

| Session | GPU | Setting | Waits for |
|---|---:|---|---|
| `c201_docx_family_E03_gpu0_20260518` | 0 | `E03`, eta=0.3 | `c201_eta03_m6_20260518` |
| `c201_docx_family_E05_gpu1_20260518` | 1 | `E05`, eta=0.5 | `c201_missing_source_modes_20260518` |

This full family differs from the earlier M0--M6 stress tests in one important
way: all source-update operators use `--source_update_evidence p2`, so the
write-back evidence is the final second-pass NSE evidence distribution. This
keeps the extractor fixed and tests only whether carrying extracted evidence
across epochs helps or contaminates the working source.

Per setting, the family contains:

| Group | Points | Flags |
|---|---:|---|
| A reset | 1 | `--source_update_mode none` |
| B soft evidence write-back | 6 | `--source_update_mode soft_evidence --source_update_alpha {0.01,0.05,0.1,0.3,0.5,1.0}` |
| C all-sample hard insert | 5 | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold {0.55,0.65,0.75,0.85,0.95}` |
| D unreliable-only hard insert | 5 | `--source_update_mode hard_insert --source_update_scope unreliable_highconf --source_update_threshold {0.55,0.65,0.75,0.85,0.95}` |
| E add-remove refinement proxy | 2 | `--source_update_mode add_remove --source_update_scope all --source_update_threshold 0.85 --source_remove_threshold {0.05,0.10}` |
| F remove-only purification proxy | 2 | `--source_update_mode hard_remove --source_update_scope all --source_remove_threshold {0.05,0.10}` |
| G Topology-DAES internals | 3 | `--topology_rel_gamma 0.0`, `--daes_entropy_coeff 0.0`, or both |

Reference-library verification for these operators is recorded in
`docs/NSE_5080_experiment_result_storage_reproduction_20260518.md`.

## 2026-05-23 V2 Persistent-State Proxy Results

Live c201 check on `2026-05-23 22:02 CST` showed both RTX 5080 GPUs idle and
no active tmux experiment sessions. The v2 five-proxy suite completed all three
seeds for every condition under:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_v2_20260521
```

All v2 rows use CIFAR-100, `pr=0.05`, `nr=0.3`, 500 epochs, seeds `1 2 3`,
ResNet-18, the same NSE extractor, `--source_update_evidence p2`,
`--max_w_model 0.5`, and `--model_warmup_epochs 20`.

### V2 Accuracy Summary

The matching reset baseline remains the verified audit run:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_unified_ablation_audit/CIFAR100/pr005_nr03/A0_full/e500_seed123/master_log.txt
```

| Condition | Mechanism | Best Acc | Final Acc | Delta Final vs NSE |
|---|---|---:|---:|---:|
| NSE-Reset baseline | epoch-wise source restoration | `79.41 ± 0.23` | `79.22 ± 0.14` | `0.00` |
| `PALS_SARI_V2_PSS` | conservative all-sample top-1 augmentation, `0.95 -> 0.85` | `79.00 ± 0.23` | `78.84 ± 0.32` | `-0.38` |
| `UPLLRS_V2_PSS` | persistent non-candidate promotion from NSE-estimated noisy pool | `78.68 ± 0.10` | `78.51 ± 0.12` | `-0.71` |
| `FREDIS_V2_PSS` | score-gap refinement plus disambiguation, conservative thresholds | `78.42 ± 0.17` | `78.10 ± 0.20` | `-1.12` |
| `IRNet_V2_PSS` | candidate/non-candidate score-gap correction | `78.26 ± 0.34` | `78.04 ± 0.27` | `-1.18` |
| `PiCOPlus_V2_PSS` | persistent soft pseudo-target carry-over | `59.90 ± 0.67` | `59.68 ± 0.65` | `-19.54` |

### V2 Final-Epoch Extraction Diagnostics

| Condition | Prec(A) | Cov(A) | NRR-A | Count(A) |
|---|---:|---:|---:|---:|
| `FREDIS_V2_PSS` | `0.9163` | `0.8730` | `0.6403` | `43649` |
| `IRNet_V2_PSS` | `0.9168` | `0.8729` | `0.6413` | `43644` |
| `PALS_SARI_V2_PSS` | `0.9247` | `0.8807` | `0.6467` | `44036` |
| `UPLLRS_V2_PSS` | `0.9211` | `0.8840` | `0.6471` | `44201` |
| `PiCOPlus_V2_PSS` | `0.6062` | `0.9745` | `0.5892` | `48726` |

### V2 Persistent-State Diagnostics

| Condition | WriteCov / PromoteCov | WrongWrite / CumWrongPromote | SrcRec-N | CleanDamage | SourceDrift | SoftContam |
|---|---:|---:|---:|---:|---:|---:|
| `FREDIS_V2_PSS` | `0.00004` | `0.0000` | `0.1818` | `0.0117` | `0.3936` | `0.6791` |
| `IRNet_V2_PSS` | `0.000007` | `0.0000` | `0.1727` | `0.0000` | `0.0645` | `0.8357` |
| `PALS_SARI_V2_PSS` | `0.000007` | `0.0000` | `0.1053` | `0.0000` | `0.0365` | `0.8453` |
| `UPLLRS_V2_PSS` | `0.0293` promote coverage | `0.3335` cumulative wrong promotion | `0.0000` | `0.0000` | `0.0000` | `0.8607` |
| `PiCOPlus_V2_PSS` | `1.0000` soft target update coverage | `0.4017` wrong top-1 state | `1.0000` | `0.0000` | `1.7873` | `0.8477` |

### V2 Interpretation

The conservative v2 proxies are much less destructive than the older aggressive
write-back stress tests, which is expected because they were designed to follow
the prior papers more cautiously. Even so, none improves over the reset NSE
baseline.

The hard candidate-state proxies show small but consistent accuracy loss. The
most cautious PALS/SARI-style augmentation is closest to reset NSE, but still
drops `0.38` final accuracy points. FREDIS-style add/remove and IRNet-style
correction drop about `1.1` points. This supports the paper's narrower claim:
when the extractor is fixed, persisting candidate-correction state is not needed
for strong NPLL performance.

UPLLRS-style persistent sample promotion also underperforms reset NSE. It does
not mutate candidate membership (`SourceDrift=0`), but it accumulates a
promotion state covering about `2.93%` of samples by the final epoch, with about
`33.35%` cumulative wrong promotions. This is useful evidence that the risk is
not limited to candidate-set rewrite; persistent sample-state carry-over can
also add errors.

The PiCO+-style soft pseudo-target carry-over is the clear failure case. It
updates every sample's soft state (`WriteCov=1.0`), creates very large drift
(`SourceDrift=1.7873`), and collapses final accuracy to `59.68`. This is the
strongest support for separating epoch-local evidence from persistent soft
supervision state.
