# Paper Result Reproduction Checklist

This checklist maps every NSE-owned result currently shown in `main.pdf` to the
streamlined reproducibility package. Third-party baseline rows are literature
provenance rows and are not rerun by NSE scripts.

## Active Scripts

| Result family | Script | Command template |
|---|---|---|
| Main NSE tables | [bayes_unified_main_20260523.py](../code/main/bayes_unified_main_20260523.py#L1) | [reproduce_main_results_c201.sh](../commands/reproduce_main_results_c201.sh#L1) |
| Component ablations A0--A8 | [train_nse_component_ablation_20260523.py](../code/component_ablation/train_nse_component_ablation_20260523.py#L1) | [reproduce_component_ablation_c201.sh](../commands/reproduce_component_ablation_c201.sh#L1) |
| Persistent supervision state proxies | [train_nse_persistent_state_proxies_20260523.py](../code/persistent_state/train_nse_persistent_state_proxies_20260523.py#L1) | [reproduce_persistent_state_v2_c201.sh](../commands/reproduce_persistent_state_v2_c201.sh#L1) |
| Representative single-seed validation | same active scripts | [run_single_seed_validation_c201_gpu0.sh](../commands/run_single_seed_validation_c201_gpu0.sh#L1) and [run_single_seed_validation_c201_gpu1.sh](../commands/run_single_seed_validation_c201_gpu1.sh#L1) |

## One-To-One Result Matrix

Use [paper_result_reproduction_matrix_20260523.csv](paper_result_reproduction_matrix_20260523.csv#L1) as
the row-level index. It records the paper table, row id, dataset, setting,
reported value, script family, and command group for each result.

## Main Synthetic CIFAR Rows

Rerun with `bayes_unified_main_20260523.py`, ResNet-18, 500 epochs, seeds
`1 2 3` for the paper table. For the validation pass, use seed `1`.

| Group | Rows | Command changes |
|---|---|---|
| CIFAR-10 grid | `q in {0.1,0.3,0.5}`, `eta in {0.1,0.2,0.3}` | vary `--dataset CIFAR10 --pr <q> --nr <eta>` |
| CIFAR-100 grid | `q in {0.01,0.03,0.05}`, `eta in {0.1,0.2,0.3}` | vary `--dataset CIFAR100 --pr <q> --nr <eta>` |
| Extreme noise | CIFAR-100 `q=0.05`, `eta in {0.4,0.5}` | same main script, vary `--nr` |
| No-noise PLL | CIFAR-100 `eta=0.0`, `q in {0.01,0.05,0.1}` | same main script, set `--nr 0.0` |

## CIFAR-100H And Crowdsourced Rows

These use the same main script but different dataset loaders and training
duration. Before rerunning the full matrix, verify that the data roots exist on
c201:

| Group | Rows | Command changes |
|---|---|---|
| CIFAR-100H | `q=0.5, eta=0.2` | `--dataset CIFAR100H --train_root ./data --pr 0.5 --nr 0.2` |
| Treeversity | `LPI=10`, `LPI=3` | `--dataset Treeversity --train_root ./Treeversity --network R50 --epochs 100 --batch_size 32 --slice 2 --lpi <3-or-10>` |
| Benthic | `LPI=10`, `LPI=3` | `--dataset Benthic --train_root ./Benthic --network R50 --epochs 100 --batch_size 32 --slice 2 --lpi <3-or-10>` |
| Plankton | `LPI=10`, `LPI=3` | `--dataset Plankton --train_root ./Plankton --network R50 --epochs 100 --batch_size 32 --slice 2 --lpi <3-or-10>` |

## Component Ablations A0--A8

These rows use `train_nse_component_ablation_20260523.py` on CIFAR-100
`q=0.05, eta=0.3`.

| Row | Paper meaning | Flag delta |
|---|---|---|
| A0 | full NSE | no extra flag |
| A1 | replace both Topology-DAES passes with fixed exponential kernel | `--sim_mode_1 exp --sim_mode_2 exp` |
| A2 | replace only pass 2 with fixed exponential kernel | `--sim_mode_1 topology_daes --sim_mode_2 exp` |
| A3 | replace only pass 1 with fixed exponential kernel | `--sim_mode_1 exp --sim_mode_2 topology_daes` |
| A4 | constant `r_i=0.5` | `--ablate_uniform_ri` |
| A5 | remove candidate-prior projection from model evidence | `--ablate_no_candidate_prior` |
| A6 | KNN-only second-stage evidence | `--max_w_model 0.0` |
| A7 | full model-view cap after warm-up | `--max_w_model 1.0` |
| A8 | detect salvage candidates but do not promote them | `--disable_salvage_training` |

## Persistent Supervision State Proxies

These rows use `train_nse_persistent_state_proxies_20260523.py` on CIFAR-100
`q=0.05, eta=0.3`.

| Row | Proxy mechanism | Flag delta |
|---|---|---|
| NSE-Reset | native-source reset before extraction | `--source_update_mode none` |
| FREDIS-V2-PSS | persistent refinement plus disambiguation | `--source_update_mode fredis_move` with conservative add/remove thresholds |
| IRNet-V2-PSS | score-gap noisy-sample correction | `--source_update_mode irnet_correct --irnet_tau_boundary 0.0 --irnet_min_non_candidate_conf 0.85` |
| PALS/SARI-V2-PSS | high-confidence top-1 partial-label augmentation | `--source_update_mode pals_augment --source_update_schedule linear --source_update_threshold_start 0.95 --source_update_threshold_end 0.85` |
| UPLLRS-V2-PSS | persistent reliable promotion state | `--persistent_promotion_mode hard --promotion_scope nse_estimated_noise_highconf --promotion_threshold 0.95` |
| PiCO+-V2-PSS | persistent soft pseudo-target state | `--source_update_mode pico_soft_target --source_update_alpha 0.1` |

## Current Single-Seed Validation Pass

The representative validation pass is intentionally smaller than the paper
tables. It reruns:

- GPU0: FREDIS-V2-PSS, IRNet-V2-PSS, PALS/SARI-V2-PSS.
- GPU1: UPLLRS-V2-PSS, PiCO+-V2-PSS, A0 full NSE, and the main CIFAR-100
  `q=0.05, eta=0.3` row.

Output root:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_single_seed_validation_20260523
```

Status: completed on 2026-05-24. The local parsed summary is
[single_seed_validation_summary_20260524.csv](../results/single_seed_validation_summary_20260524.csv#L1).

Acceptance checks:

- each launched row has a `master_log.txt`;
- the log contains `--- Run 1 Finished`;
- command flags match the rows above;
- single-seed values should be close to one member of the original three-seed
  distribution, not necessarily equal to the reported mean.

## Completed Representative Results

| Row | Single-seed best | Single-seed final | Paper final |
|---|---:|---:|---:|
| A0 full | 79.72 | 79.33 | 79.22 +/- 0.14 |
| Main CIFAR-100 `q=0.05, eta=0.3` | 79.72 | 79.33 | 79.22 +/- 0.14 |
| FREDIS-V2-PSS | 78.55 | 78.16 | 78.10 +/- 0.20 |
| IRNet-V2-PSS | 77.86 | 77.67 | 78.04 +/- 0.27 |
| PALS/SARI-V2-PSS | 79.17 | 79.17 | 78.84 +/- 0.32 |
| UPLLRS-V2-PSS | 78.82 | 78.69 | 78.51 +/- 0.12 |
| PiCO+-V2-PSS | 59.91 | 59.57 | 59.68 +/- 0.65 |
