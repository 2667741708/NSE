# Bayes-Unified Ablation Script Audit

Audited file:

`/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py`

Deprecated file for these two conditions:

`/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/bayes_unified_unified_ablation.py`

## Actual algorithm path

The current ablation script implements one unified two-pass propagation path:

1. Stage-1 KNN propagation uses `--sim_mode_1` through `get_weight_matrix(...)`.
2. The model prediction view is warmed up by
   `w_model = min(max_w_model, epoch / model_warmup_epochs)`.
3. The adaptive reliability gate is computed as a confidence ratio between
   KNN evidence and model evidence. `--ablate_uniform_ri` forces `r_i = 0.5`.
4. Model evidence is constrained by the dataset prior `omega` unless
   `--ablate_no_candidate_prior` is set.
5. Stage-2 propagation input is
   `normalize(r_i * P_knn + (1 - r_i) * P_model_prior)`.
6. Stage-2 KNN propagation uses `--sim_mode_2`.
7. Tri-consensus state uses model, KNN, and prototype views to identify salvage
   candidates. `--ablate_no_salvage_training` still logs detected salvage
   samples but prevents promotion into the training pool.
8. Training combines supervised loss and a dynamic consistency branch:
   `loss_s + dynamic_consistency_weight * loss_c_self`.

## Effective ablation controls

These controls are active in the current script and are suitable for reporting:

- `--sim_mode_1`, `--sim_mode_2`: compare `topology_daes` against `exp`, `daes`,
  or stage-specific replacements.
- `--ablate_uniform_ri`: removes sample-wise adaptive reliability.
- `--ablate_no_candidate_prior`: removes the candidate-prior mask from model
  evidence.
- `--max_w_model`: tests KNN-only (`0.0`), capped model evidence (`0.5`), and
  full model influence (`1.0`).
- `--model_warmup_epochs`: tests sensitivity to the model-view warm-up schedule.
- `--ablate_no_salvage_training`: measures whether the queue-salvaged samples
  help after being detected.
- `--ablate_no_reliable_mixup`: bypasses reliable weak/strong MixUp and uses
  direct supervised active labels.
- `--history_len`: tests the temporal queue horizon.
- `--consistency_weight`, `--consensus_power`: test the dynamic consistency
  branch strength.
- topology/DAES parameters such as `--topology_rel_mode`, `--kl_self_mode`,
  `--daes_base_tau`, and `--daes_entropy_coeff` can be used for finer kernel
  sensitivity studies.

## Legacy flags that should not be reported as completed ablations

The following flags are parsed, but in the audited version they are only used
for experiment naming/grouping and do not change the active training path:

- `--no_reliable_mixup`
- `--disable_salvage_training`
- `--no_rebalance`
- `--no_softmatch`
- `--no_unreliable_mixup`
- `--no_unreliable_training`
- `--no_rectify`
- `--fix_dynamic_weight`

Because of this, the paper should not claim completed no-Mix-up, no-SoftMatch,
no-rebalance, or supervised-only ablations from these flags without an additional
code patch that makes the flags actually affect the computation.

## Recommended default comparison

Use the existing completed baseline as the reference and do not rerun it:

```bash
python bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py \
  --dataset CIFAR100 --train_root ./data --lpi 10 \
  --pr 0.05 --nr 0.3 \
  --network R18 --epochs 500 --batch_size 256 \
  --lr 0.1 --wd 0.001 --momentum 0.9 \
  --lr_scheduler cosine \
  --mixup_alpha 1.0 --lsr 0.0 --consistency_weight 1.0 --ema_alpha 0.999 \
  --k_val 15 --delta 0.25 --history_len 15 --consensus_power 2.0 \
  --sim_mode_1 topology_daes --sim_mode_2 topology_daes --knn_heads 1 \
  --max_w_model 0.5 \
  --out ./out_ultimate \
  --exp_name bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed23 \
  --seeds 2 3
```

Use the ablation output root:

`/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast`

For every ablation, keep `command.txt`, `baseline_alignment.txt`, seed-level
logs, and the final collected CSV together under that root.
