# Bayes-Unified CIFAR-100 q=0.05 eta=0.3 Ablation Analysis

Baseline final accuracy: **79.17±0.15** over 2 seeds.
Baseline source: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed23`

All ablation values below use final-epoch accuracy. A10 is now summarized from three completed seeds.

| ID | Ablation | Final Acc. | Delta | Seeds | Log |
|---|---|---:|---:|---:|---|
| A01 | K1/K2 topology-DAES -> exp | 76.50±0.20 | -2.67 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A01_ablate_topology_daes_use_exp_both/master_log.txt` |
| A02 | K2 topology-DAES -> exp | 78.56±0.05 | -0.61 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A02_ablate_stage2_topology_daes_use_exp_s2/master_log.txt` |
| A03 | K1 topology-DAES -> exp | 76.51±0.12 | -2.66 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A03_ablate_stage1_topology_daes_use_exp_s1/master_log.txt` |
| A04 | adaptive r_i -> 0.5 | 79.14±0.11 | -0.03 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A04_ablate_adaptive_ri_use_uniform_ri05/master_log.txt` |
| A05 | w/o candidate prior | 78.17±0.06 | -1.00 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A05_ablate_candidate_prior_unmask_model_evidence/master_log.txt` |
| A06 | model cap 0.0 | 78.99±0.04 | -0.18 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A06_ablate_model_view_maxw0_knn_only/master_log.txt` |
| A07 | model cap 1.0 | 79.13±0.13 | -0.04 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A07_stress_model_view_maxw1_full_model_influence/master_log.txt` |
| A08 | detect salvage only | 78.67±0.29 | -0.50 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A08_ablate_salvage_training_log_only/master_log.txt` |
| A09 | history_len 5 | 78.97±0.09 | -0.20 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A09_ablate_queue_stability_short_history5/master_log.txt` |
| A10 | history_len 30 | 79.08±0.17 | -0.09 | 3 | `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A10_ablate_queue_stability_long_history30/master_log.txt` |

## Main Takeaways

1. **Topology-DAES is mainly useful in the first pass.** Replacing both passes with the exponential kernel drops final accuracy to 76.50, and replacing only the first pass gives a similar 76.51. Keeping topology-DAES in pass 1 but using exp in pass 2 is much better at 78.56. This suggests the geometry-aware vote is most critical before Bayesian fusion, where it decides the quality of the internal belief that later stages inherit.
2. **The current candidate-prior mask is beneficial at eta=0.3.** Removing the omega constraint gives 78.17, about 1.00 point below the baseline. At this noise level, the prior still suppresses enough false candidates to be useful, even though it may become risky when the true class is absent more often.
3. **Model-view fusion is robust but not the sole source of gain.** KNN-only evidence reaches 78.99 and full model influence reaches 79.13, both close to the baseline. The two-view pathway appears to stabilize the method, but the exact cap is less sensitive than the topology kernel or candidate prior.
4. **The adaptive r_i formula needs reinterpretation.** Forcing r_i=0.5 reaches 79.14, slightly above the current baseline. This does not mean the idea of reliability gating is wrong; it means the present confidence-ratio implementation may be overreacting to confidence calibration or early model/KNN imbalance. This is a strong clue for a revised r_i design.
5. **Queue salvage helps, but the effect is moderate.** Disabling promotion of salvaged samples gives 78.67, about 0.50 below baseline. Short history 5 gives 78.97 and long history 30 gives 79.08 over two seeds, so the queue is useful and not extremely sensitive to the horizon around the tested range.

## Paper-facing interpretation

The strongest supported story is that Bayes-Unified's robustness comes from geometry-aware first-pass topology voting plus a constrained Bayesian evidence path. The candidate prior remains useful at eta=0.3, while the model-view and queue modules provide stabilizing secondary gains. The r_i ablation suggests that the current adaptive gate should be described carefully: sample-wise reliability is an important design axis, but this implementation's confidence-ratio gate is not yet strictly better than a constant balanced gate under this condition.

## Command provenance

### A01 K1/K2 topology-DAES -> exp

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A01_ablate_topology_daes_use_exp_both/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A01_ablate_topology_daes_use_exp_both/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A01_ablate_topology_daes_use_exp_both/master_log.txt`

### A02 K2 topology-DAES -> exp

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A02_ablate_stage2_topology_daes_use_exp_s2/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A02_ablate_stage2_topology_daes_use_exp_s2/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A02_ablate_stage2_topology_daes_use_exp_s2/master_log.txt`

### A03 K1 topology-DAES -> exp

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A03_ablate_stage1_topology_daes_use_exp_s1/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A03_ablate_stage1_topology_daes_use_exp_s1/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A03_ablate_stage1_topology_daes_use_exp_s1/master_log.txt`

### A04 adaptive r_i -> 0.5

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A04_ablate_adaptive_ri_use_uniform_ri05/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A04_ablate_adaptive_ri_use_uniform_ri05/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A04_ablate_adaptive_ri_use_uniform_ri05/master_log.txt`

### A05 w/o candidate prior

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A05_ablate_candidate_prior_unmask_model_evidence/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A05_ablate_candidate_prior_unmask_model_evidence/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A05_ablate_candidate_prior_unmask_model_evidence/master_log.txt`

### A06 model cap 0.0

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A06_ablate_model_view_maxw0_knn_only/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A06_ablate_model_view_maxw0_knn_only/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A06_ablate_model_view_maxw0_knn_only/master_log.txt`

### A07 model cap 1.0

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A07_stress_model_view_maxw1_full_model_influence/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A07_stress_model_view_maxw1_full_model_influence/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A07_stress_model_view_maxw1_full_model_influence/master_log.txt`

### A08 detect salvage only

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A08_ablate_salvage_training_log_only/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A08_ablate_salvage_training_log_only/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A08_ablate_salvage_training_log_only/master_log.txt`

### A09 history_len 5

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A09_ablate_queue_stability_short_history5/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A09_ablate_queue_stability_short_history5/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A09_ablate_queue_stability_short_history5/master_log.txt`

### A10 history_len 30

- Command file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A10_ablate_queue_stability_long_history30/command.txt`
- Alignment file: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A10_ablate_queue_stability_long_history30/baseline_alignment.txt`
- Master log: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03/A10_ablate_queue_stability_long_history30/master_log.txt`
