# Bayes-Unified topology_daes 消融实验计划

## 固定目录

实验计划、消融日志和汇总结果统一放到：

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast
```

机制消融的可信入口应使用审计脚本副本，不覆盖原始基线脚本：

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py
```

旧脚本 `bayes_unified_unified_ablation.py` 中的 `E5_no_salvage_training`
和 `E8_no_reliable_mixup` 结果不进入论文，因为旧 flag 没有可靠改变对应训练路径。

统一命名规则：

```text
--out ./out_ultimate
--exp_name bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/<dataset>/<self_describing_ablation_name>
```

## 已有基线

以下完整基线已经有结果，本轮消融脚本不再重复运行：

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed23
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr001_nr03_maxw05
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr05_maxw05
```

主基线命令保留为参照，不执行：

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
  --exp_name bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed123 \
  --seeds 1 2 3 --cuda_dev 1
```

高噪声对比基线命令保留为参照，不执行：

```bash
python bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py \
  --dataset CIFAR100 --train_root ./data --lpi 10 \
  --pr 0.05 --nr 0.5 \
  --network R18 --epochs 500 --batch_size 256 \
  --lr 0.1 --wd 0.001 --momentum 0.9 \
  --lr_scheduler cosine \
  --mixup_alpha 1.0 --lsr 0.0 --consistency_weight 1.0 --ema_alpha 0.999 \
  --k_val 15 --delta 0.25 --history_len 15 --consensus_power 2.0 \
  --sim_mode_1 topology_daes --sim_mode_2 topology_daes --knn_heads 1 \
  --max_w_model 0.5 \
  --out ./out_ultimate \
  --exp_name bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr05_maxw05/refactored_e500_seed123 \
  --seeds 1 2 3 --cuda_dev 1
```

## 只运行消融实验

所有消融都严格继承基线训练超参：`CIFAR100/R18/500 epochs/batch 256/cosine/k=15/delta=0.25/history_len=15/consensus_power=2.0/max_w_model=0.5/seeds 1 2 3`，只改变表中列出的条件。

| ID | 结果名 | 消融条件 | 目的 |
|---|---|---|---|
| A01 | `A01_ablate_topology_daes_use_exp_both` | `--sim_mode_1 exp --sim_mode_2 exp` | 用传统 exp 核替代两阶段 topology_daes，检验几何自适应核是否抑制噪声扩散 |
| A02 | `A02_ablate_stage2_topology_daes_use_exp_s2` | `--sim_mode_1 topology_daes --sim_mode_2 exp` | 只去掉第二阶段 topology_daes |
| A03 | `A03_ablate_stage1_topology_daes_use_exp_s1` | `--sim_mode_1 exp --sim_mode_2 topology_daes` | 只去掉第一阶段 topology_daes |
| A04 | `A04_ablate_adaptive_ri_use_uniform_ri05` | `--ablate_uniform_ri` | 固定 `r_i=0.5`，去掉逐样本自适应门控 |
| A05 | `A05_ablate_candidate_prior_unmask_model_evidence` | `--ablate_no_candidate_prior` | 模型证据不再受候选先验 `omega` 约束 |
| A06 | `A06_ablate_model_view_maxw0_knn_only` | `--max_w_model 0.0` | 去掉模型视图，只保留 KNN/几何传播 |
| A07 | `A07_stress_model_view_maxw1_full_model_influence` | `--max_w_model 1.0` | 加强模型视图，观察单视图拟合/确认偏差 |
| A08 | `A08_ablate_salvage_training_log_only` | `--ablate_no_salvage_training` | 队列只记录打捞，不把打捞样本放回训练池 |
| A09 | `A09_ablate_queue_stability_short_history5` | `--history_len 5` | 缩短队列，看打捞稳定性下降 |
| A10 | `A10_ablate_queue_stability_long_history30` | `--history_len 30` | 拉长队列，看是否过于保守 |

高噪声 `pr=0.05,nr=0.5` 只跑关键对比，不跑已有完整基线：

| ID | 结果名 | 消融条件 |
|---|---|---|
| N01 | `N01_nr05_ablate_topology_daes_use_exp_both` | `--sim_mode_1 exp --sim_mode_2 exp` |
| N04 | `N04_nr05_ablate_adaptive_ri_use_uniform_ri05` | `--ablate_uniform_ri` |
| N05 | `N05_nr05_ablate_candidate_prior_unmask_model_evidence` | `--ablate_no_candidate_prior` |
| N08 | `N08_nr05_ablate_salvage_training_log_only` | `--ablate_no_salvage_training` |

## 双 GPU 队列

GPU 0：

```text
A01, A02, A03, A04, N01
```

GPU 1：

```text
A05, A06, A07, A08, A09, A10, N04, N05, N08
```

启动命令：

```bash
cd /home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409
bash scripts/launch_bayes_unified_ablation_c201_dualgpu.sh
```

## 汇总

每组结果位于：

```text
out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/<dataset>/<ablation_name>/master_log.txt
```

生成汇总表：

```bash
cd /home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409
python3 scripts/collect_bayes_unified_ablation_results.py
```
