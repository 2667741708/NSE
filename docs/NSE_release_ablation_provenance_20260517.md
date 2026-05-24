# NSE 发布版脚本与消融实验溯源审查文档

Last updated: 2026-05-17, Asia/Shanghai

## 1. 审查目的

本文档用于投稿前审查 NSE 论文中的主消融、source-restoration 诊断实验、发布版脚本和运行命令是否一致。目标是避免把旧脚本、无效 flag、未完成结果或不等价设置写进论文。

核心原则：

- 论文中的 Table 8、Table 9、Table 10 必须能追溯到脚本、命令、flag、日志证据和 seed 完成状态。
- GitHub 发布版脚本必须保留 reset/source-writeback 的可复现实验入口。
- 未完成的 SoftMix、HardRemove/AddRemove、clean/noisy 分组诊断不能以正式结果进入主文。
- 旧版无效 `E5_no_salvage_training` 和 `E8_no_reliable_mixup` 不能用于论文。

## 2. 本地论文改动位置

| 论文内容 | 本地代码位置 | 审查点 |
|---|---|---|
| 自定义 `algorithm` 浮动体 | [main.tex:L20-L23](../main.tex#L20-L23) | 不依赖本地损坏的 `algorithm.sty` / `algorithmic.sty`。 |
| Algorithm 1: epoch-wise source restoration | [sec/3_method.tex:L51-L72](../sec/3_method.tex#L51-L72) | 明确每个 epoch restore、extract、select、train、discard。 |
| Source-contamination 形式化定义 | [sec/3_method.tex:L154](../sec/3_method.tex#L154) | 用 persistent source write-back 区分 NSE。 |
| 理论置信参数 | [sec/3_method.tex:L492](../sec/3_method.tex#L492) | 已改为 `delta_conf`，避免与 selection quantile `delta` 混淆。 |
| Reset-vs-writeback 主表 | [sec/4_experiment.tex:L304](../sec/4_experiment.tex#L304) | Table 9 包含 M0/M1/M2/M3；M4 SoftMix 尚未进主文。 |
| Source-state 诊断表 | [sec/4_experiment.tex:L321-L322](../sec/4_experiment.tex#L321-L322) | Table 10 对应 Write/Wrong/Rec-N/Damage/Drift。 |

## 3. GitHub 发布版脚本清单

| 用途 | 本地路径 | 备注 |
|---|---|---|
| Windows 发布版 source-writeback 训练脚本 | [experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py:L510-L520](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L510-L520) | 当前支持 `none`, `hard_insert`, `mix`, `replace`。 |
| Windows smoke launcher | [experiments/nse_mvp_source_writeback_20260515_windows/run_smoke_e3_windows.ps1:L1-L24](../experiments/nse_mvp_source_writeback_20260515_windows/run_smoke_e3_windows.ps1#L1-L24) | 用于发布前快速验证命令入口。 |
| 主消融计划文档 | [docs/bayes_unified_ablation_plan.md:L76-L87](bayes_unified_ablation_plan.md#L76-L87) | 记录 A01-A08 设计。 |
| 主消融审计文档 | [docs/bayes_unified_ablation_audit.md:L35-L43](bayes_unified_ablation_audit.md#L35-L43) | 记录 audit flag 含义。 |
| Source-writeback 消融文档 | [docs/nse_source_writeback_ablation.md:L1-L40](nse_source_writeback_ablation.md#L1-L40) | 记录 M0-M6、当前状态和诊断指标。 |

远端可信脚本路径：

```text
c201:
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp.py

d437:
/home/d437/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp.py
```

## 4. Table 8 主消融设置映射

统一设置：CIFAR-100, `q=0.05`, `eta=0.3`, ResNet-18, 500 epochs, seeds 1/2/3, `k_val=15`, `delta=0.25`, `history_len=15`, `max_w_model=0.5`, `sim_mode_1=topology_daes`, `sim_mode_2=topology_daes`，除非表中另行修改。

可信入口脚本：

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py
```

| Table 8 行 | 论文名 | 必须改变的设置 | 论文数值 |
|---|---|---|---|
| A0 | Full NSE | 无额外 ablation flag | `79.22 +/- 0.14` |
| A1 | no Topology-DAES, both stages | `--sim_mode_1 exp --sim_mode_2 exp` | `76.50 +/- 0.20` |
| A2 | no Topology-DAES in stage 2 | `--sim_mode_1 topology_daes --sim_mode_2 exp` | `78.56 +/- 0.05` |
| A3 | no Topology-DAES in stage 1 | `--sim_mode_1 exp --sim_mode_2 topology_daes` | `76.51 +/- 0.12` |
| A4 | uniform reliability ratio | `--ablate_uniform_ri` | `79.14 +/- 0.11` |
| A5 | no candidate-prior projection | `--ablate_no_candidate_prior` | `78.17 +/- 0.06` |
| A6 | model evidence cap 0.0 | `--max_w_model 0.0` | `78.99 +/- 0.04` |
| A7 | model evidence cap 1.0 | `--max_w_model 1.0` | `79.13 +/- 0.13` |
| A8 | no salvage promotion | `--ablate_no_salvage_training` | `78.67 +/- 0.29` |

审查要点：

- A8 必须使用 `--ablate_no_salvage_training`，不得使用旧无效 `--disable_salvage_training`。
- 如需检查日志，必须看到 `[AblationCheck]`，并确认 no-salvage 情况下 `detected_salvage > 0` 但 `promoted_salvage=0`。
- A5 只去掉 model evidence branch 的 candidate-prior mask，不等价于移除最终 candidate-set admission logic。

## 5. Table 9/10 Source-Restoration 设置映射

统一设置：CIFAR-100, `q=0.05`, `eta=0.3`, ResNet-18, 500 epochs, seeds 1/2/3, `k_val=15`, `delta=0.25`, `history_len=15`, `max_w_model=0.5`, active-set training 和 Topology-DAES 保持一致。只改变跨 epoch 的 source update rule。

远端 d437 当前可信结果根目录：

```text
/home/d437/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_mvp_source_writeback_20260514/main_table_e500
```

远端 d437 可信脚本：

```text
/home/d437/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp.py
```

| 方法 | 命令差异 | 当前状态 | Table 9/10 使用 |
|---|---|---|---|
| M0 `NSE-Reset` | `--source_update_mode none --source_update_scope none` | d437 eta=.3 3/3 完成 | 已进入主文。 |
| M1 `Aggressive Top-1 Insert` | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_schedule pals_linear` | d437 eta=.3 3/3 完成 | 已进入主文。 |
| M2 `Thresholded All-Sample Insert` | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold 0.65` | d437 eta=.3 3/3 完成 | 已进入主文。 |
| M3 `Thresholded Unreliable-Only Insert` | `--source_update_mode hard_insert --source_update_scope unreliable_highconf --source_update_threshold 0.65` | d437 eta=.3 3/3 完成 | 已进入主文。 |
| M4 `Unreliable-only Persistent SoftMix` | `--source_update_mode mix --source_update_scope unreliable_highconf --source_update_threshold 0.65 --source_update_alpha 0.3` | d437 eta=.3 正在运行 | 等三种子完成后再进入主文或 appendix。 |
| M5 `Unreliable-only Persistent Replace` | `--source_update_mode replace --source_update_scope unreliable_highconf --source_update_threshold 0.65` | 未进入当前主文 | 适合 appendix 风险诊断。 |
| M6 `Reset NSE without Salvage` | `--source_update_mode none --source_update_scope none --ablate_no_salvage_training` | 未进入当前主文 | 可与 Table 8 A8 交叉验证。 |

## 6. d437 eta=.3 当前完成证据

| 方法 | Best Acc | Final Acc | Prec(A) | Cov(A) | NRR-A | WriteCov | WrongWrite | SrcRec-N | Damage | Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 NSE-Reset | `79.55 +/- 0.04` | `79.38 +/- 0.16` | `0.931509` | `0.887133` | `0.659990` | `0.000000` | `nan` | `0.000000` | `0.000000` | `0.000000` |
| M1 Aggressive Top-1 Insert | `57.50 +/- 0.81` | `57.34 +/- 0.87` | `0.584879` | `0.968327` | `0.566430` | `0.992180` | `0.423160` | `0.754033` | `0.180363` | `1.828747` |
| M2 Thresholded All-Sample Insert | `62.94 +/- 0.55` | `62.68 +/- 0.59` | `0.651713` | `0.960580` | `0.618986` | `0.841873` | `0.326512` | `0.739527` | `0.006185` | `1.804345` |
| M3 Thresholded Unreliable-Only Insert | `61.52 +/- 0.31` | `61.32 +/- 0.19` | `0.644078` | `0.926013` | `0.593653` | `0.021507` | `0.603933` | `0.691725` | `0.000048` | `1.301506` |

解释：

- M3 证明 selective/unreliable-only write-back 的覆盖率很低，但写入标签的错误率仍然高，且最终 drift 仍明显非零。
- M4 SoftMix 是 reviewer 建议的关键剩余变体，当前不能写入主文结果表。
- 当前脚本不支持 HardRemove/AddRemove。若论文要覆盖 deletion/refinement 型 carry-over，必须先实现并运行，或者把 claim 收窄为 insertion/mix/replacement stress tests。

## 7. 当前远端运行状态

截至 2026-05-17 检查：

```text
d437:
eta03_M4_MixWB03U is running.
No epoch-500 metrics yet.

c201:
eta05_M5_ReplaceWBU is running on GPU0.
eta05_M6_NSE_NoSalvage is running on GPU1.
eta05 M0-M4 have mostly completed seed logs, but final merge/summary still needs a clean parser pass.
```

状态检查命令：

```bash
ssh d437 "ps -eo pid,etimes,cmd | grep -E 'eta03_M4|run_task_queue|train_nse_source_mvp' | grep -v grep"
ssh c201 "ps -eo pid,etimes,cmd | grep -E 'train_nse_source_mvp|eta05|run_task_queue' | grep -v grep"
```

## 8. 论文中不能混用的结果

| 风险项 | 处理规则 |
|---|---|
| Table 8 A0 `79.22 +/- 0.14` 与 Table 9 M0 `79.38 +/- 0.16` | 两者是同一 CIFAR-100 设置下的独立运行/诊断 harness；Table 9 caption 已说明差异来源。 |
| SoftMix M4 | 未完成三种子前不能写入主文正式结果。 |
| HardRemove/AddRemove | 当前发布脚本未实现；不能声称已实证覆盖。 |
| old E5/E8 | 旧脚本解析了 flag 但未改变真实训练路径；不得用于论文。 |
| `PALS-style WB` 命名 | 主文应避免用作正式方法名，改称 `Aggressive Top-1 Insert`。 |

## 8.1 Baseline 与众包设置来源

当前论文对比数据集限定为 CIFAR-10、CIFAR-100、CIFAR-100H、Plankton、Treeversity、Benthic。

Baseline 数值来源规则：

- 论文表格中的 baseline 数字来自 PALS 论文及其公开 comparison tables。
- 原始 baseline 论文只用于标识方法和支撑方法机制/相关工作讨论，不用于声明实验数值来源。
- PALS comparison 中不存在的 baseline 条目保持 unreported，不写成来自对应原论文。

Crowdsourced datasets 设置：

- Treeversity、Benthic、Plankton 使用与 PALS 报告设置对齐的 ResNet-50 架构、ImageNet 预训练初始化权重和 100 epochs。
- 由于 PALS 未完全公开所有 crowdsourced fold 条件，论文固定 fold 2 为 test split，folds 1、3、4、5 为 training folds。

## 9. Clean/Noisy 分组诊断缺口

当前远端 d437/c201 正在运行的脚本只记录整体 `Prec(A)`, `Cov(A)`, `NRR-A`, `Clean0`, `Noisy0`。本地 GitHub 发布版脚本已经补充 `SubsetAudit` 日志，位置为 [train_nse_source_mvp.py:L1963-L2002](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L1963-L2002)。该改动只增加日志，不改变训练路径。

新增日志行格式：

```text
[SubsetAudit] epoch_0500 subset=clean ActiveCov=... ActivePrec=... PseudoAcc=...
[SubsetAudit] epoch_0500 subset=noisy ActiveCov=... ActivePrec=... PseudoAcc=...
```

使用要求：

- clean subset: `clean0 = (y_i in Y_i^0)`。
- noisy subset: `noisy0 = (y_i notin Y_i^0)`。
- active coverage: `mean(active_mask | subset)`。
- active precision / pseudo-label accuracy: `mean(active_label == clean_label | active_mask & subset)`。
- 该表优先只跑 M0 NSE-Reset；用于回应 true-label-absent 样本是否被 active supervision 恢复。
- 如果要用现有远端结果补表，必须先同步该发布版改动并重跑相应诊断；不能从旧远端日志反推 clean/noisy precision。

## 10. 投稿前执行清单

| 优先级 | 任务 | 完成标准 |
|---|---|---|
| P0 | 等待 d437 M4 SoftMix eta=.3 三种子完成 | 填入 Table 9/10 或 appendix；日志含 `[MVPSource] epoch=500` 和 `[MVPExtraction] epoch_0500`。 |
| P0 | 补 clean/noisy 分组诊断 | 使用带 `[SubsetAudit]` 的发布版脚本，至少跑 M0 NSE-Reset on CIFAR-100 `q=0.05, eta=0.3`。 |
| P1 | 决定 HardRemove/AddRemove | 若实现则跑；若不实现，则论文只声称 insertion/mix/replacement stress tests。 |
| P1 | 保持 Table 8/9 数值来源说明 | 不把不同 harness 的数值当成同一 run。 |
| P1 | 编译论文并检查 warning | `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`；检查 undefined refs/citations/overfull。 |
| P2 | GitHub release 打包 | 发布本地脚本、命令模板、README、本文档。 |
