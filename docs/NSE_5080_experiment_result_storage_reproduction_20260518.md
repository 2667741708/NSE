# NSE 5080 双卡实验结果存储与复现说明

Last updated: 2026-05-18 14:05 CST.

## 1. 文档边界

本文档作为论文口径的 experiment provenance 文档，只记录 5080 双卡实验的结果存储位置、结果获取方式和复现入口。

正式论文和后续清稿中，在所有 paper-facing rows 均已在 c201 的 5080
双卡上验证或重跑后，硬件口径统一写为：

```text
All reported NSE ablations were run on two NVIDIA GeForce RTX 5080 GPUs.
```

非 5080 主机上的历史运行只能作为调度和调试参考，不纳入本文档的论文证据链，也不应写入论文、caption 或投稿补充材料的正式实验来源说明。

## 2. 当前 5080 机器状态

远端机器：

```text
ssh c201
hostname: c201-MS-7E06
GPU0: NVIDIA GeForce RTX 5080
GPU1: NVIDIA GeForce RTX 5080
```

2026-05-18 10:57 CST 检查结果：

```text
GPU0: 388 MiB, 0% utilization
GPU1: 18 MiB, 0% utilization
tmux: no active experiment session
process: no active train_nse_source_mvp / bayes_unified training process
```

结论：c201 当前没有正在等待或运行的 NSE 消融训练，两个 5080 GPU 均为空闲状态。

## 3. 远端项目与结果根目录

主项目根目录：

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409
```

整理后的实验中心：

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果
```

主结果根目录：

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate
```

主表与 real-world/crowdsourced 汇总索引在本地项目中记录为：

```text
docs/bayes_unified_collected_results_with_paths.csv
```

## 4. Table 8 主机制消融结果路径

实验设置：

```text
Dataset: CIFAR-100
q / pr: 0.05
eta / nr: 0.3
network: ResNet-18
epochs: 500
batch size: 256
k: 15
delta: 0.25
history_len default: 15
max_w_model default: 0.5
seeds: 1, 2, 3 unless noted
hardware provenance: c201 RTX 5080 GPUs only
```

Full NSE baseline result roots:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed1
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed23
```

A01-A10 ablation result root:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03
```

Each ablation directory contains:

```text
master_log.txt
command.txt
baseline_alignment.txt
seed_*/run.log
```

Current 5080-only final-epoch summary:

| ID | Condition | Final Acc. | Seeds | Evidence |
|---|---|---:|---:|---|
| A0 | Full NSE | 79.22±0.14 | 3 | seed1 plus seed2/3 logs |
| A01 | K1/K2 topology-DAES -> exp | 76.50±0.20 | 3 | `A01_ablate_topology_daes_use_exp_both/master_log.txt` |
| A02 | K2 topology-DAES -> exp | 78.56±0.05 | 3 | `A02_ablate_stage2_topology_daes_use_exp_s2/master_log.txt` |
| A03 | K1 topology-DAES -> exp | 76.51±0.12 | 3 | `A03_ablate_stage1_topology_daes_use_exp_s1/master_log.txt` |
| A04 | adaptive r_i -> 0.5 | 79.14±0.11 | 3 | `A04_ablate_adaptive_ri_use_uniform_ri05/master_log.txt` |
| A05 | w/o candidate-prior mask in model evidence | 78.17±0.06 | 3 | `A05_ablate_candidate_prior_unmask_model_evidence/master_log.txt` |
| A06 | model cap 0.0 | 78.99±0.04 | 3 | `A06_ablate_model_view_maxw0_knn_only/master_log.txt` |
| A07 | model cap 1.0 | 79.13±0.13 | 3 | `A07_stress_model_view_maxw1_full_model_influence/master_log.txt` |
| A08 | detect salvage only | 78.67±0.29 | 3 | `A08_ablate_salvage_training_log_only/master_log.txt` |
| A09 | history_len 5 | 78.97±0.09 | 3 | `A09_ablate_queue_stability_short_history5/master_log.txt` |
| A10 | history_len 30 | 79.08±0.17 | 3 | `A10_ablate_queue_stability_long_history30/master_log.txt` |

## 5. Source-restoration / write-back 诊断结果路径

5080-only source-writeback result root:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_mvp_source_writeback_20260514/main_table_e500
```

Executable script:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp.py
```

Launcher directory:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514
```

Available launchers:

```text
run_main_table_e500.sh
run_one_main_task.sh
run_smoke_e3.sh
run_sweeps_seed1_e500.sh
run_task_queue.sh
start_eta05_after_audit.sh
start_gpu1_eta05_after_audit.sh
```

Current 5080-only status for CIFAR-100 q=0.05, eta=0.5:

| ID | Condition | Seeds on c201 | Best Acc. | Final Acc. | Status |
|---|---|---:|---:|---:|---|
| M0 | NSE-Reset | 2, 3 | 76.56±0.41 | 76.51±0.46 | incomplete for 5080-only 3-seed table; seed 1 should be rerun on c201 |
| M1 | Aggressive Top-1 Persistent Insert | 1, 2, 3 | 56.00±0.46 | 55.88±0.48 | complete |
| M2 | Thresholded All-Sample Persistent Insert | 1, 2, 3 | 60.44±0.37 | 60.31±0.41 | complete |
| M3 | Thresholded Unreliable-only Persistent Insert | 1, 2, 3 | 59.11±0.32 | 58.98±0.30 | complete |
| M4 | Unreliable-only Persistent SoftMix alpha=0.3 | 1, 2, 3 | 60.01±0.36 | 59.90±0.36 | complete |
| M5 | Unreliable-only Persistent Replace | 1, 2, 3 | 60.56±0.34 | 60.42±0.26 | complete |
| M6 | Reset NSE without Salvage | 1, 2, 3 | 73.70±0.27 | 73.56±0.25 | complete |

Important constraint:

```text
The eta=0.3 write-back diagnostic values currently used in older drafts should
not be described as 5080-only evidence until they are rerun on c201.
```

## 6. 如何获取实验结果

For any condition:

1. Open its `master_log.txt`.
2. Read:

```text
Individual Best Accuracies
Individual Final Epoch Accuracies
Final Reported (Best Acc)
Final Reported (Final Epoch Acc)
```

3. Check the first `Base Settings` line for:

```text
dataset, pr, nr, epochs, seeds, cuda_dev, source_update_mode,
source_update_scope, ablation flags
```

4. For source-writeback diagnostics, inspect each `seed_*/run.log` for:

```text
[MVPSource]
[MVPExtraction]
[SubsetAudit]  # only available in newer local/release scripts unless rerun remotely
```

5. Do not merge runs from non-5080 hosts into a paper-facing 5080-only table.

Useful remote commands:

```bash
ssh c201
nvidia-smi
tmux ls
ps -eo pid,etimes,cmd | grep -E 'train_nse_source_mvp|bayes_unified|run_task_queue' | grep -v grep
```

List source-writeback logs:

```bash
BASE="/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_mvp_source_writeback_20260514/main_table_e500"
find "$BASE" -maxdepth 3 -type f \( -name "master_log.txt" -o -name "run.log" -o -name "command.txt" \) | sort
```

List mechanism-ablation logs:

```bash
BASE="/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03"
find "$BASE" -maxdepth 2 -type f \( -name "master_log.txt" -o -name "command.txt" -o -name "baseline_alignment.txt" \) | sort
```

## 7. 如何在 5080 上复现

### 7.1 主机制消融 A01-A10

Trusted script:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py
```

Launchers:

```bash
cd /home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit
bash launch_audit_e500_gpu0_wait_20260513.sh
bash launch_audit_e500_gpu1_20260513.sh
```

Before rerunning, verify that old invalid flags are not used:

```text
Do not use --disable_salvage_training.
Do not use --no_reliable_mixup as a no-effect metadata-only control.
Use the audited flags implemented in bayes_unified_unified_ablation_audit.py.
```

### 7.2 Source-restoration / write-back diagnostics

Launch directory:

```bash
cd /home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514
```

Smoke test:

```bash
bash run_smoke_e3.sh
```

Main eta=0.5 table:

```bash
bash start_eta05_after_audit.sh
bash start_gpu1_eta05_after_audit.sh
```

For a clean 5080-only rerun of the paper-facing diagnostic table, rerun every row on c201, including:

```text
CIFAR100 q=0.05 eta=0.3 M0-M6 seeds 1/2/3
CIFAR100 q=0.05 eta=0.5 M0-M6 seeds 1/2/3
```

The current c201 eta=0.5 M0 row still needs seed 1 on c201 before it is a complete 5080-only 3-seed result.

## 8. 投稿前使用规则

1. Main text should say the reported NSE ablations use two RTX 5080 GPUs only after all paper-facing rows are rerun or verified on c201.
2. Do not cite historical non-5080 runs as official evidence.
3. If a table mixes hosts, mark it as internal only and do not move it into the paper.
4. Every paper-facing number must be traceable to:

```text
remote path
script path
command or launcher
master_log.txt
seed_*/run.log
hardware provenance
```

## 9. 核心论点验证状态审查

Current audit time: 2026-05-18 14:05 CST.

### 9.1 可以验证的核心论点

基于当前论文、代码和已经完成的实验，论文核心论点已经可以被当前实验体系验证，但需要保持 claim 范围精确：

```text
NSE does not rely on cross-epoch persistent source write-back. It restores the
working prior from the native weak-supervision record before each epoch and
extracts epoch-local active supervision. Persistent model-induced source
updates introduce source drift and wrong-write accumulation in the tested
insert / soft-mix / replacement stress tests.
```

当前证据足以支持：

- `NSE-Reset` 在 source drift 为 0 的情况下保持高准确率。
- persistent insertion、soft mix、replacement 等跨 epoch 写回策略都会显著降低准确率。
- persistent update 虽然能提高 source-level noisy recovery，但同时带来 WrongWrite、CleanDamage 或 SourceDrift。
- 因此，source-level recovery 与 active-supervision recovery 必须分开讨论；NSE 的恢复发生在 active supervision，而不是写回原始 source。

当前证据不应过度支持：

- 不应宣称所有 possible persistent update 都必然有害。
- 不应宣称已经覆盖 hard-remove、add-remove、periodic-reset 或 PLRC-style top-k reconstruction 的完整空间。
- 不应把尚未在 5080 上重跑的 historical non-5080 rows 写成 5080-only evidence。

### 9.2 代码覆盖情况

当前 release / audit 代码已经支持以下 source-update families：

| Family | Implemented | Script flag |
|---|---:|---|
| Reset / no write-back | yes | `--source_update_mode none` |
| Hard insert | yes | `--source_update_mode hard_insert` |
| SoftMix alpha=0.3 | yes | `--source_update_mode mix --source_update_alpha 0.3` |
| Hard replacement | yes | `--source_update_mode replace` |
| Reset without salvage | yes | `--source_update_mode none --ablate_no_salvage_training` |
| Hard remove | yes, added 2026-05-18 | `--source_update_mode hard_remove` |
| Add-remove | yes, added 2026-05-18 | `--source_update_mode add_remove` |
| Periodic reset K | yes, added 2026-05-18 | `--source_reset_interval K` |
| Top-k reconstruction | yes, added 2026-05-18 | `--source_update_mode topk_reconstruct` |

Conclusion: 当前代码现在可以启动 reset、insert、soft-mix、replace、hard-remove、add-remove、periodic-reset 和 top-k reconstruction 诊断。最终论文是否使用这些新增 families，仍取决于 c201 运行是否完成并通过日志审计。

Update at 2026-05-18 15:10 CST:

The local release script and c201 extended script now implement these missing
families for a one-pass diagnostic run:

| Family | Newly implemented | c201 exp |
|---|---:|---|
| Hard remove | yes | `eta03_M7_HardRemove` |
| Add-remove | yes | `eta03_M8_AddRemove` |
| Periodic reset K=10 | yes | `eta03_M9_PeriodicReset10` |
| Top-k reconstruction | yes | `eta03_M10_TopKReconstruct` |

The extended remote script is:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp_extended_20260518.py
```

The corresponding local release script is:

```text
experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py
```

## 10. 当前已有实际结果

### 10.1 d437 eta=0.3 historical/internal results

These rows are useful as current evidence while writing, but should be rerun on c201 before being described as 5080-only results.

Setting:

```text
Dataset: CIFAR-100
q / pr: 0.05
eta / nr: 0.3
epochs: 500
seeds: 1, 2, 3
```

| ID | Condition | Finished | Best Acc. | Final Acc. | Status |
|---|---|---:|---:|---:|---|
| M0 | NSE-Reset | 3/3 | 79.55±0.04 | 79.38±0.16 | complete |
| M1 | Aggressive Top-1 Persistent Insert | 3/3 | 57.50±0.81 | 57.34±0.87 | complete |
| M2 | Thresholded All-Sample Persistent Insert | 3/3 | 62.94±0.55 | 62.68±0.59 | complete |
| M3 | Thresholded Unreliable-only Persistent Insert | 3/3 | 61.52±0.31 | 61.32±0.19 | complete |
| M4 | Unreliable-only Persistent SoftMix alpha=0.3 | 3/3 | 62.33±0.20 | 62.16±0.17 | complete |
| M5 | Unreliable-only Persistent Replace | 1/3 | pending | pending | seed 2 running |
| M6 | Reset NSE without Salvage | 0/3 | pending | pending | queued / not started on d437 |

Latest d437 M5 status at 2026-05-18 13:18 CST:

```text
exp: eta03_M5_ReplaceWBU
seed 1: finished 500/500
seed 2: running, around epoch 41/500
seed 3: not started
GPU: TITAN RTX, about 98% utilization
```

### 10.2 c201 eta=0.5 5080 results

Setting:

```text
Dataset: CIFAR-100
q / pr: 0.05
eta / nr: 0.5
epochs: 500
hardware: c201, two RTX 5080 GPUs
```

| ID | Condition | Finished on c201 | Best Acc. | Final Acc. | Status |
|---|---|---:|---:|---:|---|
| M0 | NSE-Reset | 2/3 | 76.56±0.41 | 76.51±0.46 | c201 seed 1 missing; do not overwrite this exp |
| M1 | Aggressive Top-1 Persistent Insert | 3/3 | 56.00±0.46 | 55.88±0.48 | complete |
| M2 | Thresholded All-Sample Persistent Insert | 3/3 | 60.44±0.37 | 60.31±0.41 | complete |
| M3 | Thresholded Unreliable-only Persistent Insert | 3/3 | 59.11±0.32 | 58.98±0.30 | complete |
| M4 | Unreliable-only Persistent SoftMix alpha=0.3 | 3/3 | 60.01±0.36 | 59.90±0.36 | complete |
| M5 | Unreliable-only Persistent Replace | 3/3 | 60.56±0.34 | 60.42±0.26 | complete |
| M6 | Reset NSE without Salvage | 3/3 | 73.70±0.27 | 73.56±0.25 | complete |

Interpretation:

- Under more severe missing-true-label noise, reset still performs far above persistent write-back variants.
- M6 shows salvage contributes materially under reset: removing salvage drops from the M0 two-seed final accuracy around 76.5 to 73.6.
- The incomplete c201 M0 seed 1 should not be rerun under the same `eta05_M0_NSE` exp name, because the script writes `master_log.txt` with mode `w`.

## 11. 当前缺失项与处理状态

### 11.1 已经启动补跑的缺失实验

Following the rule "do not rerun an existing exp name", only an unstarted exp was launched on c201.

```text
Host: c201
GPU: 0, RTX 5080
tmux session: c201_eta03_m6_20260518
exp_name: eta03_M6_NSE_NoSalvage
dataset: CIFAR100
q / pr: 0.05
eta / nr: 0.3
seeds: 1, 2, 3
source_update_mode: none
source_update_scope: none
extra flag: --ablate_no_salvage_training
result root:
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_mvp_source_writeback_20260514/main_table_e500/eta03_M6_NSE_NoSalvage
```

Start confirmation at 2026-05-18 14:04 CST:

```text
GPU0: 3361 MiB, 97% utilization
created files: master_log.txt, seed_1/run.log
latest observed log: epoch_0001
[AblationCheck] no_salvage_training=True | detected_salvage=0 | promoted_salvage=0
[MVPExtraction] epoch_0001 | PrecA=0.495117 CovA=0.129020 NRR_A=0.000000
```

### 11.2 Missing experiments not launched automatically

| Missing item | Why not launched now | Required action |
|---|---|---|
| eta=0.5 M0 seed 1 on c201 | Same exp name `eta05_M0_NSE` already exists; rerunning would overwrite `master_log.txt`. User explicitly requested not to rerun same exp. | Create a new uniquely named seed-completion exp if needed, then merge manually. |
| eta=0.3 M5 on c201 | Same conceptual exp is currently running on d437; not launched to avoid duplicate concurrent evidence. | Wait for d437 or rerun later under 5080-only policy with a new plan. |
| eta=0.3 M0-M4 on c201 | These rows already exist on d437. User requested current data first and no duplicate same exp. | Rerun later only when converting all paper-facing rows to 5080-only. |
| HardRemove / AddRemove | Implemented and launched on c201 as `eta03_M7_HardRemove` / `eta03_M8_AddRemove`. | Wait for 3-seed completion before adding to paper tables. |
| Periodic reset K | Implemented and launched on c201 as `eta03_M9_PeriodicReset10`. | Wait for 3-seed completion before adding to paper tables. |
| Clean/noisy subset audit for old remote runs | Existing remote logs mostly have `[MVPExtraction]`, not the newer `[SubsetAudit]`. | Rerun with the newer release script if this table is needed. |

Update at 2026-05-18 15:10 CST:

These missing families have been implemented in the local release script and
the c201 extended script, then launched as new exp names so no existing
`master_log.txt` is overwritten.

```text
tmux session: c201_missing_source_modes_20260518
GPU: c201 GPU1, RTX 5080
queue order:
1. eta03_M7_HardRemove
2. eta03_M8_AddRemove
3. eta03_M9_PeriodicReset10
4. eta03_M10_TopKReconstruct
```

All rows use:

```text
Dataset: CIFAR100
q / pr: 0.05
eta / nr: 0.3
epochs: 500
seeds: 1, 2, 3
```

Variant definitions:

| Exp | Source rule | Important flags |
|---|---|---|
| `eta03_M7_HardRemove` | Remove low-confidence labels from the working source support. | `--source_update_mode hard_remove --source_update_scope all --source_remove_threshold 0.05` |
| `eta03_M8_AddRemove` | Remove low-confidence source labels and insert high-confidence model labels for unreliable samples. | `--source_update_mode add_remove --source_update_scope unreliable_highconf --source_update_threshold 0.65 --source_remove_threshold 0.05` |
| `eta03_M9_PeriodicReset10` | Persistent hard insert with source reset every 10 epochs. | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold 0.65 --source_reset_interval 10` |
| `eta03_M10_TopKReconstruct` | PLRC-style top-k reconstruction stress test. | `--source_update_mode topk_reconstruct --source_update_scope all --source_reconstruct_k 5` |

Start confirmation:

```text
eta03_M7_HardRemove result directory created.
GPU1: about 2990 MiB, 96% utilization at launch check.
First observed log contains [SubsetAudit] epoch_0001.
```

## 12. Paper-readiness conclusion

For the current scoped claim, the experimental evidence is sufficient to support the mechanism:

```text
Epoch-wise restoration plus epoch-local active extraction is a safer source
handling principle than carrying model-induced source updates across epochs in
the tested insertion, soft-mix, and replacement stress tests.
```

The manuscript should phrase the conclusion as "tested persistent write-back variants" rather than "all persistent source modification methods." For a CCF-A / top-venue submission, the strongest final state is:

1. Keep Table 8 as component ablation.
2. Extend Table 9/10 with SoftMix and optionally Replace if page space allows.
3. Mark hard-remove/add-remove/periodic-reset as not included unless implemented.
4. Before final submission, rerun all paper-facing source-writeback rows on c201 5080 if the hardware statement says all ablations used two RTX 5080 GPUs.

## 13. Docx ablation family launched on c201

The supplemental ablation opinion document
`C:/Users/BILIBILI/Downloads/消融实验意见.docx` requests a larger controlled
family over two settings:

```text
E03: CIFAR-100, q/pr=0.05, eta/nr=0.3
E05: CIFAR-100, q/pr=0.05, eta/nr=0.5
seeds: 1, 2, 3
epochs: 500
backbone: ResNet-18
hardware target: c201, two RTX 5080 GPUs
```

The family is implemented through the local release script:

```text
D:\文件\论文项目\CE泄露V3_Branch_Experiments - 副本\experiments\nse_mvp_source_writeback_20260515_windows\train_nse_source_mvp.py
```

and synchronized to c201 as:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/train_nse_source_mvp_family_20260518.py
```

The c201 launcher is:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/launch_c201_docx_family_20260518.sh
```

Result root:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_writeback_docx_family_20260518
```

The family launcher uses `--source_update_evidence p2`, so the source-update
operators are driven by the final second-pass NSE evidence distribution rather
than by raw classifier predictions. This makes the comparison stricter: all rows
share the same NSE extractor and differ only in whether and how the extracted
evidence is written back to the next epoch's working source.

### 13.1 Queues started

Two waiting tmux queues were created on 2026-05-18 20:16 CST:

| tmux session | GPU | Setting | Waits for | Status at creation |
|---|---:|---|---|---|
| `c201_docx_family_E03_gpu0_20260518` | 0 | E03, `eta=0.3` | `c201_eta03_m6_20260518` | waiting |
| `c201_docx_family_E05_gpu1_20260518` | 1 | E05, `eta=0.5` | `c201_missing_source_modes_20260518` | waiting |

Launcher logs:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_writeback_docx_family_20260518/_launcher_logs/E03_gpu0_20260518_201644.log
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_writeback_docx_family_20260518/_launcher_logs/E05_gpu1_20260518_201644.log
```

First launcher check:

```text
GPU0: 62% utilization, 4459 MiB
GPU1: 96% utilization, 4088 MiB
both launchers are waiting for their prerequisite tmux sessions
```

Current prerequisite status checked at 2026-05-18 20:18 CST:

```text
c201_eta03_m6_20260518:
  seed 1 finished: Best 79.18, Final 79.01
  seed 2 finished: Best 78.36, Final 78.29
  seed 3 running

c201_missing_source_modes_20260518:
  eta03_M7_HardRemove seed 1 finished: Best 72.89, Final 72.89
  eta03_M7_HardRemove seed 2 finished: Best 72.56, Final 72.17
  eta03_M7_HardRemove seed 3 running
  eta03_M8/M9/M10 not started yet
```

Based on current c201 timing, one seed takes roughly 2.3--2.5 hours for this
script. Each setting contains 24 experiment points times 3 seeds. Therefore the
full E03/E05 family is expected to take about one week after the prerequisite
queues finish, assuming both RTX 5080 GPUs remain dedicated to these queues.

### 13.2 Experiment points

For each setting, the launcher runs:

| Group | Points | Implementation |
|---|---:|---|
| A reset baseline | 1 | `--source_update_mode none --source_update_scope none` |
| B soft evidence write-back | 6 | `--source_update_mode soft_evidence --source_update_scope all --source_update_alpha {0.01,0.05,0.1,0.3,0.5,1.0}` |
| C all-sample hard insert | 5 | `--source_update_mode hard_insert --source_update_scope all_highconf --source_update_threshold {0.55,0.65,0.75,0.85,0.95}` |
| D unreliable-only hard insert | 5 | `--source_update_mode hard_insert --source_update_scope unreliable_highconf --source_update_threshold {0.55,0.65,0.75,0.85,0.95}` |
| E add-remove refinement proxy | 2 | `--source_update_mode add_remove --source_update_scope all --source_update_threshold 0.85 --source_remove_threshold {0.05,0.10}` |
| F remove-only purification proxy | 2 | `--source_update_mode hard_remove --source_update_scope all --source_remove_threshold {0.05,0.10}` |
| G Topology-DAES internals | 3 | `--topology_rel_gamma 0.0`, `--daes_entropy_coeff 0.0`, or both |

Script changes added for this family:

- `--source_update_evidence {model,p2}`.
- `--source_update_mode soft_evidence`.
- `--daes_entropy_coeff`, where `0.0` gives fixed-temperature DAES.
- `add_remove` now separates the high-confidence insertion mask from the
  low-evidence removal mask when `--source_update_scope all` is used.

## 14. Source-update operator provenance from the local reference PDFs

The following mapping was verified on 2026-05-18 from the local reference
library under `D:\文件\论文项目\参考文献列表`. The mapping is intentionally
conservative: the experiments are controlled proxies and stress tests, not
claimed reimplementations of the prior methods.

| Operator family in our ablation | Prior mechanism it abstracts | Local reference evidence | Caveat |
|---|---|---|---|
| Hard insert / all-sample insert | Add a confident predicted label into the candidate/partial label state. | PALS/SARI states that confident top-1 classifier predictions augment the partial label for the upcoming iteration. FREDIS describes refinement as moving correct labels from non-candidate labels into candidate labels. IRNet uses label correction by moving the predicted non-candidate label into the candidate set for detected noisy samples. | Our operator uses NSE second-pass evidence and a fixed threshold sweep; it is not an implementation of PALS, FREDIS, or IRNet. |
| Unreliable-only hard insert | Apply insertion only to samples outside the current active/reliable training set. | UPLLRS separates training data into reliable and unreliable subsets and uses different treatment for those subsets. | This is a scope diagnostic, not the UPLLRS recursive separation algorithm. |
| Remove-only purification | Persistently delete low-evidence candidate labels. | POP progressively purifies partial labels by moving false candidate labels out and training on the purified labels in the next epoch. FREDIS includes disambiguation from candidate labels to non-candidate labels. | Our deletion rule is a threshold proxy driven by NSE evidence. |
| Add-remove refinement | Combine candidate insertion and candidate deletion. | FREDIS explicitly combines refinement and disambiguation. | Our row tests the source-state risk of the add/remove pattern under the NSE extractor. |
| Soft evidence write-back | Blend the previous working source with model/evidence-based soft label importance. | ALIM trades off the initial candidate set and model outputs through a soft label-importance mechanism. | ALIM is not a persistent source-write-back method; this row tests whether soft cross-epoch carry-over is safe when turned into a persistent source state. |
| Top-k reconstruction | Reconstruct a shorter candidate label set from model/neighbor evidence. | Noise-separation guided candidate label reconstruction separates normal/noisy samples and reconstructs candidate label sets for training. | Our top-k row is a PLRC-style stress test, not the full sample-separation reconstruction method. |
| Periodic reset | Reset frequency diagnostic for source restoration. | This is derived from NSE's reset-before-extraction principle rather than a prior method. | It should be described as our diagnostic, not a baseline method. |
| Persistent pseudo-target state | Maintain/update soft pseudo-targets across training. | PiCO gradually updates pseudo targets using prototype-based disambiguation; PiCO+ extends robust PLL with noisy-sample treatment. | PiCO/PiCO+ should not be cited as candidate-set write-back evidence. They support the broader distinction between candidate-source state and model-side target state. |

Paper wording should therefore say:

```text
The operators are controlled proxies inspired by candidate refinement,
purification, partial-label augmentation, soft label-importance adjustment, and
candidate reconstruction mechanisms in prior NPLL/PLL methods.
```

Avoid saying:

```text
These rows are direct reimplementations of POP, FREDIS, PALS/SARI, ALIM, or PLRC.
```
