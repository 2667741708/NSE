# 基线脚本溯源 / Baseline Script Provenance

## 脚本文件 / Script File
- **文件名**: `v3_2passKNN_refactored_celeak_modelonly_activeonly.py`
- **总行数**: 295 行
- **获取时间**: 2026-04-12T22:38:00+08:00

## 来源 / Source
- **服务器**: C201 (10.20.22.105)
- **远程路径**: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/v3_2passKNN_refactored_celeak_modelonly_activeonly.py`
- **依赖基脚本**: `v3_2passKNN_refactored_bayes_unified.py` (通过 `importlib` 动态加载)

## 核心算法特征 / Core Algorithm Features
1. **Model-Only Leaky Prior**: Leaky omega 仅应用于模型预测分支, KNN 分支保持纯净
2. **CE Reliability Fusion**: 通过交叉熵度量 KNN 与模型预测的一致性, 计算可靠度权重 r_i
3. **Active-Only Training**: 仅使用被标记为 reliable 的样本进行训练, 跳过不可靠样本
4. **Two-Pass KNN Propagation**: 双轮 KNN 传播, 第一轮使用原始标签, 第二轮使用自适应融合结果

## 关键超参数 / Key Hyperparameters
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prior_leakage` | 0.01 | Leaky prior 泄漏系数 |
| `adap_rel_gamma` | 2.0 | CE 可靠度衰减因子 |
| `adap_rel_eps` | 1e-12 | 数值稳定性 epsilon |
| `model_warmup_epochs` | 20 | 模型预热 epoch 数 |
| `max_w_model` | 1.0 | 模型权重上限 |
| `adap_ce_direction` | knn_over_model | CE 计算方向 |

## 日志目录 / Log Directory
- **远程路径**: `/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/v3_2passKNN_refactored_celeak_modelonly_activeonly/`
