# 第3章 方法与实验配置说明

## 3.1 项目名称与主脚本

本项目当前对应算法名称为 **Reliability-Aware Bayesian Dual-View Fusion for Noisy Partial Label Learning**，实验主脚本为：

`bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py`

该版本不再对应旧的 CE-Leak、单侧泄漏先验或 Active-only 单流丢弃方案，而是统一的“可靠性感知贝叶斯证据融合”版本。其核心目标是：保留原始候选标签或众包先验，不显式覆写候选集；同时在两阶段 KNN 传播之间引入双视图模型预测，使模型语义信息以可靠度门控方式参与第二阶段传播。

## 3.2 总体框架

给定样本 $x_i$ 与候选标签集合 $Y_i$，算法每个 epoch 执行如下流程：

1. 使用弱增强视图提取特征，构建 KNN 图；
2. 使用弱增强和强增强两个视图分别预测，取平均得到模型软分布 $P_{\mathrm{model}}$；
3. 通过第一阶段 DAES-KNN 传播得到几何软分布 $P_{\mathrm{knn}}$；
4. 根据 KNN 置信度与模型置信度计算逐样本可靠度 $r_i$；
5. 在数据集先验 $\Omega$ 约束下融合 KNN 证据与模型证据，形成第二阶段传播输入；
6. 执行第二阶段 DAES-KNN 传播，得到最终伪标签与可靠集；
7. 训练阶段对可靠样本使用双视图 MixUp 监督，对非可靠样本使用 SoftMatch 加权的一致性软目标。

## 3.3 双视图模型预测

脚本中的 `get_features` 函数对每个样本同时使用弱增强图像 $x_i^w$ 与强增强图像 $x_i^s$。弱增强特征用于 KNN 拓扑：

$$
z_i = \frac{f_\theta(x_i^w)}{\|f_\theta(x_i^w)\|_2}.
$$

模型预测则由两个视图平均得到：

$$
P_{\mathrm{model},i}
= \frac{1}{2}
\left[
\mathrm{softmax}(h_\phi(f_\theta(x_i^w)))
+
\mathrm{softmax}(h_\phi(f_\theta(x_i^s)))
\right].
$$

这样做的作用是：KNN 图保持弱增强视图下更稳定的几何结构，而模型端利用强弱双视图降低单一增强带来的预测波动。

## 3.4 DAES-KNN 第一阶段传播

DAES（Dynamic Adaptive Entropy Similarity）根据邻域标签分布熵自适应调节相似度温度。对于样本 $i$ 的邻居集合 $\mathcal{N}_K(i)$，先计算邻域平均标签分布 $\bar{P}_i$，再计算归一化熵：

$$
H_i = -\frac{1}{\log C}\sum_{c=1}^{C}\bar{P}_{i,c}\log(\bar{P}_{i,c}+\epsilon).
$$

动态温度为：

$$
\tau_i = \tau_0 + \lambda_H H_i^2.
$$

之后将特征相似度转化为传播权重：

$$
W_{ij} =
\exp\left(
\frac{S_{ij}^{\gamma_s}}{\tau_i}
-
\max_{j'}\frac{S_{ij'}^{\gamma_s}}{\tau_i}
\right).
$$

第一阶段传播从候选标签矩阵或众包先验 $P^{(0)}$ 出发：

$$
P_{\mathrm{knn},i}
=
\mathrm{Norm}
\left(
\sum_{j\in\mathcal{N}_K(i)} W^{(1)}_{ij}P^{(0)}_j
\right).
$$

## 3.5 渐进式模型融合

模型预测在训练前期不直接强势介入，而是通过预热权重逐渐增强：

$$
w_t = \min\left(w_{\max}, \frac{t}{T_{\mathrm{warm}}}\right).
$$

脚本中对应参数为：

- `--model_warmup_epochs`
- `--max_w_model`

有效模型分布为：

$$
P^{\mathrm{eff}}_{\mathrm{model},i}
=
w_t P_{\mathrm{model},i}
+
(1-w_t)P_{\mathrm{knn},i}.
$$

当训练初期模型尚不可靠时，$P^{\mathrm{eff}}_{\mathrm{model}}$ 接近 $P_{\mathrm{knn}}$；训练稳定后，模型语义预测逐步参与融合。

## 3.6 自适应可靠度 $r_i$

当前脚本最终采用的是置信度比例形式的 $r_i$，不是旧草稿中的交叉熵门控形式。令：

$$
q_i^{\mathrm{knn}} = \max_c P_{\mathrm{knn},i,c},
\qquad
q_i^{\mathrm{model}} = \max_c P_{\mathrm{model},i,c}.
$$

则：

$$
r_i =
\frac{q_i^{\mathrm{knn}}}
{q_i^{\mathrm{knn}} + q_i^{\mathrm{model}} + \epsilon}.
$$

当 KNN 分布更尖锐时，$r_i$ 较大，第二阶段更相信几何传播；当模型预测更自信时，$1-r_i$ 较大，模型语义信息被更多引入。

## 3.7 统一贝叶斯证据融合

令 $\Omega_i$ 为数据集先验：

- CIFAR / CIFAR100H / CUB200 等标准 NPLL 数据集使用候选标签 mask；
- Benthic / Plankton / Treeversity 等真实众包数据集使用 crowd prior。

先将模型预测投影到原始先验支持上：

$$
\hat{P}_{\mathrm{model},i}
=
\mathrm{Norm}
\left(
P^{\mathrm{eff}}_{\mathrm{model},i}\odot\Omega_i
\right).
$$

第二阶段传播输入为：

$$
P^{(2)}_{\mathrm{in},i}
=
\mathrm{Norm}
\left(
r_iP_{\mathrm{knn},i}
+
(1-r_i)\hat{P}_{\mathrm{model},i}
\right).
$$

这就是脚本注释中“Unified Bayesian Evidential Fusion”的实际实现。它替代了旧版本中针对 CIFAR、CUB、众包数据分别写不同 refine branch 的方式，使所有数据集默认进入同一条融合路径。

## 3.8 第二阶段传播与可靠集筛选

第二阶段继续使用 DAES 权重，以 $P^{(2)}_{\mathrm{in}}$ 为参考分布进行传播：

$$
P^{(2)}_i =
\mathrm{softmax}
\left(
\sum_{j\in\mathcal{N}_K(i)}
W^{(2)}_{ij}P^{(2)}_{\mathrm{in},j}
\right).
$$

最终伪标签为：

$$
\hat{y}_i = \arg\max_c P^{(2)}_{i,c}.
$$

当 $\hat{y}_i$ 被候选标签或众包先验支持，且置信度通过 `--delta` 控制的筛选阈值时，该样本进入可靠集。

## 3.9 训练目标

训练阶段保留统一 DataLoader，但在 batch 内根据可靠掩码拆分可靠样本与非可靠样本。

### 可靠样本

可靠样本使用弱/强双视图 MixUp 监督。标签平滑目标为：

$$
\tilde{y}_i =
(1-\epsilon_{\mathrm{ls}})e_{\hat{y}_i}
+
\frac{\epsilon_{\mathrm{ls}}}{C}\mathbf{1}.
$$

监督损失为弱增强与强增强 MixUp 交叉熵的平均：

$$
\mathcal{L}_{\mathrm{sup}}
=
\frac{1}{2}
\left[
\mathrm{CE}(\tilde{x}^w,\tilde{y})
+
\mathrm{CE}(\tilde{x}^s,\tilde{y})
\right].
$$

### 非可靠样本

非可靠样本不再直接丢弃。脚本使用弱视图模型预测，经类别再平衡后作为软目标：

$$
T_i =
\mathrm{Norm}
\left(
P^w_{\mathrm{model},i}\odot\rho
\right),
\qquad
\rho =
\frac{\hat{p}_{\mathrm{target}}}
{\hat{p}_{\mathrm{model}}+\epsilon}.
$$

随后用 SoftMatch 权重 $\omega_i^{\mathrm{sm}}$ 控制该样本的一致性损失：

$$
\mathcal{L}_{\mathrm{unrel}}
=
\mathbb{E}_i
\left[
\omega_i^{\mathrm{sm}}
\mathrm{CE}
\left(
\sigma(h_\phi(f_\theta(\tilde{x}^s_i))), \tilde{T}_i
\right)
\right].
$$

总损失为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{sup}}
+
\lambda_{\mathrm{cons}}(t)\mathcal{L}_{\mathrm{unrel}}.
$$

其中动态一致性权重由非可靠样本比例和模型-KNN共识比例共同决定：

$$
\lambda_{\mathrm{cons}}(t)
=
r_{\mathrm{unrel}}(t)\cdot a_{\mathrm{mk}}(t)^p\cdot\lambda_0.
$$

因此，可靠样本提供主要监督锚点，非可靠样本只有在模型置信度和模型-KNN共识足够高时才以软权重参与训练。

## 3.10 实验配置摘要

脚本中推荐配置包括：

- CIFAR 系列：`ResNet-18`，`epochs=500`，`batch_size=256`，`lr_scheduler=cosine`；
- 真实众包数据集：`ResNet-50`，`epochs=100`，`batch_size=32`，`lr_scheduler=step`；
- 常用 DAES 配置：`--sim_mode_1 topology_daes --sim_mode_2 topology_daes`；
- 可靠集筛选：`--k_val 15 --delta 0.25`（CIFAR），众包实验常用 `--k_val 5 --delta 1.0`；
- 渐进融合：`--model_warmup_epochs 20`，`--max_w_model` 可按数据集设为 `0.5` 或更低；
- 当前参考运行通常使用 `--lsr 0.0 --history_len 15 --consensus_power 2.0`。
