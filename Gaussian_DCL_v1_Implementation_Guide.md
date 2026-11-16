# Gaussian-DCL v1.0：实现说明、数学定义与工程验收文档

> 项目：IMG — Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval  
> 研究方向：将互补不相似性学习迁移到视频时刻定位的时间位置分类头  
> 开发分支：`feature/gaussian-cdl`  
> 版本定位：Gaussian-DCL 初始工程实现（尚未完成真实数据集训练验证）

---

## 1. 本版本目标与边界

本版本在 IMG 已有的时间定位分类头上增加 Gaussian-DCL 辅助损失。它不改变 IMG 的网络结构、预测过程或推理接口，只改变训练阶段附加的互补监督目标分布。

### 1.1 本版本已经确定的设计

| 项目 | 当前实现 |
|---|---|
| 互补目标的中心 | 当前样本对应的 GT 时间位置 |
| Start 标签 | `s_labels` |
| End 标签 | `e_labels` |
| 时间距离 | `abs(t - y) / (L - 1)`，其中 `L` 是该样本有效时间长度 |
| Gaussian 相似度 | `exp(-d² / (2σ²))` |
| 不相似度权重 | `1 - exp(-d² / (2σ²))` |
| 互补目标 | 对所有有效非 GT 位置的不相似度权重归一化 |
| 预测分布 | `softmax(-logits)`，沿用当前 Uniform CDL 的 DCL 预测方向 |
| 带宽 | 一个固定正标量，全样本、AV/V/A、Start/End 共用 |
| 带宽梯度 | 不参与反向传播，不是可学习参数 |
| GT 内外权重 | 不引入 |
| Padding | 从目标分布和反向 Softmax 中排除 |
| Audio 分支 | 保留 IMG 原有 `boola` 样本级加权 |
| 默认类型 | `uniform`，保证旧 Uniform CDL 调用兼容 |

### 1.2 本版本明确不做的事情

- 不增加任何网络层或可学习参数。
- 不修改 `IMGNet.py`、`layers.py` 的定位头结构。
- 不修改数据加载、GT 标签定义和推理阶段。
- 不改变 IMG 原有定位损失、Saliency Loss、KD Loss、Important-Aware Loss。
- 不自动选择 Gaussian 带宽数值。
- 不声称 Gaussian-DCL 已经提升定位指标；这一点必须由真实训练和评估结果验证。

---

## 2. 数学定义

### 2.1 符号

对一个 batch 中的第 `b` 个视频样本，定义：

- \(L_b\)：该视频的有效时间位置数量；
- \(T\)：当前 batch padding 后的时间维度；
- \(y_b\)：真实时间位置索引，Start 使用 `s_labels[b]`，End 使用 `e_labels[b]`；
- \(t\)：候选时间位置索引；
- \(m_{b,t}\)：有效位置掩码，有效位置为 1，Padding 为 0；
- \(\sigma\)：所有样本和所有定位分支共用的固定 Gaussian 带宽。

当前实现假设 IMG 的 `video_mask` 是从有效长度生成的前缀掩码，即有效位置为 `[0, L_b-1]`。这与当前 IMG 数据加载和 `convert_length_to_mask(vfeat_lens)` 的使用方式一致。

### 2.2 相对时间距离

当 `L_b > 1` 时：

$$
d_{b,t}=\frac{|t-y_b|}{L_b-1}
$$

因此有效时间轴上的最大距离被归一化到 1。不同长度的视频使用相对时间尺度，而不是直接比较原始帧/片段索引距离。

对于只有一个有效位置的样本，分母通过 `clamp_min(1)` 防止除零；该样本没有非 GT 类别，最终互补目标为全零，Gaussian-DCL 损失也为零。

### 2.3 Gaussian 相似度与不相似度

Gaussian 相似度：

$$
G_{b,t}=\exp\left(-\frac{d_{b,t}^{2}}{2\sigma^{2}}\right)
$$

将其转换为不相似度权重：

$$
w_{b,t}=1-G_{b,t}
=1-\exp\left(-\frac{d_{b,t}^{2}}{2\sigma^{2}}\right)
$$

当 `t = y_b` 时，距离为 0，因此权重为 0。距离 GT 越远，未归一化的不相似度权重越大。

### 2.4 互补目标分布

只在有效且非 GT 的位置上归一化：

$$
r_{b,t}=
\begin{cases}
\frac{w_{b,t}}{\sum_{j\in\mathcal V_b,\,j\ne y_b}w_{b,j}},
& t\in\mathcal V_b,\ t\ne y_b,\\
0, & \text{otherwise}.
\end{cases}
$$

其中 `V_b` 表示样本 `b` 的有效时间位置集合。

对于至少有两个有效时间位置的样本：

$$
r_{b,y_b}=0,\qquad
\sum_t r_{b,t}=1
$$

该分布不是对原始分类标签做平滑，也不是把 GT 位置赋予概率质量；它是一个只覆盖非真实类别的互补目标分布。

### 2.5 DCL 预测分布与损失

对定位头输出 logits `z_b`，沿用当前 Uniform CDL 的反向 logits 机制：

$$
q_{b,t}=\mathrm{Softmax}(-z_b)_t
$$

实际计算时，先对 `-logits` 屏蔽 Padding，再做 `log_softmax`。这样可避免 IMG 定位头将 Padding logits 设为极小负数后，取负时变成极大正数的问题。

Gaussian-DCL 单样本损失：

$$
\mathcal L_{G,b}=-\sum_t r_{b,t}\log q_{b,t}
$$

返回 batch 内每个样本的 loss，而不是直接在函数内部求 batch mean。这是为了保留 IMG 现有 Audio 分支 `boola` 的逐样本加权逻辑。

### 2.6 梯度

在目标分布 `r` 固定且不参与梯度传播时，令：

$$
q=\mathrm{Softmax}(-z),\qquad
\mathcal L_G=-\sum_t r_t\log q_t
$$

对原始 logits `z_i` 求导：

$$
\boxed{\frac{\partial\mathcal L_G}{\partial z_i}=q_i-r_i}
$$

因此优化会推动反向 Softmax 分布 `q` 接近 Gaussian 构造的互补目标分布 `r`。注意这里的梯度是相对于原始 logits `z` 的梯度，不应误写成 `r-q`。

---

## 3. 带宽的含义与极限性质

### 3.1 全局共享带宽

本版本只有一个 `--gaussian_sigma` 参数：

$$
\sigma_{AV,S}=\sigma_{AV,E}=\sigma_{V,S}=\sigma_{V,E}
=\sigma_{A,S}=\sigma_{A,E}=\sigma
$$

同一个 batch 内所有视频也共享它。该参数是固定实验超参数，不被优化器更新。

### 3.2 小带宽极限

当 `σ → 0+`，对任意非 GT 的有效位置（其距离严格大于 0）：

$$
w_{b,t}\rightarrow 1
$$

因此：

$$
r_{b,t}\rightarrow\frac{1}{L_b-1},\quad t\ne y_b
$$

也就是趋近当前 Uniform CDL 使用的均匀互补目标。实现测试包含小带宽目标分布和 loss 向 Uniform CDL 靠近的验证。

### 3.3 大带宽极限

当 `σ` 很大时，利用指数函数的一阶展开：

$$
1-\exp\left(-\frac{d^2}{2\sigma^2}\right)
\approx\frac{d^2}{2\sigma^2}
$$

归一化后公共因子抵消，因此：

$$
r_{b,t}\rightarrow
\frac{d_{b,t}^2}{\sum_{j\ne y_b}d_{b,j}^2}
$$

即互补质量趋向按归一化距离平方分配。代码针对极大带宽下 float32 权重整体下溢的情况，使用这一极限分布作为数值稳定回退。

### 3.4 带宽不是越大或越小越好

- 很小的带宽使方案接近 Uniform CDL，时间距离差异弱化。
- 很大的带宽使目标分布接近距离平方加权。
- 中间带宽控制近邻与远距离位置之间的互补质量差异。

具体数值尚未由本版本确定。应在验证集上选择，且 Baseline、Uniform CDL、Gaussian-DCL 的其他训练条件保持一致。

---

## 4. 代码修改说明

### 4.1 文件清单

| 文件 | 修改 | 作用 |
|---|---|---|
| `main.py` | 修改 | 增加 CDL 类型和 Gaussian 带宽参数；在原 CDL 接入位置调用所选损失 |
| `model/losses.py` | 修改 | 保留 `uniform_cdl_loss`，新增 Gaussian 目标构造与损失计算 |
| `tests/test_gaussian_cdl.py` | 新增 | 验证 Gaussian 目标分布、loss、梯度、Mask、极限和非法输入 |
| `tests/test_uniform_cdl.py` | 保留 | 回归检查已有 Uniform CDL |
| `IMGNet.py` | 不修改 | 不改变三个定位分支 |
| `layers.py` | 不修改 | 不改变 Start/End 分类头 |
| `data_loader.py` | 不修改 | 沿用原有 GT 标签 |
| `runner_utils.py` | 不修改 | 沿用现有 Mask 和评估流程 |

### 4.2 新增命令行参数

`main.py` 新增：

```bash
--cdl_type {uniform,gaussian}
--gaussian_sigma FLOAT
