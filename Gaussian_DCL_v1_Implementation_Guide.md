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

- 
- $$
  L_b：\text{该视频的有效时间位置数量；}
  $$

- $$
  T：\text{当前 batch padding 后的时间维度；}
  $$

- $$
  y_b：\text{真实时间位置索引，Start 使用 } s\_labels[b]\text{，End 使用 } e\_labels[b]\text{；}
  $$

- $$
  t：\text{候选时间位置索引；}
  $$

- $$
  m_{b,t}：\text{有效位置掩码，有效位置为 1，Padding 为 0；}
  $$

- $$
  \sigma：\text{所有样本和所有定位分支共用的固定 Gaussian 带宽。}
  $$

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
=1-\exp\left(-\frac{d_{b,t}^{2}}{2\sigma^{2}}\right).
$$
当 `t = y_b` 时，距离为 0，因此权重为 0。距离 GT 越远，未归一化的不相似度权重越大。

### 2.4 互补目标分布

只在有效且非 GT 的位置上归一化：

$$
r_{b,t}=
\begin{cases}
\displaystyle\frac{w_{b,t}}
{\sum_{j\in\mathcal V_b,\,j\ne y_b}w_{b,j}},
& t\in\mathcal V_b,\ t\ne y_b,\\[10pt]
0, & \text{otherwise}.
\end{cases}
$$
其中 `V_b` 表示样本 `b` 的有效时间位置集合。

对于至少有两个有效时间位置的样本：

$$
r_{b,y_b}=0,\qquad
\sum_t r_{b,t}=1.
$$
该分布不是对原始分类标签做平滑，也不是把 GT 位置赋予概率质量；它是一个只覆盖非真实类别的互补目标分布。

### 2.5 DCL 预测分布与损失

对定位头输出 logits `z_b`，沿用当前 Uniform CDL 的反向 logits 机制：

$$
q_{b,t}=\operatorname{Softmax}(-z_b)_t.
$$
实际计算时，先对 `-logits` 屏蔽 Padding，再做 `log_softmax`。这样可避免 IMG 定位头将 Padding logits 设为极小负数后，取负时变成极大正数的问题。

Gaussian-DCL 单样本损失：

$$
\mathcal L_{G,b}=-\sum_t r_{b,t}\log q_{b,t}.
$$
返回 batch 内每个样本的 loss，而不是直接在函数内部求 batch mean。这是为了保留 IMG 现有 Audio 分支 `boola` 的逐样本加权逻辑。

### 2.6 梯度

在目标分布 `r` 固定且不参与梯度传播时，令：

$$
q=\operatorname{Softmax}(-z),\qquad
\mathcal L_G=-\sum_t r_t\log q_t.
$$
对原始 logits `z_i` 求导：

$$
\boxed{\frac{\partial\mathcal L_G}{\partial z_i}=q_i-r_i.}
$$
因此优化会推动反向 Softmax 分布 `q` 接近 Gaussian 构造的互补目标分布 `r`。注意这里的梯度是相对于原始 logits `z` 的梯度，不应误写成 `r-q`。

---

## 3. 带宽的含义与极限性质

### 3.1 全局共享带宽

本版本只有一个 `--gaussian_sigma` 参数：

$$
\sigma_{AV,S}=\sigma_{AV,E}=\sigma_{V,S}=\sigma_{V,E}
=\sigma_{A,S}=\sigma_{A,E}=\sigma.
$$
同一个 batch 内所有视频也共享它。该参数是固定实验超参数，不被优化器更新。

### 3.2 小带宽极限

当 `σ → 0+`，对任意非 GT 的有效位置（其距离严格大于 0）：

$$
w_{b,t}\rightarrow 1.
$$
因此：

$$
r_{b,t}\rightarrow\frac{1}{L_b-1},\quad t\ne y_b.
$$
也就是趋近当前 Uniform CDL 使用的均匀互补目标。实现测试包含小带宽目标分布和 loss 向 Uniform CDL 靠近的验证。

### 3.3 大带宽极限

当 `σ` 很大时，利用指数函数的一阶展开：

$$
1-\exp\left(-\frac{d^2}{2\sigma^2}\right)
\approx\frac{d^2}{2\sigma^2}.
$$
归一化后公共因子抵消，因此：

$$
r_{b,t}\rightarrow
\frac{d_{b,t}^2}{\sum_{j\ne y_b}d_{b,j}^2}.
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
```

默认：

```text
cdl_type = uniform
gaussian_sigma = None
```

当 `cdl_type=gaussian` 且 `w_cdl > 0` 时，必须显式提供有限且大于 0 的 `gaussian_sigma`。否则 argparse 会报错，不会静默使用未经确认的带宽。

### 4.3 损失函数接口

新增函数：

```python
gaussian_complementary_targets(targets, valid_mask, bandwidth)
gaussian_cdl_loss(logits, targets, valid_mask, bandwidth)
```

输入形状：

| 参数 | 形状 | 说明 |
|---|---|---|
| `logits` | `[B, T]` | 单个定位头的时间分类 logits |
| `targets` | `[B]` | 每个样本的 GT 时间索引 |
| `valid_mask` | `[B, T]` | 有效位置掩码 |
| `bandwidth` | 标量 | 固定全局带宽 |

输出：

- `gaussian_complementary_targets` 返回 `[B, T]` 的 float32 目标分布。
- `gaussian_cdl_loss` 返回 `[B]` 的逐样本 loss。

目标分布只由 GT、Mask 和固定带宽构造，并在返回前 `detach()`，不建立目标构造路径的梯度。

### 4.4 训练接入位置

原训练流程先计算六组定位 logits：

- AV Start / End
- Visual Start / End
- Audio Start / End

随后在原 Uniform CDL 的位置按 `cdl_type` 选择损失函数。Start 使用 `s_labels`，End 使用 `e_labels`。三个分支共享 `video_mask` 和同一个带宽。

分支聚合沿用当前实现：

$$
L_{CDL}=\operatorname{mean}(L_{AV,S}+L_{AV,E})
+\operatorname{mean}(L_{V,S}+L_{V,E})
+\frac{\sum_b boola_b(L_{A,S,b}+L_{A,E,b})}
{\sum_b boola_b+10^{-6}}.
$$
总损失增加项：

$$
L_{total}\leftarrow L_{total}+w_{CDL}L_{CDL}.
$$
这里保留了 IMG 原有 Audio 样本级重要性筛选 `boola`。它不是 GT 内外区域权重；本版本没有引入 GT 内外权重。

---

## 5. Padding、变长序列和边界处理

### 5.1 Padding

IMG 的分类头可能将 Padding logits 设置为极小值。因为 CDL 使用 `softmax(-logits)`，若先取负再不屏蔽 Padding，Padding 可能得到异常大的正 logits。因此实现顺序必须是：

1. `negative_logits = -logits.float()`；
2. 将 `~valid_mask` 位置填为 float32 最小值；
3. 对时间维度执行 `log_softmax`；
4. 在 loss 汇总前将无效位置对应的 log 概率设为 0，避免 `0 * 极端值` 的数值问题。

Padding 既不参与互补目标归一化，也不参与反向 Softmax 的归一化。

### 5.2 变长视频

每个样本独立计算有效长度 `L_b = valid_mask[b].sum()`，距离归一化分母也是逐样本的 `L_b - 1`，不是 batch 的 padding 后最大长度 `T`。

### 5.3 GT 边界位置

GT 位于序列开头或末尾时，仍使用同一公式。由于距离使用绝对值，目标分布不要求 GT 两侧都存在候选位置。

### 5.4 单有效位置

如果一个样本只有一个有效位置，且该位置就是 GT，则不存在任何非 GT 有效类别。代码定义该样本 Gaussian-DCL loss 为 0，梯度也为 0。

### 5.5 输入约束

代码检查：

- logits 必须是二维 `[B,T]`；
- targets 必须是 `[B]`；
- valid_mask 必须与 logits 同形状；
- GT 索引必须在 `[0,T)` 内；
- GT 必须落在有效位置；
- 带宽必须是有限正标量；
- 带宽不能是需要梯度的 Tensor。

---

## 6. 如何在本地 IMG 项目应用

### 6.1 应用前确认 Git 状态

在项目根目录执行：

```bash
git branch --show-current
git rev-parse HEAD
git status -sb
git diff --stat
```

必须确认当前分支是：

```text
feature/gaussian-cdl
```

不要覆盖已有未提交改动。尤其不要使用 `git add .` 将 `.idea/`、`logs/` 或个人实验脚本一并提交。

### 6.2 备份

```bash
mkdir -p ../IMG_gaussian_backup
cp main.py ../IMG_gaussian_backup/main.py
cp model/losses.py ../IMG_gaussian_backup/losses.py
```

### 6.3 替换和新增文件

将本交付包中的文件放到项目对应位置：

```text
main.py                         -> 项目根目录/main.py
model/losses.py                 -> 项目/model/losses.py
tests/test_gaussian_cdl.py      -> 项目/tests/test_gaussian_cdl.py
```

`tests/` 目录不存在时先创建。请不要替换 `tests/test_uniform_cdl.py`。

### 6.4 静态检查

```bash
python -m py_compile main.py model/losses.py
```

### 6.5 单元测试

在项目根目录执行：

```bash
python -m unittest discover -s tests -p 'test_*cdl.py' -v
```

当前交付环境对这两个测试文件执行的结果为：**17 项测试全部通过**。这是损失函数级单元测试，不代表已经完成 IMG 数据集训练、GPU 集成或定位指标验证。

### 6.6 检查修改范围

```bash
git diff --stat
git diff -- main.py model/losses.py
git status -sb
```

确认只有预期文件被修改或新增。

---

## 7. 运行方式

### 7.1 Uniform CDL（旧行为）

不传 `--cdl_type` 时默认使用 Uniform：

```bash
python main.py ... --w_cdl 1.0
```

原有 Uniform 实验脚本原则上不需要增加新参数。

### 7.2 Gaussian-DCL

```bash
python main.py ... \
  --w_cdl 1.0 \
  --cdl_type gaussian \
  --gaussian_sigma <由验证实验指定的正数>
```

`<...>` 是占位符，不能原样作为 shell 参数传入。带宽数值尚未在本版本中决定。

### 7.3 关闭 CDL

```bash
python main.py ... --w_cdl 0.0
```

当 `w_cdl=0` 时不计算 CDL 损失。

---

## 8. 单元测试覆盖范围

当前测试覆盖如下：

| 测试 | 验证目标 |
|---|---|
| 目标归一化与 GT/Padding 排除 | 每个有互补类的样本目标和为 1；GT 与 Padding 权重为 0 |
| 距离单调性与对称性 | 距离更远的位置获得更大的互补目标权重；对称位置相等 |
| 逐样本有效长度 | 变长视频按各自有效长度归一化 |
| 手工 loss 对照 | 损失等于目标分布与 `log_softmax(-logits)` 的交叉熵 |
| Padding 不影响 loss | 改变 Padding logits 不改变 loss，且 Padding 梯度为 0 |
| 梯度公式 | 数值梯度符合 `q - r` |
| 小带宽目标极限 | 目标分布趋近 Uniform one-cold 分布 |
| 小带宽 loss 极限 | Gaussian-DCL loss 趋近当前 Uniform CDL loss |
| 大带宽极限 | 目标分布趋近归一化距离平方分布 |
| 极大带宽稳定性 | 极端数值下目标仍有限且归一化 |
| 单有效位置 | 无互补类时 loss 和梯度为 0 |
| GT 合法性 | 拒绝越界或落在 Padding 的 GT |
| 带宽合法性 | 拒绝 0、负数、NaN、Inf 和可学习 Tensor |
| Uniform 回归 | 原 Uniform CDL 相关测试仍然通过 |

---

## 9. 真实训练前的集成验收清单

单元测试通过后，仍必须用 IMG 的真实 batch 做以下检查：

1. 打印并确认六组 logits 的形状均为 `[B,T]`。
2. 确认 `s_labels`、`e_labels` 均在各自视频有效长度内。
3. 确认 `video_mask` 的有效长度与 `vfeat_lens` 一致。
4. 分别检查 AV、Visual、Audio 的 Start/End Gaussian-DCL loss 是否有限。
5. 检查目标分布每行和是否为 1（单有效位置样本除外）。
6. 检查目标 GT 位置和 Padding 位置的互补目标均为 0。
7. 检查 Audio `boola` 加权和分母 `valid + 1e-6` 与原流程一致。
8. 确认 `total_loss` 包含 `w_cdl * cdl_loss`。
9. 执行 `backward()` 后检查定位分支参数梯度有限且非全零（应结合具体 batch 判断）。
10. 确认 optimizer step 正常，无 NaN/Inf。
11. 确认 `--cdl_type uniform` 仍可复现原 Uniform 调用路径。
12. 记录实际运行命令、Commit、随机种子、带宽、`w_cdl` 和训练日志。

不要只凭“训练能启动”判断接入正确。尤其要核实六组定位头均有梯度，以及 Audio 的 `boola` 筛选没有被绕过。

---

## 10. Git 版本管理建议

应用并验收代码后，先检查：

```bash
git status -sb
git diff --check
git diff --stat
```

只暂存预期文件：

```bash
git add main.py model/losses.py tests/test_gaussian_cdl.py
```

提交建议：

```bash
git commit -m "Add Gaussian temporal complementary loss"
git push origin feature/gaussian-cdl
```

如果需要恢复到应用前备份：

```bash
cp ../IMG_gaussian_backup/main.py main.py
cp ../IMG_gaussian_backup/losses.py model/losses.py
```

如果已经提交，优先使用 `git revert <commit>` 生成可追溯回退提交；不要在未确认工作区状态时使用 `git reset --hard`。

---

## 11. 当前已知限制与待验证问题

1. **带宽数值未确定。** 本版本只实现固定全局带宽机制，不包含验证集搜索结论。
2. **距离尺度依赖当前时间索引定义。** 当前采用视频有效位置索引并除以 `L-1`；如果后续确认模型标签对应的时间位置并非均匀采样索引，需要重新审查该距离定义。
3. **Mask 假设是前缀有效。** 当前 IMG 的长度 Mask 符合这一形式；若未来改成中间存在空洞的非连续有效 Mask，距离分母仍按有效长度计算，但位置坐标语义需要重新定义。
4. **目前没有真实训练证据。** 尚不能判断 Gaussian-DCL 是否优于 Uniform CDL 或 Baseline。
5. **尚未分析与 IMG 原始损失的梯度冲突。** 需要在真实训练 batch 上比较原始定位损失与 Gaussian-DCL 对共享定位头的梯度方向和相对范数。
6. **当前测试不是端到端模型测试。** 仍需确认训练脚本、GPU 环境、真实数据和 checkpoint 保存流程。

---

## 12. 当前版本结论

Gaussian-DCL v1.0 已实现一个明确的时间距离感知互补目标：GT 位置不接收互补目标质量，非 GT 有效位置根据其相对时间距离获得不同权重，并归一化为概率分布；预测端继续使用 `softmax(-logits)`，损失以逐样本形式返回并接入 IMG 现有三分支训练聚合。

本版本的下一阶段不是立即宣称方法有效，而是完成本地代码应用、单元测试复跑、真实 batch 集成验收和小规模训练检查。带宽搜索和正式对比实验应在这些工程验收通过后再制定。
## 运行方式
conda activate IMG
先设置高斯带宽：export GAUSSIAN_SIGMA=0.1 这里的 0.1 仅为命令示例，不代表已经确定的实验参数。
bash run_img_gaussian_dcl_v1_seed0515.sh
也可以指定 GPU：
GPU_IDX=0 GAUSSIAN_SIGMA=0.1 \
bash run_img_gaussian_dcl_v1_seed0515.sh
合成一条：GAUSSIAN_SIGMA=0.1 bash run_img_gaussian_dcl_v1_seed0515.sh
