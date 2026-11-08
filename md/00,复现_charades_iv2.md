# Baseline-03 实验统计报告：03_charades_iv2

> 实验标识：`03_charades_iv2`\
> 模型名称：`IMG_CS_IV2_0515`\
> 统计依据：`configs(2).json` 与三份评估日志。\
> 统计原则：保留日志原始数值；每项指标单独统计全日志最大值；不把不同
> Epoch 的指标拼接成同一模型结果。

## 1. 实验基本信息

  项目                          记录
  ----------------------------- ----------------------------------------------
  实验编号                      `03_charades_iv2`
  实验阶段                      Baseline / 论文忠实复现阶段
  数据任务                      `charades`
  视觉特征                      `iv2`（IV2）
  随机种子                      `515`
  模型名称                      `IMG_CS_IV2_0515`
  训练 Epoch                    100
  Batch Size                    16
  Initial Learning Rate         0.0005
  计划训练 Steps                77,600
  日志数量                      3 份（AV、Audio、Visual）
  每份日志评估记录              200 条
  总评估记录                    600 条
  评估频率                      每个 Epoch 记录 2 次；相邻评估间隔 388 Steps
  三份日志 Step 序列            完全一致
  当前上传的本实验 Checkpoint   未发现 IV2 对应权重文件

说明：当前上传文件列表中的 `.t7` 权重名称为
`IMG_CS_I3D_0515_*.t7`，不属于本组 `IMG_CS_IV2_0515`。因此不将上一组 I3D
权重归入本组。

## 2. 训练配置摘要

  配置项                    值
  --------------------- ------
  `video_feature_dim`      768
  `audio_feature_dim`     2048
  `text_feature_dim`      4096
  `dim`                    128
  `num_heads`                8
  `layers`                   3
  `slot_layers`              3
  `drop_rate`              0.2
  `max_pos_len`             64
  `max_pos_len_a`          128
  `warm_epoch`              20
  `warmup_proportion`      0.0
  `clip_norm`              1.0
  `event_num`                3
  `max_n`                    2
  `word_dim`               300
  `word_size`             1271
  `char_dim`                50
  `char_size`               36

### 完整配置快照

以下保留上传的 JSON 配置原始键和值：

``` json
{
  "audio_feature_dim": 2048,
  "batch_size": 16,
  "char_dim": 50,
  "char_size": 36,
  "clip_norm": 1.0,
  "cons_lamb": 1,
  "constractive": false,
  "dim": 128,
  "drop_rate": 0.2,
  "epochs": 100,
  "event_num": 3,
  "gpu_idx": "0",
  "ia_lamb": 5,
  "init_lr": 0.0005,
  "kl_lamb": 10,
  "lamb": 1,
  "layers": 3,
  "max_n": 2,
  "max_pos_len": 64,
  "max_pos_len_a": 128,
  "mode": "train",
  "model_dir": "experiments/main_seed_0515_20261102_144018/03_charades_iv2/checkpoints",
  "model_name": "IMG_CS_IV2_0515",
  "num_heads": 8,
  "num_train_steps": 77600,
  "our_loss_th": 0.2,
  "period": 100,
  "sa_lamb": 0.5,
  "save_dir": "datasets",
  "save_predictions": null,
  "seed": 515,
  "slot_layers": 3,
  "suffix": null,
  "task": "charades",
  "test_path": null,
  "text_feature_dim": 4096,
  "tmp": 3,
  "video_feature_dim": 768,
  "visual_features": "iv2",
  "warm_epoch": 20,
  "warmup_proportion": 0.0,
  "word_dim": 300,
  "word_size": 1271
}
```

## 3. 全训练过程各指标最佳值

下表对每个分支的 200 条评估记录分别取单项指标最大值。每项指标的最佳
Epoch/Step
单独记录；同一分支不同指标可能来自不同模型检查点，因此本表不是单一
Checkpoint 的联合指标表。

  分支     指标             日志最大值 达到最大值的位置（Epoch / Step）
  -------- -------------- ------------ ----------------------------------
  AV       R@1, IoU=0.3          82.50 44 / 34144
  AV       R@1, IoU=0.5          71.99 44 / 34144
  AV       R@1, IoU=0.7          53.74 36 / 27936
  AV       mean IoU              62.31 36 / 27936; 51 / 39576
  Audio    R@1, IoU=0.3          60.94 54 / 41516
  Audio    R@1, IoU=0.5          45.86 73 / 56260
  Audio    R@1, IoU=0.7          29.14 69 / 53544
  Audio    mean IoU              42.56 54 / 41516
  Visual   R@1, IoU=0.3          82.50 35 / 26772
  Visual   R@1, IoU=0.5          71.48 69 / 53544
  Visual   R@1, IoU=0.7          54.03 54 / 41904
  Visual   mean IoU              62.55 44 / 34144

### 以 R@1, IoU=0.7 为参照的各分支最高点

  分支       Best Epoch   Best Step   R@1@0.7   mean IoU（同一 Step）
  -------- ------------ ----------- --------- -----------------------
  AV                 36       27936     53.74                   62.31
  Audio              69       53544     29.14                   42.45
  Visual             54       41904     54.03                   62.34

注意：这里的"日志最高点"表示该指标在本次训练评估日志中的最大观测值。配置中的
`test_path` 为 `null`，日志也未明确标注评估
split，因此不能仅凭这些记录将其认定为独立验证集或测试集的最佳结果。

## 4. 最后一次评估结果（Epoch 100）

以下结果均来自 Step 77600，即三份日志各自的最后一条记录。

  分支       Epoch    Step   R@1@0.3   R@1@0.5   R@1@0.7   mean IoU
  -------- ------- ------- --------- --------- --------- ----------
  AV           100   77600     79.70     67.98     47.45      59.03
  Audio        100   77600     59.22     44.70     28.52      41.74
  Visual       100   77600     80.22     69.70     50.86      60.69

## 5. Checkpoint 文件情况

本次上传文件中未发现命名为 `IMG_CS_IV2_0515_*.t7` 的模型权重。因此：

-   无法对本组 Checkpoint 执行 `torch.load`、State Dict 键结构、Tensor
    数量或 SHA256 权重校验。
-   无法确认日志中的各项最高点是否对应已保存的权重。
-   当前报告仅对配置和评估日志进行统计，不将上一组 I3D
    实验的权重误归入本组。

建议后续补充本组实际保存的 `.t7` / `.pth` 权重文件及其保存
Step，以便完成权重---指标对应检查。

## 6. 统计解释与限制

1.  **单随机种子实验。** 当前配置只记录 Seed
    515，因此这是一次训练运行，不能据此计算多随机种子实验的
    Mean、Std、置信区间或显著性结论。
2.  **评估记录不是独立重复实验。** 每份日志中的 200
    条记录来自同一训练过程，不能当作 200 次独立实验计算实验间方差。
3.  **评估数据划分待确认。** 配置中的 `test_path` 为
    `null`，日志没有明确写出评估 split。本报告不擅自将评估结果标记为
    Validation 或 Test。
4.  **最佳指标与最终 Epoch 指标不同。** 日志最佳值出现在不同
    Epoch；Epoch 100
    的最后一次评估结果单独列示，不与训练过程中的最佳值混用。
5.  **Checkpoint 选择依据待确认。**
    当前没有上传本组权重，无法验证日志最高点对应的模型是否保存，也无法确认训练时采用的模型选择指标。
6.  **论文结果对比尚未纳入。**
    本报告只统计本次上传的配置和日志，不在缺少已核对论文表格对应项时填写论文原始结果或性能差值。

## 7. 当前实验结论

-   `03_charades_iv2` 的 AV、Audio、Visual 三份评估日志均覆盖 Epoch
    1--100，各有 200 条记录，共 600 条；三份日志的 Step 序列完全一致。
-   AV 分支的日志最高 R@1@0.7 为 **53.74**，出现在 Epoch 36 / Step
    27936；同一 Step 的 mean IoU 为 62.31。
-   Audio 分支的日志最高 R@1@0.7 为 **29.14**，出现在 Epoch 69 / Step
    53544；同一 Step 的 mean IoU 为 42.45。
-   Visual 分支的日志最高 R@1@0.7 为 **54.03**，出现在 Epoch 54 / Step
    41904；同一 Step 的 mean IoU 为 62.34。
-   Epoch 100 / Step 77600 的 AV、Audio、Visual 分支 R@1@0.7 分别为
    **47.45**、**28.52**、**50.86**。
-   在 R@1@0.7 这一指标上，Visual 分支的全程最高值略高于 AV 分支；Audio
    分支明显较低。该观察仅描述本次单种子日志，不构成统计显著性结论。
-   当前上传文件中未发现本组 IV2 对应
    Checkpoint；因此本组可登记为完成日志统计的单种子 Baseline
    记录，但暂不能完成最佳权重核验。

## 8. 原始来源文件

  文件                     用途
  ------------------------ -------------------------------------------
  `configs(2).json`        本组实验配置
  `eval_results(2).txt`    AV 分支评估日志（按项目文件命名约定）
  `eval_resultsa(2).txt`   Audio 分支评估日志（按项目文件命名约定）
  `eval_resultsv(2).txt`   Visual 分支评估日志（按项目文件命名约定）

------------------------------------------------------------------------

本报告由本次上传的配置文件与三份评估日志解析生成。原始文件未被修改；缺失的
Checkpoint 未作推断或补造。
