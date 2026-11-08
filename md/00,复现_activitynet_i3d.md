# Baseline-04 实验统计报告：04_activitynet_i3d

> 实验标识：`04_activitynet_i3d`\
> 模型名称：`IMG_AN_I3D_0515`\
> 统计依据：`configs(4).json` 与三份评估日志。\
> 统计原则：保留日志原始数值；每项指标单独统计全日志最大值；不把不同
> Epoch 的指标拼接成同一模型结果。

## 1. 实验基本信息

  项目                          记录
  ----------------------------- ----------------------------------
  实验编号                      `04_activitynet_i3d`
  实验阶段                      Baseline / 论文忠实复现阶段
  数据任务                      `activitynet`
  视觉特征                      `i3d`
  随机种子                      `515`
  模型名称                      `IMG_AN_I3D_0515`
  训练 Epoch                    100
  Batch Size                    16
  Initial Learning Rate         0.0005
  计划训练 Steps                233,900
  日志数量                      3 份（AV、Audio、Visual）
  每份日志评估记录              200 条
  总评估记录                    600 条
  评估频率                      每个 Epoch 记录 2 次
  相邻评估 Step 间隔            1,169 或 1,170 Steps
  三份日志 Step 序列            完全一致
  最后一次评估                  Epoch 100 / Step 233900
  `test_path`                   `null`（评估数据划分需另行确认）
  当前上传的本实验 Checkpoint   未发现 ActivityNet 对应权重文件

说明：当前上传文件中存在 `IMG_CS_I3D_0515_*.t7` 权重，但其命名对应
Charades / I3D，不属于本组 `IMG_AN_I3D_0515`。因此不将这些权重归入
ActivityNet 实验。

## 2. 训练配置摘要

  配置项                                    值
  --------------------- ----------------------
  `video_feature_dim`                     1024
  `audio_feature_dim`                      128
  `text_feature_dim`      未在本配置中单独记录
  `dim`                                    128
  `num_heads`                                8
  `layers`                                   3
  `slot_layers`                              3
  `drop_rate`                              0.2
  `max_pos_len`                            128
  `max_pos_len_a`                           71
  `warm_epoch`                              20
  `warmup_proportion`                      0.0
  `clip_norm`                              1.0
  `event_num`                                3
  `max_n`                                    2
  `word_dim`                               300
  `word_size`                            11894
  `char_dim`                               100
  `char_size`                               55
  `cons_lamb`                                1
  `ia_lamb`                                  5
  `kl_lamb`                                 10
  `sa_lamb`                                0.5

### 完整配置快照

以下保留上传 JSON 配置中的全部键和值：

``` json
{
  "audio_feature_dim": 128,
  "batch_size": 16,
  "char_dim": 100,
  "char_size": 55,
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
  "max_pos_len": 128,
  "max_pos_len_a": 71,
  "mode": "train",
  "model_dir": "experiments/main_seed_0515_20261102_144018/04_activitynet_i3d/checkpoints",
  "model_name": "IMG_AN_I3D_0515",
  "num_heads": 8,
  "num_train_steps": 233900,
  "our_loss_th": 0.1,
  "period": 100,
  "sa_lamb": 0.5,
  "save_dir": "datasets",
  "save_predictions": null,
  "seed": 515,
  "slot_layers": 3,
  "suffix": null,
  "task": "activitynet",
  "test_path": null,
  "tmp": 2,
  "video_feature_dim": 1024,
  "visual_features": "i3d",
  "warm_epoch": 20,
  "warmup_proportion": 0.0,
  "word_dim": 300,
  "word_size": 11894
}
```

## 3. 全训练过程各指标最佳值

下表对每个分支的 200 条评估记录分别取单项指标最大值。每项指标的最佳
Epoch / Step
单独记录；同一分支不同指标可能来自不同模型检查点，因此本表不是单一
Checkpoint 的联合指标表。

  分支     指标             日志最大值 达到最大值的位置（Epoch / Step）
  -------- -------------- ------------ ----------------------------------
  AV       R@1, IoU=0.3          60.53 34 / 78356
  AV       R@1, IoU=0.5          44.19 34 / 78356
  AV       R@1, IoU=0.7          28.72 43 / 100577
  AV       mean IoU              44.73 34 / 78356
  Audio    R@1, IoU=0.3          57.83 56 / 129814
  Audio    R@1, IoU=0.5          41.56 89 / 207001
  Audio    R@1, IoU=0.7          26.46 97 / 226883
  Audio    mean IoU              42.70 34 / 78356; 54 / 126306
  Visual   R@1, IoU=0.3          59.84 34 / 78356
  Visual   R@1, IoU=0.5          43.34 34 / 78356
  Visual   R@1, IoU=0.7          28.10 51 / 118119
  Visual   mean IoU              44.09 34 / 78356

### 以 R@1, IoU=0.7 为参照的各分支最高点

  -------------------------------------------------------------------------------------
  分支     Best Epoch  Best Step   R@1@0.3（同一   R@1@0.5（同一    R@1@0.7        mean
                                          Step）          Step）              IoU（同一
                                                                                 Step）
  -------- ---------- ---------- --------------- --------------- ---------- -----------
  AV               43     100577           59.82           43.93      28.72       44.29

  Audio            97     226883           56.88           41.22      26.46       42.04

  Visual           51     118119           59.56           43.10      28.10       44.05
  -------------------------------------------------------------------------------------

注意：这里的"日志最高点"表示该指标在本次训练评估日志中的最大观测值。配置中的
`test_path` 为 `null`，日志也未明确标注评估
split，因此不能仅凭这些记录将其认定为独立验证集或测试集的最佳结果。

## 4. 最后一次评估结果（Epoch 100）

以下结果均来自 Step 233900，即三份日志各自的最后一条记录。

  分支       Epoch     Step   R@1@0.3   R@1@0.5   R@1@0.7   mean IoU
  -------- ------- -------- --------- --------- --------- ----------
  AV           100   233900     56.54     41.04     26.41      41.42
  Audio        100   233900     56.87     41.29     26.32      41.95
  Visual       100   233900     57.19     41.47     26.75      42.23

## 5. Checkpoint 文件情况

本次上传文件中未发现命名为 `IMG_AN_I3D_0515_*.t7` 或其他可确认属于本组
ActivityNet 实验的模型权重。因此：

-   无法对本组 Checkpoint 执行 `torch.load`、State Dict 键结构、Tensor
    数量或 SHA256 权重校验。
-   无法确认日志中的各项最高点是否对应已保存的权重。
-   已上传的 `IMG_CS_I3D_0515_*.t7` 属于 Charades I3D 命名，不应混入
    ActivityNet 组。

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

-   AV 分支的日志最高 R@1@0.7 为 **28.72**，出现在 Epoch 43 / Step
    100577；同一 Step 的 mean IoU 为 44.29。
-   Audio 分支的日志最高 R@1@0.7 为 **26.46**，出现在 Epoch 97 / Step
    226883；同一 Step 的 mean IoU 为 42.04。
-   Visual 分支的日志最高 R@1@0.7 为 **28.10**，出现在 Epoch 51 / Step
    118119；同一 Step 的 mean IoU 为 44.05。
-   Epoch 100 / Step 233900 的 AV、Audio、Visual 分支 R@1@0.7 分别为
    **26.41**、**26.32**、**26.75**。
-   按本次日志的 R@1@0.7 全程最高值排序：AV（28.72）\> Visual（28.10）\>
    Audio（26.46）。
-   按 Epoch 100 的 R@1@0.7 排序：Visual（26.75）\> AV（26.41）\>
    Audio（26.32）。因此"全程最佳"与"最终 Epoch"给出的分支排序不同。
-   AV 的 R@1@0.7 最佳点出现在 Epoch 43，Visual 出现在 Epoch 51，而
    Audio 出现在 Epoch 97；尤其 Audio
    的最佳点较晚，模型选择时应同时检查统一选择规则，避免仅凭某个分支的单项最高值作结论。
-   三份日志均覆盖 Epoch 1--100，各有 200 条记录，共 600 条；三份日志的
    Step 序列完全一致。
-   当前上传文件中未发现本组 ActivityNet 对应
    Checkpoint；因此本组可登记为完成配置和日志统计的单种子 Baseline
    记录，但暂不能完成最佳权重核验。

## 8. 原始来源文件

  文件                     用途
  ------------------------ -------------------------------------------
  `configs(4).json`        本组实验配置
  `eval_results(4).txt`    AV 分支评估日志（按项目文件命名约定）
  `eval_resultsa(4).txt`   Audio 分支评估日志（按项目文件命名约定）
  `eval_resultsv(4).txt`   Visual 分支评估日志（按项目文件命名约定）

关键日志位置（文件内行号）：

-   AV：Epoch 34 / Step 78356 对应第 133--134 行；R@1@0.7 最佳点 Epoch
    43 / Step 100577 对应第 171--172 行；最终结果第 399--400 行。
-   Audio：Epoch 34 / Step 78356 对应第 133--134 行；R@1@0.7 最佳点
    Epoch 97 / Step 226883 对应第 387--388 行；最终结果第 399--400 行。
-   Visual：Epoch 34 / Step 78356 对应第 133--134 行；R@1@0.7 最佳点
    Epoch 51 / Step 118119 对应第 201--202 行；最终结果第 399--400 行。

------------------------------------------------------------------------

本报告由本次上传的配置文件与三份评估日志解析生成。原始文件未被修改；缺失的
Checkpoint 未作推断或补造。
