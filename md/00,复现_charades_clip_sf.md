# Baseline-02 实验统计报告：02_charades_clip_sf

> 实验标识：`02_charades_clip_sf`\
> 模型名称：`IMG_CS_CLIPSF_0515`\
> 统计依据：本次上传的 `configs(1).json`、三份评估日志。\
> 统计原则：保留日志原始数值；每项指标单独统计全日志最大值；不把不同
> Epoch 的指标拼接成同一模型结果。

## 1. 实验基本信息

  项目                          记录
  ----------------------------- -----------------------------
  实验编号                      `02_charades_clip_sf`
  实验阶段                      Baseline / 论文忠实复现阶段
  数据任务                      `charades`
  视觉特征                      `clip+sf`（CLIP + SF）
  随机种子                      `515`
  模型名称                      `IMG_CS_CLIPSF_0515`
  训练 Epoch                    100
  Batch Size                    16
  Initial Learning Rate         0.0005
  计划训练 Steps                77,600
  日志数量                      3 份（AV、Audio、Visual）
  每份日志评估记录              200 条
  总评估记录                    600 条
  评估频率                      每个 Epoch 记录 2 次
  当前上传的本实验 Checkpoint   未发现对应权重文件

说明：当前上传文件中存在的 `.t7` 权重名称属于上一组 `IMG_CS_I3D_0515`
实验，不属于本组 `IMG_CS_CLIPSF_0515`，因此不纳入本组 Checkpoint 统计。

## 2. 训练配置摘要

  配置项                    值
  --------------------- ------
  `video_feature_dim`     2816
  `audio_feature_dim`     2048
  `text_feature_dim`       512
  `dim`                    128
  `num_heads`                8
  `layers`                   3
  `slot_layers`              3
  `drop_rate`              0.2
  `max_pos_len`            128
  `max_pos_len_a`          256
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
  "max_pos_len": 128,
  "max_pos_len_a": 256,
  "mode": "train",
  "model_dir": "experiments/main_seed_0515_20261102_144018/02_charades_clip_sf/checkpoints",
  "model_name": "IMG_CS_CLIPSF_0515",
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
  "text_feature_dim": 512,
  "tmp": 3,
  "video_feature_dim": 2816,
  "visual_features": "clip+sf",
  "warm_epoch": 20,
  "warmup_proportion": 0.0,
  "word_dim": 300,
  "word_size": 1271
}
```

## 3. 全训练过程各指标最佳值

下表对每个分支的 200 条评估记录分别取单项指标最大值。每行的 Epoch/Step
是该项指标自身达到最大值的位置；同一分支不同指标可能来自不同
Epoch，因此本表不是单一 Checkpoint 的联合指标表。

  分支     指标             日志最大值   Epoch    Step
  -------- -------------- ------------ ------- -------
  AV       R@1, IoU=0.3          75.35      14   10864
  AV       R@1, IoU=0.5          60.35      17   12804
  AV       R@1, IoU=0.7          43.01      17   12804
  AV       mean IoU              54.97      17   12804
  Audio    R@1, IoU=0.3          60.99      46   35696
  Audio    R@1, IoU=0.5          47.04      46   35696
  Audio    R@1, IoU=0.7          29.27      83   64408
  Audio    mean IoU              43.12      64   49664
  Visual   R@1, IoU=0.3          74.01      14   10864
  Visual   R@1, IoU=0.5          59.14      28   21728
  Visual   R@1, IoU=0.7          41.51      28   21728
  Visual   mean IoU              53.74      28   21728

### 以 R@1, IoU=0.7 为参照的各分支最高点

  --------------------------------------------------------------------------------------------
  分支        Best Epoch    Best Step      R@1@0.7         mean 对应权重是否在本次上传文件中
                                                      IoU（同一 
                                                         Step） 
  --------- ------------ ------------ ------------ ------------ ------------------------------
  AV                  17        12804        43.01        54.97 否

  Audio               83        64408        29.27        42.55 否

  Visual              28        21728        41.51        53.74 否
  --------------------------------------------------------------------------------------------

注意：这里的"日志最高点"表示该指标在训练过程评估日志中的最大观测值。当前配置中的
`test_path` 为 `null`，日志也没有明确标明评估
split，因此不能仅凭这些记录将其认定为独立验证集或测试集的最佳结果。

## 4. 最后一次评估结果（Epoch 100）

以下结果均来自 Step 77600，即三份日志各自的最后一条记录。

  分支       Epoch    Step   R@1@0.3   R@1@0.5   R@1@0.7   mean IoU
  -------- ------- ------- --------- --------- --------- ----------
  AV           100   77600     70.30     57.90     38.90      51.39
  Audio        100   77600     59.62     46.18     28.79      42.34
  Visual       100   77600     69.41     56.40     38.04      50.76

## 5. Checkpoint 文件情况

本次上传文件中未发现命名为 `IMG_CS_CLIPSF_0515_*.t7` 的模型权重。因此：

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

-   `02_charades_clip_sf` 的三份评估日志均覆盖 Epoch 1--100，共 200
    条记录/分支，三份日志的评估 Step 序列一致。
-   AV 分支的日志最高 R@1@0.7 为 **43.01**，出现在 Epoch 17 / Step
    12804；同一 Step 的 mean IoU 为 54.97。
-   Visual 分支的日志最高 R@1@0.7 为 **41.51**，出现在 Epoch 28 / Step
    21728；同一 Step 的 mean IoU 为 53.74。
-   Audio 分支的日志最高 R@1@0.7 为 **29.27**，出现在 Epoch 83 / Step
    64408；同一 Step 的 mean IoU 为 42.55。
-   Epoch 100 / Step 77600 的 AV、Audio、Visual 分支 R@1@0.7 分别为
    38.90、28.79、38.04。
-   当前上传文件中未发现本组 CLIP+SF 对应
    Checkpoint；因此本组可登记为完成日志统计的单种子 Baseline
    记录，但暂不能完成最佳权重核验。

## 8. 原始来源文件

  文件                     用途
  ------------------------ -------------------------------------------
  `configs(1).json`        本组实验配置
  `eval_results(1).txt`    AV 分支评估日志（按项目文件命名约定）
  `eval_resultsa(1).txt`   Audio 分支评估日志（按项目文件命名约定）
  `eval_resultsv(1).txt`   Visual 分支评估日志（按项目文件命名约定）

------------------------------------------------------------------------

本报告由本次上传的配置文件与三份评估日志解析生成。原始文件未被修改；缺失的
Checkpoint 未作推断或补造。
