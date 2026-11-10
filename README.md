<div align="center">
cd /data/IMG_rebuild




# 丢弃 main.py 的本地修改
git restore main.py

# 切换到新分支
git switch feature/uniform-cdl

# 更新到远程最新版本
git pull --ff-only origin feature/uniform-cdl

# 验证
git status




## Training

Baseline training (CDL disabled):

```bash
python main.py --task charades --mode train --gpu_idx 0 --w_cdl 0.0
```

Train IMG with the paper's Uniform Complementary Dissimilarity Loss (CDL):

```bash
python main.py --task charades --mode train --gpu_idx 0 --w_cdl 1.0
```

The default `--w_cdl` is `0.0` to keep existing baseline runs unchanged. Set it to
`1.0` to enable the Uniform CDL objective (the default weight used by the official
CDL ImageNet example); other non-negative values can be used for ablations.

For each of IMG's audiovisual, visual, and audio localization heads, Uniform CDL is
computed independently for the start and end logits and added to the original
cross-entropy objective:

```
L_total = L_IMG + w_cdl * (L_CDL_start + L_CDL_end)
```

IMG uses variable-length video sequences within a padded batch. The implementation
therefore normalizes the one-cold target over each sample's valid temporal positions
only; padded positions are excluded from the complementary softmax and loss. The
audio-head CDL retains IMG's existing importance-aware `boola` weighting, matching
the audio-head CE/KD training path. All original IMG losses and model components
remain enabled.

Run the unit tests for the loss implementation with:

```bash
python -m unittest tests.test_uniform_cdl
```

## Inference


```bash
python main.py --task <charades|activitynet|charadesAM> --mode test --gpu_idx <GPU INDEX>
```

Change the config `model_name` in `main.py` to the model_name of `your checkpoint`.



