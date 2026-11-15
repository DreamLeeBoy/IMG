#!/usr/bin/env bash
# Run the four main IMG experiments with random seed 0515 (decimal 515).
#
# Experiments:
#   1) Charades-STA + I3D
#   2) ActivityNet + I3D
#   3) Charades-STA + CLIP+SlowFast
#   4) Charades-STA + InternVideo2
#
# Charades-AudioMatter is intentionally NOT included.
#
# Run:
#   conda activate IMG
#   bash run_img_main_experiments_seed0515.sh
#
# Optional overrides:
#   GPU_IDX=1 bash run_img_main_experiments_seed0515.sh
#   EPOCHS=1 BATCH_SIZE=2 bash run_img_main_experiments_seed0515.sh
#   SEED=0515 bash run_img_main_experiments_seed0515.sh

# Deliberately do NOT enable `set -e`: failed experiments must not
# prevent the remaining experiments from being attempted.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDX="${GPU_IDX:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INIT_LR="${INIT_LR:-0.0005}"
SAVE_DIR="${SAVE_DIR:-datasets}"

# ============================================================
# Gaussian-DCL settings (edit these values before running)
# ============================================================
CDL_TYPE="gaussian"
W_CDL="2.0"
GAUSSIAN_SIGMA="0.3"

# Keep the label "0515" for folder names, but pass integer 515 to Python.
SEED_LABEL="${SEED:-0515}"
if [[ ! "$SEED_LABEL" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SEED must contain digits only; received: $SEED_LABEL" >&2
    exit 2
fi
SEED_VALUE=$((10#$SEED_LABEL))

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
    echo "[ERROR] main.py not found. Put this script in the IMG repository root." >&2
    exit 2
fi
if [[ ! -f "$ROOT_DIR/model/losses.py" ]]; then
    echo "[ERROR] model/losses.py not found." >&2
    exit 2
fi

# Fail early if the checked-out code does not contain the Gaussian-DCL implementation.
for required_token in "--w_cdl" "--cdl_type" "--gaussian_sigma" "gaussian_cdl_loss"; do
    if ! grep -Fq -- "$required_token" "$ROOT_DIR/main.py"; then
        echo "[ERROR] main.py is missing Gaussian-DCL interface: $required_token" >&2
        exit 2
    fi
done
if ! grep -Fq "def gaussian_cdl_loss" "$ROOT_DIR/model/losses.py"; then
    echo "[ERROR] model/losses.py does not define gaussian_cdl_loss." >&2
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable '$PYTHON_BIN' not found. Activate the IMG environment first." >&2
    exit 2
fi

if ! "$PYTHON_BIN" - "$GAUSSIAN_SIGMA" "$W_CDL" <<'PY'
import math
import sys
sigma = float(sys.argv[1])
weight = float(sys.argv[2])
if not math.isfinite(sigma) or sigma <= 0:
    raise SystemExit("GAUSSIAN_SIGMA must be finite and greater than 0.")
if not math.isfinite(weight) or weight < 0:
    raise SystemExit("W_CDL must be finite and greater than or equal to 0.")
PY
then
    echo "[ERROR] Invalid Gaussian-DCL hyperparameter; no experiment was started." >&2
    exit 2
fi

# Check PyTorch and the requested CUDA device before starting data loading.
if ! "$PYTHON_BIN" - "$GPU_IDX" <<'PY'
import sys
import torch
gpu = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active Python environment.")
if gpu < 0 or gpu >= torch.cuda.device_count():
    raise SystemExit(
        f"GPU_IDX={gpu} is invalid; visible GPU count={torch.cuda.device_count()}."
    )
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; "
      f"GPU {gpu}={torch.cuda.get_device_name(gpu)}")
PY
then
    echo "[ERROR] PyTorch/CUDA preflight failed; no experiment was started." >&2
    exit 2
fi

# Resolve the dataset-cache directory for the GloVe/cache pre-check.
if [[ "$SAVE_DIR" = /* ]]; then
    CACHE_ROOT="$SAVE_DIR"
else
    CACHE_ROOT="$ROOT_DIR/$SAVE_DIR"
fi

# Files/directories needed by the four requested configurations.
REQUIRED_PATHS=(
    # Charades-STA annotations
    "data/dataset/charades/charades.json"
    "data/dataset/charades/charades_sta_train_qid.txt"
    "data/dataset/charades/charades_sta_test_qid.txt"

    # Charades I3D + PANNs
    "data/features/charades/i3d_video"
    "data/features/charades/audio"
    "data/features/charades/feature_shapes_i3d_v.json"
    "data/features/charades/feature_shapes_a.json"

    # Charades CLIP + SlowFast + CLIP text
    "data/features/charades/clip_features/visual_features"
    "data/features/charades/clip_features/slowfast_features"
    "data/features/charades/clip_features/text_features"
    "data/features/charades/feature_shapes_clip_v.json"

    # Charades InternVideo2 + LLaMA2 text
    "data/dataset/charades/charades_sta_train_tvr_format.jsonl"
    "data/dataset/charades/charades_sta_test_tvr_format.jsonl"
    "data/features/charades/iv2_features/visual_features_6b"
    "data/features/charades/iv2_features/llama2_txt"
    "data/features/charades/feature_shapes_iv2_v.json"

    # ActivityNet I3D + VGGish
    "data/dataset/activitynet/train_qid.json"
    "data/dataset/activitynet/val_1_qid.json"
    "data/dataset/activitynet/val_2_qid.json"
    "data/features/activitynet/i3d_video"
    "data/features/activitynet/audio/VGGish.pickle"
    "data/features/activitynet/feature_shapes_i3d_v.json"
    "data/features/activitynet/feature_shapes_a.json"
)

MISSING=0
for item in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$ROOT_DIR/$item" ]]; then
        echo "[MISSING] $item" >&2
        MISSING=1
    fi
done

# These are the cache filenames used by IMG's data_gen.py/data_gen_iv2.py.
CACHE_FILES=(
    "charades_i3d_128.pkl"
    "activitynet_i3d_128.pkl"
    "charades_clip+sf_128.pkl"
    "charades_iv2_64.pkl"
)

NEED_GLOVE=0
for cache_file in "${CACHE_FILES[@]}"; do
    if [[ ! -f "$CACHE_ROOT/$cache_file" ]]; then
        NEED_GLOVE=1
    fi
done

if [[ "$NEED_GLOVE" -eq 1 && ! -f "$ROOT_DIR/data/features/glove.840B.300d.txt" ]]; then
    echo "[MISSING] data/features/glove.840B.300d.txt (needed to build one or more dataset caches)" >&2
    MISSING=1
fi

if [[ "$MISSING" -ne 0 ]]; then
    echo "[ERROR] Required files/directories are missing. Fix the paths above before training." >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_REL="experiments/img_gaussian_dcl_v1_seed_${SEED_LABEL}_${STAMP}"
OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_REL"
mkdir -p "$OUTPUT_ROOT"

MANIFEST="$OUTPUT_ROOT/run_manifest.csv"
printf 'experiment,task,visual_features,seed,model_name,cdl_type,w_cdl,gaussian_sigma,training_status,log_file,model_dir,tensorboard_dir\n' > "$MANIFEST"

echo "============================================================"
echo "IMG main experiments"
echo "Repository   : $ROOT_DIR"
echo "Python       : $PYTHON_BIN"
echo "GPU index    : $GPU_IDX"
echo "Epochs       : $EPOCHS"
echo "Batch size   : $BATCH_SIZE"
echo "Learning rate: $INIT_LR"
echo "CDL type     : $CDL_TYPE"
echo "CDL weight   : $W_CDL"
echo "Gaussian sigma: $GAUSSIAN_SIGMA"
echo "Random seed  : $SEED_LABEL (integer passed to Python: $SEED_VALUE)"
echo "Save dir     : $SAVE_DIR"
echo "Output root  : $OUTPUT_ROOT"
echo "Experiments  : 4"
echo "============================================================"

FAILED=0
ATTEMPTED=0

run_experiment() {
    local experiment="$1"
    local task="$2"
    local visual_features="$3"
    local model_name="$4"
    local char_dim="$5"
    local visual_len="$6"
    local audio_len="$7"

    local exp_rel="$OUTPUT_REL/$experiment"
    local exp_dir="$ROOT_DIR/$exp_rel"
    local model_base_rel="$exp_rel/checkpoints"
    local model_folder="${model_name}_${task}_${visual_features}_v_${visual_len}_a_${audio_len}"
    local model_dir_rel="$model_base_rel/$model_folder/model"
    local tb_dir_rel="$exp_rel/tensorboard"
    local log_file="$exp_dir/train.log"
    local tb_source="$ROOT_DIR/runs/$model_base_rel/$model_folder"

    mkdir -p "$exp_dir"
    ATTEMPTED=$((ATTEMPTED + 1))

    echo
    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] START $experiment"
    echo "Task           : $task"
    echo "Visual feature : $visual_features"
    echo "Model name     : $model_name"
    echo "Seed           : $SEED_VALUE"
    echo "Experiment dir : $exp_dir"
    echo "Log            : $log_file"
    echo "------------------------------------------------------------"

    if PYTHONHASHSEED="$SEED_VALUE" "$PYTHON_BIN" -u main.py \
        --task "$task" \
        --mode train \
        --gpu_idx "$GPU_IDX" \
        --seed "$SEED_VALUE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --init_lr "$INIT_LR" \
        --visual_features "$visual_features" \
        --char_dim "$char_dim" \
        --model_name "$model_name" \
        --model_dir "$model_base_rel" \
        --save_dir "$SAVE_DIR" \
        --w_cdl "$W_CDL" \
        --cdl_type "$CDL_TYPE" \
        --gaussian_sigma "$GAUSSIAN_SIGMA" \
        2>&1 | tee "$log_file"; then
        status="SUCCESS"
        echo "[$(date '+%F %T')] FINISH $experiment"
    else
        status="FAILED"
        FAILED=1
        echo "[$(date '+%F %T')] FAILED $experiment; see $log_file" >&2
    fi

    # main.py writes TensorBoard under runs/<model_dir>/<model-folder>.
    # Move it into the corresponding experiment directory.
    if [[ -d "$tb_source" ]]; then
        mkdir -p "$ROOT_DIR/$tb_dir_rel"
        mv "$tb_source" "$ROOT_DIR/$tb_dir_rel/"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment" "$task" "$visual_features" "$SEED_VALUE" "$model_name" \
        "$CDL_TYPE" "$W_CDL" "$GAUSSIAN_SIGMA" "$status" \
        "$log_file" "$model_dir_rel" "$tb_dir_rel" >> "$MANIFEST"
}

# 1) Charades-STA + I3D + PANNs: v_128, a_256.
run_experiment \
    "charades_i3d" "charades" "i3d" "IMGCS${SEED_LABEL}" "50" "128" "256"

# 2) ActivityNet + I3D + VGGish: v_128, a_71.
run_experiment \
    "activitynet_i3d" "activitynet" "i3d" "IMGAN${SEED_LABEL}" "100" "128" "71"

# 3) Charades-STA + CLIP + SlowFast + PANNs: v_128, a_256.
run_experiment \
    "charades_clip_sf" "charades" "clip+sf" "IMGCS${SEED_LABEL}" "50" "128" "256"

# 4) Charades-STA + InternVideo2 + PANNs: v_64, a_128.
run_experiment \
    "charades_iv2" "charades" "iv2" "IMGCS${SEED_LABEL}" "50" "64" "128"

SUMMARY="$OUTPUT_ROOT/summary_best_checkpoint.csv"
AGGREGATE="$OUTPUT_ROOT/summary_mean_std.csv"

if ! "$PYTHON_BIN" - "$ROOT_DIR" "$MANIFEST" "$SUMMARY" "$AGGREGATE" <<'PY'
import csv
import os
import re
import statistics
import sys
from collections import defaultdict

root, manifest_path, summary_path, aggregate_path = sys.argv[1:5]

# Format written by util/runner_utils.py -> eval_test().
pattern = re.compile(
    r"Epoch\s+(\d+),\s*Step\s+(\d+):\s*\n"
    r"\s*Rank@1,\s*IoU=0\.3:\s*([\d.]+)\s+"
    r"Rank@1,\s*IoU=0\.5:\s*([\d.]+)\s+"
    r"Rank@1,\s*IoU=0\.7:\s*([\d.]+)\s+"
    r"mean IoU:\s*([\d.]+)",
    re.MULTILINE,
)

details = []
with open(manifest_path, newline="", encoding="utf-8") as f:
    for run in csv.DictReader(f):
        result_file = os.path.join(root, run["model_dir"], "eval_results.txt")
        checkpoint_dir = os.path.join(root, run["model_dir"])
        checkpoint_count = (
            len([x for x in os.listdir(checkpoint_dir) if x.endswith(".t7")])
            if os.path.isdir(checkpoint_dir) else 0
        )

        best = None
        if os.path.isfile(result_file):
            with open(result_file, encoding="utf-8", errors="replace") as rf:
                matches = list(pattern.finditer(rf.read()))
            if matches:
                # main.py keeps the checkpoint with the highest R1@0.7.
                best = max(matches, key=lambda m: float(m.group(5)))

        row = {
            "experiment": run["experiment"],
            "task": run["task"],
            "visual_features": run["visual_features"],
            "seed": int(run["seed"]),
            "model_name": run["model_name"],
            "cdl_type": run["cdl_type"],
            "w_cdl": float(run["w_cdl"]),
            "gaussian_sigma": float(run["gaussian_sigma"]),
            "training_status": run["training_status"],
            "best_epoch_by_R1@0.7": "",
            "best_step_by_R1@0.7": "",
            "R1@0.3": "",
            "R1@0.5": "",
            "R1@0.7": "",
            "mIoU": "",
            "checkpoint_count": checkpoint_count,
            "model_dir": run["model_dir"],
            "log_file": run["log_file"],
            "result_status": "NO_RESULTS",
        }

        if best:
            row.update({
                "best_epoch_by_R1@0.7": int(best.group(1)),
                "best_step_by_R1@0.7": int(best.group(2)),
                "R1@0.3": float(best.group(3)),
                "R1@0.5": float(best.group(4)),
                "R1@0.7": float(best.group(5)),
                "mIoU": float(best.group(6)),
                "result_status": "AVAILABLE",
            })

        details.append(row)

columns = [
    "experiment", "task", "visual_features", "seed", "model_name",
    "cdl_type", "w_cdl", "gaussian_sigma", "training_status",
    "best_epoch_by_R1@0.7", "best_step_by_R1@0.7",
    "R1@0.3", "R1@0.5", "R1@0.7", "mIoU", "checkpoint_count",
    "model_dir", "log_file", "result_status",
]

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(details)

metrics = ["R1@0.3", "R1@0.5", "R1@0.7", "mIoU"]
by_experiment = defaultdict(list)
for row in details:
    if row["training_status"] == "SUCCESS" and row["result_status"] == "AVAILABLE":
        by_experiment[row["experiment"]].append(row)

with open(aggregate_path, "w", newline="", encoding="utf-8-sig") as f:
    fields = ["experiment", "successful_runs"]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std_sample"])
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for experiment in (
        "charades_i3d",
        "activitynet_i3d",
        "charades_clip_sf",
        "charades_iv2",
    ):
        rows = by_experiment.get(experiment, [])
        out = {"experiment": experiment, "successful_runs": len(rows)}
        for metric in metrics:
            vals = [float(r[metric]) for r in rows if r[metric] != ""]
            out[f"{metric}_mean"] = round(statistics.mean(vals), 4) if vals else ""
            # One seed cannot provide a meaningful sample standard deviation.
            out[f"{metric}_std_sample"] = (
                round(statistics.stdev(vals), 4) if len(vals) >= 2 else ""
            )
        writer.writerow(out)

print(f"\nPer-experiment summary : {summary_path}")
print(f"Aggregate summary      : {aggregate_path}")
print("Best checkpoint selection: maximum R1@0.7, matching main.py.")
print("Note: with one seed (0515), sample standard deviation is left blank.")
PY
then
    FAILED=1
    echo "[ERROR] Result-summary generation failed." >&2
fi

echo
echo "============================================================"
echo "Experiments attempted  : $ATTEMPTED / 4"
echo "Output root            : $OUTPUT_ROOT"
echo "Run manifest           : $MANIFEST"
echo "Best-checkpoint summary: $SUMMARY"
echo "Mean/std summary       : $AGGREGATE"
echo "============================================================"

if [[ "$ATTEMPTED" -ne 4 ]]; then
    echo "[ERROR] Only $ATTEMPTED of 4 experiments were attempted." >&2
    FAILED=1
fi

if [[ "$FAILED" -ne 0 ]]; then
    echo "[WARNING] One or more experiments failed or were not attempted. Check each train.log and the summary CSV." >&2
    exit 1
fi

echo "[SUCCESS] All four Gaussian-DCL experiments completed successfully."
