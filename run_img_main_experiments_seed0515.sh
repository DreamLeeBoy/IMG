#!/usr/bin/env bash

set -uo pipefail

# ============================================================
# IMG Main Experiments
# Random Seed: 0515
# Total Experiments: 4
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDX="${GPU_IDX:-0}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INIT_LR="${INIT_LR:-0.0005}"

SEED=515
SEED_TAG="0515"

STAMP="$(date +%Y%m%d_%H%M%S)"

OUTPUT_REL="experiments/main_seed_${SEED_TAG}_${STAMP}"
OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_REL"

mkdir -p "$OUTPUT_ROOT"

MANIFEST="$OUTPUT_ROOT/run_manifest.csv"
SUMMARY="$OUTPUT_ROOT/summary_main_results.csv"

printf 'experiment,task,visual_features,seed,training_status,log_file,model_dir,eval_file\n' > "$MANIFEST"

FAILED=0

# ============================================================
# 1. Environment Check
# ============================================================

echo "============================================================"
echo "IMG Main Experiments - Seed ${SEED_TAG}"
echo "============================================================"

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
    echo "[ERROR] main.py not found."
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python not found."
    exit 2
fi

# ============================================================
# 2. Required Files Check
# ============================================================

REQUIRED_FILES=(

    # Charades annotations
    data/dataset/charades/charades.json
    data/dataset/charades/charades_sta_train_qid.txt
    data/dataset/charades/charades_sta_test_qid.txt
    data/dataset/charades/charades_sta_train_tvr_format.jsonl
    data/dataset/charades/charades_sta_test_tvr_format.jsonl

    # ActivityNet annotations
    data/dataset/activitynet/train_qid.json
    data/dataset/activitynet/val_1_qid.json
    data/dataset/activitynet/val_2_qid.json

    # GloVe
    data/features/glove.840B.300d.txt

    # Charades feature metadata
    data/features/charades/feature_shapes_a.json
    data/features/charades/feature_shapes_i3d_v.json
    data/features/charades/feature_shapes_clip_v.json
    data/features/charades/feature_shapes_iv2_v.json

    # ActivityNet feature metadata
    data/features/activitynet/feature_shapes_a.json
    data/features/activitynet/feature_shapes_i3d_v.json

    # ActivityNet audio
    data/features/activitynet/audio/VGGish.pickle
)

MISSING=0

for item in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$ROOT_DIR/$item" ]]; then
        echo "[MISSING] $item"
        MISSING=1
    fi
done

# Check feature directories and ensure they contain files.

"$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])

checks = {
    "Charades I3D": ("data/features/charades/i3d_video", "*.npy"),
    "Charades PANNs": ("data/features/charades/audio", "*.npy"),

    "Charades CLIP": (
        "data/features/charades/clip_features/visual_features", "*.npy"
    ),

    "Charades SlowFast": (
        "data/features/charades/clip_features/slowfast_features", "*.npz"
    ),

    "Charades CLIP Text": (
        "data/features/charades/clip_features/text_features", "*.npy"
    ),

    "Charades InternVideo2": (
        "data/features/charades/iv2_features/visual_features_6b", "*.pt"
    ),

    "Charades LLaMA2": (
        "data/features/charades/iv2_features/llama2_txt", "*.pt"
    ),

    "ActivityNet I3D": (
        "data/features/activitynet/i3d_video", "*.npy"
    ),
}

failed = False

print("\n========== Feature Availability ==========")

for name, (directory, pattern) in checks.items():

    path = root / directory
    count = len(list(path.glob(pattern))) if path.exists() else 0

    print(f"{name}: {count}")

    if count == 0:
        print(f"[MISSING] {directory}")
        failed = True

sys.exit(1 if failed else 0)
PY

if [[ $? -ne 0 ]]; then
    echo "[ERROR] Feature directory check failed."
    exit 2
fi

if [[ "$MISSING" -ne 0 ]]; then
    echo "[ERROR] Required files are missing."
    exit 2
fi

# Check Python dependencies and NLTK tokenizer.

if ! "$PYTHON_BIN" - <<'PY'
import torch
import numpy
import nltk
import IPython
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from nltk.tokenize import word_tokenize

try:
    word_tokenize("this is a test", language="english")
except LookupError as e:
    print("[ERROR] NLTK tokenizer resource missing.")
    print(e)
    raise SystemExit(1)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
PY
then
    echo "[ERROR] Python environment check failed."
    exit 2
fi

echo
echo "[PASS] All required files and environment checks passed."

# ============================================================
# 3. Training Function
# ============================================================

run_experiment() {

    local experiment="$1"
    local task="$2"
    local visual="$3"
    local char_dim="$4"
    local model_name="$5"
    local visual_len="$6"
    local audio_len="$7"

    local EXP_REL="$OUTPUT_REL/$experiment"
    local EXP_DIR="$ROOT_DIR/$EXP_REL"

    local MODEL_BASE_REL="$EXP_REL/checkpoints"

    local MODEL_FOLDER="${model_name}_${task}_${visual}_v_${visual_len}_a_${audio_len}"

    local MODEL_DIR_REL="$MODEL_BASE_REL/$MODEL_FOLDER/model"

    local LOG_FILE="$EXP_DIR/train.log"

    local EVAL_FILE="$ROOT_DIR/$MODEL_DIR_REL/eval_results.txt"

    mkdir -p "$EXP_DIR"

    echo
    echo "============================================================"
    echo "START: $experiment"
    echo "============================================================"

    echo "Task          : $task"
    echo "Visual        : $visual"
    echo "Seed          : $SEED_TAG"
    echo "Epochs        : $EPOCHS"
    echo "Batch size    : $BATCH_SIZE"
    echo "Learning rate : $INIT_LR"
    echo "GPU           : $GPU_IDX"
    echo "Model         : $model_name"
    echo "Log           : $LOG_FILE"

    local status="FAILED"

    if PYTHONHASHSEED="$SEED" "$PYTHON_BIN" -u main.py \
        --task "$task" \
        --mode train \
        --gpu_idx "$GPU_IDX" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --init_lr "$INIT_LR" \
        --visual_features "$visual" \
        --char_dim "$char_dim" \
        --model_name "$model_name" \
        --model_dir "$MODEL_BASE_REL" \
        2>&1 | tee "$LOG_FILE"; then

        status="SUCCESS"

        echo
        echo "[SUCCESS] $experiment"

    else

        status="FAILED"
        FAILED=1

        echo
        echo "[FAILED] $experiment"
        echo "Check: $LOG_FILE"

    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment" \
        "$task" \
        "$visual" \
        "$SEED" \
        "$status" \
        "$LOG_FILE" \
        "$MODEL_DIR_REL" \
        "$EVAL_FILE" >> "$MANIFEST"

}

# ============================================================
# 4. Four Main Experiments
# ============================================================

echo
echo "Starting four main experiments..."

# ------------------------------------------------------------
# Experiment 1
# Charades-STA + I3D + PANNs
# ------------------------------------------------------------

run_experiment \
    "01_charades_i3d" \
    "charades" \
    "i3d" \
    50 \
    "IMG_CS_I3D_0515" \
    128 \
    256

# ------------------------------------------------------------
# Experiment 2
# Charades-STA + CLIP + SlowFast + PANNs
# ------------------------------------------------------------

run_experiment \
    "02_charades_clip_sf" \
    "charades" \
    "clip+sf" \
    50 \
    "IMG_CS_CLIPSF_0515" \
    128 \
    256

# ------------------------------------------------------------
# Experiment 3
# Charades-STA + InternVideo2 + PANNs + LLaMA2
# ------------------------------------------------------------

run_experiment \
    "03_charades_iv2" \
    "charades" \
    "iv2" \
    50 \
    "IMG_CS_IV2_0515" \
    64 \
    128

# ------------------------------------------------------------
# Experiment 4
# ActivityNet + I3D + VGGish
# ------------------------------------------------------------

run_experiment \
    "04_activitynet_i3d" \
    "activitynet" \
    "i3d" \
    100 \
    "IMG_AN_I3D_0515" \
    128 \
    71

# ============================================================
# 5. Extract Best Results
# ============================================================

echo
echo "============================================================"
echo "Extracting experimental results..."
echo "============================================================"

"$PYTHON_BIN" - \
    "$ROOT_DIR" \
    "$MANIFEST" \
    "$SUMMARY" <<'PY'

import csv
import os
import re
import sys

root, manifest, summary = sys.argv[1:4]

pattern = re.compile(
    r"Epoch\s+(\d+),\s*Step\s+(\d+):\s*\n"
    r"Rank@1,\s*IoU=0\.3:\s*([\d.]+)\s+"
    r"Rank@1,\s*IoU=0\.5:\s*([\d.]+)\s+"
    r"Rank@1,\s*IoU=0\.7:\s*([\d.]+)\s+"
    r"mean IoU:\s*([\d.]+)",
    re.MULTILINE
)

results = []

with open(manifest, newline="", encoding="utf-8") as f:

    for run in csv.DictReader(f):

        result_file = run["eval_file"]

        best = None

        if os.path.isfile(result_file):

            with open(
                result_file,
                encoding="utf-8",
                errors="replace"
            ) as rf:

                matches = list(pattern.finditer(rf.read()))

            if matches:
                best = max(
                    matches,
                    key=lambda m: float(m.group(5))
                )

        row = {
            "experiment": run["experiment"],
            "task": run["task"],
            "visual_features": run["visual_features"],
            "seed": run["seed"],
            "training_status": run["training_status"],

            "best_epoch": "",
            "best_step": "",

            "R1@0.3": "",
            "R1@0.5": "",
            "R1@0.7": "",
            "mIoU": "",

            "result_status": "NO_RESULTS",

            "model_dir": run["model_dir"],
            "log_file": run["log_file"],
        }

        if best:

            row.update({
                "best_epoch": int(best.group(1)),
                "best_step": int(best.group(2)),

                "R1@0.3": float(best.group(3)),
                "R1@0.5": float(best.group(4)),
                "R1@0.7": float(best.group(5)),
                "mIoU": float(best.group(6)),

                "result_status": "AVAILABLE",
            })

        results.append(row)

columns = [
    "experiment",
    "task",
    "visual_features",
    "seed",
    "training_status",
    "best_epoch",
    "best_step",
    "R1@0.3",
    "R1@0.5",
    "R1@0.7",
    "mIoU",
    "result_status",
    "model_dir",
    "log_file",
]

with open(
    summary,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)

print("\n========== Main Experiment Results ==========")

for row in results:

    print("\n", row["experiment"])
    print("Training:", row["training_status"])
    print("Results:", row["result_status"])

    if row["result_status"] == "AVAILABLE":

        print("Best epoch:", row["best_epoch"])
        print("R1@0.3:", row["R1@0.3"])
        print("R1@0.5:", row["R1@0.5"])
        print("R1@0.7:", row["R1@0.7"])
        print("mIoU:", row["mIoU"])

print("\nSummary saved:", summary)

PY

# ============================================================
# 6. Final Report
# ============================================================

echo
echo "============================================================"
echo "ALL FOUR MAIN EXPERIMENTS HAVE BEEN ATTEMPTED"
echo "============================================================"

echo "Seed       : $SEED_TAG"
echo "Output     : $OUTPUT_ROOT"
echo "Manifest   : $MANIFEST"
echo "Summary    : $SUMMARY"

if [[ "$FAILED" -ne 0 ]]; then

    echo
    echo "[WARNING] Some experiments failed."
    echo "Please inspect their train.log files."

    exit 1

fi

echo
echo "[SUCCESS] All four training jobs completed."
echo "============================================================"
