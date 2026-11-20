#!/usr/bin/env bash
# Charades-STA + InternVideo2 Gaussian-CDL grid: 3 weights x 4 sigmas = 12 runs.
# Run: conda activate IMG && bash run_charades_iv2_gaussian_cdl_grid_seed0515.sh
# Optional: GPU_IDX=1 EPOCHS=1 BATCH_SIZE=2 bash run_charades_iv2_gaussian_cdl_grid_seed0515.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDX="${GPU_IDX:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INIT_LR="${INIT_LR:-0.0005}"
SAVE_DIR="${SAVE_DIR:-datasets}"

# Edit these arrays to change the experiment grid.
CDL_TYPE="gaussian"
W_CDL_VALUES=(0.5 1.0 1.5)
GAUSSIAN_SIGMA_VALUES=(0.05 0.1 0.2 0.3)

SEED_LABEL="${SEED:-0515}"
if [[ ! "$SEED_LABEL" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SEED must contain digits only: $SEED_LABEL" >&2
    exit 2
fi
SEED_VALUE=$((10#$SEED_LABEL))

for file in main.py model/losses.py; do
    [[ -f "$ROOT_DIR/$file" ]] || { echo "[ERROR] Missing $file" >&2; exit 2; }
done
for token in "--w_cdl" "--cdl_type" "--gaussian_sigma" "gaussian_cdl_loss"; do
    grep -Fq -- "$token" "$ROOT_DIR/main.py" || {
        echo "[ERROR] main.py lacks Gaussian-CDL interface: $token" >&2; exit 2;
    }
done
grep -Fq "def gaussian_cdl_loss" "$ROOT_DIR/model/losses.py" || {
    echo "[ERROR] gaussian_cdl_loss not found in model/losses.py" >&2; exit 2;
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[ERROR] Python executable '$PYTHON_BIN' not found; activate IMG first." >&2; exit 2;
}

# Validate all parameter values before starting any run.
"$PYTHON_BIN" - "${W_CDL_VALUES[@]}" -- "${GAUSSIAN_SIGMA_VALUES[@]}" <<'PY'
import math, sys
args = sys.argv[1:]
split = args.index("--")
weights = [float(x) for x in args[:split]]
sigmas = [float(x) for x in args[split + 1:]]
assert weights and sigmas, "Both parameter arrays must be non-empty."
assert all(math.isfinite(x) and x >= 0 for x in weights), "Invalid w_cdl value."
assert all(math.isfinite(x) and x > 0 for x in sigmas), "Invalid gaussian_sigma value."
print(f"Validated {len(weights)} x {len(sigmas)} = {len(weights)*len(sigmas)} runs.")
PY
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Invalid hyperparameter grid." >&2
    exit 2
fi

# CUDA preflight.
"$PYTHON_BIN" - "$GPU_IDX" <<'PY'
import sys, torch
gpu = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
if gpu < 0 or gpu >= torch.cuda.device_count():
    raise SystemExit(f"GPU_IDX={gpu} invalid; visible GPU count={torch.cuda.device_count()}.")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU {gpu}={torch.cuda.get_device_name(gpu)}")
PY
if [[ $? -ne 0 ]]; then
    echo "[ERROR] CUDA preflight failed; no experiment was started." >&2
    exit 2
fi

# Required Charades-STA + InternVideo2 files only.
if [[ "$SAVE_DIR" = /* ]]; then CACHE_ROOT="$SAVE_DIR"; else CACHE_ROOT="$ROOT_DIR/$SAVE_DIR"; fi
REQUIRED_PATHS=(
    "data/dataset/charades/charades.json"
    "data/dataset/charades/charades_sta_train_qid.txt"
    "data/dataset/charades/charades_sta_test_qid.txt"
    "data/dataset/charades/charades_sta_train_tvr_format.jsonl"
    "data/dataset/charades/charades_sta_test_tvr_format.jsonl"
    "data/features/charades/iv2_features/visual_features_6b"
    "data/features/charades/iv2_features/llama2_txt"
    "data/features/charades/audio"
    "data/features/charades/feature_shapes_iv2_v.json"
    "data/features/charades/feature_shapes_a.json"
)
MISSING=0
for item in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$ROOT_DIR/$item" ]]; then echo "[MISSING] $item" >&2; MISSING=1; fi
done
if [[ ! -f "$CACHE_ROOT/charades_iv2_64.pkl" && ! -f "$ROOT_DIR/data/features/glove.840B.300d.txt" ]]; then
    echo "[MISSING] data/features/glove.840B.300d.txt (needed to build charades_iv2_64.pkl)" >&2
    MISSING=1
fi
if [[ "$MISSING" -ne 0 ]]; then
    echo "[ERROR] Required Charades-STA/IV2 files are missing." >&2
    exit 2
fi

TASK="charades"
VISUAL_FEATURES="iv2"
MODEL_NAME="IMGCS${SEED_LABEL}"
CHAR_DIM="50"
VISUAL_LEN="64"
AUDIO_LEN="128"
TOTAL_RUNS=$((${#W_CDL_VALUES[@]} * ${#GAUSSIAN_SIGMA_VALUES[@]}))

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_REL="experiments/img_gaussian_cdl_charades_iv2_grid_seed_${SEED_LABEL}_${STAMP}"
OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_REL"
mkdir -p "$OUTPUT_ROOT"
MANIFEST="$OUTPUT_ROOT/run_manifest.csv"
printf 'experiment,task,visual_features,seed,model_name,cdl_type,w_cdl,gaussian_sigma,training_status,log_file,model_dir,tensorboard_dir\n' > "$MANIFEST"

echo "============================================================"
echo "Charades-STA + InternVideo2 Gaussian-CDL grid"
echo "GPU=$GPU_IDX | seed=$SEED_VALUE | epochs=$EPOCHS | batch_size=$BATCH_SIZE"
echo "w_cdl: ${W_CDL_VALUES[*]}"
echo "sigma: ${GAUSSIAN_SIGMA_VALUES[*]}"
echo "Runs: $TOTAL_RUNS"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

FAILED=0
ATTEMPTED=0

run_experiment() {
    local weight="$1" sigma="$2"
    local w_tag="${weight//./p}" s_tag="${sigma//./p}"
    local experiment="charades_iv2_w${w_tag}_sigma${s_tag}"
    local exp_rel="$OUTPUT_REL/$experiment"
    local exp_dir="$ROOT_DIR/$exp_rel"
    local model_base_rel="$exp_rel/checkpoints"
    local model_folder="${MODEL_NAME}_${TASK}_${VISUAL_FEATURES}_v_${VISUAL_LEN}_a_${AUDIO_LEN}"
    local model_dir_rel="$model_base_rel/$model_folder/model"
    local tb_dir_rel="$exp_rel/tensorboard"
    local log_file="$exp_dir/train.log"
    local tb_source="$ROOT_DIR/runs/$model_base_rel/$model_folder"

    mkdir -p "$exp_dir"
    ATTEMPTED=$((ATTEMPTED + 1))
    echo
    echo "[$(date '+%F %T')] START $experiment (w_cdl=$weight, sigma=$sigma)"

    if PYTHONHASHSEED="$SEED_VALUE" "$PYTHON_BIN" -u main.py \
        --task "$TASK" --mode train --gpu_idx "$GPU_IDX" --seed "$SEED_VALUE" \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --init_lr "$INIT_LR" \
        --visual_features "$VISUAL_FEATURES" --char_dim "$CHAR_DIM" \
        --model_name "$MODEL_NAME" --model_dir "$model_base_rel" --save_dir "$SAVE_DIR" \
        --w_cdl "$weight" --cdl_type "$CDL_TYPE" --gaussian_sigma "$sigma" \
        2>&1 | tee "$log_file"; then
        status="SUCCESS"
        echo "[$(date '+%F %T')] FINISH $experiment"
    else
        status="FAILED"
        FAILED=1
        echo "[$(date '+%F %T')] FAILED $experiment; see $log_file" >&2
    fi

    if [[ -d "$tb_source" ]]; then
        mkdir -p "$ROOT_DIR/$tb_dir_rel"
        mv "$tb_source" "$ROOT_DIR/$tb_dir_rel/"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment" "$TASK" "$VISUAL_FEATURES" "$SEED_VALUE" "$MODEL_NAME" \
        "$CDL_TYPE" "$weight" "$sigma" "$status" "$log_file" \
        "$model_dir_rel" "$tb_dir_rel" >> "$MANIFEST"
}

# All 12 combinations, run sequentially. Same seed and settings; only the
# two requested hyperparameters vary.
for weight in "${W_CDL_VALUES[@]}"; do
    for sigma in "${GAUSSIAN_SIGMA_VALUES[@]}"; do
        run_experiment "$weight" "$sigma"
    done
done

# Parse each run's best checkpoint (maximum R1@0.7, same criterion as IMG).
SUMMARY="$OUTPUT_ROOT/summary_best_checkpoint.csv"
"$PYTHON_BIN" - "$ROOT_DIR" "$MANIFEST" "$SUMMARY" <<'PY'
import csv, os, re, sys
root, manifest, output = sys.argv[1:4]
pattern = re.compile(
    r"Epoch\s+(\d+),\s*Step\s+(\d+):\s*\n"
    r"\s*Rank@1,\s*IoU=0\.3:\s*([\d.]+)\s+"
    r"Rank@1,\s*IoU=0\.5:\s*([\d.]+)\s+"
    r"Rank@1,\s*IoU=0\.7:\s*([\d.]+)\s+"
    r"mean IoU:\s*([\d.]+)", re.MULTILINE)
columns = ["experiment","task","visual_features","seed","model_name","cdl_type",
           "w_cdl","gaussian_sigma","training_status","best_epoch_by_R1@0.7",
           "best_step_by_R1@0.7","R1@0.3","R1@0.5","R1@0.7","mIoU",
           "checkpoint_count","model_dir","log_file","result_status"]
rows = []
with open(manifest, newline="", encoding="utf-8") as f:
    for run in csv.DictReader(f):
        model_dir = os.path.join(root, run["model_dir"])
        result_file = os.path.join(model_dir, "eval_results.txt")
        checkpoints = len([x for x in os.listdir(model_dir) if x.endswith(".t7")]) if os.path.isdir(model_dir) else 0
        best = None
        if os.path.isfile(result_file):
            with open(result_file, encoding="utf-8", errors="replace") as rf:
                matches = list(pattern.finditer(rf.read()))
            if matches: best = max(matches, key=lambda m: float(m.group(5)))
        row = {k: run.get(k, "") for k in ["experiment","task","visual_features","seed",
               "model_name","cdl_type","w_cdl","gaussian_sigma","training_status","model_dir","log_file"]}
        row.update({"best_epoch_by_R1@0.7":"","best_step_by_R1@0.7":"",
                    "R1@0.3":"","R1@0.5":"","R1@0.7":"","mIoU":"",
                    "checkpoint_count":checkpoints,"result_status":"NO_RESULTS"})
        if best:
            row.update({"best_epoch_by_R1@0.7":int(best.group(1)),
                        "best_step_by_R1@0.7":int(best.group(2)),
                        "R1@0.3":float(best.group(3)),"R1@0.5":float(best.group(4)),
                        "R1@0.7":float(best.group(5)),"mIoU":float(best.group(6)),
                        "result_status":"AVAILABLE"})
        rows.append(row)
with open(output, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
print(f"Best-checkpoint summary: {output}")
PY
if [[ $? -ne 0 ]]; then FAILED=1; echo "[ERROR] Summary generation failed." >&2; fi

echo
echo "============================================================"
echo "Attempted: $ATTEMPTED / $TOTAL_RUNS"
echo "Output:    $OUTPUT_ROOT"
echo "Manifest:  $MANIFEST"
echo "Summary:   $SUMMARY"
echo "============================================================"
if [[ "$ATTEMPTED" -ne "$TOTAL_RUNS" ]]; then FAILED=1; fi
if [[ "$FAILED" -ne 0 ]]; then
    echo "[WARNING] One or more runs failed. Check each train.log." >&2
    exit 1
fi
echo "[SUCCESS] All $TOTAL_RUNS Charades-STA + InternVideo2 runs completed."
