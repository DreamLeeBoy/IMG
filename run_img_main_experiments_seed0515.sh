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
            "w_cdl": run["w_cdl"],
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
    "w_cdl",
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
    print("CDL weight:", row["w_cdl"])
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
echo "Uniform CDL: $CDL_WEIGHT"
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
