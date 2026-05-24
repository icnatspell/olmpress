#!/usr/bin/env bash
# ResNet-50 structured pruning sweep via olmpress / Olive.
#
# For each importance × ratio pair:
#   1. Instantiates workflow_template.yaml and runs `olmpress run` to prune.
#   2. Calls eval_accuracy.py on the saved TorchScript model.
#   3. Prints a comparison table and optionally saves a CSV.
#
# Usage (from project root):
#   bash examples/resnet50_pruning/run_sweep.sh
#   bash examples/resnet50_pruning/run_sweep.sh --csv workflows/resnet50-pruning/sweep_results.csv

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

HERE="examples/resnet50_pruning"
TEMPLATE="${HERE}/workflow_template.yaml"
EVALUATE="${HERE}/eval_accuracy.py"

IMPORTANCES=(magnitude group_magnitude lamp fpgm taylor hessian)
RATIOS=(0.05 0.10 0.15 0.20 0.25 0.30)
GLOBAL_PRUNING=true
CALIBRATION_STEPS=10
NUM_SAMPLES=2000
CSV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TMP=$(mktemp /tmp/olmpress_sweep_XXXXXX.yaml)
trap 'rm -f "$TMP"' EXIT

mkdir -p workflows/resnet50-pruning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Parse VALUE from "top1=VALUE" in a string.
extract_top1() { printf '%s\n' "$1" | grep -o 'top1=[0-9.]*' | cut -d= -f2; }

pct() { awk "BEGIN { printf \"%.1f%%\", $1 * 100 }"; }

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

printf "Evaluating baseline (%s) …\n" "microsoft/resnet-50"
BASELINE_OUT=$(uv run python "$EVALUATE" --num-samples "$NUM_SAMPLES" 2>&1)
BASELINE_TOP1=$(extract_top1 "$BASELINE_OUT")
printf "  baseline top-1=%s\n\n" "$(pct "$BASELINE_TOP1")"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

declare -a RECORDS  # "importance|ratio|top1"

for IMPORTANCE in "${IMPORTANCES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        OUTPUT_DIR="workflows/resnet50-pruning/outputs/${IMPORTANCE}_${RATIO}"
        printf "  %-18s %s  pruning …" "$IMPORTANCE" "$RATIO"

        export IMPORTANCE RATIO GLOBAL_PRUNING CALIBRATION_STEPS OUTPUT_DIR
        envsubst < "$TEMPLATE" > "$TMP"
        uv run olmpress run --config "$TMP" > /dev/null 2>&1

        MODEL_PT="${OUTPUT_DIR}/model.pt"
        printf " evaluating …"
        EVAL_OUT=$(uv run python "$EVALUATE" --model "$MODEL_PT" --num-samples "$NUM_SAMPLES" 2>&1)
        TOP1=$(extract_top1 "$EVAL_OUT")

        printf " top-1=%s\n" "$(pct "$TOP1")"
        RECORDS+=("${IMPORTANCE}|${RATIO}|${TOP1}")
    done
done

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

SEP="----------------------------------------------------"
printf "\n%s\n" "$SEP"
printf "  ResNet-50 pruning sweep  (global_pruning=%s)\n" "$GLOBAL_PRUNING"
printf "%s\n\n" "$SEP"
printf "%-18s  %6s  %7s\n" "importance" "ratio" "top-1"
printf "%-18s  %6s  %7s\n" "------------------" "------" "-------"
printf "%-18s  %6s  %7s\n" "baseline" "0%" "$(pct "$BASELINE_TOP1")"
for rec in "${RECORDS[@]}"; do
    IFS='|' read -r IMP R T1 <<< "$rec"
    printf "%-18s  %6s  %7s\n" \
        "$IMP" \
        "$(awk "BEGIN { printf \"%.0f%%\", $R * 100 }")" \
        "$(pct "$T1")"
done
printf "%s\n" "$SEP"

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

if [[ -n "$CSV" ]]; then
    printf 'importance,pruning_ratio,top1\n' > "$CSV"
    printf 'baseline,0,%s\n' "$BASELINE_TOP1" >> "$CSV"
    for rec in "${RECORDS[@]}"; do
        IFS='|' read -r IMP R T1 <<< "$rec"
        printf '%s,%s,%s\n' "$IMP" "$R" "$T1" >> "$CSV"
    done
    printf '\nResults saved to %s\n' "$CSV"
fi
