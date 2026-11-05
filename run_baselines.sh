#!/usr/bin/env bash
set -euo pipefail

# Run the new "In Search of Goodness" baseline goodness functions
# (softmax_energy_margin, game_theoretic) across all configurations,
# plus standard + FFCL variants, on Fashion-MNIST 4x2000.
#
# Skips any experiment that already has a result file.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${1:-fashion_mnist}"
N_LAYERS="${2:-4}"
HIDDEN="${3:-2000}"
EPOCHS=60
RESULTS_DIR="experiment_results"

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs found, using CPU"
    NUM_GPUS=1
fi
echo "Using $NUM_GPUS GPUs"

# New baseline goodness functions from "In Search of Goodness" paper
NEW_GOODNESS=("softmax_energy_margin" "game_theoretic")
ACTIVATIONS=("gelu" "swish")
NORM_GATES=("ng" "no_ng")
MODES=("" "ffcl_")  # "" = standard, "ffcl_" = FFCL

VARIANTS=()
for mode in "${MODES[@]}"; do
    for act in "${ACTIVATIONS[@]}"; do
        for gn in "${NEW_GOODNESS[@]}"; do
            for ng in "${NORM_GATES[@]}"; do
                VARIANTS+=("${mode}${act}_${gn}_${ng}")
            done
        done
    done
done

echo "Total new variants to run: ${#VARIANTS[@]}"
echo "Dataset: $DATASET  Arch: ${N_LAYERS}x${HIDDEN}"
echo ""

LAUNCHED=0
SKIPPED=0

for v in "${VARIANTS[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${v}_${DATASET}_${N_LAYERS}x${HIDDEN}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "SKIP $v (result exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    GPU=$((LAUNCHED % NUM_GPUS))
    echo "LAUNCH $v on GPU $GPU"

    python3 ff_experiments.py run \
        --dataset "$DATASET" \
        --variant "$v" \
        --n_layers "$N_LAYERS" \
        --hidden "$HIDDEN" \
        --epochs "$EPOCHS" \
        --gpu "$GPU" \
        --results_dir "$RESULTS_DIR" &

    LAUNCHED=$((LAUNCHED + 1))

    # Don't overload: wait when all GPUs are busy
    if [ "$((LAUNCHED % NUM_GPUS))" -eq 0 ]; then
        echo "  -- waiting for batch of $NUM_GPUS to finish --"
        wait
    fi
done

wait
echo ""
echo "Done. Launched=$LAUNCHED  Skipped=$SKIPPED"
echo ""

# Show updated results
python3 ff_experiments.py collect --results_dir "$RESULTS_DIR"
