#!/usr/bin/env bash
set -euo pipefail

# Sparsity sweep: Standard vs FFCL, both with k-sweep and α-sweep
# Config: gelu / no_ng / Fashion-MNIST 4x2000

cd "$(dirname "$0")"

DATASET="fashion_mnist"
N_LAYERS=4
HIDDEN=2000
EPOCHS=60
RESULTS_DIR="sweep_results"
NUM_GPUS=8

K_VALUES="0.0025 0.005 0.01 0.02 0.05 0.10 0.20"
ALPHA_VALUES="1.0 1.25 1.5 1.75 2.0"

mkdir -p "$RESULTS_DIR"

JOBS=()

for MODE in std ffcl; do
    FFCL_FLAG=""
    if [ "$MODE" = "ffcl" ]; then FFCL_FLAG="--ffcl"; fi

    for k in $K_VALUES; do
        RESULT_FILE="${RESULTS_DIR}/k_${k}_${MODE}_${DATASET}_${N_LAYERS}x${HIDDEN}.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "SKIP ${MODE} k=$k (result exists)"
        else
            JOBS+=("k $k $FFCL_FLAG")
        fi
    done

    for a in $ALPHA_VALUES; do
        RESULT_FILE="${RESULTS_DIR}/alpha_${a}_${MODE}_${DATASET}_${N_LAYERS}x${HIDDEN}.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "SKIP ${MODE} α=$a (result exists)"
        else
            JOBS+=("alpha $a $FFCL_FLAG")
        fi
    done
done

TOTAL=${#JOBS[@]}
echo ""
echo "Launching $TOTAL sweep experiments across $NUM_GPUS GPUs"
echo ""

LAUNCHED=0
PIDS=()

for job in "${JOBS[@]}"; do
    SWEEP_TYPE=$(echo "$job" | awk '{print $1}')
    PARAM=$(echo "$job" | awk '{print $2}')
    FFCL_FLAG=$(echo "$job" | awk '{print $3}')
    GPU=$((LAUNCHED % NUM_GPUS))

    MODE="std"
    if [ -n "$FFCL_FLAG" ]; then MODE="ffcl"; fi
    echo "  [$((LAUNCHED+1))/$TOTAL] GPU $GPU: ${MODE} ${SWEEP_TYPE}=${PARAM}"

    python3 run_sparsity_sweep.py \
        --sweep_type "$SWEEP_TYPE" \
        --param_value "$PARAM" \
        --dataset "$DATASET" \
        --n_layers "$N_LAYERS" \
        --hidden "$HIDDEN" \
        --epochs "$EPOCHS" \
        --gpu "$GPU" \
        --results_dir "$RESULTS_DIR" \
        $FFCL_FLAG &

    PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))

    if [ "$((LAUNCHED % NUM_GPUS))" -eq 0 ]; then
        echo "  -- waiting for batch of $NUM_GPUS --"
        for pid in "${PIDS[@]}"; do wait "$pid"; done
        PIDS=()
    fi
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

echo ""
echo "=== All $TOTAL experiments done ==="
echo ""

# Print summary
python3 -c "
import json
from pathlib import Path

results = []
for f in sorted(Path('$RESULTS_DIR').glob('*.json')):
    with open(f) as fh:
        results.append(json.load(fh))

for sweep in ['k', 'alpha']:
    sweep_r = [r for r in results if r['sweep_type'] == sweep]
    if not sweep_r:
        continue

    if sweep == 'k':
        title = 'k SWEEP (hard selection — contrast_topk)'
        col = 'k_frac'
    else:
        title = 'α SWEEP (soft sparsity — entmax-weighted energy)'
        col = 'alpha'

    print('=' * 75)
    print(f'  {title}')
    print(f'  gelu / no_ng / Fashion-MNIST 4x2000')
    print('=' * 75)

    for mode_label, is_ffcl in [('Standard', False), ('FFCL', True)]:
        subset = sorted([r for r in sweep_r if r['ffcl'] == is_ffcl],
                        key=lambda r: r['param_value'])
        if not subset:
            continue
        best = max(subset, key=lambda r: r['test_accuracy'])
        print(f'\n  {mode_label}:')
        if sweep == 'k':
            print(f\"  {'k_frac':>8s}  {'k_abs':>6s}  {'Acc%':>8s}  {'Loss':>8s}  {'Time':>6s}\")
            print('  ' + '-' * 50)
            for r in subset:
                k_abs = max(2, int(2000 * r['param_value']))
                m = ' *' if r == best else ''
                print(f\"  {r['param_value']:>8.4f}  {k_abs:>6d}  {r['test_accuracy']*100:>7.2f}%  {r['test_loss']:>8.4f}  {r['train_time_sec']:>5.0f}s{m}\")
        else:
            print(f\"  {'alpha':>8s}  {'Acc%':>8s}  {'Loss':>8s}  {'Time':>6s}\")
            print('  ' + '-' * 42)
            for r in subset:
                m = ' *' if r == best else ''
                print(f\"  {r['param_value']:>8.2f}  {r['test_accuracy']*100:>7.2f}%  {r['test_loss']:>8.4f}  {r['train_time_sec']:>5.0f}s{m}\")
    print()
"
