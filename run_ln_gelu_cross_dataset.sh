#!/usr/bin/env bash
# Repeat the paper's cross-dataset experiments (Table 5) with ln_gelu activation.
# 5 ln_gelu variants + vanilla baseline × 6 datasets, seed=42, 4×2000.
# Round-robins across available GPUs (default 8).
set -euo pipefail
cd "$(dirname "$0")"

# Use CUDA-enabled Python (base python3 often has no CUDA)
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for _p in "${HOME}/miniconda3/envs/torch-env/bin/python" \
            "${HOME}/anaconda3/envs/torch-env/bin/python"; do
    if [ -x "$_p" ] && "$_p" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      PYTHON="$_p"; break
    fi
  done
  [ -z "$PYTHON" ] && PYTHON=python3
fi
echo "Python: $PYTHON"

NUM_GPUS=${NUM_GPUS:-8}
RESULTS_DIR="${RESULTS_DIR:-experiment_results/ln_gelu_cross_dataset}"

VARIANTS=(
  ln_gelu_burstiness_no_ng
  ffcl_ln_gelu_burstiness_no_ng
)

DATASETS=(mnist fashion_mnist cifar10 usps svhn emnist)

N_LAYERS=4
HIDDEN=2000
EPOCHS=60
SEED=42

echo "=================================================================="
echo "  ln_gelu Cross-Dataset Experiments (paper Table 5 mirror)"
echo "  ${#VARIANTS[@]} variants × ${#DATASETS[@]} datasets = $(( ${#VARIANTS[@]} * ${#DATASETS[@]} )) runs"
echo "  arch=${N_LAYERS}x${HIDDEN}  epochs=${EPOCHS}  seed=${SEED}"
echo "  GPUs: ${NUM_GPUS}    results → ${RESULTS_DIR}"
echo "=================================================================="
echo ""

mkdir -p "$RESULTS_DIR" "${RESULTS_DIR}/logs"

PIDS=()
NAMES=()
gpu_idx=0

for ds in "${DATASETS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${v}_${ds}_${N_LAYERS}x${HIDDEN}.json"
    if [ -f "$RESULT_FILE" ]; then
      echo "  SKIP (exists): ${v} / ${ds}"
      continue
    fi

    GPU=$((gpu_idx % NUM_GPUS))
    LOG="${RESULTS_DIR}/logs/${v}_${ds}.log"
    echo "  LAUNCH gpu=${GPU}: ${v} / ${ds}"

    CUDA_VISIBLE_DEVICES=${GPU} "$PYTHON" -u ff_experiments.py run \
      --variant "$v" \
      --dataset "$ds" \
      --n_layers "$N_LAYERS" \
      --hidden "$HIDDEN" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --gpu 0 \
      --results_dir "$RESULTS_DIR" \
      > "$LOG" 2>&1 &

    PIDS+=($!)
    NAMES+=("${v}/${ds}")
    gpu_idx=$((gpu_idx + 1))

    # If we've filled all GPUs, wait for current batch before launching more
    if (( ${#PIDS[@]} >= NUM_GPUS )); then
      FAIL=0
      for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
          echo "  FAILED: ${NAMES[$i]}  (see ${RESULTS_DIR}/logs/)"
          FAIL=$((FAIL + 1))
        else
          echo "  OK: ${NAMES[$i]}"
        fi
      done
      PIDS=()
      NAMES=()
      echo ""
    fi
  done
done

# Wait for any remaining jobs
if (( ${#PIDS[@]} > 0 )); then
  FAIL=0
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "  FAILED: ${NAMES[$i]}"
      FAIL=$((FAIL + 1))
    else
      echo "  OK: ${NAMES[$i]}"
    fi
  done
fi

echo ""
echo "=================================================================="
echo "  All experiments complete. Collecting results..."
echo "=================================================================="
echo ""
"$PYTHON" ff_experiments.py collect --results_dir "$RESULTS_DIR"
