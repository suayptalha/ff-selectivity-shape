#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS=${1:-4}  # default 4 GPUs, override with: bash run_fashion_parallel.sh 8

VARIANTS=(
  ffcl_gelu_contrast_topk_ng ffcl_gelu_contrast_topk_no_ng
  ffcl_gelu_entropy_ng ffcl_gelu_entropy_no_ng
  ffcl_gelu_layernorm_topk_ng ffcl_gelu_layernorm_topk_no_ng
  ffcl_gelu_sos_ng ffcl_gelu_sos_no_ng
  ffcl_gelu_topk_ng ffcl_gelu_topk_no_ng
  ffcl_gelu_variance_ng ffcl_gelu_variance_no_ng
  ffcl_swish_contrast_topk_ng ffcl_swish_contrast_topk_no_ng
  ffcl_swish_entropy_ng ffcl_swish_entropy_no_ng
  ffcl_swish_layernorm_topk_ng ffcl_swish_layernorm_topk_no_ng
  ffcl_swish_sos_ng ffcl_swish_sos_no_ng
  ffcl_swish_topk_ng ffcl_swish_topk_no_ng
  ffcl_swish_variance_ng ffcl_swish_variance_no_ng
  gelu_layernorm_topk_ng gelu_layernorm_topk_no_ng
  swish_layernorm_topk_ng swish_layernorm_topk_no_ng
)

DATASET=fashion_mnist
N_LAYERS=4
HIDDEN=2000

echo "=== Fashion-MNIST 4x2000 FFCL ablation ==="
echo "  ${#VARIANTS[@]} variants across ${NUM_GPUS} GPUs"
echo ""

SKIPPED=0
LAUNCHED=0

for i in "${!VARIANTS[@]}"; do
  V="${VARIANTS[$i]}"
  RESULT_FILE="experiment_results/${V}_${DATASET}_${N_LAYERS}x${HIDDEN}.json"

  if [ -f "$RESULT_FILE" ]; then
    echo "  SKIP (exists): $V"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  GPU=$((i % NUM_GPUS))
  echo "  LAUNCH gpu=$GPU: $V"
  python3 ff_experiments.py run --variant "$V" \
    --dataset "$DATASET" --n_layers "$N_LAYERS" --hidden "$HIDDEN" \
    --gpu "$GPU" &
  LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "  Launched: $LAUNCHED  Skipped: $SKIPPED"
echo "  Waiting for all to finish..."
wait
echo ""
echo "=== All experiments complete ==="
echo ""

python3 ff_experiments.py collect
