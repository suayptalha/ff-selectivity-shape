#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${HOME}/miniconda3/envs/torch-env/bin/python"
RDIR="experiment_results/ln_swish_cross_dataset"
mkdir -p "$RDIR/logs"

run() {
  local gpu="$1" variant="$2" dataset="$3" tag="$4"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u ff_experiments.py run \
    --variant "$variant" --dataset "$dataset" \
    --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
    --results_dir "$RDIR" > "$RDIR/logs/${tag}.log" 2>&1 &
}

echo "=== Batch 1: interaction table + first cross-dataset ==="
run 0 ln_swish_sos_no_ng             fashion_mnist  sos_fmnist
run 1 ln_swish_topk_no_ng            fashion_mnist  topk_fmnist
run 2 ln_swish_entmax15_energy_no_ng  fashion_mnist  entmax_fmnist
run 3 ln_swish_burstiness_no_ng       fashion_mnist  burst_fmnist
run 4 ln_swish_burstiness_no_ng       mnist          burst_mnist
run 5 ffcl_ln_swish_burstiness_no_ng  mnist          ffcl_burst_mnist
run 6 ffcl_ln_swish_burstiness_no_ng  fashion_mnist  ffcl_burst_fmnist
run 7 ln_swish_burstiness_no_ng       cifar10        burst_cifar10
wait
echo "Batch 1 done"

echo "=== Batch 2: remaining cross-dataset ==="
run 0 ffcl_ln_swish_burstiness_no_ng  cifar10   ffcl_burst_cifar10
run 1 ln_swish_burstiness_no_ng       usps      burst_usps
run 2 ffcl_ln_swish_burstiness_no_ng  usps      ffcl_burst_usps
run 3 ln_swish_burstiness_no_ng       svhn      burst_svhn
run 4 ffcl_ln_swish_burstiness_no_ng  svhn      ffcl_burst_svhn
run 5 ln_swish_burstiness_no_ng       emnist    burst_emnist
run 6 ffcl_ln_swish_burstiness_no_ng  emnist    ffcl_burst_emnist
wait
echo "Batch 2 done"
echo "ALL 16 DONE"
