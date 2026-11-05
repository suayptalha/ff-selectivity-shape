#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${HOME}/miniconda3/envs/torch-env/bin/python"

# Clear corrupted dataset caches
rm -rf data/USPS data/usps* data/EMNIST data/emnist* 2>/dev/null || true
echo "Cleared corrupted dataset caches"

RDIR1="experiment_results/ln_swish_cross_dataset"
RDIR2="experiment_results/ln_gelu_cross_dataset"
mkdir -p "$RDIR1/logs" "$RDIR2/logs"

echo "=== Launching 6 experiments ==="

# 2 missing cross-dataset: Std LN-Swish burstiness on USPS and EMNIST
CUDA_VISIBLE_DEVICES=0 "$PY" -u ff_experiments.py run \
  --variant ln_swish_burstiness_no_ng --dataset usps \
  --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
  --results_dir "$RDIR1" > "$RDIR1/logs/burst_usps.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 "$PY" -u ff_experiments.py run \
  --variant ln_swish_burstiness_no_ng --dataset emnist \
  --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
  --results_dir "$RDIR1" > "$RDIR1/logs/burst_emnist.log" 2>&1 &

# 4 burstiness detail: LN-burstiness with LN-GELU and LN-Swish × Std/FFCL on F-MNIST
CUDA_VISIBLE_DEVICES=2 "$PY" -u ff_experiments.py run \
  --variant ln_gelu_layernorm_burstiness_no_ng --dataset fashion_mnist \
  --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
  --results_dir "$RDIR2" > "$RDIR2/logs/ln_burst_fmnist.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 "$PY" -u ff_experiments.py run \
  --variant ffcl_ln_gelu_layernorm_burstiness_no_ng --dataset fashion_mnist \
  --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
  --results_dir "$RDIR2" > "$RDIR2/logs/ffcl_ln_burst_fmnist.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 "$PY" -u ff_experiments.py run \
  --variant ln_swish_layernorm_burstiness_no_ng --dataset fashion_mnist \
  --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
  --results_dir "$RDIR1" > "$RDIR1/logs/ln_burst_fmnist.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 "$PY" -u ff_experiments.py run \
  --variant ffcl_ln_swish_layernorm_burstiness_no_ng --dataset fashion_mnist \
  --n_layers 4 --hidden 2000 --seed 42 --gpu 0 \
  --results_dir "$RDIR1" > "$RDIR1/logs/ffcl_ln_burst_fmnist.log" 2>&1 &

wait
echo "ALL 6 DONE"
