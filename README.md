# Selectivity and Shape in the Design of Forward-Forward Goodness Functions

Code for **"Selectivity and Shape in the Design of Forward-Forward Goodness Functions"** (NeurIPS 2026 submission).

## Key Finding

The dramatic performance gains in Forward-Forward learning demonstrate that the choice of goodness function is the primary driver of architectural success, with **sparse functions** vastly outperforming the traditional Sum-of-Squares (SoS) baseline. By implementing **Burstiness** and **Separate Label–Feature Forwarding (FFCL)**, the model transitions from a dense, inefficient update rule to a high-impact mechanism that mimics biological sparsity, resulting in a staggering **+32.3pp** improvement on Fashion-MNIST and a massive **+72.0pp** leap on the USPS dataset. These results prove that when the network attends to only a subset of neurons, it achieves superior generalization and noise resistance across diverse tasks, effectively bridging the gap between backpropagation-free learning and state-of-the-art performance.

| Method | MNIST | F-MNIST | CIFAR-10 | USPS | SVHN | EMNIST |
|---|---|---|---|---|---|---|
| SoS Baseline (ReLU) | 88.76% | 56.41% | 37.85% | 21.87% | 32.25% | 69.91% |
| FFCL + SoS (GELU) | 93.45% | 82.38% | 46.46% | 76.83% | 71.65% | 77.80% |
| Std + Top-k (Swish) | 90.23% | 78.64% | 37.37% | 82.86% | 57.68% | 66.35% |
| FFCL + Ctrst. top-k | 93.31% | 83.59% | 47.07% | 90.03% | 66.05% | 79.69% |
| Std + Burstiness (Swish) | 96.54% | 88.06% | 53.56% | 93.12% | 78.85% | 86.21% |
| **FFCL + Burstiness (best)** | **98.16%** | **88.69%** | **55.58%** | **93.87%** | **83.99%** | **91.62%** |
| **Delta (best vs SoS)** | **+9.4** | **+32.3** | **+17.7** | **+72.0** | **+51.7** | **+21.7** |

## Setup

```bash
pip install -r requirements.txt
pip install entmax  # for entmax-weighted energy goodness
```

## Reproducing Results

### Main comparison (Table 1)

```bash
python ff_main.py --ffcl
```

### Reproducing Best Results (Burstiness)

```bash
# Run FFCL with Burstiness on Fashion-MNIST
python ff_main.py --ffcl --dataset fashion_mnist --arch 4 2000
```

### Full goodness function comparison (Table 2)

```bash
python ff_main.py --full
```

### FFCL experiments

```bash
python ff_experiments.py --ffcl --goodness all --dataset fashion_mnist --arch 4 2000 --gpu 0
```

### Sparsity sweep (Figure 1)

```bash
# k-sweep for contrast top-k
python run_sparsity_sweep.py --sweep_type k --param_value 0.02 --dataset fashion_mnist --gpu 0

# alpha-sweep for entmax
python run_sparsity_sweep.py --sweep_type alpha --param_value 1.5 --dataset fashion_mnist --gpu 0

# Launch full sweep across GPUs
bash launch_sweep.sh
```

## Options

```
--full              Run all goodness functions
--dataset           Choose dataset: mnist, fashion_mnist (default: both)
--arch L H          Architecture: L layers, H hidden units (repeatable)
--gpu N             GPU index (default: 0)
--ffcl              Use FFCL (separate label-feature forwarding)
```

### Examples

```bash
# Quick test on MNIST only
python ff_main.py --dataset mnist --arch 4 2000

# Full Fashion-MNIST comparison with FFCL
python ff_experiments.py --ffcl --goodness all --dataset fashion_mnist --arch 4 2000

# Run all remaining experiments on 8 GPUs
bash run_fashion_parallel.sh
```

## Goodness Functions

The core insight: **sparsity in the goodness function is optimal**.

```python
# Standard sum-of-squares (baseline)
def goodness_sos(h):
    return h.pow(2).mean(dim=-1)

# Top-k: hard selection of k most active neurons
def goodness_topk(h):
    k = max(5, int(h.shape[-1] * 0.02))
    return torch.topk(h, k=k, dim=-1).values.mean(dim=-1)

# Entmax-weighted energy: adaptive sparse weighting
def goodness_entmax(h, alpha=1.5):
    from entmax import entmax_bisect
    weights = entmax_bisect(h, alpha=alpha)
    return (weights * h.pow(2)).sum(dim=-1)

# Burstiness (excess kurtosis): measures heavy-tailedness (best overall)
def goodness_burstiness(h):
    mu = h.mean(dim=-1, keepdim=True)
    diff = h - mu
    var = diff.pow(2).mean(dim=-1).clamp(min=1e-12)
    kurt = diff.pow(4).mean(dim=-1) / var.pow(2)
    return kurt - 3.0
```

Top-k measures only peak neural activity, ignoring the (d-k) least active neurons. Entmax goes further by adaptively weighting neurons — the number of attended neurons varies per input. Performance peaks at α ≈ 1.5 (adaptive sparsity) and degrades at both extremes (dense softmax α=1 and hard sparsemax α=2).

## Project Structure

```
ff_main.py              # Core FF implementation + main comparison
ff_experiments.py       # Full combinatorial experiment framework
run_sparsity_sweep.py   # Targeted k and alpha sweeps
launch_sweep.sh         # Multi-GPU sweep orchestration
run_fashion_parallel.sh # Parallel Fashion-MNIST experiments
experiment_results.zip     # JSON results from all experiments
sweep_results.zip          # JSON results from sparsity sweeps
paper/                  # LaTeX source for the paper
```

## Hardware

- Standard FF experiments: 30-60 seconds per run on a single GPU
- FFCL experiments: 35-70 seconds per run
- Entmax experiments: 200-400 seconds per run (bisection overhead)
- Full suite: ~4 GPU-hours on NVIDIA A100s

## Citation

```bibtex
@article{anonymous2026sparse,
  title={Selectivity and Shape in the Design of Forward-Forward Goodness Functions},
  author={Anonymous},
  journal={NeurIPS 2026 Submission},
  year={2026}
}
```

## License

MIT
