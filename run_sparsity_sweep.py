#!/usr/bin/env python3
"""
Sparsity sweep: vary k (hard selection) and α (entmax soft sparsity)
on the best FFCL config (gelu / contrast_topk / no_ng) for Fashion-MNIST 4x2000.

Produces data for a paper figure showing accuracy vs. sparsity level.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from entmax import entmax_bisect

from ff_experiments import (
    load_dataset, FFLayer, FFCLLayer, _l2_normalize,
    train_ff, train_ff_ffcl, evaluate_ff, evaluate_ff_ffcl,
    _vanilla_norm_gate, _vanilla_neg_gen, embed_labels,
)


# ======================================================================
# Parameterized goodness functions
# ======================================================================

def goodness_contrast_topk_k(h, k_frac):
    """contrast_topk with configurable k as a fraction of hidden dim."""
    k = max(2, int(h.shape[-1] * k_frac))
    top = torch.topk(h, k=k, dim=-1).values.mean(dim=-1)
    bot = torch.topk(-h, k=k, dim=-1).values.mean(dim=-1)
    return top - bot


def goodness_entmax_energy(h, alpha):
    """Entmax-weighted sum of squares with configurable α.
    α=1.0 ≈ softmax (dense), α=2.0 = sparsemax (hard sparse)."""
    weights = entmax_bisect(h, alpha=alpha, dim=-1)
    return (weights * h.pow(2)).sum(dim=-1)


# ======================================================================
# Single sweep run
# ======================================================================

def run_single(sweep_type, param_value, ffcl, dataset, n_layers, hidden,
               epochs, lr, threshold, batch_size, seed, gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    tr_img, tr_lbl, te_img, te_lbl, input_dim, n_classes = load_dataset(dataset)
    mu, std = tr_img.mean(), tr_img.std()
    tr_img = (tr_img - mu) / (std + 1e-8)
    te_img = (te_img - mu) / (std + 1e-8)
    tr_img, tr_lbl = tr_img.to(device), tr_lbl.to(device)
    te_img, te_lbl = te_img.to(device), te_lbl.to(device)

    if sweep_type == "k":
        goodness_fn = partial(goodness_contrast_topk_k, k_frac=param_value)
        label = f"contrast_topk k={param_value*100:.1f}% ({max(2, int(hidden * param_value))})"
    else:
        goodness_fn = partial(goodness_entmax_energy, alpha=param_value)
        label = f"entmax α={param_value:.2f}"

    mode_tag = "FFCL" if ffcl else "Standard"
    print(f"\n{'='*60}")
    print(f"  {mode_tag} — {label}")
    print(f"  gelu / no_ng / {dataset} / {n_layers}x{hidden}")
    print(f"{'='*60}", flush=True)

    if ffcl:
        layers = []
        for i in range(n_layers):
            d_in = input_dim if i == 0 else hidden
            layers.append(FFCLLayer(d_in, hidden, n_classes, activation="gelu").to(device))

        t0 = time.time()
        train_ff_ffcl(layers, tr_img, tr_lbl, n_classes, device,
                      goodness_fn=goodness_fn, norm_gate_fn=_vanilla_norm_gate,
                      epochs=epochs, batch_size=batch_size, lr=lr, threshold=threshold)
        train_sec = time.time() - t0

        test_acc, test_loss = evaluate_ff_ffcl(
            layers, te_img, te_lbl, n_classes, device,
            goodness_fn=goodness_fn, norm_gate_fn=_vanilla_norm_gate)
    else:
        first_dim = input_dim + n_classes
        layers = []
        for i in range(n_layers):
            d_in = first_dim if i == 0 else hidden
            layers.append(FFLayer(d_in, hidden, activation="gelu").to(device))

        t0 = time.time()
        train_ff(layers, tr_img, tr_lbl, n_classes, device,
                 goodness_fn=goodness_fn, norm_gate_fn=_vanilla_norm_gate,
                 neg_gen_fn=_vanilla_neg_gen,
                 epochs=epochs, batch_size=batch_size, lr=lr, threshold=threshold)
        train_sec = time.time() - t0

        test_acc, test_loss = evaluate_ff(
            layers, te_img, te_lbl, n_classes, device,
            goodness_fn=goodness_fn, norm_gate_fn=_vanilla_norm_gate)

    n_params = sum(p.numel() for layer in layers for p in layer.parameters())

    print(f"\n  ✓ Test Accuracy: {test_acc*100:.2f}%", flush=True)
    print(f"    Train Time:   {train_sec:.1f}s\n", flush=True)

    return {
        "sweep_type": sweep_type,
        "param_value": param_value,
        "ffcl": ffcl,
        "label": label,
        "test_accuracy": round(test_acc, 6),
        "test_loss": round(test_loss, 6),
        "train_time_sec": round(train_sec, 1),
        "n_params": n_params,
        "dataset": dataset,
        "n_layers": n_layers,
        "hidden": hidden,
        "epochs": epochs,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_type", required=True, choices=["k", "alpha"])
    parser.add_argument("--param_value", required=True, type=float)
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ffcl", action="store_true")
    parser.add_argument("--results_dir", default="sweep_results")
    args = parser.parse_args()

    results = run_single(
        args.sweep_type, args.param_value, args.ffcl, args.dataset,
        args.n_layers, args.hidden, args.epochs, args.lr,
        args.threshold, args.batch_size, args.seed, args.gpu,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "ffcl" if args.ffcl else "std"
    fname = f"{args.sweep_type}_{args.param_value}_{mode_tag}_{args.dataset}_{args.n_layers}x{args.hidden}.json"
    with open(results_dir / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {results_dir / fname}")


if __name__ == "__main__":
    main()
