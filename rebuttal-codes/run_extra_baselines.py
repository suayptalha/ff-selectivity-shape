#!/usr/bin/env python3
"""
Forward-Forward Rebuttal Extra Baselines Suite
===============================================
Runs fast comparative benchmark tests inside the `rebuttal-codes` folder:
1. SymBa (Symmetric Forward-Forward loss)
2. SCFF (Self-Contrastive Forward-Forward style feature-contrast)
3. Standard FF + Burstiness (Ours)
4. Standard FF + SoS (Baseline)

Evaluates on Fashion-MNIST and MNIST across 5 seeds using standard per-layer goodness scoring.
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset_data(dataset_name="fashion_mnist", data_dir="../data", val_size=5000):
    dpath = Path(data_dir)
    dpath.mkdir(exist_ok=True)

    if dataset_name == "fashion_mnist":
        tr_full = datasets.FashionMNIST(str(dpath), train=True, download=True)
        te_full = datasets.FashionMNIST(str(dpath), train=False, download=True)
        in_dim, n_classes = 784, 10
    elif dataset_name == "mnist":
        tr_full = datasets.MNIST(str(dpath), train=True, download=True)
        te_full = datasets.MNIST(str(dpath), train=False, download=True)
        in_dim, n_classes = 784, 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    tr_img = tr_full.data.float().view(-1, in_dim) / 255.0
    te_img = te_full.data.float().view(-1, in_dim) / 255.0
    tr_lbl = tr_full.targets.long()
    te_lbl = te_full.targets.long()

    gen = torch.Generator().manual_seed(12345)
    perm = torch.randperm(len(tr_img), generator=gen)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return (tr_img[train_idx].to(DEVICE), tr_lbl[train_idx].to(DEVICE),
            tr_img[val_idx].to(DEVICE), tr_lbl[val_idx].to(DEVICE),
            te_img.to(DEVICE), te_lbl.to(DEVICE), in_dim, n_classes)

def goodness_sos(h):
    return h.pow(2).mean(dim=-1)

def goodness_burstiness(h):
    mu = h.mean(dim=-1, keepdim=True)
    diff = h - mu
    var = diff.pow(2).mean(dim=-1).clamp(min=1e-12)
    kurt = diff.pow(4).mean(dim=-1) / var.pow(2)
    return kurt - 3.0

class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        h = self.linear(x)
        if self.activation == "gelu":
            return F.gelu(h)
        return F.relu(h)

def l2_normalize(h):
    return h / (h.norm(dim=-1, keepdim=True) + 1e-6)

def embed_labels(images, labels, n_classes, scale=10.0):
    oh = F.one_hot(labels, n_classes).float() * scale
    return l2_normalize(torch.cat([images, oh], dim=-1))

def train_and_eval_extra(train_x, train_y, eval_x, eval_y, n_classes, in_dim,
                         method="burstiness", activation="relu",
                         label_scale=10.0, threshold=1.0, lr=0.001, epochs=60, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_layers, hidden_dim = 4, 2000

    in_dims = [in_dim + n_classes] + [hidden_dim] * (n_layers - 1)
    layers = [FFLayer(in_dims[i], hidden_dim, activation).to(DEVICE) for i in range(n_layers)]
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in layers]

    n = len(train_x)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, 500):
            idx = perm[start:start + 500]
            imgs, lbls = train_x[idx], train_y[idx]
            wrong = (lbls + torch.randint(1, n_classes, lbls.shape, device=DEVICE)) % n_classes

            h_pos = embed_labels(imgs, lbls, n_classes, scale=label_scale)
            h_neg = embed_labels(imgs, wrong, n_classes, scale=label_scale)

            for layer, opt in zip(layers, opts):
                a_pos = layer(h_pos)
                a_neg = layer(h_neg)

                if method == "sos":
                    # Standard Hinton FF loss with softplus threshold
                    g_pos = goodness_sos(a_pos)
                    g_neg = goodness_sos(a_neg)
                    loss = F.softplus(threshold - g_pos).mean() + F.softplus(g_neg - threshold).mean()
                elif method == "burstiness":
                    # Our proposed shape-sensitive (excess kurtosis) loss
                    g_pos = goodness_burstiness(a_pos)
                    g_neg = goodness_burstiness(a_neg)
                    loss = F.softplus(threshold - g_pos).mean() + F.softplus(g_neg - threshold).mean()
                elif method == "symba":
                    # SymBa (Lee & Song, 2023): Symmetric contrast loss without fixed threshold
                    # L_symba = log(1 + exp(g_neg - g_pos))
                    g_pos = goodness_sos(a_pos)
                    g_neg = goodness_sos(a_neg)
                    loss = F.softplus(g_neg - g_pos).mean()
                elif method == "scff":
                    # SCFF (Chen et al., 2025): Self-Contrastive InfoNCE / BCE loss
                    # L_scff = -log( exp(g_pos) / (exp(g_pos) + exp(g_neg)) ) = log(1 + exp(g_neg - g_pos))
                    # Evaluated with log-sum-exp stabilization:
                    g_pos = goodness_sos(a_pos)
                    g_neg = goodness_sos(a_neg)
                    loss = F.binary_cross_entropy_with_logits(g_pos - g_neg, torch.ones_like(g_pos))
                else:
                    raise ValueError(f"Unknown method: {method}")

                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    h_pos = l2_normalize(a_pos.detach())
                    h_neg = l2_normalize(a_neg.detach())

    # Evaluation phase using standard raw goodness sum
    N = len(eval_x)
    g_per_layer = torch.zeros(N, n_classes, device=DEVICE)

    for c in range(n_classes):
        c_lbls = torch.full((N,), c, dtype=torch.long, device=DEVICE)
        x = embed_labels(eval_x, c_lbls, n_classes, scale=label_scale)
        h = x
        b_good = torch.zeros(N, device=DEVICE)

        for layer in layers:
            h = layer(h)
            if method == "burstiness":
                b_good += goodness_burstiness(h)
            else:
                b_good += goodness_sos(h)
            h = l2_normalize(h)

        g_per_layer[:, c] = b_good

    acc = (g_per_layer.argmax(dim=1) == eval_y).float().mean().item() * 100.0
    return acc

def run_all_benchmarks(dataset_name="fashion_mnist"):
    print("=" * 80)
    print(f"EXTRA BASELINES BENCHMARK ({dataset_name.upper()})")
    print("=" * 80)

    train_x, train_y, val_x, val_y, test_x, test_y, in_dim, n_classes = load_dataset_data(dataset_name)
    methods = ["sos", "symba", "scff", "burstiness"]
    seeds = [42, 43, 44, 45, 46]
    results = {}

    for m in methods:
        print(f"\n---> Running Method: {m.upper()}")
        accs = []
        for seed in seeds:
            acc = train_and_eval_extra(
                train_x, train_y, test_x, test_y, n_classes, in_dim,
                method=m, activation="relu", label_scale=10.0, threshold=1.0,
                lr=0.001, epochs=60, seed=seed
            )
            accs.append(acc)
            print(f"  [Seed {seed}] Acc: {acc:.2f}%")

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        results[m] = {"mean": mean_acc, "std": std_acc, "per_seed": accs}
        print(f"===> RESULT {m.upper()}: {mean_acc:.2f} ± {std_acc:.2f}%")

    out_file = Path(__file__).parent / f"extra_baselines_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "fashion_mnist"
    run_all_benchmarks(ds)

