#!/usr/bin/env python3
"""
Forward-Forward Rebuttal Evaluation Suite
=========================================
Runs validation tuning and 5-seed test evaluations for the NeurIPS rebuttal.
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

# Ensure stdout line buffering for real-time logging
sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_splits(dataset_name="fashion_mnist", data_dir="data", val_size=5000):
    dpath = Path.cwd() / data_dir
    dpath.mkdir(exist_ok=True)

    if dataset_name == "fashion_mnist":
        tr_full = datasets.FashionMNIST(str(dpath), train=True, download=True)
        te_full = datasets.FashionMNIST(str(dpath), train=False, download=True)
        in_dim, n_classes = 784, 10
    elif dataset_name == "mnist":
        tr_full = datasets.MNIST(str(dpath), train=True, download=True)
        te_full = datasets.MNIST(str(dpath), train=False, download=True)
        in_dim, n_classes = 784, 10
    elif dataset_name == "cifar10":
        tr_full = datasets.CIFAR10(str(dpath), train=True, download=True)
        te_full = datasets.CIFAR10(str(dpath), train=False, download=True)
        tr_img = torch.from_numpy(np.array(tr_full.data)).float().reshape(-1, 3072) / 255.0
        te_img = torch.from_numpy(np.array(te_full.data)).float().reshape(-1, 3072) / 255.0
        tr_lbl = torch.tensor(tr_full.targets, dtype=torch.long)
        te_lbl = torch.tensor(te_full.targets, dtype=torch.long)

        gen = torch.Generator().manual_seed(12345)
        perm = torch.randperm(len(tr_img), generator=gen)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        return (tr_img[train_idx].to(DEVICE), tr_lbl[train_idx].to(DEVICE),
                tr_img[val_idx].to(DEVICE), tr_lbl[val_idx].to(DEVICE),
                te_img.to(DEVICE), te_lbl.to(DEVICE), 3072, 10)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    tr_img = tr_full.data.float().view(-1, in_dim) / 255.0
    te_img = te_full.data.float().view(-1, in_dim) / 255.0
    tr_lbl = tr_full.targets.long()
    te_lbl = te_full.targets.long()

    # Deterministic train/val split (seed 12345)
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

GOODNESS_MAP = {
    "sos": goodness_sos,
    "burstiness": goodness_burstiness
}

class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.out_dim = out_dim

    def forward(self, x):
        h = self.linear(x)
        if self.activation == "gelu":
            return F.gelu(h)
        elif self.activation == "swish":
            return h * torch.sigmoid(h)
        return F.relu(h)

class FFCLLayer(nn.Module):
    """Per-layer label forwarding (direct label injection)."""
    def __init__(self, in_dim, out_dim, n_classes, activation="gelu"):
        super().__init__()
        self.feature_linear = nn.Linear(in_dim, out_dim)
        self.label_linear = nn.Linear(n_classes, out_dim, bias=False)
        self.out_dim = out_dim
        self.activation = activation

    def _activate(self, h):
        if self.activation == "gelu":
            return F.gelu(h)
        elif self.activation == "swish":
            return h * torch.sigmoid(h)
        return F.relu(h)

    def forward(self, h_feat, label_oh):
        h = self._activate(self.feature_linear(h_feat))
        h_combined = h + self.label_linear(label_oh)
        return h, h_combined

def l2_normalize(h):
    return h / (h.norm(dim=-1, keepdim=True) + 1e-6)

def embed_labels(images, labels, n_classes, scale=5.0):
    oh = F.one_hot(labels, n_classes).float() * scale
    return l2_normalize(torch.cat([images, oh], dim=-1))

def train_and_eval(train_x, train_y, eval_x, eval_y, n_classes, in_dim,
                   goodness_name="burstiness", activation="gelu", pathway="std",
                   label_scale=5.0, threshold=2.0, lr=0.001, epochs=60, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    goodness_fn = GOODNESS_MAP[goodness_name]
    n_layers, hidden_dim = 4, 2000

    if pathway == "std":
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
                    loss = (F.softplus(threshold - goodness_fn(a_pos)).mean() +
                            F.softplus(goodness_fn(a_neg) - threshold).mean())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        h_pos = l2_normalize(a_pos.detach())
                        h_neg = l2_normalize(a_neg.detach())

        N = len(eval_x)
        g_per_layer = torch.zeros(N, n_classes, device=DEVICE)
        g_concat = torch.zeros(N, n_classes, device=DEVICE)

        for c in range(n_classes):
            c_lbls = torch.full((N,), c, dtype=torch.long, device=DEVICE)
            x = embed_labels(eval_x, c_lbls, n_classes, scale=label_scale)
            h = x
            layer_acts = []
            b_good = torch.zeros(N, device=DEVICE)

            for layer in layers:
                h = layer(h)
                b_good += goodness_fn(h)
                layer_acts.append(h)
                h = l2_normalize(h)

            g_per_layer[:, c] = b_good
            g_concat[:, c] = b_good + goodness_fn(torch.cat(layer_acts, dim=-1))

        acc_std = (g_per_layer.argmax(dim=1) == eval_y).float().mean().item() * 100.0
        acc_concat = (g_concat.argmax(dim=1) == eval_y).float().mean().item() * 100.0
        return acc_std, acc_concat

    else: # Per-layer Label Forwarding
        in_dims = [in_dim] + [hidden_dim] * (n_layers - 1)
        layers = [FFCLLayer(in_dims[i], hidden_dim, n_classes, activation).to(DEVICE) for i in range(n_layers)]
        opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in layers]

        n = len(train_x)
        for epoch in range(epochs):
            perm = torch.randperm(n, device=DEVICE)
            for start in range(0, n, 500):
                idx = perm[start:start + 500]
                imgs, lbls = train_x[idx], train_y[idx]
                wrong = (lbls + torch.randint(1, n_classes, lbls.shape, device=DEVICE)) % n_classes

                pos_oh = F.one_hot(lbls, n_classes).float() * (label_scale / 5.0)
                neg_oh = F.one_hot(wrong, n_classes).float() * (label_scale / 5.0)

                h_pos = l2_normalize(imgs)
                h_neg = l2_normalize(imgs)

                for layer, opt in zip(layers, opts):
                    feat_pos, comb_pos = layer(h_pos, pos_oh)
                    feat_neg, comb_neg = layer(h_neg, neg_oh)

                    loss = (F.softplus(threshold - goodness_fn(comb_pos)).mean() +
                            F.softplus(goodness_fn(comb_neg) - threshold).mean())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        h_pos = l2_normalize(feat_pos.detach())
                        h_neg = l2_normalize(feat_neg.detach())

        N = len(eval_x)
        g_per_layer = torch.zeros(N, n_classes, device=DEVICE)
        g_concat = torch.zeros(N, n_classes, device=DEVICE)

        for c in range(n_classes):
            c_oh = F.one_hot(torch.full((N,), c, dtype=torch.long, device=DEVICE), n_classes).float() * (label_scale / 5.0)
            h = l2_normalize(eval_x)
            layer_acts = []
            b_good = torch.zeros(N, device=DEVICE)

            for layer in layers:
                feat, comb = layer(h, c_oh)
                b_good += goodness_fn(comb)
                layer_acts.append(comb)
                h = l2_normalize(feat)

            g_per_layer[:, c] = b_good
            g_concat[:, c] = b_good + goodness_fn(torch.cat(layer_acts, dim=-1))

        acc_std = (g_per_layer.argmax(dim=1) == eval_y).float().mean().item() * 100.0
        acc_concat = (g_concat.argmax(dim=1) == eval_y).float().mean().item() * 100.0
        return acc_std, acc_concat

def run_rebuttal():
    print("=" * 80)
    print("CANONICAL NEURIPS REBUTTAL EVALUATION SUITE")
    print("=" * 80)

    train_x, train_y, val_x, val_y, test_x, test_y, in_dim, n_classes = load_splits("fashion_mnist")
    seeds = [42, 43, 44, 45, 46]

    configs = [
        ("sos", "relu", "std"),
        ("burstiness", "relu", "std"),
        ("sos", "gelu", "std"),
        ("burstiness", "gelu", "std"),
        ("sos", "gelu", "per_layer_label"),
        ("burstiness", "gelu", "per_layer_label"),
    ]

    canonical_results = {}
    scales = [2.0, 5.0, 10.0]
    thresholds = [1.0, 2.0, 5.0]

    for goodness, act, pathway in configs:
        key = f"{goodness}_{act}_{pathway}"
        print(f"\n==================================================")
        print(f" PROCESSING CONFIG: {key.upper()}")
        print(f"==================================================")

        # 1. Validation Tuning (Matching 60 Epoch Budget)
        best_val, best_hp = -1.0, None
        for s in scales:
            for t in thresholds:
                v_std, v_concat = train_and_eval(
                    train_x, train_y, val_x, val_y, n_classes, in_dim,
                    goodness_name=goodness, activation=act, pathway=pathway,
                    label_scale=s, threshold=t, lr=0.001, epochs=60, seed=42
                )
                print(f"  Val Config: scale={s:<4} thresh={t:<4} --> Val Std Acc: {v_std:.2f}% (Concat: {v_concat:.2f}%)")
                if v_std > best_val:
                    best_val = v_std
                    best_hp = {"label_scale": s, "threshold": t, "lr": 0.001}

        print(f"===> SELECTED VAL HYPERPARAMETERS: {best_hp} (Val Acc: {best_val:.2f}%)")

        # 2. Final 5-Seed Evaluation on Untouched 10k Test Set
        test_std_accs = []
        test_concat_accs = []

        for seed in seeds:
            t_std, t_concat = train_and_eval(
                train_x, train_y, test_x, test_y, n_classes, in_dim,
                goodness_name=goodness, activation=act, pathway=pathway,
                label_scale=best_hp["label_scale"], threshold=best_hp["threshold"],
                lr=best_hp["lr"], epochs=60, seed=seed
            )
            test_std_accs.append(t_std)
            test_concat_accs.append(t_concat)
            print(f"  Seed {seed}: Test Std Acc = {t_std:.2f}% | Concat Acc = {t_concat:.2f}%")

        std_arr = np.array(test_std_accs)
        concat_arr = np.array(test_concat_accs)

        canonical_results[key] = {
            "val_selected_hps": best_hp,
            "val_selected_accuracy": float(best_val),
            "test_std_mean": float(std_arr.mean()),
            "test_std_std": float(std_arr.std(ddof=1)),
            "test_concat_mean": float(concat_arr.mean()),
            "test_concat_std": float(concat_arr.std(ddof=1)),
            "per_seed_std_accs": test_std_accs,
            "per_seed_concat_accs": test_concat_accs
        }

        print(f"\n===> FINAL 5-SEED TEST RESULT for {key}:")
        print(f"     Standard Per-Layer Scoring: {std_arr.mean():.2f}% +/- {std_arr.std(ddof=1):.2f}%")
        print(f"     Concatenated Ensemble Score: {concat_arr.mean():.2f}% +/- {concat_arr.std(ddof=1):.2f}%")

    out_file = Path("rebuttal_results.json")
    with open(out_file, "w") as f:
        json.dump(canonical_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL CANONICAL REBUTTAL EXPERIMENTS COMPLETED! Saved to {out_file.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    run_rebuttal()
