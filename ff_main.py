#!/usr/bin/env python3
"""
Goodness Functions for Forward-Forward Learning
============================================

Reproduces the main experiments from:
  "Selectivity and Shape in the Design of Forward-Forward Goodness Functions"

Usage:
  python ff_main.py                  # Run main comparison (Table 1)
  python ff_main.py --full             # Run full goodness comparison (Table 2)
  python ff_main.py --dataset mnist    # Single dataset
  python ff_main.py --arch 4 2000     # Single architecture (layers, hidden)
"""

import argparse
import time
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from entmax import sparsemax, entmax15
from torchvision import datasets

# ======================================================================
# Data
# ======================================================================

def load_dataset(name):
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    if name == "mnist":
        tr = datasets.MNIST(str(data_dir), train=True, download=True)
        te = datasets.MNIST(str(data_dir), train=False, download=True)
        tr_img = tr.data.float().view(-1, 784) / 255.0
        te_img = te.data.float().view(-1, 784) / 255.0
        return tr_img, tr.targets.long(), te_img, te.targets.long(), 784, 10

    if name == "fashion_mnist":
        tr = datasets.FashionMNIST(str(data_dir), train=True, download=True)
        te = datasets.FashionMNIST(str(data_dir), train=False, download=True)
        tr_img = tr.data.float().view(-1, 784) / 255.0
        te_img = te.data.float().view(-1, 784) / 255.0
        return tr_img, tr.targets.long(), te_img, te.targets.long(), 784, 10

    raise ValueError(f"Unknown dataset: {name}")


# ======================================================================
# FF Layer
# ======================================================================

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
        elif self.activation == "ln_gelu":
            return F.gelu(F.layer_norm(h, [self.out_dim]))
        elif self.activation == "swish":
            return h * torch.sigmoid(h)
        elif self.activation == "ln_swish":
            h = F.layer_norm(h, [self.out_dim])
            return h * torch.sigmoid(h)
        return F.relu(h)


class FFCLLayer(nn.Module):
    """FFCL-style layer: separate feature and label pathways."""

    def __init__(self, in_dim, out_dim, n_classes, activation="gelu"):
        super().__init__()
        self.feature_linear = nn.Linear(in_dim, out_dim)
        self.label_linear = nn.Linear(n_classes, out_dim, bias=False)
        self.out_dim = out_dim
        self.activation = activation

    def _activate(self, h):
        if self.activation == "gelu":
            return F.gelu(h)
        if self.activation == "ln_gelu":
            return F.gelu(F.layer_norm(h, [self.out_dim]))
        if self.activation == "swish":
            return h * torch.sigmoid(h)
        if self.activation == "ln_swish":
            h = F.layer_norm(h, [self.out_dim])
            return h * torch.sigmoid(h)
        return F.relu(h)

    def forward(self, h_feat, label_oh):
        h = self._activate(self.feature_linear(h_feat))
        h_combined = h + self.label_linear(label_oh)
        return h, h_combined


# ======================================================================
# Goodness functions
# ======================================================================

def goodness_sos(h):
    """Sum of squares (Hinton's original)."""
    return h.pow(2).mean(dim=-1)


def goodness_topk(h):
    """Mean of top-k activations (k = 2% of dim, min 5)."""
    k = max(5, int(h.shape[-1] * 0.02))
    return torch.topk(h, k=k, dim=-1).values.mean(dim=-1)


def goodness_contrast_topk(h):
    """Spread between most-active and least-active neurons."""
    k = max(5, int(h.shape[-1] * 0.01))
    top = torch.topk(h, k=k, dim=-1).values.mean(dim=-1)
    bot = torch.topk(-h, k=k, dim=-1).values.mean(dim=-1)
    return top - bot


def goodness_variance(h):
    return h.var(dim=-1)


def goodness_entropy(h):
    """Negative entropy of softmax distribution."""
    p = torch.softmax(h, dim=-1)
    return -(p * torch.log_softmax(h, dim=-1)).sum(dim=-1)


def goodness_layernorm_topk(h):
    """LayerNorm → top-k: normalizes before selecting top-k neurons."""
    h = F.layer_norm(h, [h.shape[-1]])
    k = max(5, int(h.shape[-1] * 0.02))
    return torch.topk(h, k=k, dim=-1).values.mean(dim=-1)


def goodness_softmax_energy_margin(h, temperature=1.0, scale=0.1):
    """From 'In Search of Goodness' — SoS + softmax-energy (negative logsumexp)."""
    base = h.pow(2).mean(dim=-1)
    logits = h / max(temperature, 1e-8)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits.clamp(min=-88.0, max=88.0)
    entropy_term = -torch.logsumexp(logits, dim=-1)
    base_mean = base.mean().clamp(min=1e-8)
    return base + scale * base_mean * entropy_term


def goodness_game_theoretic(h, importance_weight=0.1):
    """From 'In Search of Goodness' — importance-weighted SoS."""
    base = h.pow(2).mean(dim=-1)
    magnitude = torch.abs(h)
    importance = magnitude / (magnitude.sum(dim=-1, keepdim=True) + 1e-8)
    weighted = (h.pow(2) * (1.0 + importance)).mean(dim=-1)
    return base + importance_weight * (weighted - base)


def goodness_sparsemax_energy(h):
    """Sparsemax over activations → sparse-weighted sum of squares."""
    weights = sparsemax(h, dim=-1)
    return (weights * h.pow(2)).sum(dim=-1)


def goodness_entmax15_energy(h):
    """α=1.5 entmax → adaptive-sparse-weighted sum of squares."""
    weights = entmax15(h, dim=-1)
    return (weights * h.pow(2)).sum(dim=-1)


def goodness_entmax_topk(h):
    """Entmax-1.5 sparse weights → top-k of weighted activations."""
    weights = entmax15(h, dim=-1)
    weighted = weights * h
    k = max(5, int(h.shape[-1] * 0.02))
    return torch.topk(weighted, k=k, dim=-1).values.mean(dim=-1)


def goodness_burstiness(h):
    """Excess kurtosis (burstiness): measures heavy-tailedness."""
    mu = h.mean(dim=-1, keepdim=True)
    diff = h - mu
    var = diff.pow(2).mean(dim=-1).clamp(min=1e-12)
    kurt = (diff.pow(4).mean(dim=-1) / var.pow(2))
    return kurt - 3.0


def goodness_layernorm_burstiness(h):
    """LayerNorm → burstiness."""
    h = F.layer_norm(h, [h.shape[-1]])
    return goodness_burstiness(h)


GOODNESS_FNS = {
    "sos": goodness_sos,
    "topk": goodness_topk,
    "contrast_topk": goodness_contrast_topk,
    "variance": goodness_variance,
    "entropy": goodness_entropy,
    "layernorm_topk": goodness_layernorm_topk,
    "softmax_energy_margin": goodness_softmax_energy_margin,
    "game_theoretic": goodness_game_theoretic,
    "sparsemax_energy": goodness_sparsemax_energy,
    "entmax15_energy": goodness_entmax15_energy,
    "entmax_topk": goodness_entmax_topk,
    "burstiness": goodness_burstiness,
    "layernorm_burstiness": goodness_layernorm_burstiness,
}


# ======================================================================
# Helpers
# ======================================================================

def l2_normalize(h):
    return h / (h.norm(dim=-1, keepdim=True) + 1e-6)


def embed_labels(images, labels, n_classes, scale=5.0):
    oh = F.one_hot(labels, n_classes).float() * scale
    x = torch.cat([images, oh], dim=-1)
    return l2_normalize(x)


def make_negatives(images, labels, n_classes):
    wrong = (labels + torch.randint(1, n_classes, labels.shape,
                                    device=labels.device)) % n_classes
    return embed_labels(images, wrong, n_classes)


# ======================================================================
# Training (standard: labels concatenated at input)
# ======================================================================

def train_ff(layers, train_images, train_labels, n_classes, device,
             goodness_fn, epochs=60, batch_size=500, lr=0.001, threshold=2.0):

    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in layers]
    n = len(train_images)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            imgs, lbls = train_images[idx], train_labels[idx]

            pos_in = embed_labels(imgs, lbls, n_classes)
            neg_in = make_negatives(imgs, lbls, n_classes)
            h_pos, h_neg = pos_in, neg_in

            for layer, opt in zip(layers, optimizers):
                a_pos = layer(h_pos)
                a_neg = layer(h_neg)

                loss = (F.softplus(threshold - goodness_fn(a_pos)).mean() +
                        F.softplus(goodness_fn(a_neg) - threshold).mean())

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_batches += 1

                with torch.no_grad():
                    h_pos = l2_normalize(a_pos.detach())
                    h_neg = l2_normalize(a_neg.detach())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={total_loss/max(n_batches,1):.4f}")


# ======================================================================
# Training (FFCL: separate label/feature pathways at every layer)
# ======================================================================

def train_ff_ffcl(layers, train_images, train_labels, n_classes, device,
                  goodness_fn, epochs=60, batch_size=500, lr=0.001, threshold=2.0):

    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in layers]
    n = len(train_images)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            imgs, lbls = train_images[idx], train_labels[idx]

            pos_label_oh = F.one_hot(lbls, n_classes).float()
            wrong = (lbls + torch.randint(1, n_classes, lbls.shape,
                                          device=device)) % n_classes
            neg_label_oh = F.one_hot(wrong, n_classes).float()

            h_pos = l2_normalize(imgs)
            h_neg = l2_normalize(imgs)

            for layer, opt in zip(layers, optimizers):
                feat_pos, combined_pos = layer(h_pos, pos_label_oh)
                feat_neg, combined_neg = layer(h_neg, neg_label_oh)

                loss = (F.softplus(threshold - goodness_fn(combined_pos)).mean() +
                        F.softplus(goodness_fn(combined_neg) - threshold).mean())

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_batches += 1

                with torch.no_grad():
                    h_pos = l2_normalize(feat_pos.detach())
                    h_neg = l2_normalize(feat_neg.detach())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={total_loss/max(n_batches,1):.4f}")


# ======================================================================
# Evaluation — standard (multi-pass with ensemble scoring)
# ======================================================================

@torch.no_grad()
def evaluate_ff(layers, images, labels, n_classes, device,
                goodness_fn, batch_size=2000):
    N = len(images)
    goodness_matrix = torch.zeros(N, n_classes, device=device)

    for c in range(n_classes):
        c_labels = torch.full((N,), c, dtype=torch.long, device=device)

        for b in range(0, N, batch_size):
            b_end = min(b + batch_size, N)
            x = embed_labels(images[b:b_end], c_labels[b:b_end], n_classes)
            h = x
            layer_acts = []
            b_good = torch.zeros(b_end - b, device=device)

            for layer in layers:
                h = layer(h)
                b_good += goodness_fn(h)
                layer_acts.append(h)
                h = l2_normalize(h)

            if len(layer_acts) > 1:
                b_good += goodness_fn(torch.cat(layer_acts, dim=-1))

            goodness_matrix[b:b_end, c] = b_good

    preds = goodness_matrix.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc


# ======================================================================
# Evaluation — FFCL (label injected at every layer)
# ======================================================================

@torch.no_grad()
def evaluate_ff_ffcl(layers, images, labels, n_classes, device,
                     goodness_fn, batch_size=2000):
    N = len(images)
    goodness_matrix = torch.zeros(N, n_classes, device=device)

    for c in range(n_classes):
        label_oh = F.one_hot(
            torch.full((N,), c, dtype=torch.long, device=device), n_classes
        ).float()

        for b in range(0, N, batch_size):
            b_end = min(b + batch_size, N)
            h = l2_normalize(images[b:b_end])
            b_label_oh = label_oh[b:b_end]
            layer_acts = []
            b_good = torch.zeros(b_end - b, device=device)

            for layer in layers:
                feat, combined = layer(h, b_label_oh)
                b_good += goodness_fn(combined)
                layer_acts.append(combined)
                h = l2_normalize(feat)

            if len(layer_acts) > 1:
                b_good += goodness_fn(torch.cat(layer_acts, dim=-1))

            goodness_matrix[b:b_end, c] = b_good

    preds = goodness_matrix.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc


# ======================================================================
# Single experiment
# ======================================================================

def run_single(dataset, activation, goodness_name, n_layers, hidden,
               gpu=0, seed=42, epochs=60, lr=0.001, threshold=2.0,
               ffcl=False):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    goodness_fn = GOODNESS_FNS[goodness_name]
    tr_img, tr_lbl, te_img, te_lbl, in_dim, n_cls = load_dataset(dataset)

    mu, std = tr_img.mean(), tr_img.std()
    tr_img = (tr_img - mu) / (std + 1e-8)
    te_img = (te_img - mu) / (std + 1e-8)
    tr_img, tr_lbl = tr_img.to(device), tr_lbl.to(device)
    te_img, te_lbl = te_img.to(device), te_lbl.to(device)

    if ffcl:
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
            layers.append(FFCLLayer(d_in, hidden, n_cls,
                                    activation=activation).to(device))

        t0 = time.time()
        train_ff_ffcl(layers, tr_img, tr_lbl, n_cls, device, goodness_fn,
                      epochs=epochs, lr=lr, threshold=threshold)
        train_sec = time.time() - t0

        acc = evaluate_ff_ffcl(layers, te_img, te_lbl, n_cls, device, goodness_fn)
    else:
        first_dim = in_dim + n_cls
        layers = []
        for i in range(n_layers):
            d_in = first_dim if i == 0 else hidden
            layers.append(FFLayer(d_in, hidden, activation=activation).to(device))

        t0 = time.time()
        train_ff(layers, tr_img, tr_lbl, n_cls, device, goodness_fn,
                 epochs=epochs, lr=lr, threshold=threshold)
        train_sec = time.time() - t0

        acc = evaluate_ff(layers, te_img, te_lbl, n_cls, device, goodness_fn)

    n_params = sum(p.numel() for l in layers for p in l.parameters())

    return {
        "dataset": dataset,
        "activation": activation,
        "goodness": goodness_name,
        "arch": f"{n_layers}x{hidden}",
        "accuracy": acc,
        "params": n_params,
        "time": train_sec,
        "ffcl": ffcl,
    }


# ======================================================================
# Main experiment drivers
# ======================================================================

def run_main_comparison(datasets, archs, gpu=0, ffcl=False):
    """Table 1: Baseline vs Top-k comparison (+FFCL variants)."""
    configs = [
        ("relu", "sos", False, "FF-SoS (baseline)"),
        ("gelu", "topk", False, "FF + Top-k (GELU)"),
        ("swish", "topk", False, "FF + Top-k (Swish)"),
        ("gelu", "softmax_energy_margin", False, "FF + SoftmaxEnergyMargin (GELU)"),
        ("gelu", "game_theoretic", False, "FF + GameTheoretic (GELU)"),
    ]

    if ffcl:
        configs += [
            ("gelu", "layernorm_topk", False, "FF + LN-Top-k (GELU)"),
            ("gelu", "topk", True, "FFCL + Top-k (GELU)"),
            ("gelu", "layernorm_topk", True, "FFCL + LN-Top-k (GELU)"),
            ("swish", "layernorm_topk", True, "FFCL + LN-Top-k (Swish)"),
            ("gelu", "burstiness", True, "FFCL + Burstiness (GELU)"),
            ("ln_gelu", "burstiness", True, "FFCL + Burstiness (LN-GELU)"),
            ("ln_swish", "burstiness", True, "FFCL + Burstiness (LN-Swish)"),
        ]

    print("\n" + "=" * 80)
    title = "  Main Results — Top-k vs SoS Baseline"
    if ffcl:
        title += " + FFCL"
    print(title)
    print("=" * 80)

    for n_layers, hidden in archs:
        for ds in datasets:
            print(f"\n  --- {ds.upper()} {n_layers}x{hidden} ---")
            for act, good, use_ffcl, label in configs:
                print(f"\n  Running: {label} ...", flush=True)
                r = run_single(ds, act, good, n_layers, hidden,
                               gpu=gpu, ffcl=use_ffcl)
                print(f"  {label:<30s}  {r['accuracy']*100:6.2f}%  "
                      f"({r['params']:,} params, {r['time']:.0f}s)")

    print()


def run_full_comparison(datasets, archs, gpu=0, ffcl=False):
    """Table 2: All goodness functions (+FFCL variants)."""
    activations = ["gelu", "swish"]
    goodness_fns = ["sos", "topk", "contrast_topk", "variance", "entropy",
                    "layernorm_topk"]

    all_results = []

    for n_layers, hidden in archs:
        for ds in datasets:
            print(f"\n{'='*70}")
            print(f"  {ds.upper()} — {n_layers}x{hidden}")
            print(f"{'='*70}")

            # Vanilla baseline
            print(f"\n  Running: relu/sos (baseline) ...", flush=True)
            r = run_single(ds, "relu", "sos", n_layers, hidden, gpu=gpu)
            print(f"  {'relu/sos':<30s}  {r['accuracy']*100:6.2f}%")
            all_results.append(r)

            # Standard variants
            for act, gn in product(activations, goodness_fns):
                tag = f"{act}/{gn}"
                print(f"\n  Running: {tag} ...", flush=True)
                r = run_single(ds, act, gn, n_layers, hidden, gpu=gpu)
                print(f"  {tag:<30s}  {r['accuracy']*100:6.2f}%")
                all_results.append(r)

            # FFCL variants
            if ffcl:
                for act, gn in product(activations, goodness_fns):
                    tag = f"FFCL {act}/{gn}"
                    print(f"\n  Running: {tag} ...", flush=True)
                    r = run_single(ds, act, gn, n_layers, hidden,
                                   gpu=gpu, ffcl=True)
                    print(f"  {tag:<30s}  {r['accuracy']*100:6.2f}%")
                    all_results.append(r)

    print("\n\n" + "=" * 80)
    print("  SUMMARY: Average accuracy per goodness function")
    print("  (averaged over GELU/Swish activations)")
    print("=" * 80)

    for n_layers, hidden in archs:
        arch = f"{n_layers}x{hidden}"
        for ds in datasets:
            non_baseline = [r for r in all_results
                            if r["arch"] == arch and r["dataset"] == ds
                            and r["activation"] != "relu"]

            std_results = [r for r in non_baseline if not r.get("ffcl")]
            ffcl_results = [r for r in non_baseline if r.get("ffcl")]

            print(f"\n  {ds} ({arch}) — Standard:")
            for gn in goodness_fns:
                grs = [r for r in std_results if r["goodness"] == gn]
                if grs:
                    avg = sum(r["accuracy"] for r in grs) / len(grs) * 100
                    best = max(r["accuracy"] for r in grs) * 100
                    print(f"    {gn:<16s}  avg={avg:6.2f}%  best={best:6.2f}%")

            if ffcl_results:
                print(f"\n  {ds} ({arch}) — FFCL:")
                for gn in goodness_fns:
                    grs = [r for r in ffcl_results if r["goodness"] == gn]
                    if grs:
                        avg = sum(r["accuracy"] for r in grs) / len(grs) * 100
                        best = max(r["accuracy"] for r in grs) * 100
                        print(f"    {gn:<16s}  avg={avg:6.2f}%  best={best:6.2f}%")

    print()


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Top-k Goodness for Forward-Forward Learning")
    parser.add_argument("--full", action="store_true",
                        help="Run full goodness comparison (Table 2)")
    parser.add_argument("--ffcl", action="store_true",
                        help="Include FFCL variants (separate label/feature forwarding "
                             "+ LayerNorm top-k)")
    parser.add_argument("--dataset", nargs="+",
                        default=["mnist", "fashion_mnist"],
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--arch", nargs=2, type=int, action="append",
                        help="Architecture as: --arch LAYERS HIDDEN (repeatable)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    archs = args.arch if args.arch else [(2, 500), (4, 2000)]

    if args.full:
        run_full_comparison(args.dataset, archs, gpu=args.gpu, ffcl=args.ffcl)
    else:
        run_main_comparison(args.dataset, archs, gpu=args.gpu, ffcl=args.ffcl)
