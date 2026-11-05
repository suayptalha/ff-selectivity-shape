#!/usr/bin/env python3
"""
ff_experiments.py — Forward-Forward combinatorial ablation experiments.

Evaluates all 30 combinations of evolved components:
  3 activations (relu, gelu, swish) × 5 goodness × 2 norm-gate (on/off)

Evaluation matches the evolution pipeline: goodness summed across all
layers PLUS a concat-of-all-layers goodness term.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from entmax import sparsemax, entmax15




# ======================================================================
# Data
# ======================================================================

def load_dataset(name):
    from torchvision import datasets
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

    if name == "cifar10":
        tr = datasets.CIFAR10(str(data_dir), train=True, download=True)
        te = datasets.CIFAR10(str(data_dir), train=False, download=True)
        tr_img = torch.from_numpy(np.array(tr.data)).float().reshape(-1, 3072) / 255.0
        te_img = torch.from_numpy(np.array(te.data)).float().reshape(-1, 3072) / 255.0
        tr_lbl = torch.tensor(tr.targets, dtype=torch.long)
        te_lbl = torch.tensor(te.targets, dtype=torch.long)
        return tr_img, tr_lbl, te_img, te_lbl, 3072, 10

    if name == "usps":
        tr = datasets.USPS(str(data_dir), train=True, download=True)
        te = datasets.USPS(str(data_dir), train=False, download=True)
        tr_img = torch.from_numpy(np.array(tr.data)).float().reshape(-1, 256) / 255.0
        te_img = torch.from_numpy(np.array(te.data)).float().reshape(-1, 256) / 255.0
        tr_lbl = torch.tensor(tr.targets, dtype=torch.long)
        te_lbl = torch.tensor(te.targets, dtype=torch.long)
        return tr_img, tr_lbl, te_img, te_lbl, 256, 10

    if name == "svhn":
        tr = datasets.SVHN(str(data_dir), split="train", download=True)
        te = datasets.SVHN(str(data_dir), split="test", download=True)
        tr_img = torch.from_numpy(tr.data).float().reshape(-1, 3072) / 255.0
        te_img = torch.from_numpy(te.data).float().reshape(-1, 3072) / 255.0
        tr_lbl = torch.tensor(tr.labels, dtype=torch.long)
        te_lbl = torch.tensor(te.labels, dtype=torch.long)
        return tr_img, tr_lbl, te_img, te_lbl, 3072, 10

    if name == "emnist":
        tr = datasets.EMNIST(str(data_dir), split="letters", train=True, download=True)
        te = datasets.EMNIST(str(data_dir), split="letters", train=False, download=True)
        tr_img = tr.data.float().view(-1, 784) / 255.0
        te_img = te.data.float().view(-1, 784) / 255.0
        tr_lbl = tr.targets.long() - 1
        te_lbl = te.targets.long() - 1
        return tr_img, tr_lbl, te_img, te_lbl, 784, 26

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
        if self.activation == "layernorm_relu":
            h = F.relu(F.layer_norm(h, [self.out_dim]))
        elif self.activation == "ln_gelu":
            h = F.gelu(F.layer_norm(h, [self.out_dim]))
        elif self.activation == "ln_swish":
            h = F.layer_norm(h, [self.out_dim])
            h = h * torch.sigmoid(h)
        elif self.activation == "gelu":
            h = F.gelu(h)
        elif self.activation == "swish":
            h = h * torch.sigmoid(h)
        else:
            h = F.relu(h)
        return h


class FFCLLayer(nn.Module):
    """FFCL-style layer with separate feature and label pathways.

    Features are processed through the main linear + activation.
    Labels are projected into the same space via a separate linear and
    added to the features for goodness computation. Only pure features
    are passed to the next layer, keeping representations label-free.
    """

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
        if self.activation == "ln_swish":
            h = F.layer_norm(h, [self.out_dim])
            return h * torch.sigmoid(h)
        if self.activation == "swish":
            return h * torch.sigmoid(h)
        return F.relu(h)

    def forward(self, h_feat, label_oh):
        h = self._activate(self.feature_linear(h_feat))
        h_combined = h + self.label_linear(label_oh)
        return h, h_combined


# ======================================================================
# Vanilla (baseline) component functions
# ======================================================================

def _goodness_sos(activations):
    """Sum of squares (Hinton's original)."""
    return activations.pow(2).mean(dim=-1)


def _goodness_variance(activations):
    return activations.var(dim=-1)


def _goodness_topk(activations):
    """Mean of top-k activations. k scales with dim (2%, min 5)."""
    k = max(5, int(activations.shape[-1] * 0.02))
    return torch.topk(activations, k=k, dim=-1).values.mean(dim=-1)


def _goodness_entropy(activations):
    """Negative entropy of softmax distribution."""
    return -(torch.softmax(activations, dim=-1) *
             torch.log_softmax(activations, dim=-1)).sum(dim=-1)


def _goodness_contrast_topk(activations):
    """Spread between most-active and least-active neurons. k scales with dim."""
    k = max(5, int(activations.shape[-1] * 0.01))
    top = torch.topk(activations, k=k, dim=-1).values.mean(dim=-1)
    bot = torch.topk(-activations, k=k, dim=-1).values.mean(dim=-1)
    return top - bot


def _goodness_layernorm_topk(activations):
    """LayerNorm → top-k: normalizes activation distribution before selecting
    the top-k neurons, making the threshold stable across layers and epochs."""
    h = F.layer_norm(activations, [activations.shape[-1]])
    k = max(5, int(h.shape[-1] * 0.02))
    return torch.topk(h, k=k, dim=-1).values.mean(dim=-1)


def _goodness_softmax_energy_margin(h, temperature=1.0, scale=0.1):
    """From "In Search of Goodness" (Shah & Tripathi, 2025).
    Base sum-of-squares plus a softmax-energy (negative logsumexp) term that
    encourages peaky activation distributions.  Adapted to our mean-based
    convention so thresholds stay compatible."""
    base = h.pow(2).mean(dim=-1)
    logits = h / max(temperature, 1e-8)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits.clamp(min=-88.0, max=88.0)
    entropy_term = -torch.logsumexp(logits, dim=-1)
    base_mean = base.mean().clamp(min=1e-8)
    return base + scale * base_mean * entropy_term


def _goodness_game_theoretic(h, importance_weight=0.1):
    """From "In Search of Goodness" (Shah & Tripathi, 2025).
    Weights each neuron's squared activation by its relative magnitude
    (a cheap proxy for Shapley-value feature importance)."""
    base = h.pow(2).mean(dim=-1)
    magnitude = torch.abs(h)
    importance = magnitude / (magnitude.sum(dim=-1, keepdim=True) + 1e-8)
    weighted = (h.pow(2) * (1.0 + importance)).mean(dim=-1)
    return base + importance_weight * (weighted - base)


def _goodness_sparsemax_energy(h):
    """Sparsemax over activations → sparse-weighted sum of squares.
    Sparsemax produces exact zeros, giving hard sparse weighting."""
    weights = sparsemax(h, dim=-1)
    return (weights * h.pow(2)).sum(dim=-1)


def _goodness_entmax15_energy(h):
    """α=1.5 entmax over activations → adaptive-sparse-weighted sum of squares.
    Midpoint between softmax (dense) and sparsemax (hard sparse)."""
    weights = entmax15(h, dim=-1)
    return (weights * h.pow(2)).sum(dim=-1)


def _goodness_entmax_topk(h):
    """Entmax-1.5 to get sparse weights, then top-k of the weighted activations.
    Combines learned sparse weighting with hard selection."""
    weights = entmax15(h, dim=-1)
    weighted = weights * h
    k = max(5, int(h.shape[-1] * 0.02))
    return torch.topk(weighted, k=k, dim=-1).values.mean(dim=-1)


def _goodness_burstiness(h):
    """Excess kurtosis (burstiness): scale-invariant measure of heavy-tailed
    activation distributions.  g = E[(h-μ)^4] / Var(h)^2 - 3."""
    mu = h.mean(dim=-1, keepdim=True)
    diff = h - mu
    var = diff.pow(2).mean(dim=-1).clamp(min=1e-12)
    kurt = diff.pow(4).mean(dim=-1) / var.pow(2)
    return kurt - 3.0


def _goodness_layernorm_burstiness(h):
    """LayerNorm → burstiness."""
    h = F.layer_norm(h, [h.shape[-1]])
    return _goodness_burstiness(h)


GOODNESS_FUNCTIONS = {
    'sos': _goodness_sos,
    'variance': _goodness_variance,
    'topk': _goodness_topk,
    'entropy': _goodness_entropy,
    'contrast_topk': _goodness_contrast_topk,
    'layernorm_topk': _goodness_layernorm_topk,
    'softmax_energy_margin': _goodness_softmax_energy_margin,
    'game_theoretic': _goodness_game_theoretic,
    'sparsemax_energy': _goodness_sparsemax_energy,
    'entmax15_energy': _goodness_entmax15_energy,
    'entmax_topk': _goodness_entmax_topk,
    'burstiness': _goodness_burstiness,
    'layernorm_burstiness': _goodness_layernorm_burstiness,
}


def _norm_gate(h):
    """Norm-gating: scale activations by sigmoid of their L2 norm."""
    return h * torch.sigmoid(h.norm(dim=-1, keepdim=True))


def _vanilla_norm_gate(h):
    return h


def _l2_normalize(h):
    return h / (h.norm(dim=-1, keepdim=True) + 1e-6)


def embed_labels(images, labels, n_classes, scale=5.0):
    oh = F.one_hot(labels, n_classes).float() * scale
    x = torch.cat([images, oh], dim=-1)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _vanilla_neg_gen(images, labels, n_classes):
    wrong = (labels + torch.randint(1, n_classes, labels.shape,
                                    device=labels.device)) % n_classes
    return embed_labels(images, wrong, n_classes)


# ======================================================================
# Training
# ======================================================================

def train_ff(layers, train_images, train_labels, n_classes, device,
             goodness_fn, norm_gate_fn, neg_gen_fn,
             epochs=20, batch_size=500, lr=0.001, threshold=2.0):

    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in layers]
    n_samples = len(train_images)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            imgs, lbls = train_images[idx], train_labels[idx]

            pos_in = embed_labels(imgs, lbls, n_classes)
            neg_in = neg_gen_fn(imgs, lbls, n_classes)

            h_pos, h_neg = pos_in, neg_in

            for layer, opt in zip(layers, optimizers):
                a_pos = layer(h_pos)
                a_neg = layer(h_neg)

                g_pos = goodness_fn(a_pos)
                g_neg = goodness_fn(a_neg)

                loss = (F.softplus(threshold - g_pos).mean() +
                        F.softplus(g_neg - threshold).mean())

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_batches += 1

                with torch.no_grad():
                    h_pos = norm_gate_fn(a_pos.detach())
                    h_neg = norm_gate_fn(a_neg.detach())
                    h_pos = _l2_normalize(h_pos)
                    h_neg = _l2_normalize(h_neg)

        avg = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}", flush=True)


# ======================================================================
# FFCL Training (separate label/feature pathways)
# ======================================================================

def train_ff_ffcl(layers, train_images, train_labels, n_classes, device,
                  goodness_fn, norm_gate_fn,
                  epochs=20, batch_size=500, lr=0.001, threshold=2.0):
    """FFCL-style training: labels are injected at every layer via a separate
    projection, not concatenated at the input. Only pure features propagate
    between layers."""

    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in layers]
    n_samples = len(train_images)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            imgs, lbls = train_images[idx], train_labels[idx]

            pos_label_oh = F.one_hot(lbls, n_classes).float()
            wrong = (lbls + torch.randint(1, n_classes, lbls.shape,
                                          device=device)) % n_classes
            neg_label_oh = F.one_hot(wrong, n_classes).float()

            h_pos = _l2_normalize(imgs)
            h_neg = _l2_normalize(imgs)

            for layer, opt in zip(layers, optimizers):
                feat_pos, combined_pos = layer(h_pos, pos_label_oh)
                feat_neg, combined_neg = layer(h_neg, neg_label_oh)

                g_pos = goodness_fn(norm_gate_fn(combined_pos))
                g_neg = goodness_fn(norm_gate_fn(combined_neg))

                loss = (F.softplus(threshold - g_pos).mean() +
                        F.softplus(g_neg - threshold).mean())

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_batches += 1

                with torch.no_grad():
                    h_pos = _l2_normalize(feat_pos.detach())
                    h_neg = _l2_normalize(feat_neg.detach())

        avg = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}", flush=True)


# ======================================================================
# FFCL Multi-pass evaluation
# ======================================================================

@torch.no_grad()
def evaluate_ff_ffcl(layers, images, labels, n_classes, device,
                     goodness_fn, norm_gate_fn, batch_size=2000):
    """Multi-pass evaluation for FFCL: injects each class hypothesis at
    every layer and sums goodness across layers + concat goodness."""
    N = len(images)
    goodness_matrix = torch.zeros(N, n_classes, device=device)

    for c in range(n_classes):
        label_oh = F.one_hot(
            torch.full((N,), c, dtype=torch.long, device=device), n_classes
        ).float()

        class_good = torch.zeros(N, device=device)

        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            b_img = images[b_start:b_end]
            b_label_oh = label_oh[b_start:b_end]

            h = _l2_normalize(b_img)
            layer_acts = []
            norm_gated_acts = []
            b_good = torch.zeros(b_end - b_start, device=device)

            for layer in layers:
                feat, combined = layer(h, b_label_oh)
                b_good += goodness_fn(combined)
                layer_acts.append(combined)
                ng = norm_gate_fn(combined)
                norm_gated_acts.append(ng)
                h = _l2_normalize(feat)

            if len(layer_acts) > 1:
                b_good += goodness_fn(torch.cat(layer_acts, dim=-1))
                b_good += goodness_fn(torch.cat(norm_gated_acts, dim=-1))

            class_good[b_start:b_end] = b_good

        goodness_matrix[:, c] = class_good

    preds = goodness_matrix.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    val_loss = F.cross_entropy(goodness_matrix, labels).item()
    return accuracy, val_loss


# ======================================================================
# Multi-pass evaluation (standard)
# ======================================================================

@torch.no_grad()
def evaluate_ff(layers, images, labels, n_classes, device,
                goodness_fn, norm_gate_fn, batch_size=2000):
    """Multi-pass FF evaluation matching evolution pipeline.
    Sums goodness across: each layer individually + concat of all layers.
    Also includes norm-gated concat (matching evolution's output_3).
    """
    N = len(images)
    goodness_matrix = torch.zeros(N, n_classes, device=device)

    for c in range(n_classes):
        c_labels = torch.full((N,), c, dtype=torch.long, device=device)
        class_good = torch.zeros(N, device=device)

        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            b_img = images[b_start:b_end]
            b_lbl = c_labels[b_start:b_end]

            x = embed_labels(b_img, b_lbl, n_classes)
            h = x
            layer_acts = []
            norm_gated_acts = []
            b_good = torch.zeros(b_end - b_start, device=device)

            for layer in layers:
                h = layer(h)
                b_good += goodness_fn(h)
                layer_acts.append(h)
                ng = norm_gate_fn(h)
                norm_gated_acts.append(ng)
                h = _l2_normalize(ng)

            if len(layer_acts) > 1:
                b_good += goodness_fn(torch.cat(layer_acts, dim=-1))
                b_good += goodness_fn(torch.cat(norm_gated_acts, dim=-1))

            class_good[b_start:b_end] = b_good

        goodness_matrix[:, c] = class_good

    preds = goodness_matrix.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    val_loss = F.cross_entropy(goodness_matrix, labels).item()
    return accuracy, val_loss


# ======================================================================
# Experiment configuration
# ======================================================================

# Evolved layer_1 activations (from supergraph)
EVOLVED_ACTIVATIONS = ['gelu', 'swish', 'ln_gelu', 'ln_swish']

# Evolved layer_2 input modes: direct from layer_1, or through norm_gate
EVOLVED_NORM_GATE = [False, True]  # False = layer_2_0, True = layer_2_1


def build_all_variants():
    """Build all combinatorial variants from the evolved supergraph.

    Standard: 2 activations × 11 goodness × 2 norm-gate = 44
    FFCL:     2 activations × 11 goodness × 2 norm-gate = 44
    Plus vanilla baseline = 89 total.
    """
    variants = {}

    # Vanilla FF baseline (not evolved — for reference)
    variants['vanilla'] = {
        'desc': 'vanilla (relu/sos/no_ng)',
        'activation': 'relu',
        'goodness_fn': _goodness_sos,
        'goodness_desc': 'sos',
        'norm_gate_fn': _vanilla_norm_gate,
        'has_norm_gate': False,
        'neg_gen_fn': _vanilla_neg_gen,
        'neg_gen_desc': 'wrong_label',
        'ffcl': False,
    }

    # Standard variants (label concatenated at input)
    for act in EVOLVED_ACTIVATIONS:
        for gname, gfn in GOODNESS_FUNCTIONS.items():
            for ng in EVOLVED_NORM_GATE:
                ng_tag = "ng" if ng else "no_ng"
                key = f"{act}_{gname}_{ng_tag}"
                variants[key] = {
                    'desc': f"{act} / {gname} / {ng_tag}",
                    'activation': act,
                    'goodness_fn': gfn,
                    'goodness_desc': gname,
                    'norm_gate_fn': _norm_gate if ng else _vanilla_norm_gate,
                    'has_norm_gate': ng,
                    'neg_gen_fn': _vanilla_neg_gen,
                    'neg_gen_desc': 'wrong_label',
                    'ffcl': False,
                }

    # FFCL variants (separate label/feature forwarding at every layer)
    for act in EVOLVED_ACTIVATIONS:
        for gname, gfn in GOODNESS_FUNCTIONS.items():
            for ng in EVOLVED_NORM_GATE:
                ng_tag = "ng" if ng else "no_ng"
                key = f"ffcl_{act}_{gname}_{ng_tag}"
                variants[key] = {
                    'desc': f"FFCL {act} / {gname} / {ng_tag}",
                    'activation': act,
                    'goodness_fn': gfn,
                    'goodness_desc': gname,
                    'norm_gate_fn': _norm_gate if ng else _vanilla_norm_gate,
                    'has_norm_gate': ng,
                    'neg_gen_fn': _vanilla_neg_gen,
                    'neg_gen_desc': 'wrong_label',
                    'ffcl': True,
                }

    return variants


# ======================================================================
# Run single experiment
# ======================================================================

def run_experiment(args, variants=None):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if variants is None:
        variants = build_all_variants()

    if args.variant not in variants:
        print(f"  Unknown variant '{args.variant}'. Available: {list(variants.keys())}")
        sys.exit(1)

    v = variants[args.variant]

    is_ffcl = v.get('ffcl', False)

    print(f"\n{'='*60}")
    print(f"  {v['desc']}")
    print(f"  dataset={args.dataset}  arch={args.n_layers}x{args.hidden}")
    print(f"  epochs={args.epochs}  lr={args.lr}  gpu={args.gpu}  seed={args.seed}")
    print(f"  activation={v['activation']}  goodness={v['goodness_desc']}")
    print(f"  norm_gate={v['has_norm_gate']}  ffcl={is_ffcl}")
    print(f"{'='*60}", flush=True)

    tr_img, tr_lbl, te_img, te_lbl, input_dim, n_classes = load_dataset(args.dataset)

    mu, std = tr_img.mean(), tr_img.std()
    tr_img = (tr_img - mu) / (std + 1e-8)
    te_img = (te_img - mu) / (std + 1e-8)

    tr_img, tr_lbl = tr_img.to(device), tr_lbl.to(device)
    te_img, te_lbl = te_img.to(device), te_lbl.to(device)

    if is_ffcl:
        layers = []
        for i in range(args.n_layers):
            d_in = input_dim if i == 0 else args.hidden
            layers.append(FFCLLayer(d_in, args.hidden, n_classes,
                                    activation=v["activation"]).to(device))

        t0 = time.time()
        train_ff_ffcl(layers, tr_img, tr_lbl, n_classes, device,
                      goodness_fn=v["goodness_fn"],
                      norm_gate_fn=v["norm_gate_fn"],
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, threshold=args.threshold)
        train_sec = time.time() - t0

        test_acc, test_loss = evaluate_ff_ffcl(
            layers, te_img, te_lbl, n_classes, device,
            goodness_fn=v["goodness_fn"],
            norm_gate_fn=v["norm_gate_fn"])
    else:
        first_dim = input_dim + n_classes
        layers = []
        for i in range(args.n_layers):
            d_in = first_dim if i == 0 else args.hidden
            layers.append(FFLayer(d_in, args.hidden,
                                  activation=v["activation"]).to(device))

        t0 = time.time()
        train_ff(layers, tr_img, tr_lbl, n_classes, device,
                 goodness_fn=v["goodness_fn"],
                 norm_gate_fn=v["norm_gate_fn"],
                 neg_gen_fn=v["neg_gen_fn"],
                 epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, threshold=args.threshold)
        train_sec = time.time() - t0

        test_acc, test_loss = evaluate_ff(
            layers, te_img, te_lbl, n_classes, device,
            goodness_fn=v["goodness_fn"],
            norm_gate_fn=v["norm_gate_fn"])

    n_params = sum(p.numel() for layer in layers for p in layer.parameters())

    results = {
        "variant": args.variant,
        "description": v["desc"],
        "dataset": args.dataset,
        "n_layers": args.n_layers,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "seed": args.seed,
        "test_accuracy": round(test_acc, 6),
        "test_loss": round(test_loss, 6),
        "train_time_sec": round(train_sec, 1),
        "n_params": n_params,
        "activation": v["activation"],
        "goodness": v["goodness_desc"],
        "norm_gate": v["has_norm_gate"],
        "ffcl": is_ffcl,
    }

    print(f"\n  ✓ Test Accuracy: {test_acc*100:.2f}%", flush=True)
    print(f"    Test Loss:    {test_loss:.4f}", flush=True)
    print(f"    Params:       {n_params:,}", flush=True)
    print(f"    Train Time:   {train_sec:.1f}s\n", flush=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{args.variant}_{args.dataset}_{args.n_layers}x{args.hidden}.json"
    with open(results_dir / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {results_dir / fname}")
    return results


# ======================================================================
# Collect all results into summary tables
# ======================================================================

def collect_results(results_dir, supergraph=None):
    rd = Path(results_dir)
    if not rd.exists():
        print(f"No results directory: {rd}")
        return

    all_results = []
    for f in sorted(rd.glob("*.json")):
        with open(f) as fh:
            all_results.append(json.load(fh))

    if not all_results:
        print("No result files found.")
        return

    archs = sorted(set(f"{r['n_layers']}x{r['hidden']}" for r in all_results))
    datasets = sorted(set(r["dataset"] for r in all_results))

    for arch in archs:
        ar = [r for r in all_results if f"{r['n_layers']}x{r['hidden']}" == arch]
        ds_in_arch = sorted(set(r["dataset"] for r in ar))

        for ds in ds_in_arch:
            ds_results = [r for r in ar if r["dataset"] == ds]
            ds_results.sort(key=lambda r: -r["test_accuracy"])

            print(f"\n{'=' * 90}")
            print(f"  {ds.upper()} — {arch} — Ranked by Test Accuracy")
            print("=" * 90)
            print(f"  {'#':<4s} {'Activation':<10s} {'Goodness':<16s} {'NormGate':<10s}"
                  f" {'Acc%':>8s}  {'Loss':>8s}  {'Time':>6s}")
            print("  " + "-" * 82)

            best_acc = ds_results[0]["test_accuracy"] if ds_results else 0
            for rank, r in enumerate(ds_results, 1):
                acc = r["test_accuracy"] * 100
                delta = acc - best_acc * 100
                ng = "yes" if r.get("norm_gate", False) else "no"
                marker = " *" if rank == 1 else ""
                print(f"  {rank:<4d} {r['activation']:<10s} {r['goodness']:<16s} {ng:<10s}"
                      f" {acc:>7.2f}%  {r['test_loss']:>8.4f}  {r['train_time_sec']:>5.0f}s{marker}")

    # --- Summary: best config per dataset ---
    print(f"\n{'=' * 90}")
    print("  BEST CONFIG PER DATASET")
    print("=" * 90)
    for arch in archs:
        for ds in datasets:
            matches = [r for r in all_results
                       if f"{r['n_layers']}x{r['hidden']}" == arch and r["dataset"] == ds]
            if matches:
                best = max(matches, key=lambda r: r["test_accuracy"])
                ng = "norm_gate" if best.get("norm_gate", False) else "no_ng"
                print(f"  {ds:<16s} {arch:<8s}  {best['activation']}/{best['goodness']}/{ng}"
                      f"  → {best['test_accuracy']*100:.2f}%")

    # --- Pivot: goodness function comparison ---
    print(f"\n{'=' * 90}")
    print("  GOODNESS FUNCTION COMPARISON (averaged over activations & norm-gate)")
    print("=" * 90)
    for arch in archs:
        for ds in datasets:
            matches = [r for r in all_results
                       if f"{r['n_layers']}x{r['hidden']}" == arch and r["dataset"] == ds]
            if not matches:
                continue
            print(f"\n  {ds} ({arch}):")
            gnames = sorted(set(r["goodness"] for r in matches))
            for gn in gnames:
                gmatches = [r for r in matches if r["goodness"] == gn]
                avg_acc = sum(r["test_accuracy"] for r in gmatches) / len(gmatches) * 100
                best_acc = max(r["test_accuracy"] for r in gmatches) * 100
                print(f"    {gn:<16s}  avg={avg_acc:6.2f}%  best={best_acc:6.2f}%  (n={len(gmatches)})")

    print()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF experiments for paper")
    sub = parser.add_subparsers(dest="cmd")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--dataset", default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10",
                                 "usps", "svhn", "emnist"])
    p_run.add_argument("--variant", required=True)
    p_run.add_argument("--n_layers", type=int, default=2)
    p_run.add_argument("--hidden", type=int, default=500)
    p_run.add_argument("--epochs", type=int, default=60)
    p_run.add_argument("--lr", type=float, default=0.001)
    p_run.add_argument("--batch_size", type=int, default=500)
    p_run.add_argument("--threshold", type=float, default=2.0)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--gpu", type=int, default=0)
    p_run.add_argument("--results_dir", default="experiment_results")

    # --- list ---
    p_list = sub.add_parser("list", help="List all 30 combinatorial variants")

    # --- collect ---
    p_col = sub.add_parser("collect", help="Collect and display all results")
    p_col.add_argument("--results_dir", default="experiment_results")

    args = parser.parse_args()

    if args.cmd == "run":
        run_experiment(args)
    elif args.cmd == "list":
        variants = build_all_variants()
        print(f"\nAvailable variants ({len(variants)}):")
        for name, v in variants.items():
            print(f"  {name:<35s}  {v['desc']}")
        print()
    elif args.cmd == "collect":
        collect_results(args.results_dir)
    else:
        parser.print_help()
