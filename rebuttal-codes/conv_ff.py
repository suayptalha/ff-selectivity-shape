#!/usr/bin/env python3
"""
Convolutional Forward-Forward with Shape-Sensitive Goodness
===========================================================

Convolutional extension of:
  "Selectivity and Shape in the Design of Forward-Forward Goodness Functions"

This library mirrors the fully-connected FF logic in ``ff_main.py`` exactly:
  * layer/block-local training (no gradient crosses a block boundary),
  * L2-normalized propagation between blocks,
  * the softplus goodness-contrast loss  softplus(tau - g(h+)) + softplus(g(h-) - tau),
  * two label pathways: standard input-level label maps, and separate
    per-layer (FFCL-style) label forwarding,
  * PRIMARY inference: standard raw per-block goodness sum  S(c) = sum_l g(h_l^c),
    argmax over candidate classes (NO concatenated-activation ensemble term,
    matching the reviewers' request for standard FF inference).

Convolution-specific additions used for the rebuttal:
  * convolutional architectures Conv2/Conv4/Conv8/WideConv8 and a block-local
    ResNet18 (each BasicBlock is one local FF objective),
  * global-flattened convolutional goodness (SoS mean, burstiness, top-k,
    entmax-weighted energy, variance, moment-p),
  * RMS pooling for downsampling,
  * per-sample inter-block L2 / RMS normalization over (C,H,W),
  * an orthogonal channel-mixing negative control (energy-preserving,
    shape-destroying) for the falsifiable prediction H7/Sec.23.1,
  * runtime and peak-GPU-memory measurement.

Only the code is provided here; drive it with ``run_conv_experiments.py``.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from entmax import entmax15
    _HAS_ENTMAX = True
except Exception:                                   # entmax optional
    _HAS_ENTMAX = False

from torchvision import datasets


# ======================================================================
# Data (with a fixed train / validation / test split)
# ======================================================================

# split sizes follow the reviewer's Section 4.1 table
_VAL_SIZE = {
    "mnist": 5000,
    "fashion_mnist": 5000,
    "cifar10": 5000,
    "cifar100": 5000,
    "svhn": 5000,
}


def _load_hf_cifar(name):
    """Fetch CIFAR-10 / CIFAR-100 from the HuggingFace Hub instead of the
    torchvision mirror (more reliable download).  Returns uint8 arrays
    (N, H, W, 3) and integer label arrays.  Requires `pip install datasets`."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` is required for CIFAR "
            "(`pip install datasets`).") from e

    repo = "uoft-cs/cifar10" if name == "cifar10" else "uoft-cs/cifar100"
    img_key = "img"
    lbl_key = "label" if name == "cifar10" else "fine_label"
    ds = load_dataset(repo)

    def to_np(split):
        d = ds[split]
        imgs = np.stack([np.asarray(im) for im in d[img_key]])   # (N,32,32,3) uint8
        lbls = np.asarray(d[lbl_key], dtype=np.int64)
        return imgs, lbls

    tr_img, tr_lbl = to_np("train")
    te_img, te_lbl = to_np("test")
    return tr_img, tr_lbl, te_img, te_lbl


def _to_chw(arr, channels):
    """arr: uint8/float array -> float tensor (N, C, H, W) in [0, 1]."""
    t = torch.as_tensor(np.asarray(arr))
    if t.dtype != torch.float32:
        t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    if t.dim() == 3:                                # (N, H, W) grayscale
        t = t.unsqueeze(1)
    elif t.dim() == 4 and t.shape[-1] in (1, 3):    # (N, H, W, C) -> (N, C, H, W)
        t = t.permute(0, 3, 1, 2).contiguous()
    return t


def load_dataset_conv(name, data_root=None, val_split_seed=0):
    """Return train/val/test tensors as (N, C, H, W), plus in_ch and n_classes.

    Normalization uses per-channel mean/std computed on the *training* split
    only (validation is carved out before statistics are computed).
    """
    root = Path(data_root) if data_root else Path(__file__).resolve().parent / "data"
    root.mkdir(exist_ok=True)
    root = str(root)

    if name == "mnist":
        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        tr_x, tr_y = _to_chw(tr.data, 1), tr.targets.long()
        te_x, te_y = _to_chw(te.data, 1), te.targets.long()
        n_cls = 10
    elif name == "fashion_mnist":
        tr = datasets.FashionMNIST(root, train=True, download=True)
        te = datasets.FashionMNIST(root, train=False, download=True)
        tr_x, tr_y = _to_chw(tr.data, 1), tr.targets.long()
        te_x, te_y = _to_chw(te.data, 1), te.targets.long()
        n_cls = 10
    elif name in ("cifar10", "cifar100"):
        tr_img, tr_lbl, te_img, te_lbl = _load_hf_cifar(name)
        tr_x, tr_y = _to_chw(tr_img, 3), torch.as_tensor(tr_lbl).long()
        te_x, te_y = _to_chw(te_img, 3), torch.as_tensor(te_lbl).long()
        n_cls = 10 if name == "cifar10" else 100
    elif name == "svhn":
        tr = datasets.SVHN(root, split="train", download=True)
        te = datasets.SVHN(root, split="test", download=True)
        tr_x, tr_y = _to_chw(tr.data, 3), torch.as_tensor(tr.labels).long()
        te_x, te_y = _to_chw(te.data, 3), torch.as_tensor(te.labels).long()
        n_cls = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # deterministic validation carve-out (independent of the run seed)
    n_val = _VAL_SIZE.get(name, 5000)
    g = torch.Generator().manual_seed(val_split_seed)
    perm = torch.randperm(tr_x.shape[0], generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    va_x, va_y = tr_x[val_idx], tr_y[val_idx]
    tr_x, tr_y = tr_x[tr_idx], tr_y[tr_idx]

    # per-channel normalization from the training split only
    mean = tr_x.mean(dim=(0, 2, 3), keepdim=True)
    std = tr_x.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    tr_x = (tr_x - mean) / std
    va_x = (va_x - mean) / std
    te_x = (te_x - mean) / std

    in_ch = tr_x.shape[1]
    return (tr_x, tr_y, va_x, va_y, te_x, te_y, in_ch, n_cls)


# ======================================================================
# Convolutional goodness functions  (operate on h : (B, C, H, W))
# ======================================================================
#
# All statistics are computed on the globally flattened feature map z = vec(h),
# z in R^{C*H*W}, and accumulated in FP32 (moments in particular) as required.

_TOPK_RHO = 0.02
_ENTMAX_ALPHA = 1.5


def _flat(h):
    return h.flatten(1).float()                     # (B, d), FP32


def _beta_p(p):
    """Gaussian normalizing constant: (p-1)!! for even p, else 0."""
    if p % 2 == 1:
        return 0.0
    beta = 1.0
    for k in range(1, p, 2):
        beta *= k
    return float(beta)


def cg_sos(h):
    """Mean squared activation (Hinton SoS, scaled by 1/d for scale stability)."""
    return _flat(h).pow(2).mean(dim=1)


def cg_variance(h):
    return _flat(h).var(dim=1)


def cg_topk(h, rho=_TOPK_RHO):
    z = _flat(h)
    k = max(5, int(z.shape[1] * rho))
    return z.topk(k, dim=1).values.mean(dim=1)


def cg_layernorm_topk(h, rho=_TOPK_RHO):
    z = _flat(h)
    z = F.layer_norm(z, [z.shape[1]])
    k = max(5, int(z.shape[1] * rho))
    return z.topk(k, dim=1).values.mean(dim=1)


def cg_entmax15_energy(h):
    if not _HAS_ENTMAX:
        raise RuntimeError("entmax not installed; `pip install entmax`.")
    z = _flat(h)
    w = entmax15(z, dim=1)
    return (w * z.pow(2)).sum(dim=1)


def cg_burstiness(h):
    """Excess kurtosis; scale-invariant, parameter-free."""
    z = _flat(h)
    mu = z.mean(dim=1, keepdim=True)
    d = z - mu
    var = d.pow(2).mean(dim=1).clamp(min=1e-6)
    kurt = d.pow(4).mean(dim=1) / var.pow(2)
    return kurt - 3.0


def cg_layernorm_burstiness(h):
    z = _flat(h)
    z = F.layer_norm(z, [z.shape[1]])
    mu = z.mean(dim=1, keepdim=True)
    d = z - mu
    var = d.pow(2).mean(dim=1).clamp(min=1e-6)
    return d.pow(4).mean(dim=1) / var.pow(2) - 3.0


def _moment_p(h, p):
    z = _flat(h)
    mu = z.mean(dim=1, keepdim=True)
    d = z - mu
    var = d.pow(2).mean(dim=1).clamp(min=1e-6)
    return d.pow(p).mean(dim=1) / var.pow(p / 2.0) - _beta_p(p)


def make_goodness(name):
    """Return a callable g: (B,C,H,W) -> (B,).  ``name`` may be 'moment6' etc."""
    if name.startswith("moment"):
        p = int(name.replace("moment", ""))
        return lambda h: _moment_p(h, p)
    if name.startswith("topk"):                     # e.g. topk2 == rho=2%
        if name == "topk":
            return cg_topk
        rho = float(name.replace("topk", "")) / 100.0
        return lambda h: cg_topk(h, rho=rho)
    fns = {
        "sos": cg_sos,
        "variance": cg_variance,
        "layernorm_topk": cg_layernorm_topk,
        "entmax15_energy": cg_entmax15_energy,
        "burstiness": cg_burstiness,
        "layernorm_burstiness": cg_layernorm_burstiness,
    }
    if name not in fns:
        raise ValueError(f"Unknown goodness: {name}")
    return fns[name]


# ======================================================================
# Normalization / pooling helpers
# ======================================================================

def rms_pool2d(h, kernel=2):
    """RMSPool: sqrt(mean(h^2)) over each pooling window."""
    return F.avg_pool2d(h.pow(2), kernel_size=kernel, stride=kernel).clamp(min=1e-12).sqrt()


def l2norm_conv(h, mode="l2", eps=1e-6):
    """Per-sample normalization over (C, H, W).

    mode='l2'  : fixes total L2 norm (direct counterpart of the MLP's between-
                 layer L2 normalization).
    mode='rms' : fixes root-mean-square activation (dimension-normalized).
    """
    flat = h.flatten(1)
    if mode == "rms":
        denom = flat.pow(2).mean(dim=1).add(eps).sqrt()
    else:
        denom = flat.pow(2).sum(dim=1).add(eps).sqrt()
    return h / denom.view(-1, 1, 1, 1)


def rms_norm_gate(h, eps=1e-6):
    """RMS norm-gate: h' = sigmoid(rms(h)) * h  (convolutional analogue of the
    MLP norm-gating; RMS keeps the gate from saturating for large C*H*W)."""
    flat = h.flatten(1)
    r = flat.pow(2).mean(dim=1).add(eps).sqrt()
    return h * torch.sigmoid(r).view(-1, 1, 1, 1)


def _activate(h, kind, gn=None):
    """Activation with optional pre-activation conv-LayerNorm (GroupNorm(1,C))."""
    if kind in ("convln_gelu", "convln_swish"):
        h = gn(h)
        if kind == "convln_gelu":
            return F.gelu(h)
        return h * torch.sigmoid(h)
    if kind == "gelu":
        return F.gelu(h)
    if kind == "swish":
        return h * torch.sigmoid(h)
    return F.relu(h)


# ======================================================================
# Orthogonal channel-mixing negative control (Sec. 23.1 / H7)
# ======================================================================

class ChannelMixer:
    """Fixed orthogonal 1x1 mixing Q (Q^T Q = I) applied per spatial location
    *only when computing goodness*.  Energy (sum of channel squares) is exactly
    preserved, so SoS is invariant; coordinate-wise kurtosis generally is not.

    theta_deg controls mixing strength via adjacent-channel pairwise rotations.
    """

    def __init__(self, theta_deg):
        self.theta = math.radians(theta_deg)
        self._cache = {}

    def _q(self, C, device, dtype):
        key = (C, device, dtype)
        if key not in self._cache:
            Q = torch.eye(C, device=device, dtype=dtype)
            c, s = math.cos(self.theta), math.sin(self.theta)
            for i in range(0, C - 1, 2):
                Q[i, i], Q[i, i + 1] = c, -s
                Q[i + 1, i], Q[i + 1, i + 1] = s, c
            self._cache[key] = Q
        return self._cache[key]

    def __call__(self, h):
        if self.theta == 0.0:
            return h
        Q = self._q(h.shape[1], h.device, h.dtype)
        return torch.einsum("oc,bchw->bohw", Q, h)


# ======================================================================
# FF blocks
# ======================================================================

class ConvFFBlock(nn.Module):
    """A single convolutional FF block: conv (3x3, pad 1) -> [pre-act LN] ->
    activation -> [RMS downsample].  Returns (feat, h_lab):

      * feat  : label-free features that propagate to the next block,
      * h_lab : features used for the local goodness (identical to feat for
                the standard pathway; feat + lambda * label-embed for the
                separate per-layer pathway).
    """

    def __init__(self, in_ch, out_ch, n_classes, activation, downsample,
                 pathway, lam=1.0):
        super().__init__()
        use_bias = activation not in ("convln_gelu", "convln_swish")
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=use_bias)
        self.gn = nn.GroupNorm(1, out_ch) if not use_bias else None
        self.activation = activation
        self.downsample = downsample
        self.pathway = pathway
        self.lam = lam
        if pathway == "separate":
            # per-class, per-channel label embedding  E in R^{K x C}
            self.label_embed = nn.Parameter(torch.randn(n_classes, out_ch) * 0.1)

    def forward(self, x, label_idx=None):
        h = self.conv(x)
        h = _activate(h, self.activation, self.gn)
        if self.downsample:
            h = rms_pool2d(h, 2)
        feat = h
        if self.pathway == "separate" and label_idx is not None:
            e = self.label_embed[label_idx]                 # (B, C)
            h_lab = feat + self.lam * e.view(*e.shape, 1, 1)
            return feat, h_lab
        return feat, feat


class BasicFFBlock(nn.Module):
    """ResNet BasicBlock as ONE block-local FF objective.  Gradient may flow
    through the two internal convolutions and the skip projection, but never
    across the block boundary (the output is detached before the next block).
    Uses GroupNorm(1,C) instead of BatchNorm to avoid cross-example coupling
    between positive and negative samples (Sec. 7.4)."""

    def __init__(self, in_ch, out_ch, stride, n_classes, activation, pathway,
                 lam=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, out_ch)
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(1, out_ch),
            )
        else:
            self.short = nn.Identity()
        # plain gelu/swish inside the residual block (relu falls back to relu)
        self.act = "gelu" if activation in ("convln_gelu", "gelu") else \
                   ("swish" if activation in ("convln_swish", "swish") else "relu")
        self.pathway = pathway
        self.lam = lam
        if pathway == "separate":
            self.label_embed = nn.Parameter(torch.randn(n_classes, out_ch) * 0.1)

    def _a(self, h):
        if self.act == "gelu":
            return F.gelu(h)
        if self.act == "swish":
            return h * torch.sigmoid(h)
        return F.relu(h)

    def forward(self, x, label_idx=None):
        out = self._a(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self._a(out + self.short(x))
        feat = out
        if self.pathway == "separate" and label_idx is not None:
            e = self.label_embed[label_idx]
            h_lab = feat + self.lam * e.view(*e.shape, 1, 1)
            return feat, h_lab
        return feat, feat


class ConvFFNet(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)


# ----------------------------------------------------------------------
# Architecture registry
# ----------------------------------------------------------------------

# (out_channels, downsample_after_block)
_ARCHS = {
    "conv2": [(64, True), (128, False)],
    "conv4": [(64, True), (128, True), (256, True), (512, False)],
    "conv6": [(64, False), (64, True), (128, True),
              (256, False), (256, True), (512, False)],
    "conv8": [(64, False), (64, True), (128, False), (128, True),
              (256, False), (256, True), (512, False), (512, True)],
    "wideconv8": [(96, False), (96, True), (192, False), (192, True),
                  (384, False), (384, True), (768, False), (768, True)],
}

# ResNet18: (out_ch, stride) per BasicBlock, after a 3x3 stride-1 CIFAR stem
_RESNET18_BLOCKS = [(64, 1), (64, 1), (128, 2), (128, 1),
                    (256, 2), (256, 1), (512, 2), (512, 1)]


def build_model(arch, in_ch, n_classes, activation, pathway, lam=1.0):
    in0 = in_ch + (n_classes if pathway == "standard" else 0)
    if arch == "resnet18":
        blocks = [ConvFFBlock(in0, 64, n_classes, activation,
                              downsample=False, pathway=pathway, lam=lam)]
        prev = 64
        for out_ch, stride in _RESNET18_BLOCKS:
            blocks.append(BasicFFBlock(prev, out_ch, stride, n_classes,
                                       activation, pathway, lam=lam))
            prev = out_ch
        return ConvFFNet(blocks)

    if arch not in _ARCHS:
        raise ValueError(f"Unknown architecture: {arch}")
    blocks, prev = [], in0
    for out_ch, ds in _ARCHS[arch]:
        blocks.append(ConvFFBlock(prev, out_ch, n_classes, activation,
                                  downsample=ds, pathway=pathway, lam=lam))
        prev = out_ch
    return ConvFFNet(blocks)


def list_architectures():
    return list(_ARCHS.keys()) + ["resnet18"]


# ======================================================================
# Standard-pathway helper: build input label maps
# ======================================================================

def _label_map_input(images, label_idx, n_classes, label_scale):
    """Concatenate K constant label channels to the image (standard pathway)."""
    B, _, H, W = images.shape
    ymap = images.new_zeros(B, n_classes, H, W)
    ymap[torch.arange(B, device=images.device), label_idx] = label_scale
    return torch.cat([images, ymap], dim=1)


# ======================================================================
# Training  (interleaved block-local; mirrors ff_main.train_ff / _ffcl)
# ======================================================================

def train_conv_ff(model, images, labels, n_classes, goodness_fn, pathway,
                  device, epochs=30, batch_size=128, lr=1e-3, threshold=2.0,
                  label_scale=5.0, norm_mode="l2", norm_gate=False,
                  grad_clip=None, mixer=None, log_every=10, verbose=True):
    """Block-local FF training. No gradient crosses a block boundary: each block
    is updated on its own local goodness loss and its output is detached before
    the next block (identical protocol to the fully-connected ff_main.py)."""
    opts = [torch.optim.Adam(b.parameters(), lr=lr) for b in model.blocks]
    n = images.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            imgs, lbls = images[idx], labels[idx]
            wrong = (lbls + torch.randint(1, n_classes, lbls.shape,
                                          device=device)) % n_classes

            if pathway == "standard":
                x_pos = _label_map_input(imgs, lbls, n_classes, label_scale)
                x_neg = _label_map_input(imgs, wrong, n_classes, label_scale)
                for blk, opt in zip(model.blocks, opts):
                    feat_pos, hlab_pos = blk(x_pos)
                    feat_neg, hlab_neg = blk(x_neg)
                    gp = goodness_fn(mixer(hlab_pos) if mixer else hlab_pos)
                    gn_ = goodness_fn(mixer(hlab_neg) if mixer else hlab_neg)
                    loss = (F.softplus(threshold - gp).mean() +
                            F.softplus(gn_ - threshold).mean())
                    opt.zero_grad()
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(blk.parameters(), grad_clip)
                    opt.step()
                    total += loss.item(); nb += 1
                    with torch.no_grad():
                        p = l2norm_conv(feat_pos.detach(), norm_mode)
                        q = l2norm_conv(feat_neg.detach(), norm_mode)
                        x_pos = rms_norm_gate(p) if norm_gate else p
                        x_neg = rms_norm_gate(q) if norm_gate else q
            else:  # separate per-layer label forwarding (single label-free stream)
                x = imgs
                for blk, opt in zip(model.blocks, opts):
                    feat, hlab_pos = blk(x, lbls)
                    _, hlab_neg = blk(x, wrong)
                    gp = goodness_fn(mixer(hlab_pos) if mixer else hlab_pos)
                    gn_ = goodness_fn(mixer(hlab_neg) if mixer else hlab_neg)
                    loss = (F.softplus(threshold - gp).mean() +
                            F.softplus(gn_ - threshold).mean())
                    opt.zero_grad()
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(blk.parameters(), grad_clip)
                    opt.step()
                    total += loss.item(); nb += 1
                    with torch.no_grad():
                        f = l2norm_conv(feat.detach(), norm_mode)
                        x = rms_norm_gate(f) if norm_gate else f

        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"      epoch {epoch+1:3d}/{epochs}  loss={total/max(nb,1):.4f}",
                  flush=True)


# ======================================================================
# Inference  (PRIMARY = standard raw per-block goodness sum)
# ======================================================================

@torch.no_grad()
def _goodness_per_block(model, images, label_idx, n_classes, goodness_fn,
                        pathway, label_scale, norm_mode, norm_gate, mixer=None):
    """Return an (B, n_blocks) matrix of per-block goodness for one candidate
    class (given by label_idx)."""
    if pathway == "standard":
        x = _label_map_input(images, label_idx, n_classes, label_scale)
    else:
        x = images
    goods = []
    for blk in model.blocks:
        if pathway == "separate":
            feat, hlab = blk(x, label_idx)
        else:
            feat, hlab = blk(x)
        goods.append(goodness_fn(mixer(hlab) if mixer else hlab))
        f = l2norm_conv(feat, norm_mode)
        x = rms_norm_gate(f) if norm_gate else f
    return torch.stack(goods, dim=1)                # (B, n_blocks)


@torch.no_grad()
def evaluate_conv_ff(model, images, labels, n_classes, goodness_fn, pathway,
                     device, batch_size=500, label_scale=5.0, norm_mode="l2",
                     norm_gate=False, mixer=None, scoring="raw",
                     calib=None):
    """Multi-pass FF inference.

    scoring:
      'raw'   : S(c) = sum_l g(h_l^c)                 [PRIMARY, standard FF]
      'last'  : S(c) = g(h_L^c)                        (last block only)
      'calib' : S(c) = sum_l (g_l - mu_l) / (sigma_l)  (validation-calibrated)
                requires ``calib`` = (mu, sigma) tensors of shape (n_blocks,)
    Returns (accuracy, per_block_stats) where per_block_stats is the mean
    goodness matrix over candidate=true class (used for calibration fitting).
    """
    N = len(images)
    scores = torch.zeros(N, n_classes, device=device)
    for c in range(n_classes):
        for b in range(0, N, batch_size):
            e = min(b + batch_size, N)
            lbl = torch.full((e - b,), c, dtype=torch.long, device=device)
            gmat = _goodness_per_block(model, images[b:e], lbl, n_classes,
                                       goodness_fn, pathway, label_scale,
                                       norm_mode, norm_gate, mixer)
            if scoring == "last":
                s = gmat[:, -1]
            elif scoring == "calib":
                mu, sig = calib
                s = ((gmat - mu) / (sig + 1e-6)).sum(dim=1)
            else:
                s = gmat.sum(dim=1)
            scores[b:e, c] = s
    acc = (scores.argmax(dim=1) == labels).float().mean().item()
    return acc


@torch.no_grad()
def fit_layer_calibration(model, images, labels, n_classes, goodness_fn,
                          pathway, device, batch_size=500, label_scale=5.0,
                          norm_mode="l2", norm_gate=False):
    """Estimate per-block goodness mean/std on the (validation) set, evaluating
    each example at its TRUE class label -> used for calibrated scoring."""
    N = len(images)
    n_blocks = len(model.blocks)
    acc = torch.zeros(N, n_blocks, device=device)
    for b in range(0, N, batch_size):
        e = min(b + batch_size, N)
        acc[b:e] = _goodness_per_block(model, images[b:e], labels[b:e],
                                       n_classes, goodness_fn, pathway,
                                       label_scale, norm_mode, norm_gate)
    return acc.mean(dim=0), acc.std(dim=0)


# ======================================================================
# End-to-end run  (train + evaluate, with runtime & memory measurement)
# ======================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_conv_experiment(arch, dataset, goodness, pathway, activation="gelu",
                        seed=0, epochs=30, batch_size=128, lr=1e-3,
                        threshold=2.0, label_scale=5.0, lam=1.0,
                        norm_mode="l2", norm_gate=False, grad_clip=None,
                        gpu=0, data=None, channel_mix_theta=0.0,
                        scoring="raw", verbose=True):
    """Train one convolutional FF model and evaluate it.

    Returns a result dict with test accuracy, params, train time, peak GPU
    memory (MB), and the config, so the driver can aggregate over seeds.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    if data is None:
        data = load_dataset_conv(dataset)
    tr_x, tr_y, va_x, va_y, te_x, te_y, in_ch, n_cls = data
    tr_x, tr_y = tr_x.to(device), tr_y.to(device)
    te_x, te_y = te_x.to(device), te_y.to(device)

    goodness_fn = make_goodness(goodness)
    mixer = ChannelMixer(channel_mix_theta) if channel_mix_theta else None

    model = build_model(arch, in_ch, n_cls, activation, pathway, lam=lam).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    train_conv_ff(model, tr_x, tr_y, n_cls, goodness_fn, pathway, device,
                  epochs=epochs, batch_size=batch_size, lr=lr,
                  threshold=threshold, label_scale=label_scale,
                  norm_mode=norm_mode, norm_gate=norm_gate,
                  grad_clip=grad_clip, mixer=mixer, verbose=verbose)
    train_sec = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6
               if device.type == "cuda" else 0.0)

    calib = None
    if scoring == "calib":
        va = va_x.to(device)
        calib = fit_layer_calibration(model, va, va_y.to(device), n_cls,
                                      goodness_fn, pathway, device,
                                      label_scale=label_scale,
                                      norm_mode=norm_mode, norm_gate=norm_gate)

    t1 = time.time()
    acc = evaluate_conv_ff(model, te_x, te_y, n_cls, goodness_fn, pathway,
                           device, label_scale=label_scale, norm_mode=norm_mode,
                           norm_gate=norm_gate, mixer=mixer, scoring=scoring,
                           calib=calib)
    infer_sec = time.time() - t1

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "arch": arch, "dataset": dataset, "goodness": goodness,
        "pathway": pathway, "activation": activation, "seed": seed,
        "accuracy": acc, "params": n_params, "train_sec": train_sec,
        "infer_sec": infer_sec, "peak_mem_mb": peak_mb,
        "channel_mix_theta": channel_mix_theta, "scoring": scoring,
        "config": {"epochs": epochs, "lr": lr, "threshold": threshold,
                   "label_scale": label_scale, "norm_mode": norm_mode,
                   "norm_gate": norm_gate, "lam": lam, "batch_size": batch_size},
    }
