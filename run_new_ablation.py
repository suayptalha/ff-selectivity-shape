#!/usr/bin/env python3
"""Run all new FFCL + LayerNorm-TopK ablation experiments.

Skips any experiment that already has a saved result file.
Runs Fashion-MNIST first (both archs), then optionally CIFAR-10
for the best-performing new config.
"""

import json
import sys
import time
from pathlib import Path
from itertools import product

from ff_experiments import (
    build_all_variants, run_experiment, collect_results
)


def existing_results(results_dir):
    rd = Path(results_dir)
    if not rd.exists():
        return set()
    return {f.stem for f in rd.glob("*.json")}


def result_filename(variant, dataset, n_layers, hidden):
    return f"{variant}_{dataset}_{n_layers}x{hidden}"


def make_args(variant, dataset, n_layers, hidden, epochs=60, lr=0.001,
              batch_size=500, threshold=2.0, seed=42, gpu=0,
              results_dir="experiment_results"):
    """Create an args-like namespace for run_experiment."""
    import argparse
    return argparse.Namespace(
        variant=variant, dataset=dataset, n_layers=n_layers, hidden=hidden,
        epochs=epochs, lr=lr, batch_size=batch_size, threshold=threshold,
        seed=seed, gpu=gpu, results_dir=results_dir,
    )


def run_batch(variants_to_run, dataset, archs, gpu=0, results_dir="experiment_results"):
    all_variants = build_all_variants()
    done = existing_results(results_dir)
    results = []

    for n_layers, hidden in archs:
        for vname in variants_to_run:
            fname = result_filename(vname, dataset, n_layers, hidden)
            if fname in done:
                print(f"  SKIP (exists): {fname}")
                continue

            if vname not in all_variants:
                print(f"  SKIP (unknown): {vname}")
                continue

            args = make_args(vname, dataset, n_layers, hidden, gpu=gpu,
                             results_dir=results_dir)
            try:
                r = run_experiment(args, variants=all_variants)
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {vname} — {e}")

    return results


def print_comparison(results_dir, dataset, n_layers, hidden):
    rd = Path(results_dir)
    all_r = []
    for f in rd.glob(f"*_{dataset}_{n_layers}x{hidden}.json"):
        with open(f) as fh:
            all_r.append(json.load(fh))

    if not all_r:
        return

    all_r.sort(key=lambda r: -r["test_accuracy"])

    print(f"\n{'='*80}")
    print(f"  ALL RESULTS: {dataset.upper()} {n_layers}x{hidden} (ranked)")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Variant':<42} {'Acc%':>8}  {'Loss':>8}  {'Time':>6}")
    print(f"  {'-'*74}")

    for i, r in enumerate(all_r, 1):
        ffcl_tag = " [FFCL]" if r.get("ffcl") else ""
        ng_tag = "/ng" if r.get("norm_gate") else ""
        label = f"{r['activation']}/{r['goodness']}{ng_tag}{ffcl_tag}"
        acc = r["test_accuracy"] * 100
        print(f"  {i:<4} {label:<42} {acc:>7.2f}%  {r['test_loss']:>8.4f}  {r['train_time_sec']:>5.0f}s")

    # Highlight new vs old best
    old_best = [r for r in all_r if not r.get("ffcl") and r.get("goodness") != "layernorm_topk"]
    new_best = [r for r in all_r if r.get("ffcl") or r.get("goodness") == "layernorm_topk"]

    if old_best and new_best:
        ob = max(old_best, key=lambda r: r["test_accuracy"])
        nb = max(new_best, key=lambda r: r["test_accuracy"])
        delta = (nb["test_accuracy"] - ob["test_accuracy"]) * 100
        print(f"\n  Previous best: {ob['activation']}/{ob['goodness']} "
              f"= {ob['test_accuracy']*100:.2f}%")
        ffcl_t = " [FFCL]" if nb.get("ffcl") else ""
        print(f"  New best:      {nb['activation']}/{nb['goodness']}{ffcl_t} "
              f"= {nb['test_accuracy']*100:.2f}%  (Δ = {delta:+.2f}pp)")

    return all_r


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--phase", choices=["fashion", "cifar", "all"], default="fashion")
    parser.add_argument("--arch", nargs=2, type=int, default=None,
                        help="Single arch to run, e.g. --arch 4 2000")
    parser.add_argument("--results_dir", default="experiment_results")
    cli = parser.parse_args()

    all_v = build_all_variants()

    # New variants: all ffcl_* plus standard layernorm_topk
    new_variants = sorted([
        k for k, v in all_v.items()
        if v.get("ffcl") or v.get("goodness_desc") == "layernorm_topk"
    ])
    print(f"New variants to test: {len(new_variants)}")
    for v in new_variants:
        print(f"  {v}: {all_v[v]['desc']}")

    if cli.arch:
        archs = [tuple(cli.arch)]
    else:
        archs = [(4, 2000), (2, 500)]

    if cli.phase in ("fashion", "all"):
        print(f"\n{'#'*80}")
        print(f"  PHASE 1: Fashion-MNIST ablation ({len(new_variants)} variants × {len(archs)} archs)")
        print(f"{'#'*80}\n")

        run_batch(new_variants, "fashion_mnist", archs,
                  gpu=cli.gpu, results_dir=cli.results_dir)

        for nl, hid in archs:
            print_comparison(cli.results_dir, "fashion_mnist", nl, hid)

    if cli.phase in ("cifar", "all"):
        print(f"\n{'#'*80}")
        print(f"  PHASE 2: CIFAR-10 — best new configs")
        print(f"{'#'*80}\n")

        run_batch(new_variants, "cifar10", archs,
                  gpu=cli.gpu, results_dir=cli.results_dir)

        for nl, hid in archs:
            print_comparison(cli.results_dir, "cifar10", nl, hid)
