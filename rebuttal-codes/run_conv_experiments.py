#!/usr/bin/env python3
"""
Rebuttal driver for the convolutional Forward-Forward experiments.
==================================================================

Produces the *necessary* convolutional evidence for the NeurIPS author
response (Section 34 central set + supporting tables).  Every experiment is
multi-seed and reports mean +/- std.

Experiments
-----------
  rebuttal   Main rebuttal table (Sec. 34): Conv8 + block-local ResNet18,
             tuned-SoS vs burstiness, standard & separate-label pathways,
             matched GELU, standard raw layer-sum inference, 5 seeds,
             train time & peak memory, + paired seed gains.
  goodness   Full goodness-function comparison on Conv8 (Table C2), both
             pathways, multi-seed.
  control    Orthogonal channel-mixing negative control (Table C13 / Fig C5):
             energy-preserving, shape-destroying rotation should leave SoS
             ~unchanged but hurt burstiness.
  scoring    Inference-rule ablation (Table C11): raw vs last-block vs
             validation-calibrated layer sum.
  scaling    Architecture scaling (Table C9): Conv2/Conv4/Conv8/WideConv8/
             ResNet18 for SoS vs burstiness.
  tune       Validation-set hyperparameter search for SoS AND burstiness
             (matched budget); writes best configs to conv_tuned.json, which
             `rebuttal` then consumes.

Usage (run on your server; all default to CIFAR-10):
  python run_conv_experiments.py tune     --dataset cifar10 --gpu 0
  python run_conv_experiments.py rebuttal --dataset cifar10 --gpu 0 --seeds 0 1 2 3 4
  python run_conv_experiments.py goodness --dataset cifar10 --gpu 0
  python run_conv_experiments.py control  --dataset cifar10 --gpu 0
  python run_conv_experiments.py scoring  --dataset cifar10 --gpu 0
  python run_conv_experiments.py scaling  --dataset cifar10 --gpu 0

Nothing is run here; this file only defines the experiments.
"""

import argparse
import json
import statistics as st
from itertools import product
from pathlib import Path

import conv_ff as C


RESULTS_DIR = Path(__file__).resolve().parent / "conv_results"
TUNED_FILE = Path(__file__).resolve().parent / "conv_tuned.json"


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _agg(vals):
    """mean, std (population->sample std with n>=2)."""
    m = sum(vals) / len(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _save(name, obj):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\n[saved] {RESULTS_DIR / (name + '.json')}")


def _load_tuned():
    if TUNED_FILE.exists():
        return json.loads(TUNED_FILE.read_text())
    return {}


# default (untuned) hyperparameters; `tune` overrides these per goodness
_DEFAULTS = {
    "sos":        dict(lr=1e-3, threshold=2.0, label_scale=5.0, lam=1.0),
    "burstiness": dict(lr=1e-3, threshold=2.0, label_scale=5.0, lam=1.0),
}


def _hp(goodness, tuned):
    base = dict(_DEFAULTS.get(goodness, _DEFAULTS["burstiness"]))
    base.update(tuned.get(goodness, {}))
    return base


def _run_seeds(seeds, data, verbose=False, **kw):
    """Run one config over seeds; return list of result dicts."""
    out = []
    for s in seeds:
        r = C.run_conv_experiment(seed=s, data=data, verbose=verbose, **kw)
        out.append(r)
        print(f"    seed {s}: acc={r['accuracy']*100:6.2f}%  "
              f"({r['train_sec']:.0f}s, {r['peak_mem_mb']:.0f}MB)", flush=True)
    return out


# ======================================================================
# tune: validation-based hyperparameter search (matched budget)
# ======================================================================

def cmd_tune(args):
    """Search a compact grid on the VALIDATION split for SoS and burstiness,
    using a matched number of configurations.  Selection is by validation
    accuracy (test set never inspected).  Short epochs keep the search cheap;
    the winner is re-run at full length by `rebuttal`."""
    data = C.load_dataset_conv(args.dataset)
    tr_x, tr_y, va_x, va_y, te_x, te_y, in_ch, n_cls = data
    # tune on Conv8 with the standard pathway + matched GELU
    arch, pathway, act = "conv8", "standard", "gelu"

    # matched grid (same #configs for both goodness functions)
    lrs = [1e-3, 3e-3]
    thresholds = [1.0, 2.0, 4.0]
    label_scales = [2.0, 5.0, 10.0]

    tuned = _load_tuned()
    for goodness in ["sos", "burstiness"]:
        print(f"\n=== tuning {goodness} on {args.dataset} "
              f"({arch}/{pathway}/{act}) ===")
        best, best_val = None, -1.0
        grid = list(product(lrs, thresholds, label_scales))
        for i, (lr, thr, ls) in enumerate(grid):
            # train on train split, score on validation split
            data_val = (tr_x, tr_y, va_x, va_y, va_x, va_y, in_ch, n_cls)
            r = C.run_conv_experiment(
                arch=arch, dataset=args.dataset, goodness=goodness,
                pathway=pathway, activation=act, seed=args.tune_seed,
                epochs=args.tune_epochs, batch_size=args.batch_size,
                lr=lr, threshold=thr, label_scale=ls, gpu=args.gpu,
                data=data_val, verbose=False)
            val_acc = r["accuracy"]
            print(f"  [{i+1:2d}/{len(grid)}] lr={lr} thr={thr} ls={ls}"
                  f"  val={val_acc*100:6.2f}%", flush=True)
            if val_acc > best_val:
                best_val = val_acc
                best = dict(lr=lr, threshold=thr, label_scale=ls, lam=1.0)
        best["val_acc"] = best_val
        best["n_configs"] = len(grid)
        tuned[goodness] = best
        print(f"  -> best {goodness}: {best}")

    TUNED_FILE.write_text(json.dumps(tuned, indent=2))
    print(f"\n[saved] {TUNED_FILE}")


# ======================================================================
# rebuttal: main Section-34 table
# ======================================================================

def cmd_rebuttal(args):
    data = C.load_dataset_conv(args.dataset)
    tuned = _load_tuned()
    if not tuned:
        print("[warn] no conv_tuned.json found; using default hyperparameters. "
              "Run `tune` first for a validation-tuned SoS baseline.")

    # (arch, pathway) rows; each x {tuned SoS, burstiness}
    rows = [
        ("conv8", "standard"),
        ("conv8", "separate"),
        ("resnet18", "separate"),
    ]
    goods = ["sos", "burstiness"]

    all_results, table = [], []
    # store per-(arch,pathway) seed accuracies for paired-gain computation
    seed_acc = {}
    for arch, pathway in rows:
        for goodness in goods:
            hp = _hp(goodness, tuned)
            print(f"\n### {arch} / {pathway} / {goodness}  hp={hp}")
            res = _run_seeds(
                args.seeds, data,
                arch=arch, dataset=args.dataset, goodness=goodness,
                pathway=pathway, activation="gelu", epochs=args.epochs,
                batch_size=args.batch_size, lr=hp["lr"],
                threshold=hp["threshold"], label_scale=hp["label_scale"],
                lam=hp["lam"], gpu=args.gpu, scoring="raw")
            all_results.extend(res)
            accs = [r["accuracy"] * 100 for r in res]
            seed_acc[(arch, pathway, goodness)] = accs
            m, s = _agg(accs)
            tt, _ = _agg([r["train_sec"] for r in res])
            mem, _ = _agg([r["peak_mem_mb"] for r in res])
            table.append(dict(arch=arch, pathway=pathway, goodness=goodness,
                              acc_mean=m, acc_std=s, train_sec=tt, peak_mb=mem,
                              accs=accs))

    # paired burstiness - SoS gains (same seeds)
    print("\n" + "=" * 78)
    print("  MAIN REBUTTAL TABLE  (dataset = %s, %d seeds, raw layer-sum inference)"
          % (args.dataset, len(args.seeds)))
    print("=" * 78)
    hdr = (f"  {'Architecture':<20} {'Pathway':<10} {'Goodness':<11} "
           f"{'Acc%':>13}  {'PairGain':>9}  {'Train(s)':>9}  {'Mem(MB)':>8}")
    print(hdr)
    print("  " + "-" * len(hdr))
    for row in table:
        gain = ""
        if row["goodness"] == "burstiness":
            sos = seed_acc.get((row["arch"], row["pathway"], "sos"))
            if sos:
                diffs = [b - a for b, a in zip(row["accs"], sos)]
                gm, gs = _agg(diffs)
                gain = f"+{gm:.2f}±{gs:.2f}"
        print(f"  {row['arch']:<20} {row['pathway']:<10} {row['goodness']:<11} "
              f"{row['acc_mean']:6.2f}±{row['acc_std']:<5.2f}  {gain:>9}  "
              f"{row['train_sec']:9.0f}  {row['peak_mb']:8.0f}")

    _save(f"rebuttal_{args.dataset}",
          dict(table=table, seed_acc={str(k): v for k, v in seed_acc.items()},
               raw=all_results, seeds=args.seeds))


# ======================================================================
# goodness: full goodness-function comparison on Conv8 (Table C2)
# ======================================================================

def cmd_goodness(args):
    data = C.load_dataset_conv(args.dataset)
    tuned = _load_tuned()
    goods = ["sos", "variance", "topk", "layernorm_topk",
             "entmax15_energy", "moment6", "burstiness"]
    if not C._HAS_ENTMAX:
        goods.remove("entmax15_energy")

    table, raw = [], []
    for pathway in ["standard", "separate"]:
        for goodness in goods:
            hp = _hp(goodness, tuned)
            print(f"\n### conv8 / {pathway} / {goodness}")
            res = _run_seeds(
                args.seeds, data,
                arch="conv8", dataset=args.dataset, goodness=goodness,
                pathway=pathway, activation="gelu", epochs=args.epochs,
                batch_size=args.batch_size, lr=hp["lr"],
                threshold=hp["threshold"], label_scale=hp["label_scale"],
                gpu=args.gpu, scoring="raw")
            raw.extend(res)
            m, s = _agg([r["accuracy"] * 100 for r in res])
            table.append(dict(pathway=pathway, goodness=goodness,
                              acc_mean=m, acc_std=s))

    print("\n" + "=" * 60)
    print("  GOODNESS COMPARISON  (Conv8, %s, %d seeds)"
          % (args.dataset, len(args.seeds)))
    print("=" * 60)
    print(f"  {'Goodness':<20} {'Standard':>14} {'Separate':>14}")
    for goodness in goods:
        std = next((r for r in table if r["pathway"] == "standard"
                    and r["goodness"] == goodness), None)
        sep = next((r for r in table if r["pathway"] == "separate"
                    and r["goodness"] == goodness), None)
        sc = f"{std['acc_mean']:.2f}±{std['acc_std']:.2f}" if std else "-"
        pc = f"{sep['acc_mean']:.2f}±{sep['acc_std']:.2f}" if sep else "-"
        print(f"  {goodness:<20} {sc:>14} {pc:>14}")

    _save(f"goodness_{args.dataset}", dict(table=table, raw=raw))


# ======================================================================
# control: orthogonal channel-mixing negative control (Table C13)
# ======================================================================

def cmd_control(args):
    """H7 falsifiable prediction: an energy-preserving orthogonal channel
    rotation should leave SoS ~unchanged but degrade burstiness."""
    data = C.load_dataset_conv(args.dataset)
    tuned = _load_tuned()
    thetas = [0.0, 15.0, 30.0, 45.0]
    table, raw = [], []
    for goodness in ["sos", "burstiness"]:
        hp = _hp(goodness, tuned)
        for theta in thetas:
            print(f"\n### conv8/standard/{goodness}  theta={theta}deg")
            res = _run_seeds(
                args.seeds, data,
                arch="conv8", dataset=args.dataset, goodness=goodness,
                pathway="standard", activation="gelu", epochs=args.epochs,
                batch_size=args.batch_size, lr=hp["lr"],
                threshold=hp["threshold"], label_scale=hp["label_scale"],
                gpu=args.gpu, channel_mix_theta=theta, scoring="raw")
            raw.extend(res)
            m, s = _agg([r["accuracy"] * 100 for r in res])
            table.append(dict(goodness=goodness, theta=theta,
                              acc_mean=m, acc_std=s))

    print("\n" + "=" * 60)
    print("  ORTHOGONAL CHANNEL-MIXING CONTROL  (Conv8, %s, %d seeds)"
          % (args.dataset, len(args.seeds)))
    print("=" * 60)
    print(f"  {'theta(deg)':<12} {'SoS':>14} {'Burstiness':>14}")
    for theta in thetas:
        s_ = next(r for r in table if r["goodness"] == "sos" and r["theta"] == theta)
        b_ = next(r for r in table if r["goodness"] == "burstiness" and r["theta"] == theta)
        print(f"  {theta:<12} {s_['acc_mean']:6.2f}±{s_['acc_std']:<5.2f} "
              f"  {b_['acc_mean']:6.2f}±{b_['acc_std']:<5.2f}")
    _save(f"control_{args.dataset}", dict(table=table, raw=raw))


# ======================================================================
# scoring: inference-rule ablation (Table C11)
# ======================================================================

def cmd_scoring(args):
    data = C.load_dataset_conv(args.dataset)
    tuned = _load_tuned()
    table, raw = [], []
    for goodness in ["sos", "burstiness"]:
        hp = _hp(goodness, tuned)
        for scoring in ["raw", "last", "calib"]:
            print(f"\n### conv8/separate/{goodness}  scoring={scoring}")
            res = _run_seeds(
                args.seeds, data,
                arch="conv8", dataset=args.dataset, goodness=goodness,
                pathway="separate", activation="gelu", epochs=args.epochs,
                batch_size=args.batch_size, lr=hp["lr"],
                threshold=hp["threshold"], label_scale=hp["label_scale"],
                gpu=args.gpu, scoring=scoring)
            raw.extend(res)
            m, s = _agg([r["accuracy"] * 100 for r in res])
            table.append(dict(goodness=goodness, scoring=scoring,
                              acc_mean=m, acc_std=s))

    print("\n" + "=" * 60)
    print("  INFERENCE-RULE ABLATION  (Conv8/separate, %s, %d seeds)"
          % (args.dataset, len(args.seeds)))
    print("=" * 60)
    print(f"  {'Scoring':<12} {'SoS':>14} {'Burstiness':>14}")
    for scoring in ["raw", "last", "calib"]:
        s_ = next(r for r in table if r["goodness"] == "sos" and r["scoring"] == scoring)
        b_ = next(r for r in table if r["goodness"] == "burstiness" and r["scoring"] == scoring)
        print(f"  {scoring:<12} {s_['acc_mean']:6.2f}±{s_['acc_std']:<5.2f} "
              f"  {b_['acc_mean']:6.2f}±{b_['acc_std']:<5.2f}")
    _save(f"scoring_{args.dataset}", dict(table=table, raw=raw))


# ======================================================================
# scaling: architecture scaling (Table C9)
# ======================================================================

def cmd_scaling(args):
    data = C.load_dataset_conv(args.dataset)
    tuned = _load_tuned()
    archs = ["conv2", "conv4", "conv8", "wideconv8", "resnet18"]
    table, raw = [], []
    for arch in archs:
        for goodness in ["sos", "burstiness"]:
            hp = _hp(goodness, tuned)
            print(f"\n### {arch}/separate/{goodness}")
            res = _run_seeds(
                args.seeds, data,
                arch=arch, dataset=args.dataset, goodness=goodness,
                pathway="separate", activation="gelu", epochs=args.epochs,
                batch_size=args.batch_size, lr=hp["lr"],
                threshold=hp["threshold"], label_scale=hp["label_scale"],
                gpu=args.gpu, scoring="raw")
            raw.extend(res)
            m, s = _agg([r["accuracy"] * 100 for r in res])
            table.append(dict(arch=arch, goodness=goodness, acc_mean=m,
                              acc_std=s, params=res[0]["params"]))

    print("\n" + "=" * 66)
    print("  ARCHITECTURE SCALING  (separate-label, %s, %d seeds)"
          % (args.dataset, len(args.seeds)))
    print("=" * 66)
    print(f"  {'Arch':<12} {'Params':>10} {'SoS':>14} {'Burstiness':>14}")
    for arch in archs:
        s_ = next(r for r in table if r["arch"] == arch and r["goodness"] == "sos")
        b_ = next(r for r in table if r["arch"] == arch and r["goodness"] == "burstiness")
        print(f"  {arch:<12} {s_['params']:>10,} "
              f"{s_['acc_mean']:6.2f}±{s_['acc_std']:<5.2f} "
              f"{b_['acc_mean']:6.2f}±{b_['acc_std']:<5.2f}")
    _save(f"scaling_{args.dataset}", dict(table=table, raw=raw))


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="cifar10",
                        choices=["mnist", "fashion_mnist", "cifar10",
                                 "cifar100", "svhn"])
        sp.add_argument("--gpu", type=int, default=0)
        sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        sp.add_argument("--epochs", type=int, default=30)
        sp.add_argument("--batch-size", type=int, default=128)

    sp = sub.add_parser("tune"); common(sp)
    sp.add_argument("--tune-epochs", type=int, default=10)
    sp.add_argument("--tune-seed", type=int, default=0)

    for name in ["rebuttal", "goodness", "control", "scoring", "scaling"]:
        common(sub.add_parser(name))

    args = p.parse_args()
    {
        "tune": cmd_tune, "rebuttal": cmd_rebuttal, "goodness": cmd_goodness,
        "control": cmd_control, "scoring": cmd_scoring, "scaling": cmd_scaling,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
