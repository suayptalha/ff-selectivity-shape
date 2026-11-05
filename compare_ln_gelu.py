#!/usr/bin/env python3
"""Compare ln_gelu burstiness results against paper baselines."""
import json, glob

paper = {
    'Std + Burstiness (Swish)': {
        'mnist': 96.54, 'fashion_mnist': 88.06, 'cifar10': 53.56,
        'usps': 93.12, 'svhn': 78.85, 'emnist': 86.21
    },
    'FFCL + Burstiness (GELU)': {
        'mnist': 98.09, 'fashion_mnist': 88.41, 'cifar10': 54.36,
        'usps': 93.27, 'svhn': 82.84, 'emnist': 91.62
    },
}

ln = {}
for f in glob.glob('experiment_results/ln_gelu_cross_dataset/*burstiness*.json'):
    r = json.load(open(f))
    tag = 'FFCL + Burstiness (ln_gelu)' if r['ffcl'] else 'Std + Burstiness (ln_gelu)'
    ln.setdefault(tag, {})[r['dataset']] = r['test_accuracy'] * 100

ds_order = ['mnist', 'fashion_mnist', 'cifar10', 'usps', 'svhn', 'emnist']
ds_short = ['MNIST', 'F-MNIST', 'CIFAR-10', 'USPS', 'SVHN', 'EMNIST']

hdr = f"{'Config':<34s}" + "".join(f"{s:>9s}" for s in ds_short)
print(hdr)
print("=" * 88)

for name in ['Std + Burstiness (Swish)',
             'Std + Burstiness (ln_gelu)',
             '',
             'FFCL + Burstiness (GELU)',
             'FFCL + Burstiness (ln_gelu)']:
    if not name:
        print()
        continue
    vals = paper.get(name, ln.get(name, {}))
    row = "".join(f"{vals.get(d, 0):8.2f}%" for d in ds_order)
    print(f"{name:<34s}{row}")

print()
print("Delta (ln_gelu vs paper original):")
print("-" * 88)
for paper_name, ln_name in [
    ('Std + Burstiness (Swish)', 'Std + Burstiness (ln_gelu)'),
    ('FFCL + Burstiness (GELU)', 'FFCL + Burstiness (ln_gelu)'),
]:
    pv = paper[paper_name]
    lv = ln[ln_name]
    row = "".join(f"{lv.get(d,0) - pv.get(d,0):+8.2f}p" for d in ds_order)
    print(f"{ln_name:<34s}{row}")
