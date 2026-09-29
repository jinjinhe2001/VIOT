"""Dump N MNIST densities (train or test split) through the training sampler into a pool file.

Uses ``viot.data_2d.sample_mnist_pairs`` (upsample -> random blur -> floor -> PR normalisation ->
mass normalisation), batches of 50 pairs, after ``torch.manual_seed(seed)``. The pool is the
source/target samples of consecutive batches, [N, 1, 256, 256].

Paper pool (MNIST test split, used by the MNIST momentum diagnostics):
  python scripts/data/dump_mnist_pool.py --split test --n 300 --seed 0 --out data/mnist_test_pool_256.pt
(the paper's exact command was not archived; these are the script defaults it was built with).
"""
import argparse
import os
import sys
from pathlib import Path

import torch

try:
    import viot  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from viot.data_2d import sample_mnist_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--split', default='test', choices=['train', 'test'])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--mnist-root', default='data', help='torchvision MNIST root (downloaded if missing)')
    ap.add_argument('--device', default='cuda', help='device for the resize/blur/area steps (paper: cuda)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    R = a.resolution
    xs = []
    while sum(x.shape[0] for x in xs) < a.n:
        r0, r1 = sample_mnist_pairs(50, R, R, dev, root=a.mnist_root, train=(a.split == 'train'))
        xs += [r0.cpu(), r1.cpu()]
    X = torch.cat(xs)[:a.n].contiguous()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(X, a.out)
    print('saved', a.out, tuple(X.shape), 'sum', float(X.sum((1, 2, 3)).mean()), 'split', a.split)


if __name__ == '__main__':
    main()
