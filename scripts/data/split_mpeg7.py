r"""Split the MPEG-7 pool 90/10 for the held-out retrain (paper run exp90).

``perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))``; the first
``round(0.10 * N)`` indices (sorted) are the test set, the rest (sorted) the training set.
For the paper file (N = 1294, seed 0) this gives 1165 train / 129 test shapes, identical to
``scripts/data/splits/mpeg7_256_split_s0.json``.

Writes ``<out-dir>/mpeg7_256_{train,test}_s<seed>.pt`` and ``<out-dir>/mpeg7_256_split_s<seed>.json``.

  python scripts/data/split_mpeg7.py --data data/mpeg7_256_datamean_clean.pt --out-dir data \
      --check scripts/data/splits/mpeg7_256_split_s0.json
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--data', default='data/mpeg7_256_datamean_clean.pt')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test-frac', type=float, default=0.10)
    ap.add_argument('--prefix', default='mpeg7_256')
    ap.add_argument('--check', default=None,
                    help='split json to compare the indices against (fails on mismatch)')
    a = ap.parse_args()

    d = torch.load(a.data, map_location='cpu', weights_only=False)
    x = d if torch.is_tensor(d) else d[list(d.keys())[0]]
    N = x.shape[0]
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(N, generator=g)
    n_test = int(round(a.test_frac * N))
    te, tr = perm[:n_test].sort().values, perm[n_test:].sort().values
    split = {'seed': a.seed, 'N': N, 'train_idx': tr.tolist(), 'test_idx': te.tolist()}
    if a.check:
        with open(a.check) as f:
            ref = json.load(f)
        if ref['train_idx'] != split['train_idx'] or ref['test_idx'] != split['test_idx']:
            raise SystemExit(f'split differs from {a.check} (N={N} vs {ref["N"]})')
        print(f'split matches {a.check}')
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f'{a.prefix}_{{}}_s{a.seed}'
    torch.save(x[tr].clone(), os.path.join(a.out_dir, tag.format('train') + '.pt'))
    torch.save(x[te].clone(), os.path.join(a.out_dir, tag.format('test') + '.pt'))
    with open(os.path.join(a.out_dir, f'{a.prefix}_split_s{a.seed}.json'), 'w') as f:
        json.dump(split, f)
    print(f'N={N} train={len(tr)} test={len(te)} shape={tuple(x.shape[1:])} dtype={x.dtype} '
          f'sum0={float(x[0].sum()):.3f} max0={float(x[0].max()):.3f} -> {a.out_dir}')


if __name__ == '__main__':
    main()
