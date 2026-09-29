"""Print shape, mass, peak and participation-ratio statistics of density pools (.pt tensors).

  python scripts/data/pool_stats.py data/chinese_chars_256.pt data/mpeg7_256_datamean_clean.pt

Statistics use the first 300 samples. 2D pools should have sum 1 per sample; the PR fraction
(PR / number of cells) is ~0.197 for the glyph pools and ~0.244 for MPEG-7.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('pools', nargs='+', help='.pt files holding a tensor [N, 1, ...]')
    a = ap.parse_args()
    for f in a.pools:
        x = torch.load(f, map_location='cpu', weights_only=False)
        n = x.shape[0]
        x = x[:300].float()
        dims = tuple(range(1, x.dim()))
        s = x.sum(dims)
        cells = x[0].numel()
        pr = (s ** 2) / (x.pow(2).sum(dims) + 1e-12) / cells
        print(f'{f}: N={n} {tuple(x.shape[1:])} sum={s.mean():.4f} max={x.amax(dims).mean():.6f} '
              f'PRfrac={pr.mean():.4f}+-{pr.std():.4f} min={x.min():.2e}', flush=True)


if __name__ == '__main__':
    main()
