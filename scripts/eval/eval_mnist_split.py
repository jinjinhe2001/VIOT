r"""Evaluate a 2D MNIST operator on the official MNIST test (or train) split with the training sampler.

``mnist_2d`` was trained on the 60k torchvision training split only. This script draws pairs with
the training sampler ``viot.data_2d.sample_mnist_pairs`` (upsample -> random blur sigma ~ U(0.5, 2)
-> floor -> PR normalisation -> mass normalisation) from the requested split, after
``torch.manual_seed(42)``, in batches of 10, and reports the metrics of eval_2d_heldout.py
(50-step MacCormack rollout).

Paper rows ("MNIST 2D": in-pool = train split, evaluation = test split):
  python scripts/eval/eval_mnist_split.py --model mnist_2d --split test  --out runs/eval/mnist_2d_on_mnist_test.json
  python scripts/eval/eval_mnist_split.py --model mnist_2d --split train --out runs/eval/mnist_2d_on_mnist_train.json
"""
import argparse
import json
import os
import time

import torch

from eval_common import add_model_args, load_model  # noqa: E402  (also puts the repo on sys.path)
from eval_2d_heldout import rollout_metrics, summarize  # noqa: E402
from viot.data_2d import sample_mnist_pairs  # noqa: E402
from viot.ops_2d import ADVECTION_SCHEMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, dim=2)
    ap.add_argument('--split', choices=['train', 'test'], required=True)
    ap.add_argument('--mnist-root', default='data', help='torchvision MNIST root (downloaded if missing)')
    ap.add_argument('--n-pairs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-infer-steps', type=int, default=50)
    ap.add_argument('--advection', default='maccormack', choices=list(ADVECTION_SCHEMES))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = torch.device(a.device)

    model, info = load_model(a, dim=2, device=dev)
    R = info['arch']['max_res']
    train = a.split == 'train'
    print(f'MNIST split={a.split} (root {a.mnist_root})', flush=True)

    torch.manual_seed(a.seed)
    acc, done, t0 = {}, 0, time.time()
    while done < a.n_pairs:
        b = min(a.batch, a.n_pairs - done)
        rho_0, rho_1 = sample_mnist_pairs(b, R, R, dev, root=a.mnist_root, train=train)
        m = rollout_metrics(model, rho_0, rho_1, a.n_infer_steps, scheme=a.advection)
        for k, v in m.items():
            acc.setdefault(k, []).append(v.cpu())
        done += b
    acc = {k: torch.cat(v) for k, v in acc.items()}
    print(f'done {a.n_pairs} pairs in {time.time() - t0:.1f}s', flush=True)
    summary = summarize(acc)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'tag': a.tag, 'ckpt': info['ckpt'], 'model': info['model'], 'arch': info['arch'],
                   'advection': a.advection, 'split': a.split, 'n_pairs': a.n_pairs, 'seed': a.seed,
                   'n_infer_steps': a.n_infer_steps, 'summary': summary,
                   'per_pair': {k: v.tolist() for k, v in acc.items()}}, f)
    print('saved', a.out)


if __name__ == '__main__':
    main()
