"""Held-out evaluation of a 2D VIOT operator on a shape pool (paper Table "Generalization").

Protocol (identical to the paper's held-out evaluation): with ``torch.Generator().manual_seed(42)``,
draw ``--n-pairs`` source indices and then ``--n-pairs`` target indices uniformly with replacement
from the pool; mass-normalise both; roll out ``--n-infer-steps`` = 50 steps (dt = 1/50) with
MacCormack advection, clamping and mass renormalisation after every step; batches of 10 pairs.
Metrics per pair: terminal L2 = ||rho_T - rho_1||_2 on the unit-mass scale, relative L2
(/ ||rho_1||_2), final mass in %, mean |div v| (circular central differences), mean enstrophy
mean(omega^2) and mean kinetic energy mean(rho |v|^2), each averaged over the rollout steps.
Prints mean / std / median and writes all per-pair values and the sampled indices to ``--out``.

Examples (see scripts/eval/README.md for every row of the paper table):
  python scripts/eval/eval_2d_heldout.py --model cjk_2d --data-path data/cjk_heldout_chars_256.pt \
      --out runs/eval/cjk_2d_on_cjk_heldout_chars.json
  python scripts/eval/eval_2d_heldout.py --ckpt runs/mpeg7_2d_heldout_split/spectral_bb_visc_model_best.pt \
      --data-path data/mpeg7_256_test_s0.pt --out runs/eval/mpeg7_split_on_test129.json

Note: every 2D row of the paper table, including latin_font_2d (trained with first-order
semi-Lagrangian advection), was evaluated with MacCormack advection, the default here.
"""
import argparse
import json
import os
import time

import torch

from eval_common import add_model_args, load_model  # noqa: E402  (also puts the repo on sys.path)
from viot.ops_2d import ADVECTION_SCHEMES, advect, divergence_2d, vorticity_2d  # noqa: E402


@torch.no_grad()
def rollout_metrics(model, rho_0, rho_1, n_steps, scheme='maccormack'):
    """Roll out ``n_steps`` steps and return per-pair metric tensors (see the module docstring)."""
    dt = 1.0 / n_steps
    rho = rho_0.clone()
    m0 = rho.sum(dim=(-2, -1), keepdim=True)
    div_acc = torch.zeros(rho.shape[0], device=rho.device)
    enst_acc = torch.zeros_like(div_acc)
    ke_acc = torch.zeros_like(div_acc)
    for step in range(n_steps):
        t = torch.full((rho.shape[0],), step / n_steps, device=rho.device, dtype=rho.dtype)
        v = model.forward_velocity_only(rho, t, rho_1)
        div_acc += divergence_2d(v).abs().mean(dim=(1, 2, 3))
        enst_acc += (vorticity_2d(v) ** 2).mean(dim=(1, 2, 3))
        ke_acc += (rho * (v ** 2).sum(1, keepdim=True)).mean(dim=(1, 2, 3))
        rho = advect(rho, v, dt, scheme=scheme).clamp(min=0)
        rho = rho * m0 / (rho.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    l2 = ((rho - rho_1) ** 2).sum(dim=(1, 2, 3)).sqrt()
    rel = l2 / (rho_1 ** 2).sum(dim=(1, 2, 3)).sqrt()
    mass = rho.sum(dim=(1, 2, 3)) / (rho_0.sum(dim=(1, 2, 3)) + 1e-12) * 100
    return dict(terminal_l2=l2, rel_l2=rel, mass_pct=mass, mean_abs_div=div_acc / n_steps,
                enstrophy=enst_acc / n_steps, ke=ke_acc / n_steps)


def summarize(acc):
    summary = {k: {'mean': float(v.mean()), 'std': float(v.std()), 'median': float(v.median())} for k, v in acc.items()}
    print(f'{"metric":<14} {"mean":>12} {"std":>12} {"median":>12}')
    for k, v in summary.items():
        print(f'{k:<14} {v["mean"]:>12.6g} {v["std"]:>12.4g} {v["median"]:>12.6g}')
    return summary


def load_pool(path):
    pool = torch.load(path, map_location='cpu', weights_only=False)
    if not torch.is_tensor(pool):
        pool = pool[list(pool.keys())[0]]
    return pool.float()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, dim=2)
    ap.add_argument('--data-path', required=True, help='.pt tensor [N,1,H,W] shape pool to draw pairs from')
    ap.add_argument('--n-pairs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--n-infer-steps', type=int, default=50)
    ap.add_argument('--advection', default='maccormack', choices=list(ADVECTION_SCHEMES),
                    help='advection scheme of the rollout (paper protocol: maccormack for every 2D model)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True, help='output JSON path')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    dev = torch.device(a.device)
    model, info = load_model(a, dim=2, device=dev)

    pool = load_pool(a.data_path)
    N = pool.shape[0]
    g = torch.Generator().manual_seed(a.seed)
    idx_src = torch.randint(0, N, (a.n_pairs,), generator=g)
    idx_tgt = torch.randint(0, N, (a.n_pairs,), generator=g)
    print(f'pool {a.data_path}: N={N} shape={tuple(pool.shape[1:])} | pairs={a.n_pairs} seed={a.seed} '
          f'| identical src==tgt: {(idx_src == idx_tgt).sum().item()} | advection={a.advection}', flush=True)

    acc = {}
    t0 = time.time()
    for b in range(0, a.n_pairs, a.batch):
        s, t_ = idx_src[b:b + a.batch], idx_tgt[b:b + a.batch]
        rho_0 = pool[s].to(dev)
        rho_1 = pool[t_].to(dev)
        rho_0 = rho_0 / (rho_0.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        rho_1 = rho_1 / (rho_1.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        m = rollout_metrics(model, rho_0, rho_1, a.n_infer_steps, scheme=a.advection)
        for k, v in m.items():
            acc.setdefault(k, []).append(v.cpu())
    acc = {k: torch.cat(v) for k, v in acc.items()}
    print(f'done {a.n_pairs} pairs in {time.time() - t0:.1f}s', flush=True)

    summary = summarize(acc)
    out = {'tag': a.tag, 'ckpt': info['ckpt'], 'model': info['model'], 'arch': info['arch'],
           'advection': a.advection, 'data_path': a.data_path, 'N_pool': N, 'n_pairs': a.n_pairs,
           'seed': a.seed, 'n_infer_steps': a.n_infer_steps, 'summary': summary,
           'per_pair': {k: v.tolist() for k, v in acc.items()},
           'idx_src': idx_src.tolist(), 'idx_tgt': idx_tgt.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f)
    print('saved', a.out)


if __name__ == '__main__':
    main()
