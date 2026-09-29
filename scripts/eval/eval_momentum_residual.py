r"""Leray-projected Navier-Stokes momentum residual of a 2D VIOT rollout (the "implicit control force").

For a rollout v_0..v_{T-1} (T = n_steps) computed with eval_2d_heldout.py's inference loop, form
    m = dv/dt + (v . grad) v - mu * lap v          (central time differences, spectral space derivatives)
and report  r = || P[m] || / || m ||  where P = I - k k^T/|k|^2 is the Leray projector (it removes the
part a pressure gradient can absorb). r ~ 0 <=> unforced NS solution; r ~ 1 <=> fully forced.
Also reports the spectral mean |div v| (exactness of the curl parameterisation) and the same residual
at a second step count to check time-discretisation convergence.

Pairs: with ``torch.Generator().manual_seed(42)``, ``--n-pairs`` (20) source then target indices from
the pool, batches of 5; MacCormack rollout with clamp + mass renormalisation.

  python scripts/eval/eval_momentum_residual.py --model mpeg7_2d --data-path data/mpeg7_256_datamean_clean.pt \
      --n-pairs 20 --steps 50 100 --out runs/eval/mom_mpeg7_2d.json

This is the r = ||P m|| / ||m|| diagnostic of the paper's momentum discussion. The q / F_RMS /
delta_M / delta_H columns of the arXiv momentum table come from a later diagnostic suite that is
not part of this release (see scripts/eval/README.md).
"""
import argparse
import json
import math
import os

import torch

from eval_common import add_model_args, load_model  # noqa: E402  (also puts the repo on sys.path)
from eval_2d_heldout import load_pool  # noqa: E402
from viot.ops_2d import ADVECTION_SCHEMES, advect  # noqa: E402


@torch.no_grad()
def rollout_velocities(model, rho_0, rho_1, n_steps, scheme='maccormack'):
    dt = 1.0 / n_steps
    rho = rho_0.clone()
    m0 = rho.sum(dim=(-2, -1), keepdim=True)
    vs = []
    for step in range(n_steps):
        t = torch.full((rho.shape[0],), step / n_steps, device=rho.device, dtype=rho.dtype)
        v = model.forward_velocity_only(rho, t, rho_1)
        vs.append(v)
        rho = advect(rho, v, dt, scheme=scheme).clamp(min=0)
        rho = rho * m0 / (rho.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    return torch.stack(vs)  # [T,B,2,H,W]


def residual_stats(V, mu):
    T, B, C, H, W = V.shape
    dev = V.device
    dt = 1.0 / T
    ky = (2 * math.pi * torch.fft.fftfreq(H, device=dev)).view(H, 1)
    kx = (2 * math.pi * torch.fft.fftfreq(W, device=dev)).view(1, W)
    ksq = ky ** 2 + kx ** 2
    ksq_safe = ksq.clone()
    ksq_safe[0, 0] = 1.0

    def d_dy(f):
        return torch.fft.ifft2(torch.fft.fft2(f) * (1j * ky)).real

    def d_dx(f):
        return torch.fft.ifft2(torch.fft.fft2(f) * (1j * kx)).real

    def lap(f):
        return torch.fft.ifft2(torch.fft.fft2(f) * (-ksq)).real

    vy, vx = V[:, :, 0], V[:, :, 1]
    dvy = (vy[2:] - vy[:-2]) / (2 * dt)
    dvx = (vx[2:] - vx[:-2]) / (2 * dt)
    vym, vxm = vy[1:-1], vx[1:-1]
    my = dvy + vxm * d_dx(vym) + vym * d_dy(vym) - mu * lap(vym)
    mx = dvx + vxm * d_dx(vxm) + vym * d_dy(vxm) - mu * lap(vxm)
    My, Mx = torch.fft.fft2(my), torch.fft.fft2(mx)
    kdotM = ky * My + kx * Mx
    Myp, Mxp = My - ky * kdotM / ksq_safe, Mx - kx * kdotM / ksq_safe
    # per-pair relative residual (over time and space), then global
    num_pp = (Myp.abs() ** 2 + Mxp.abs() ** 2).sum(dim=(0, 2, 3)).sqrt()
    den_pp = (My.abs() ** 2 + Mx.abs() ** 2).sum(dim=(0, 2, 3)).sqrt()
    rel_pp = num_pp / (den_pp + 1e-20)
    rel_global = ((Myp.abs() ** 2 + Mxp.abs() ** 2).sum().sqrt() / ((My.abs() ** 2 + Mx.abs() ** 2).sum().sqrt() + 1e-20)).item()
    divv = (d_dx(vxm) + d_dy(vym)).abs().mean().item()
    return dict(rel_global=rel_global, rel_mean=rel_pp.mean().item(), rel_std=rel_pp.std().item(),
                spectral_mean_abs_div=divv)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, dim=2)
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--n-pairs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=5)
    ap.add_argument('--steps', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--mu', type=float, default=0.0, help='physical viscosity in the residual (0 = Euler momentum)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--advection', default='maccormack', choices=list(ADVECTION_SCHEMES))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, info = load_model(a, dim=2, device=dev)
    pool = load_pool(a.data_path)
    N = pool.shape[0]
    g = torch.Generator().manual_seed(a.seed)
    idx_src = torch.randint(0, N, (a.n_pairs,), generator=g)
    idx_tgt = torch.randint(0, N, (a.n_pairs,), generator=g)
    out = {'tag': a.tag, 'ckpt': info['ckpt'], 'model': info['model'], 'arch': info['arch'], 'advection': a.advection,
           'data_path': a.data_path, 'n_pairs': a.n_pairs, 'mu': a.mu, 'by_steps': {}}
    for T in a.steps:
        agg = {'rel_pp': [], 'div': []}
        rel_g_num = 0.0
        for b in range(0, a.n_pairs, a.batch):
            s, t_ = idx_src[b:b + a.batch], idx_tgt[b:b + a.batch]
            rho_0 = pool[s].to(dev)
            rho_1 = pool[t_].to(dev)
            rho_0 = rho_0 / (rho_0.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            rho_1 = rho_1 / (rho_1.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            V = rollout_velocities(model, rho_0, rho_1, T, scheme=a.advection)
            st = residual_stats(V, a.mu)
            agg['rel_pp'].append(st['rel_mean'])
            agg['div'].append(st['spectral_mean_abs_div'])
            rel_g_num += st['rel_global']
        r = {'rel_mean_over_batches': sum(agg['rel_pp']) / len(agg['rel_pp']),
             'rel_global_mean_over_batches': rel_g_num / len(agg['rel_pp']),
             'spectral_mean_abs_div': sum(agg['div']) / len(agg['div'])}
        out['by_steps'][str(T)] = r
        print(f'{a.tag} steps={T}: rel momentum residual = {r["rel_mean_over_batches"]:.4f} '
              f'(global {r["rel_global_mean_over_batches"]:.4f}) | spectral |div| = {r["spectral_mean_abs_div"]:.2e}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', a.out)


if __name__ == '__main__':
    main()
