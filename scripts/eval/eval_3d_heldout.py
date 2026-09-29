r"""Held-out evaluation of a 3D VIOT operator (paper Table "Generalization", 3D rows).

Uses the training sampler (``viot.data_3d.sample_voxel_pairs`` = the trainer's raw + peak-norm +
domain-check sampler, no augmentation) and the as-trained first-order semi-Lagrangian advection,
so the numbers are on the paper's scale (peak-normalised densities, terminal L2 =
||rho_T - rho_1||_2 summed over voxels). After ``torch.manual_seed(42)``, pairs are drawn in
batches of ``--batch`` (4) until ``--n-pairs`` (100); each batch draws its source indices, then its
target indices, from the global torch RNG (so the batch size changes which pairs are drawn). The
rollout clamps and renormalises mass after every step. Metrics: terminal L2, relative L2, mass %,
mean |div v|, mean enstrophy mean_x |omega|^2, mean KE mean(rho |v|^2) (steps averaged).

Examples (see scripts/eval/README.md for every row of the paper table):
  python scripts/eval/eval_3d_heldout.py --model sphere2airplane_3d \
      --src-path data/sphere_match_airplanes_wtt_s02_128.pt --tgt-path data/airplanes_wtt_test_128.pt \
      --out runs/eval/sphere2airplane_3d_on_test808.json
  python scripts/eval/eval_3d_heldout.py --model humman_3d \
      --src-path data/humman_interp_heldout_n14000_128.pt --tgt-path data/humman_interp_heldout_n14000_128.pt \
      --out runs/eval/humman_3d_on_interp_heldout.json
If the GPU runs out of memory, lower --batch (this draws different pairs than the paper protocol).
When --src-path equals --tgt-path the pool is loaded once (the HuMMan pool alone is ~42 GB).
"""
import argparse
import json
import os
import time

import torch

from eval_common import add_model_args, load_model  # noqa: E402  (also puts the repo on sys.path)
from viot.data_3d import load_voxels, sample_voxel_pairs  # noqa: E402
from viot.ops_3d import ADVECTION_SCHEMES, advect_3d, divergence_3d, vorticity_3d  # noqa: E402


@torch.no_grad()
def rollout_metrics(model, rho_0, rho_1, n_steps, scheme='semi_lagrangian'):
    dt = 1.0 / n_steps
    rho = rho_0.clone()
    m0 = rho.sum(dim=(-3, -2, -1), keepdim=True)
    B = rho.shape[0]
    div_acc = torch.zeros(B, device=rho.device)
    enst_acc = torch.zeros(B, device=rho.device)
    ke_acc = torch.zeros(B, device=rho.device)
    for step in range(n_steps):
        t = torch.full((B,), step / n_steps, device=rho.device, dtype=rho.dtype)
        v = model.forward_velocity_only(rho, t, rho_1)
        div_acc += divergence_3d(v).abs().mean(dim=(1, 2, 3, 4))
        enst_acc += (vorticity_3d(v) ** 2).sum(dim=1).mean(dim=(1, 2, 3))
        ke_acc += (rho * (v ** 2).sum(1, keepdim=True)).mean(dim=(1, 2, 3, 4))
        rho = advect_3d(rho, v, dt, scheme=scheme).clamp(min=0)
        rho = rho * m0 / (rho.sum(dim=(-3, -2, -1), keepdim=True) + 1e-12)
    l2 = ((rho - rho_1) ** 2).sum(dim=(1, 2, 3, 4)).sqrt()
    rel = l2 / (rho_1 ** 2).sum(dim=(1, 2, 3, 4)).sqrt()
    mass = rho.sum(dim=(1, 2, 3, 4)) / (rho_0.sum(dim=(1, 2, 3, 4)) + 1e-12) * 100
    return dict(terminal_l2=l2, rel_l2=rel, mass_pct=mass, mean_abs_div=div_acc / n_steps,
                enstrophy=enst_acc / n_steps, ke=ke_acc / n_steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, dim=3)
    ap.add_argument('--src-path', default='data/sphere_match_airplanes_wtt_s02_128.pt')
    ap.add_argument('--tgt-path', default='data/airplanes_wtt_test_128.pt')
    ap.add_argument('--n-pairs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-infer-steps', type=int, default=50)
    ap.add_argument('--advection', default='semi_lagrangian', choices=list(ADVECTION_SCHEMES),
                    help='advection scheme of the rollout (paper protocol: semi_lagrangian, as trained)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = torch.device(a.device)

    model, info = load_model(a, dim=3, device=dev)

    src = load_voxels(a.src_path).float()
    tgt = src if a.tgt_path == a.src_path else load_voxels(a.tgt_path).float()
    print(f'src {tuple(src.shape)} | tgt {tuple(tgt.shape)} from {a.tgt_path} | advection={a.advection}', flush=True)

    torch.manual_seed(a.seed)
    acc = {}
    t0 = time.time()
    done = 0
    while done < a.n_pairs:
        b = min(a.batch, a.n_pairs - done)
        rho_0, rho_1 = sample_voxel_pairs(src, tgt, b, dev, peak_norm=True)
        m = rollout_metrics(model, rho_0, rho_1, a.n_infer_steps, scheme=a.advection)
        for k, v in m.items():
            acc.setdefault(k, []).append(v.cpu())
        done += b
        if done % 20 == 0:
            print(f'  {done}/{a.n_pairs} pairs | {time.time() - t0:.0f}s', flush=True)
    acc = {k: torch.cat(v) for k, v in acc.items()}
    summary = {k: {'mean': float(v.mean()), 'std': float(v.std()), 'median': float(v.median())} for k, v in acc.items()}
    print(f'done {a.n_pairs} pairs in {time.time() - t0:.0f}s', flush=True)
    print(f'{"metric":<14} {"mean":>12} {"std":>12} {"median":>12}')
    for k, v in summary.items():
        print(f'{k:<14} {v["mean"]:>12.6g} {v["std"]:>12.4g} {v["median"]:>12.6g}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'tag': a.tag, 'ckpt': info['ckpt'], 'model': info['model'], 'arch': info['arch'],
                   'advection': a.advection, 'src_path': a.src_path, 'tgt_path': a.tgt_path, 'n_pairs': a.n_pairs,
                   'seed': a.seed, 'n_infer_steps': a.n_infer_steps, 'summary': summary,
                   'per_pair': {k: v.tolist() for k, v in acc.items()}}, f)
    print('saved', a.out)


if __name__ == '__main__':
    main()
