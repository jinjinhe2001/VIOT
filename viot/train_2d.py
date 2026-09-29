"""Train the 2D VIOT operator (FNO stream function, Benamou-Brenier + viscosity loss).

Usage (the recipe of the released ``mnist_2d`` model)::

    python -m viot.train_2d \\
        --resolution 256 --sampler mnist --batch-size 4 --lr 1e-4 --n-steps 80000 \\
        --n-rollout 10 --n-test 20 --n-infer-steps 50 --k-max 0.25 \\
        --arch fno --fno-width 64 --fno-modes 32 --fno-layers 8 \\
        --bb-visc-only --lambda-ke 1.0 --lambda-mu 0.01 --save-dir runs/mnist_2d

For a pre-rendered pool use ``--sampler pool --data-path pool.pt`` (the
original name ``--sampler chinese`` still works).

Objective. Each step draws a batch ``(rho_0, rho_1)``, unrolls the model for
``--n-rollout`` steps (``dt = 1 / n_rollout``) with differentiable advection,
clamping and mass renormalisation, and minimises::

    loss = MSE(HW * rho_T, HW * rho_1)                          # terminal
         + lambda_ke * mean_s mean(rho_s * |v_s|^2)             # kinetic energy
         + lambda_mu * mean_s mean(omega(v_s)^2)                # enstrophy (viscosity)

with Adam, gradient-norm clipping at 1.0 and (unless ``--no-scheduler``)
cosine decay to ``0.01 * lr``.

Outputs in ``--save-dir``:

* ``spectral_bb_visc_model_step{N}.pt`` every 1000 steps;
* ``spectral_bb_visc_model_best.pt``: at each 1000-step save, overwritten if
  the current training batch's terminal loss is the lowest so far (this is a
  training-batch criterion, not a validation score);
* ``spectral_bb_visc_model.pt``: final weights;
* ``rollout_results.pt``: an ``--n-test``-pair evaluation of the final
  weights (pairs drawn after ``torch.manual_seed(42)``, ``--n-infer-steps``
  rollout steps), followed by a printed metrics table.

All checkpoints are bare ``state_dict`` files loadable by
:class:`viot.model_2d.FNO2D`.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data_2d import get_sampler
from .model_2d import FNO2D
from .ops_2d import advect, rollout_2d, vorticity_2d

CKPT_EVERY = 1000
RUN_NAME = "spectral_bb_visc"


# ---------------------------------------------------------------------------
# End-of-training metrics (same formulas as the original trainer's table)
# ---------------------------------------------------------------------------

def _zero_pad_div(v):
    # Zero-padded central differences, as in the original trainer's metrics
    # table (the training loss and viot.ops_2d use circular padding).
    vx_pad = F.pad(v[:, 0:1], (1, 1, 1, 1), mode='constant', value=0)
    vy_pad = F.pad(v[:, 1:2], (1, 1, 1, 1), mode='constant', value=0)
    return ((vx_pad[:, :, 2:, 1:-1] - vx_pad[:, :, :-2, 1:-1]) / 2.0
            + (vy_pad[:, :, 1:-1, 2:] - vy_pad[:, :, 1:-1, :-2]) / 2.0)


def _zero_pad_vorticity(v):
    vx_pad = F.pad(v[:, 0:1], (1, 1, 1, 1), mode='constant', value=0)
    vy_pad = F.pad(v[:, 1:2], (1, 1, 1, 1), mode='constant', value=0)
    return ((vy_pad[:, :, 2:, 1:-1] - vy_pad[:, :, :-2, 1:-1]) / 2.0
            - (vx_pad[:, :, 1:-1, 2:] - vx_pad[:, :, 1:-1, :-2]) / 2.0)


def compute_metrics(results, name=RUN_NAME):
    """Print and return the end-of-training metrics for ``results`` (CPU tensors).

    Terminal Error (L2) = mean over pairs of ``||rho_T - rho_1||_2``.
    """
    frames = results[f"{name}_frames"]
    velocities = results[f"{name}_velocities"]
    rho_1 = results["test_rho_1"]
    rho_final = frames[-1]
    m = {
        "terminal_err": ((rho_final - rho_1) ** 2).sum(dim=(-2, -1)).sqrt().mean().item(),
        "mass_pct": (frames[-1].sum(dim=(-2, -1)) / (frames[0].sum(dim=(-2, -1)) + 1e-12)).mean().item() * 100,
        "mean_div": float(np.mean([_zero_pad_div(v).abs().mean().item() for v in velocities])),
        "enstrophy": float(np.mean([(_zero_pad_vorticity(v) ** 2).mean().item() for v in velocities])),
        "kinetic_energy": float(np.mean([(frames[i] * (v ** 2).sum(dim=1, keepdim=True)).mean().item()
                                         for i, v in enumerate(velocities)])),
        "inference_time_ms": results.get(f"{name}_inference_time_ms", float('nan')),
    }
    print(f"\n{'='*70}\nMETRICS\n{'='*70}")
    print(f"{'Terminal Error (L2)':<25} | {m['terminal_err']:>14.6f}")
    print(f"{'Mass Conservation (%)':<25} | {m['mass_pct']:>13.2f}%")
    print(f"{'Mean |div(v)|':<25} | {m['mean_div']:>14.6f}")
    print(f"{'Mean Enstrophy':<25} | {m['enstrophy']:>14.6f}")
    print(f"{'Kinetic Energy':<25} | {m['kinetic_energy']:>14.6f}")
    print(f"{'Inference Time (ms)':<25} | {m['inference_time_ms']:>13.1f}ms")
    print(f"{'='*70}")
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args=None):
    if args is None:
        args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    B = args.batch_size
    res = args.resolution
    n_rollout = args.n_rollout
    save_dir = args.save_dir
    scheme = args.advection

    sample_pair = get_sampler(args.sampler, res, data_path=args.data_path,
                              device=device, mnist_root=args.mnist_root)

    name = RUN_NAME
    print(f"\n{'='*60}")
    print(f"Training: {name} (loss=bb_visc, arch={args.arch}, advection={scheme})")
    print(f"  res={res}, k_max={args.k_max}")
    print(f"{'='*60}")

    model = FNO2D(
        max_res=res, k_max=args.k_max,
        width=args.fno_width, n_modes=args.fno_modes,
        n_layers=args.fno_layers, verbose=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if args.init_from is not None:
        sd_in = torch.load(args.init_from, map_location=device, weights_only=False)
        if isinstance(sd_in, dict) and "model_state_dict" in sd_in:
            model.load_state_dict(sd_in["model_state_dict"])
            prev_step = sd_in.get("step", None)
            prev_best = sd_in.get("best_loss", None)
            print(f"  Init from resume ckpt: {args.init_from} (prev step={prev_step}, prev best={prev_best})")
        else:
            model.load_state_dict(sd_in)
            print(f"  Init from model weights: {args.init_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None if args.no_scheduler else \
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_steps, eta_min=args.lr * 0.01)

    model.train()
    best_loss = float("inf")
    t_start = time.time()

    for step in range(args.n_steps):
        rho_0, rho_1 = sample_pair(B)

        optimizer.zero_grad()

        dt_r = 1.0 / n_rollout
        rho_scale = float(rho_0.shape[-1] * rho_0.shape[-2])
        rho_r = rho_0.clone()
        initial_mass = rho_r.sum(dim=(-2, -1), keepdim=True)
        ke_sum = 0.0
        enst_sum = 0.0

        for s in range(n_rollout):
            t_s = torch.full((B,), s * dt_r, device=device)
            v_s = model.forward_velocity_only(rho_r, t_s, rho_1)

            v_sq = (v_s ** 2).sum(dim=1, keepdim=True)
            ke_sum = ke_sum + (rho_r * v_sq).mean()

            omega = vorticity_2d(v_s)
            enst_sum = enst_sum + (omega ** 2).mean()

            rho_r = advect(rho_r, v_s, dt_r, scheme=scheme)
            rho_r = rho_r.clamp(min=0)
            rho_r = rho_r * initial_mass / (rho_r.sum(dim=(-2, -1), keepdim=True) + 1e-12)

        terminal = F.mse_loss(rho_r * rho_scale, rho_1 * rho_scale)
        ke = ke_sum / n_rollout
        loss = terminal + args.lambda_ke * ke
        enst = enst_sum / n_rollout
        loss = loss + args.lambda_mu * enst

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t_start
            cur_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            extra = f" | term={terminal.item():.2e} | ke={ke.item():.2e} | enst={enst.item():.2e}"
            print(f"  step {step:5d}/{args.n_steps} | loss={loss.item():.6f}"
                  f"{extra} | lr={cur_lr:.1e} | {elapsed:.1f}s")

        if (step + 1) % CKPT_EVERY == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{name}_model_step{step+1}.pt"))
            cur_loss = terminal.item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"{name}_model_best.pt"))
                print(f"  [ckpt] best saved (loss={cur_loss:.6f})")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_model.pt"))
    print(f"  Done in {time.time() - t_start:.1f}s")

    # Evaluate the final weights on n_test fixed-seed pairs
    print(f"\nEvaluating {name}...")
    model.eval()
    torch.manual_seed(42)
    test_rho_0, test_rho_1 = sample_pair(args.n_test)

    results = {"test_rho_0": test_rho_0.cpu(), "test_rho_1": test_rho_1.cpu()}

    t_inf = time.time()
    data = rollout_2d(model, test_rho_0, test_rho_1, n_steps=args.n_infer_steps,
                      scheme=scheme, return_frames=True, return_velocities=True)
    inf_ms = (time.time() - t_inf) / args.n_test * 1000
    results[f"{name}_frames"] = [f.cpu() for f in data["frames"]]
    results[f"{name}_velocities"] = [v.cpu() for v in data["velocities"]]
    results[f"{name}_inference_time_ms"] = inf_ms
    print(f"  Inference (conditional): {inf_ms:.1f} ms/sample")

    torch.save(results, os.path.join(save_dir, "rollout_results.pt"))
    compute_metrics(results, name)

    print("\nAll done!")
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the 2D VIOT operator (FNO stream function, BB + viscosity loss)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--sampler", type=str, default="mnist",
                   choices=["mnist", "pool", "chinese"],
                   help="mnist (torchvision MNIST) or pool (pre-rendered .pt, needs --data-path); "
                        "'chinese' is the original name of 'pool'")
    p.add_argument("--data-path", type=str, default=None,
                   help="Pool .pt file [N,1,H,W] for --sampler pool")
    p.add_argument("--mnist-root", type=str, default=".data",
                   help="torchvision MNIST root for --sampler mnist (default: .data)")
    p.add_argument("--n-rollout", type=int, default=10,
                   help="Rollout steps per training sample")
    p.add_argument("--n-test", type=int, default=10,
                   help="Number of end-of-training evaluation pairs")
    p.add_argument("--save-dir", type=str, default="results_spectral_2d")
    p.add_argument("--k-max", type=float, default=0.25,
                   help="Stream-function truncation radius in cycles/pixel")
    p.add_argument("--lambda-ke", type=float, default=1.0)
    p.add_argument("--lambda-mu", type=float, default=0.1)
    p.add_argument("--n-infer-steps", type=int, default=50,
                   help="Rollout steps for the end-of-training evaluation")
    p.add_argument("--advection", type=str, default="maccormack",
                   choices=["maccormack", "semi_lagrangian"],
                   help="Advection scheme for training and evaluation "
                        "(latin_font_2d was trained with semi_lagrangian)")
    p.add_argument("--no-scheduler", action="store_true",
                   help="Constant learning rate instead of cosine decay")
    p.add_argument("--bb-visc-only", action="store_true",
                   help="Accepted for compatibility; the BB + viscosity loss is always used")
    p.add_argument("--arch", type=str, default="fno", choices=["fno"],
                   help="Accepted for compatibility; only the FNO is released")
    p.add_argument("--fno-width", type=int, default=64,
                   help="FNO feature channel width (default: 64)")
    p.add_argument("--fno-modes", type=int, default=16,
                   help="FNO modes per dimension per layer (default: 16)")
    p.add_argument("--fno-layers", type=int, default=8,
                   help="Number of FNO layers (default: 8)")
    p.add_argument("--init-from", type=str, default=None,
                   help="Path to .pt (raw model state_dict OR dict with 'model_state_dict') "
                        "to initialize weights. Fresh optimizer/scheduler.")
    p.add_argument("--log-every", type=int, default=500,
                   help="Log every N steps (default: 500)")
    p.add_argument("--seed", type=int, default=None,
                   help="torch.manual_seed at start (default: unseeded, as in the paper runs)")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: cuda if available, else cpu)")
    return p.parse_args(argv)


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
