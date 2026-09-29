"""Train the 3D VIOT operator (FNO3D) with the Benamou-Brenier + viscosity objective.

Single GPU::

    python -m viot.train_3d --resolution 128 --sampler shapenet \\
        --data-path-src SRC.pt --data-path-tgt TGT.pt --bb-visc-only ...

Multi-GPU (one process per GPU; the batch size is per rank)::

    python -m torch.distributed.run --standalone --nproc_per_node=2 -m viot.train_3d ...

The exact commands of the released models are in ``scripts/train/``.

Per step, a batch of pairs is drawn (:func:`viot.data_3d.sample_voxel_pairs`)
and augmented, then ``rho_0`` is rolled out for ``--n-rollout`` steps with the
model velocity (differentiably). The loss is

    lambda_terminal * MSE(rho(1), rho_1) + lambda_ke * mean_s mean(rho_s |v_s|^2)
        + lambda_mu * mean_s mean(|curl v_s|^2)

with ``lambda_mu`` optionally ramped up linearly over ``--lambda-mu-warmup``
steps. Checkpoints (bare ``state_dict``) are written to ``--save-dir`` every
1000 steps as ``spectral_bb_visc_3d_model_step{N}.pt``;
``spectral_bb_visc_3d_model_best.pt`` is overwritten whenever the terminal loss
of the current training batch at such a save is the lowest so far, and
``spectral_bb_visc_3d_model.pt`` holds the final weights. Afterwards the final
model is evaluated on ``--n-test`` pairs (seed 42) with an ``--n-infer-steps``
rollout with mass renormalisation; trajectories go to ``rollout_results_3d.pt``
and a metric table is printed. ``--eval-only`` re-runs that evaluation on
``*_model_best.pt`` (or ``*_model.pt`` if there is no best file).

Distributed training does not wrap the model in DDP: rank 0's initial weights
are broadcast and gradients are averaged with an explicit all-reduce after
each backward pass. Rank ``r`` seeds the global RNG with ``42 + r``.
"""

import argparse
import contextlib
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint

from .data_3d import augment_batch_3d, load_voxels, sample_voxel_pairs
from .model_3d import FNO3D
from .ops_3d import ADVECTION_SCHEMES, advect_3d, rollout_3d, vorticity_3d

MODEL_NAME = "spectral_bb_visc_3d"


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _init_distributed():
    """Initialise the process group from torchrun env vars if present.

    Returns (local_rank, world_size, is_distributed); (0, 1, False) when not
    launched with more than one process. Uses NCCL (one GPU per rank), or
    gloo on the CPU when CUDA is unavailable.
    """
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        else:
            dist.init_process_group(backend="gloo")
        return local_rank, dist.get_world_size(), True
    return 0, 1, False


def _shutdown_distributed(is_distributed):
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_sampler(args, device):
    """Load the source/target voxel pools and return ``sample(B) -> (rho_0, rho_1)``."""
    if not args.data_path_src or not args.data_path_tgt:
        print("ERROR: --data-path-src and --data-path-tgt are required "
              "(pass the same file twice for self-pairs)")
        sys.exit(1)
    print(f"Loading source voxels from {args.data_path_src}...")
    shapes_src = load_voxels(args.data_path_src)
    print(f"  Loaded {shapes_src.shape[0]} source shapes, shape={shapes_src.shape}")
    print(f"Loading target voxels from {args.data_path_tgt}...")
    shapes_tgt = load_voxels(args.data_path_tgt)
    print(f"  Loaded {shapes_tgt.shape[0]} target shapes, shape={shapes_tgt.shape}")
    return lambda B: sample_voxel_pairs(shapes_src, shapes_tgt, B, device,
                                        peak_norm=args.peak_norm)


def _build_model(args, device, verbose):
    return FNO3D(
        max_res=args.resolution, k_max=args.k_max,
        width=args.fno_width, n_modes=args.fno_modes,
        n_layers=args.fno_layers, verbose=verbose,
    ).to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args=None):
    if args is None:
        args = parse_args()
    if not args.bb_visc_only:
        raise SystemExit("viot.train_3d only implements the released objective; pass --bb-visc-only")
    if not args.raw_sampler:
        raise SystemExit("viot.train_3d only implements the sampler of the released models; "
                         "pass --raw-sampler (and --peak-norm, as in scripts/train/)")

    # --- distributed setup (no-op unless launched with torchrun) ---
    local_rank, world_size, is_distributed = _init_distributed()
    is_main = local_rank == 0
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Per-rank seed so every rank draws different batches
    torch.manual_seed(42 + local_rank)

    if is_main:
        print(f"Training on {device} | world_size={world_size} "
              f"| distributed={is_distributed}")

    B = args.batch_size
    res = args.resolution
    n_rollout = args.n_rollout
    save_dir = args.save_dir
    name = MODEL_NAME

    sample_pair = get_sampler(args, device)

    use_amp = args.amp
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # --- evaluation only (rank 0) ---
    if args.eval_only:
        if is_distributed and not is_main:
            _shutdown_distributed(is_distributed)
            return
        print(f"Eval-only mode: loading models from {save_dir}/")
        models = {}
        best_path = os.path.join(save_dir, f"{name}_model_best.pt")
        final_path = os.path.join(save_dir, f"{name}_model.pt")
        if os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path):
            model_path = final_path
        else:
            model_path = None
            print(f"  Skipping {name} (no saved model found)")
        if model_path is not None:
            model = _build_model(args, device, verbose=True)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            models[name] = model
            n_params = sum(p.numel() for p in model.parameters())
            which = "best" if "best" in model_path else "final"
            print(f"  Loaded {name} ({n_params:,} params, {which})")

        if not models:
            print("ERROR: No models found.")
            _shutdown_distributed(is_distributed)
            return

        _evaluate(models, sample_pair, args, device)
        _shutdown_distributed(is_distributed)
        return

    # --- training ---
    if is_main:
        print(f"\n{'='*60}")
        print(f"Training: {name} (loss=bb_visc)")
        print(f"  resolution={res}^3, BS={B} per-rank × {world_size} = {B*world_size} global"
              f", k_max={args.k_max}")
        print(f"  fno_width={args.fno_width}, fno_modes={args.fno_modes}, "
              f"fno_layers={args.fno_layers}")
        print(f"  AMP={use_amp}, checkpoint={args.checkpoint}, advection={args.advection}")
        print(f"{'='*60}")

    model = _build_model(args, device, verbose=is_main)

    # Initialise from a checkpoint (continuation runs)
    if args.init_from is not None:
        if is_main:
            print(f"  Loading initial weights from: {args.init_from}")
        sd = torch.load(args.init_from, map_location=device, weights_only=True)
        # Drop trunc_mask (resolution-dependent buffer)
        sd_filtered = {k: v for k, v in sd.items() if 'trunc_mask' not in k}
        missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
        if is_main:
            if missing:
                print(f"    Missing keys (will keep model defaults): {missing}")
            if unexpected:
                print(f"    Unexpected keys (ignored): {unexpected}")
            print(f"    Loaded {len(sd_filtered)} params from checkpoint")

    # --init-scale: scale the (zero-initialised) last projection conv so that
    # the model does not start exactly at v = 0. Only for fresh runs.
    if args.init_scale != 1.0 and args.init_from is None:
        with torch.no_grad():
            final_proj = model.project[-1]
            final_proj.weight.mul_(args.init_scale)
            if final_proj.bias is not None:
                final_proj.bias.mul_(args.init_scale)
            if is_main:
                print(f"  Applied init-scale = {args.init_scale} to project[-1]")

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  Parameters: {n_params:,}")

    # Manual data parallelism: no DDP wrapper (DDP does not cope well with
    # n_rollout checkpointed forward calls per step); start from rank 0's
    # weights and all-reduce gradients after backward.
    if is_distributed:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None if args.no_scheduler else \
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_steps, eta_min=args.lr * 0.01)

    model.train()
    best_loss = float("inf")
    t_start = time.time()

    for step in range(args.n_steps):
        rho_0, rho_1 = sample_pair(B)
        rho_0, rho_1 = augment_batch_3d(rho_0, rho_1)

        optimizer.zero_grad()

        ctx = torch.amp.autocast('cuda') if use_amp and device.type == 'cuda' \
            else contextlib.nullcontext()
        with ctx:
            dt_r = 1.0 / n_rollout
            if args.peak_norm:
                # Peak-normalised densities are O(1) per voxel at any resolution.
                rho_scale = 1.0
            else:
                # Sum-normalised densities are ~1/N per voxel; rescale to O(1).
                rho_scale = float(res ** 3)
            rho_r = rho_0.clone()
            initial_mass = rho_r.sum(dim=(-3, -2, -1), keepdim=True)
            ke_sum = 0.0
            enst_sum = 0.0

            for s in range(n_rollout):
                t_s = torch.full((B,), s * dt_r, device=device)
                if args.checkpoint:
                    v_s = torch.utils.checkpoint.checkpoint(
                        model.forward_velocity_only, rho_r, t_s, rho_1,
                        use_reentrant=False)
                else:
                    v_s = model.forward_velocity_only(rho_r, t_s, rho_1)

                # Kinetic energy: rho * |v|^2
                v_sq = (v_s ** 2).sum(dim=1, keepdim=True)
                ke_sum = ke_sum + (rho_r * v_sq).mean()

                # Enstrophy: |curl v|^2
                omega = vorticity_3d(v_s)
                enst_sum = enst_sum + (omega ** 2).sum(dim=1).mean()

                rho_r = advect_3d(rho_r, v_s, dt_r, scheme=args.advection)
                rho_r = rho_r.clamp(min=0)
                if not args.peak_norm:
                    # (With --peak-norm there is no renormalisation: advection
                    # keeps max(rho) <= 1 and the body at ~1.)
                    rho_r = rho_r * initial_mass / (rho_r.sum(dim=(-3, -2, -1), keepdim=True) + 1e-12)

            terminal = F.mse_loss(rho_r * rho_scale, rho_1 * rho_scale)
            ke = ke_sum / n_rollout
            lam_ke = args.lambda_ke
            loss = args.lambda_terminal * terminal + lam_ke * ke

            extra = f" | term={terminal.item():.2e} | ke={ke.item():.2e} | lam_ke={lam_ke:.4f} | lam_T={args.lambda_terminal:.2f}"

            enst = enst_sum / n_rollout
            if args.lambda_mu_warmup > 0:
                lam_mu = args.lambda_mu * min(1.0, (step + 1) / args.lambda_mu_warmup)
            else:
                lam_mu = args.lambda_mu
            loss = loss + lam_mu * enst
            extra += f" | enst={enst.item():.2e} | lam_mu={lam_mu:.4f}"

        # Backward (+ gradient all-reduce across ranks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _allreduce_grads(model, is_distributed, world_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _allreduce_grads(model, is_distributed, world_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if is_main and (step % args.log_every == 0 or step == args.n_steps - 1):
            elapsed = time.time() - t_start
            cur_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            print(f"  step {step:5d}/{args.n_steps} | loss={loss.item():.6f}"
                  f"{extra} | lr={cur_lr:.1e} | {elapsed:.1f}s")

        # Checkpoint every 1000 steps (rank 0). "best" = lowest terminal loss
        # of the current training batch among these saves.
        if is_main and (step + 1) % 1000 == 0:
            os.makedirs(save_dir, exist_ok=True)
            sd_to_save = model.state_dict()
            ckpt_path = os.path.join(save_dir, f"{name}_model_step{step+1}.pt")
            torch.save(sd_to_save, ckpt_path)
            step_loss = terminal.item()
            if step_loss < best_loss:
                best_loss = step_loss
                best_path = os.path.join(save_dir, f"{name}_model_best.pt")
                torch.save(sd_to_save, best_path)
                print(f"  [ckpt] step {step+1}: saved best model (loss={step_loss:.6f})")
            else:
                print(f"  [ckpt] step {step+1}: saved checkpoint (best={best_loss:.6f})")

    models = {name: model}
    if is_main:
        print(f"  Training completed in {time.time() - t_start:.1f}s")

    # Final weights (rank 0)
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_model.pt"))
        print(f"\nFinal models saved to {save_dir}/")

    # Evaluate the final model (rank 0; other ranks wait at the barrier)
    if is_distributed:
        dist.barrier()
    if is_main:
        _evaluate(models, sample_pair, args, device)
        print("\nAll done!")

    _shutdown_distributed(is_distributed)


def _allreduce_grads(model, is_distributed, world_size):
    """Average gradients across ranks (no-op when not distributed)."""
    if is_distributed:
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(models, sample_pair, args, device):
    """Roll out ``--n-test`` pairs (seed 42), save them and print metrics."""
    n_test = args.n_test
    n_infer_steps = args.n_infer_steps
    save_dir = args.save_dir

    print(f"\n{'='*60}")
    print(f"Evaluating on test set (n_infer_steps={n_infer_steps}, advection={args.advection})")
    print(f"{'='*60}")

    torch.manual_seed(42)
    test_rho_0, test_rho_1 = sample_pair(n_test)

    results = {
        "test_rho_0": test_rho_0.cpu(),
        "test_rho_1": test_rho_1.cpu(),
    }

    for name, model in models.items():
        print(f"\n  Rolling out {name} ({n_infer_steps} steps)...")
        model.eval()
        t_inf_start = time.time()
        out = rollout_3d(model, test_rho_0, test_rho_1, n_steps=n_infer_steps,
                         scheme=args.advection, mass_renorm=True,
                         return_frames=True, return_velocities=True, store_device="cpu")
        inf_time_ms = (time.time() - t_inf_start) / n_test * 1000.0
        results[f"{name}_inference_time_ms"] = inf_time_ms
        print(f"    Inference time: {inf_time_ms:.1f} ms/sample")
        results[f"{name}_frames"] = out["frames"]
        results[f"{name}_velocities"] = out["velocities"]

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "rollout_results_3d.pt")
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")

    compute_metrics_3d(results, list(models))


def _div_zero_pad_3d(v):
    """Central-difference divergence with zero padding (metric table only)."""
    vD_pad = F.pad(v[:, 0:1], (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    vH_pad = F.pad(v[:, 1:2], (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    vW_pad = F.pad(v[:, 2:3], (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    dvD = (vD_pad[:, :, 2:, 1:-1, 1:-1] - vD_pad[:, :, :-2, 1:-1, 1:-1]) / 2.0
    dvH = (vH_pad[:, :, 1:-1, 2:, 1:-1] - vH_pad[:, :, 1:-1, :-2, 1:-1]) / 2.0
    dvW = (vW_pad[:, :, 1:-1, 1:-1, 2:] - vW_pad[:, :, 1:-1, 1:-1, :-2]) / 2.0
    return dvD + dvH + dvW


def _vorticity_mag_zero_pad_3d(v):
    """Central-difference |curl v| with zero padding (metric table only)."""
    v_pad = F.pad(v, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    dvD_dh = (v_pad[:, 0:1, 1:-1, 2:, 1:-1] - v_pad[:, 0:1, 1:-1, :-2, 1:-1]) / 2.0
    dvD_dw = (v_pad[:, 0:1, 1:-1, 1:-1, 2:] - v_pad[:, 0:1, 1:-1, 1:-1, :-2]) / 2.0
    dvH_dd = (v_pad[:, 1:2, 2:, 1:-1, 1:-1] - v_pad[:, 1:2, :-2, 1:-1, 1:-1]) / 2.0
    dvH_dw = (v_pad[:, 1:2, 1:-1, 1:-1, 2:] - v_pad[:, 1:2, 1:-1, 1:-1, :-2]) / 2.0
    dvW_dd = (v_pad[:, 2:3, 2:, 1:-1, 1:-1] - v_pad[:, 2:3, :-2, 1:-1, 1:-1]) / 2.0
    dvW_dh = (v_pad[:, 2:3, 1:-1, 2:, 1:-1] - v_pad[:, 2:3, 1:-1, :-2, 1:-1]) / 2.0
    omega_d = dvW_dh - dvH_dw
    omega_h = dvD_dw - dvW_dd
    omega_w = dvH_dd - dvD_dh
    return (omega_d ** 2 + omega_h ** 2 + omega_w ** 2).sqrt()


def compute_metrics_3d(results, names):
    """Print the end-of-training metric table and return it as a dict.

    "Terminal Error (L2)" is the batch mean of ||rho(1) - rho_1||_2 (sum over
    voxels, then sqrt); this is the number reported in the training logs.
    Divergence and enstrophy here use zero-padded differences (as in the
    original evaluation), unlike the periodic operators in :mod:`viot.ops_3d`.
    """
    print(f"\n{'='*70}")
    print("3D METRICS")
    print(f"{'='*70}")

    labels = {MODEL_NAME: "Spectral (BB+Visc)"}
    header = f"{'Metric':<25}"
    for name in names:
        header += f" | {labels.get(name, name):>14}"
    print(header)
    print("-" * len(header))

    rho_1 = results["test_rho_1"]
    metrics = {name: {} for name in names}

    for name in names:
        frames = results[f"{name}_frames"]
        velocities = results[f"{name}_velocities"]

        rho_final = frames[-1]
        terminal_err = ((rho_final - rho_1) ** 2).sum(dim=(-3, -2, -1)).sqrt().mean().item()
        metrics[name]["terminal_err"] = terminal_err

        initial_mass = frames[0].sum(dim=(-3, -2, -1))
        final_mass = frames[-1].sum(dim=(-3, -2, -1))
        metrics[name]["mass_pct"] = (final_mass / (initial_mass + 1e-12)).mean().item() * 100

        metrics[name]["mean_div"] = np.mean(
            [_div_zero_pad_3d(v).abs().mean().item() for v in velocities])
        metrics[name]["enstrophy"] = np.mean(
            [(_vorticity_mag_zero_pad_3d(v) ** 2).mean().item() for v in velocities])

        ke_values = []
        for t_idx, v in enumerate(velocities):
            rho_t = frames[t_idx]
            v_sq = (v ** 2).sum(dim=1, keepdim=True)
            ke_values.append((rho_t * v_sq).mean().item())
        metrics[name]["kinetic_energy"] = np.mean(ke_values)

        metrics[name]["inference_time_ms"] = results.get(f"{name}_inference_time_ms", float('nan'))

    rows = [
        ("Terminal Error (L2)", "terminal_err", "{:>14.6f}"),
        ("Mass Conservation (%)", "mass_pct", "{:>13.2f}%"),
        ("Mean |div(v)|", "mean_div", "{:>14.6f}"),
        ("Mean Enstrophy", "enstrophy", "{:>14.6f}"),
        ("Kinetic Energy", "kinetic_energy", "{:>14.6f}"),
        ("Inference Time (ms)", "inference_time_ms", "{:>13.1f}ms"),
    ]
    for label, key, fmt in rows:
        row = f"{label:<25}"
        for name in names:
            row += " | " + fmt.format(metrics[name][key])
        print(row)

    print(f"{'='*70}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the 3D VIOT operator (FNO3D, BB + viscosity)")
    # Training
    p.add_argument("--batch-size", type=int, default=8, help="Batch size per rank")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=64,
                   help="Grid resolution R (data must be R^3)")

    # Data
    p.add_argument("--sampler", type=str, default="shapenet", choices=["shapenet"],
                   help="Voxel-pool pair sampler (the only one in this release)")
    p.add_argument("--data-path-src", type=str, default="",
                   help="Source voxel pool .pt (rho_0)")
    p.add_argument("--data-path-tgt", type=str, default="",
                   help="Target voxel pool .pt (rho_1); may equal the source pool")
    p.add_argument("--raw-sampler", action="store_true",
                   help="Use pool values as-is (clamp only). Required: the legacy "
                        "participation-ratio resampling is not part of this release.")
    p.add_argument("--peak-norm", action="store_true",
                   help="Normalize each sample to max(rho)=1 instead of sum(rho)=1 "
                        "(used by all released models)")

    # Model
    p.add_argument("--k-max", type=float, default=0.25,
                   help="Spectral truncation radius of the vector potential (cycles/voxel)")
    p.add_argument("--fno-width", type=int, default=32, help="FNO channel width")
    p.add_argument("--fno-modes", type=int, default=12, help="FNO modes per dimension per layer")
    p.add_argument("--fno-layers", type=int, default=6, help="Number of FNO layers")
    p.add_argument("--init-scale", type=float, default=1.0,
                   help="Multiplier on the last projection layer at init (fresh runs only); "
                        ">1 gives a non-zero initial velocity")
    p.add_argument("--init-from", type=str, default=None,
                   help="Initialise from this state_dict (trunc_mask is skipped)")

    # Loss
    p.add_argument("--bb-visc-only", action="store_true",
                   help="Benamou-Brenier + viscosity objective (required)")
    p.add_argument("--lambda-ke", type=float, default=1.0,
                   help="Weight of the kinetic-energy term")
    p.add_argument("--lambda-mu", type=float, default=0.05,
                   help="Weight of the enstrophy (viscosity) term")
    p.add_argument("--lambda-mu-warmup", type=int, default=0,
                   help="If >0, ramp lambda_mu linearly from 0 over the first N steps")
    p.add_argument("--lambda-terminal", type=float, default=1.0,
                   help="Weight of the terminal MSE term")

    # Rollout / evaluation
    p.add_argument("--n-rollout", type=int, default=10,
                   help="Rollout steps per training sample")
    p.add_argument("--n-test", type=int, default=5,
                   help="Number of evaluation pairs")
    p.add_argument("--n-infer-steps", type=int, default=50,
                   help="Rollout steps for evaluation")
    p.add_argument("--advection", type=str, default="semi_lagrangian",
                   choices=list(ADVECTION_SCHEMES),
                   help="Advection scheme for training and evaluation rollouts "
                        "(default: semi_lagrangian, as the released models were trained)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; evaluate *_model_best.pt (else *_model.pt) in --save-dir")

    # Output / logging
    p.add_argument("--save-dir", type=str, default="results_spectral_3d/exp15")
    p.add_argument("--log-every", type=int, default=500, help="Log every N steps")

    # Optimisation
    p.add_argument("--no-scheduler", action="store_true",
                   help="Constant learning rate (default: cosine decay to lr/100)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient-norm clip value")

    # Performance
    p.add_argument("--amp", action="store_true", help="Automatic mixed precision (CUDA)")
    p.add_argument("--checkpoint", action="store_true",
                   help="Gradient checkpointing of each rollout step's forward pass")

    return p.parse_args(argv)


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
