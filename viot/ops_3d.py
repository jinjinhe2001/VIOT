"""3D advection schemes, rollouts and flow diagnostics.

Conventions
-----------
* Densities are ``[B, 1, D, H, W]``; velocities are ``[B, 3, D, H, W]`` with
  channels ordered (D, H, W), in voxels per unit time. A rollout of
  ``n_steps`` uses ``dt = 1 / n_steps``.
* The semi-Lagrangian schemes sample with ``grid_sample`` (trilinear,
  ``align_corners=False``, border padding). The finite-difference
  diagnostics are periodic (circular central differences, spacing 1 voxel).

Schemes (:func:`advect_3d`)
---------------------------
``"semi_lagrangian"``  First-order semi-Lagrangian step with a single backtrace.
                       **All released 3D models were trained and evaluated with
                       this scheme** (default).
``"maccormack"``       MacCormack predictor-corrector, each pass a semi-Lagrangian
                       step with a midpoint (RK2) backtrace. Used by some of the
                       paper's 3D figure and timing scripts, never for training.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "ADVECTION_SCHEMES",
    "advect_semi_lagrangian_3d",
    "advect_maccormack_3d",
    "advect_3d",
    "vorticity_3d",
    "divergence_3d",
    "rollout_3d",
]

ADVECTION_SCHEMES = ("semi_lagrangian", "maccormack")


# ---------------------------------------------------------------------------
# First-order semi-Lagrangian advection (as trained)
# ---------------------------------------------------------------------------

def advect_semi_lagrangian_3d(rho: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order semi-Lagrangian advection step (single backtrace).

    The sampling grid is always float32 and ``rho`` is sampled in float32, so
    the step is AMP-safe; the result is cast back to ``rho.dtype``.

    Args:
        rho: [B, 1, D, H, W] density
        v:   [B, 3, D, H, W] velocity (ch 0 = D, 1 = H, 2 = W)
        dt:  time step

    Returns:
        [B, 1, D, H, W] advected density
    """
    B, _, D, H, W = rho.shape

    # grid_sample 5D expects grid [B, D, H, W, 3] in (x=W, y=H, z=D) order
    zz = torch.linspace(-1 + 1 / D, 1 - 1 / D, D, device=rho.device, dtype=torch.float32)
    yy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=rho.device, dtype=torch.float32)
    xx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=rho.device, dtype=torch.float32)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing='ij')  # [D,H,W] each
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [D,H,W,3]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B,D,H,W,3]

    # Voxel units -> normalized coordinates
    scale_x = 2.0 / W
    scale_y = 2.0 / H
    scale_z = 2.0 / D

    # Backtrace: departure = x - v * dt
    departure = grid.clone()
    departure[..., 0] = grid[..., 0] - v[:, 2, :, :, :] * dt * scale_x  # x (W)
    departure[..., 1] = grid[..., 1] - v[:, 1, :, :, :] * dt * scale_y  # y (H)
    departure[..., 2] = grid[..., 2] - v[:, 0, :, :, :] * dt * scale_z  # z (D)

    rho_new = F.grid_sample(rho.float(), departure, align_corners=False,
                            padding_mode='border', mode='bilinear')
    return rho_new.to(rho.dtype)


# ---------------------------------------------------------------------------
# MacCormack advection (figure scripts only)
# ---------------------------------------------------------------------------

def _advect_midpoint_3d(rho: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """Single semi-Lagrangian step with a midpoint (RK2) backtrace."""
    B, _, D, H, W = rho.shape

    zz = torch.linspace(-1 + 1 / D, 1 - 1 / D, D, device=rho.device, dtype=rho.dtype)
    yy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=rho.device, dtype=rho.dtype)
    xx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=rho.device, dtype=rho.dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)

    scale_x = 2.0 / W
    scale_y = 2.0 / H
    scale_z = 2.0 / D

    mid = grid.clone()
    mid[..., 0] = grid[..., 0] - v[:, 2] * 0.5 * dt * scale_x
    mid[..., 1] = grid[..., 1] - v[:, 1] * 0.5 * dt * scale_y
    mid[..., 2] = grid[..., 2] - v[:, 0] * 0.5 * dt * scale_z

    v_mid = F.grid_sample(v, mid, align_corners=False,
                          padding_mode='border', mode='bilinear')

    departure = grid.clone()
    departure[..., 0] = grid[..., 0] - v_mid[:, 2] * dt * scale_x
    departure[..., 1] = grid[..., 1] - v_mid[:, 1] * dt * scale_y
    departure[..., 2] = grid[..., 2] - v_mid[:, 0] * dt * scale_z

    return F.grid_sample(rho, departure, align_corners=False,
                         padding_mode='border', mode='bilinear')


def advect_maccormack_3d(rho: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """MacCormack advection (midpoint backtrace in each pass).

        rho_hat  = SL(rho, v, +dt)
        rho_back = SL(rho_hat, v, -dt)
        rho_next = rho_hat + 0.5 * (rho - rho_back)

    Not used to train or evaluate the released 3D models (see the module
    docstring). Expects ``rho`` and ``v`` of the same floating dtype.
    """
    rho_hat = _advect_midpoint_3d(rho, v, dt)
    rho_back = _advect_midpoint_3d(rho_hat, v, -dt)
    return rho_hat + 0.5 * (rho - rho_back)


def advect_3d(rho: torch.Tensor, v: torch.Tensor, dt: float,
              scheme: str = "semi_lagrangian") -> torch.Tensor:
    """Advect ``rho`` by ``v`` for time ``dt`` with the named scheme.

    ``scheme`` is ``"semi_lagrangian"`` (default, as trained) or ``"maccormack"``.
    """
    if scheme == "semi_lagrangian":
        return advect_semi_lagrangian_3d(rho, v, dt)
    if scheme == "maccormack":
        return advect_maccormack_3d(rho, v, dt)
    raise ValueError(f"Unknown advection scheme {scheme!r}; expected one of {ADVECTION_SCHEMES}")


# ---------------------------------------------------------------------------
# Diagnostics (circular central differences, grid spacing = 1 voxel)
# ---------------------------------------------------------------------------

def vorticity_3d(v: torch.Tensor) -> torch.Tensor:
    """Vorticity omega = curl(v). v: [B,3,D,H,W] -> [B,3,D,H,W].

    This is the operator in the training objective (enstrophy =
    mean over voxels of |omega|^2).
    """
    v_pad = F.pad(v, (1, 1, 1, 1, 1, 1), mode='circular')  # [B,3,D+2,H+2,W+2]

    dvD_dd = (v_pad[:, 0:1, 2:, 1:-1, 1:-1] - v_pad[:, 0:1, :-2, 1:-1, 1:-1]) / 2.0
    dvD_dh = (v_pad[:, 0:1, 1:-1, 2:, 1:-1] - v_pad[:, 0:1, 1:-1, :-2, 1:-1]) / 2.0
    dvD_dw = (v_pad[:, 0:1, 1:-1, 1:-1, 2:] - v_pad[:, 0:1, 1:-1, 1:-1, :-2]) / 2.0

    dvH_dd = (v_pad[:, 1:2, 2:, 1:-1, 1:-1] - v_pad[:, 1:2, :-2, 1:-1, 1:-1]) / 2.0
    dvH_dh = (v_pad[:, 1:2, 1:-1, 2:, 1:-1] - v_pad[:, 1:2, 1:-1, :-2, 1:-1]) / 2.0
    dvH_dw = (v_pad[:, 1:2, 1:-1, 1:-1, 2:] - v_pad[:, 1:2, 1:-1, 1:-1, :-2]) / 2.0

    dvW_dd = (v_pad[:, 2:3, 2:, 1:-1, 1:-1] - v_pad[:, 2:3, :-2, 1:-1, 1:-1]) / 2.0
    dvW_dh = (v_pad[:, 2:3, 1:-1, 2:, 1:-1] - v_pad[:, 2:3, 1:-1, :-2, 1:-1]) / 2.0
    dvW_dw = (v_pad[:, 2:3, 1:-1, 1:-1, 2:] - v_pad[:, 2:3, 1:-1, 1:-1, :-2]) / 2.0

    omega_d = dvW_dh - dvH_dw
    omega_h = dvD_dw - dvW_dd
    omega_w = dvH_dd - dvD_dh

    return torch.cat([omega_d, omega_h, omega_w], dim=1)


def divergence_3d(v: torch.Tensor) -> torch.Tensor:
    """Divergence dv_D/dD + dv_H/dH + dv_W/dW. v: [B,3,D,H,W] -> [B,1,D,H,W]."""
    vD_pad = F.pad(v[:, 0:1], (1, 1, 1, 1, 1, 1), mode='circular')
    vH_pad = F.pad(v[:, 1:2], (1, 1, 1, 1, 1, 1), mode='circular')
    vW_pad = F.pad(v[:, 2:3], (1, 1, 1, 1, 1, 1), mode='circular')
    dvD_dD = (vD_pad[:, :, 2:, 1:-1, 1:-1] - vD_pad[:, :, :-2, 1:-1, 1:-1]) / 2.0
    dvH_dH = (vH_pad[:, :, 1:-1, 2:, 1:-1] - vH_pad[:, :, 1:-1, :-2, 1:-1]) / 2.0
    dvW_dW = (vW_pad[:, :, 1:-1, 1:-1, 2:] - vW_pad[:, :, 1:-1, 1:-1, :-2]) / 2.0
    return dvD_dD + dvH_dH + dvW_dW


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_3d(model, rho_0: torch.Tensor, rho_1: torch.Tensor, n_steps: int = 50,
               scheme: str = "semi_lagrangian", mass_renorm: bool = True,
               return_frames: bool = False, return_velocities: bool = False,
               store_device=None):
    """Transport ``rho_0`` toward ``rho_1`` with ``n_steps`` model-driven steps.

    Step ``s`` evaluates ``v = model.forward_velocity_only(rho, t, rho_1)`` at
    ``t = s / n_steps``, advects with ``dt = 1 / n_steps``, clamps negative
    values to zero and (if ``mass_renorm``) rescales each sample to its initial
    mass. With the defaults this is exactly the trainer's evaluation rollout
    (the one behind the end-of-training metrics). Runs under
    ``torch.no_grad()``; the model's train/eval mode is left unchanged (the
    released models have no dropout or batch norm, so it does not matter).

    Args:
        model: a :class:`viot.model_3d.FNO3D` (anything with ``forward_velocity_only``)
        rho_0, rho_1: [B, 1, D, H, W] source and target densities
        n_steps: number of steps (50 in the paper)
        scheme: advection scheme, see :func:`advect_3d`
        mass_renorm: rescale to the initial mass after every step. ``False``
            gives the clamp-only update the peak-normalised training rollout uses.
        return_frames: also return the ``n_steps + 1`` densities (including ``rho_0``)
        return_velocities: also return the ``n_steps`` velocities
        store_device: where to keep returned frames/velocities (default: the
            input device). ``"cpu"`` avoids holding a 128^3 trajectory on the GPU.

    Returns:
        The final density ``[B, 1, D, H, W]`` if both flags are False, otherwise
        a dict with key ``"final"`` and, as requested, ``"frames"`` and
        ``"velocities"`` (lists of tensors).
    """
    def _store(x):
        return x.clone() if store_device is None else x.to(store_device, copy=True)

    dt = 1.0 / n_steps
    rho = rho_0.clone()
    initial_mass = rho.sum(dim=(-3, -2, -1), keepdim=True)
    frames = [_store(rho)] if return_frames else None
    velocities = [] if return_velocities else None

    with torch.no_grad():
        for step in range(n_steps):
            t_val = step / n_steps
            t = torch.full((rho.shape[0],), t_val, device=rho.device, dtype=rho.dtype)
            v = model.forward_velocity_only(rho, t, rho_1)
            if velocities is not None:
                velocities.append(_store(v))
            rho = advect_3d(rho, v, dt, scheme=scheme)
            rho = rho.clamp(min=0)
            if mass_renorm:
                current_mass = rho.sum(dim=(-3, -2, -1), keepdim=True)
                rho = rho * initial_mass / (current_mass + 1e-12)
            if frames is not None:
                frames.append(_store(rho))

    if not (return_frames or return_velocities):
        return rho
    out = {"final": rho}
    if return_frames:
        out["frames"] = frames
    if return_velocities:
        out["velocities"] = velocities
    return out
