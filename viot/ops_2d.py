"""2D advection schemes, rollouts and flow diagnostics.

Conventions
-----------
* Densities are ``[B, 1, H, W]``; velocities are ``[B, 2, H, W]`` with
  channel 0 = velocity along H (rows) and channel 1 = velocity along W
  (columns), in pixels per unit time. A rollout of ``n_steps`` uses
  ``dt = 1 / n_steps``, so the density moves ``v * dt`` pixels per step.
* The semi-Lagrangian schemes sample with ``grid_sample`` (bilinear,
  ``align_corners=False``, border padding). The finite-difference
  diagnostics and the WENO scheme are periodic (circular).

Schemes (:func:`advect`)
------------------------
``"maccormack"``       MacCormack predictor-corrector, each pass a semi-Lagrangian
                       step with a midpoint (RK2) backtrace. Used to train and
                       evaluate the released 2D models ``mnist_2d``, ``cjk_2d``,
                       ``mpeg7_2d``, and by the paint-chain GUI.
``"semi_lagrangian"``  First-order semi-Lagrangian step with a single backtrace.
                       ``latin_font_2d`` was trained and evaluated with it.
``"weno"``             Conservative flux-form WENO5 with SSP-RK3 sub-stepping
                       (CFL <= 0.5, at most 16 sub-steps). Optional; used only
                       for some paper figures, never for training.
"""

import math

import torch
import torch.nn.functional as F

__all__ = [
    "ADVECTION_SCHEMES",
    "advect_maccormack",
    "advect_semi_lagrangian",
    "advect_weno",
    "advect",
    "vorticity_2d",
    "divergence_2d",
    "rollout_2d",
]

ADVECTION_SCHEMES = ("maccormack", "semi_lagrangian", "weno")


# ---------------------------------------------------------------------------
# Semi-Lagrangian advection
# ---------------------------------------------------------------------------

def advect_semi_lagrangian(rho: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order semi-Lagrangian advection step (single backtrace).

    Args:
        rho: [B, 1, H, W] density field (any number of channels works)
        v:   [B, 2, H, W] velocity field (ch0 = H-direction, ch1 = W-direction)
        dt:  time step

    Returns:
        [B, 1, H, W] advected density
    """
    B, _, H, W = rho.shape

    # Base grid of pixel centers in [-1, 1] (grid_sample convention):
    # grid is [B, H, W, 2] with (x=W, y=H).
    yy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=rho.device, dtype=rho.dtype)
    xx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=rho.device, dtype=rho.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')  # [H, W] each
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,W,2]

    # Convert velocity from pixel units to normalized coords
    scale_x = 2.0 / W
    scale_y = 2.0 / H

    # Backtrack: departure point = current position - v*dt (in normalized coords)
    # v[:,0] = velocity in H direction -> grid y; v[:,1] = velocity in W direction -> grid x
    departure = grid.clone()
    departure[..., 0] = grid[..., 0] - v[:, 1, :, :] * dt * scale_x  # grid x = W direction
    departure[..., 1] = grid[..., 1] - v[:, 0, :, :] * dt * scale_y  # grid y = H direction

    rho_new = F.grid_sample(rho, departure, align_corners=False, padding_mode='border', mode='bilinear')
    return rho_new


def _advect_midpoint_2d(rho: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """Single semi-Lagrangian step with midpoint (RK2) backtrace."""
    B, _, H, W = rho.shape
    yy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=rho.device, dtype=rho.dtype)
    xx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=rho.device, dtype=rho.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    scale_x = 2.0 / W
    scale_y = 2.0 / H

    mid = grid.clone()
    mid[..., 0] = grid[..., 0] - v[:, 1, :, :] * 0.5 * dt * scale_x
    mid[..., 1] = grid[..., 1] - v[:, 0, :, :] * 0.5 * dt * scale_y
    v_mid = F.grid_sample(v, mid, align_corners=False, padding_mode='border', mode='bilinear')

    departure = grid.clone()
    departure[..., 0] = grid[..., 0] - v_mid[:, 1, :, :] * dt * scale_x
    departure[..., 1] = grid[..., 1] - v_mid[:, 0, :, :] * dt * scale_y
    return F.grid_sample(rho, departure, align_corners=False, padding_mode='border', mode='bilinear')


def advect_maccormack(rho: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """MacCormack semi-Lagrangian advection in 2D (midpoint backtrace per pass).

    ``rho_hat = SL(rho, v, dt)``, ``rho_back = SL(rho_hat, v, -dt)``,
    ``rho_new = rho_hat + 0.5 * (rho - rho_back)``. No limiter is applied;
    :func:`rollout_2d` clamps negatives and renormalises mass after each step.
    """
    rho_hat = _advect_midpoint_2d(rho, v, dt)
    rho_back = _advect_midpoint_2d(rho_hat, v, -dt)
    return rho_hat + 0.5 * (rho - rho_back)


# ---------------------------------------------------------------------------
# WENO5 + SSP-RK3 (Eulerian, conservative, periodic)
# ---------------------------------------------------------------------------

def _weno5_left_at_face(q, dim, eps=1e-6):
    qm2 = torch.roll(q, shifts=2, dims=dim)
    qm1 = torch.roll(q, shifts=1, dims=dim)
    q0 = q
    qp1 = torch.roll(q, shifts=-1, dims=dim)
    qp2 = torch.roll(q, shifts=-2, dims=dim)
    p0 = (1.0 / 3.0) * qm2 - (7.0 / 6.0) * qm1 + (11.0 / 6.0) * q0
    p1 = -(1.0 / 6.0) * qm1 + (5.0 / 6.0) * q0 + (1.0 / 3.0) * qp1
    p2 = (1.0 / 3.0) * q0 + (5.0 / 6.0) * qp1 - (1.0 / 6.0) * qp2
    b0 = (13.0 / 12.0) * (qm2 - 2 * qm1 + q0) ** 2 \
        + 0.25 * (qm2 - 4 * qm1 + 3 * q0) ** 2
    b1 = (13.0 / 12.0) * (qm1 - 2 * q0 + qp1) ** 2 \
        + 0.25 * (qm1 - qp1) ** 2
    b2 = (13.0 / 12.0) * (q0 - 2 * qp1 + qp2) ** 2 \
        + 0.25 * (3 * q0 - 4 * qp1 + qp2) ** 2
    a0 = 0.1 / (eps + b0) ** 2
    a1 = 0.6 / (eps + b1) ** 2
    a2 = 0.3 / (eps + b2) ** 2
    s = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / s


def _weno5_right_at_face(q, dim, eps=1e-6):
    qm1 = torch.roll(q, shifts=1, dims=dim)
    q0 = q
    qp1 = torch.roll(q, shifts=-1, dims=dim)
    qp2 = torch.roll(q, shifts=-2, dims=dim)
    qp3 = torch.roll(q, shifts=-3, dims=dim)
    p0 = (1.0 / 3.0) * qp3 - (7.0 / 6.0) * qp2 + (11.0 / 6.0) * qp1
    p1 = -(1.0 / 6.0) * qp2 + (5.0 / 6.0) * qp1 + (1.0 / 3.0) * q0
    p2 = (1.0 / 3.0) * qp1 + (5.0 / 6.0) * q0 - (1.0 / 6.0) * qm1
    b0 = (13.0 / 12.0) * (qp3 - 2 * qp2 + qp1) ** 2 \
        + 0.25 * (qp3 - 4 * qp2 + 3 * qp1) ** 2
    b1 = (13.0 / 12.0) * (qp2 - 2 * qp1 + q0) ** 2 \
        + 0.25 * (qp2 - q0) ** 2
    b2 = (13.0 / 12.0) * (qp1 - 2 * q0 + qm1) ** 2 \
        + 0.25 * (3 * qp1 - 4 * q0 + qm1) ** 2
    a0 = 0.1 / (eps + b0) ** 2
    a1 = 0.6 / (eps + b1) ** 2
    a2 = 0.3 / (eps + b2) ** 2
    s = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / s


def _weno_flux_div(rho, v):
    vh = v[:, 0:1]
    vw = v[:, 1:2]
    rL_h = _weno5_left_at_face(rho, dim=-2)
    rR_h = _weno5_right_at_face(rho, dim=-2)
    vh_face = 0.5 * (vh + torch.roll(vh, shifts=-1, dims=-2))
    upwind_h = torch.where(vh_face >= 0, rL_h, rR_h)
    flux_h = vh_face * upwind_h
    div_h = flux_h - torch.roll(flux_h, shifts=1, dims=-2)
    rL_w = _weno5_left_at_face(rho, dim=-1)
    rR_w = _weno5_right_at_face(rho, dim=-1)
    vw_face = 0.5 * (vw + torch.roll(vw, shifts=-1, dims=-1))
    upwind_w = torch.where(vw_face >= 0, rL_w, rR_w)
    flux_w = vw_face * upwind_w
    div_w = flux_w - torch.roll(flux_w, shifts=1, dims=-1)
    return -(div_h + div_w)


def advect_weno(rho: torch.Tensor, v: torch.Tensor, dt: float,
                max_cfl: float = 0.5, max_substeps: int = 16,
                clamp_nonneg: bool = True) -> torch.Tensor:
    """Flux-form WENO5 advection with SSP-RK3 sub-steps (periodic boundaries).

    The step is split into ``ceil(max|v| * |dt| / max_cfl)`` sub-steps, capped
    at ``max_substeps``. With ``clamp_nonneg`` every RK stage is clamped to
    ``>= 0``. Mass is conserved up to the clamping.
    """
    v_max = float(v.abs().max())
    n_sub = max(1, int(math.ceil(v_max * abs(dt) / max_cfl)))
    n_sub = min(n_sub, max_substeps)
    sub_dt = dt / n_sub
    out = rho
    for _ in range(n_sub):
        k1 = _weno_flux_div(out, v)
        u1 = out + sub_dt * k1
        if clamp_nonneg:
            u1 = u1.clamp(min=0.0)
        k2 = _weno_flux_div(u1, v)
        u2 = 0.75 * out + 0.25 * (u1 + sub_dt * k2)
        if clamp_nonneg:
            u2 = u2.clamp(min=0.0)
        k3 = _weno_flux_div(u2, v)
        out = (1.0 / 3.0) * out + (2.0 / 3.0) * (u2 + sub_dt * k3)
        if clamp_nonneg:
            out = out.clamp(min=0.0)
    return out


def advect(rho: torch.Tensor, v: torch.Tensor, dt: float,
           scheme: str = "maccormack") -> torch.Tensor:
    """Advect ``rho`` by ``v`` for time ``dt`` with the named scheme.

    ``scheme`` is one of ``"maccormack"`` (default), ``"semi_lagrangian"``
    or ``"weno"`` (see the module docstring).
    """
    if scheme == "maccormack":
        return advect_maccormack(rho, v, dt)
    if scheme == "semi_lagrangian":
        return advect_semi_lagrangian(rho, v, dt)
    if scheme == "weno":
        return advect_weno(rho, v, dt)
    raise ValueError(f"Unknown advection scheme {scheme!r}; expected one of {ADVECTION_SCHEMES}")


# ---------------------------------------------------------------------------
# Diagnostics (circular central differences, grid spacing = 1 pixel)
# ---------------------------------------------------------------------------

def vorticity_2d(v: torch.Tensor) -> torch.Tensor:
    """Scalar vorticity: omega = dv_W/dH - dv_H/dW. v: [B,2,H,W] -> [B,1,H,W].

    This is the operator in the training objective (enstrophy = mean(omega^2)).
    """
    v_pad = F.pad(v, (1, 1, 1, 1), mode='circular')
    dvH_dW = (v_pad[:, 0:1, 1:-1, 2:] - v_pad[:, 0:1, 1:-1, :-2]) / 2.0
    dvW_dH = (v_pad[:, 1:2, 2:, 1:-1] - v_pad[:, 1:2, :-2, 1:-1]) / 2.0
    return dvW_dH - dvH_dW  # [B, 1, H, W]


def divergence_2d(v: torch.Tensor) -> torch.Tensor:
    """Divergence dv_H/dH + dv_W/dW. v: [B,2,H,W] -> [B,1,H,W].

    Circular central differences, as used by the held-out evaluation.
    """
    vH_pad = F.pad(v[:, 0:1], (1, 1, 1, 1), mode='circular')
    vW_pad = F.pad(v[:, 1:2], (1, 1, 1, 1), mode='circular')
    dvH_dH = (vH_pad[:, :, 2:, 1:-1] - vH_pad[:, :, :-2, 1:-1]) / 2.0
    dvW_dW = (vW_pad[:, :, 1:-1, 2:] - vW_pad[:, :, 1:-1, :-2]) / 2.0
    return dvH_dH + dvW_dW


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_2d(model, rho_0: torch.Tensor, rho_1: torch.Tensor, n_steps: int = 50,
               scheme: str = "maccormack", return_frames: bool = False,
               return_velocities: bool = False):
    """Transport ``rho_0`` toward ``rho_1`` with ``n_steps`` model-driven steps.

    Step ``s`` evaluates ``v = model.forward_velocity_only(rho, t, rho_1)`` at
    ``t = s / n_steps``, advects with ``dt = 1 / n_steps``, clamps negative
    values to zero and rescales each sample to its initial mass. Runs under
    ``torch.no_grad()``; the model's train/eval mode is left unchanged (the
    released models have no dropout or batch norm, so it does not matter).

    Args:
        model: a :class:`viot.model_2d.FNO2D` (anything with ``forward_velocity_only``)
        rho_0, rho_1: [B, 1, H, W] source and target densities
        n_steps: number of steps (50 in the paper)
        scheme: advection scheme, see :func:`advect`
        return_frames: also return the ``n_steps + 1`` densities (including ``rho_0``)
        return_velocities: also return the ``n_steps`` velocities

    Returns:
        The final density ``[B, 1, H, W]`` if both flags are False, otherwise a
        dict with key ``"final"`` and, as requested, ``"frames"`` (list of
        tensors) and ``"velocities"`` (list of tensors). Tensors stay on the
        input device.
    """
    dt = 1.0 / n_steps
    rho = rho_0.clone()
    initial_mass = rho.sum(dim=(-2, -1), keepdim=True)
    frames = [rho.clone()] if return_frames else None
    velocities = [] if return_velocities else None

    with torch.no_grad():
        for step in range(n_steps):
            t_val = step / n_steps
            t = torch.full((rho.shape[0],), t_val, device=rho.device, dtype=rho.dtype)
            v = model.forward_velocity_only(rho, t, rho_1)
            if velocities is not None:
                velocities.append(v.clone())
            rho = advect(rho, v, dt, scheme=scheme)
            rho = rho.clamp(min=0)
            rho = rho * initial_mass / (rho.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            if frames is not None:
                frames.append(rho.clone())

    if not (return_frames or return_velocities):
        return rho
    out = {"final": rho}
    if return_frames:
        out["frames"] = frames
    if return_velocities:
        out["velocities"] = velocities
    return out
