"""3D VIOT operator: an FNO that predicts a vector potential.

The network maps ``(rho_t, t, rho_target)`` to a 3-component vector potential
``A``. ``A`` is low-pass filtered in Fourier space (``|k| <= k_max`` in
cycles/voxel) and its curl is taken analytically, so the returned velocity is
divergence-free by construction (up to FFT round-off).

Pipeline (``FNO3D.forward``):
    1. lift: Conv3d(2 -> width, 1x1x1) on ``cat(rho_t, rho_target)``
    2. ``n_layers`` x :class:`FNOLayer3D` (spectral conv + pointwise conv,
       FiLM time conditioning, GroupNorm, GELU, residual)
    3. project: Conv3d(width, width) -> GELU -> Conv3d(width, 3) gives ``A``
       (the last conv is zero-initialised, so an untrained model outputs v = 0)
    4. ``rfftn(A)`` -> mask ``|k|^2 <= k_max^2`` -> :func:`spectral_curl_3d`
       -> ``irfftn`` -> ``v`` with shape ``[B, 3, D, H, W]``

Velocity channels are ordered (D, H, W) and measured in voxels per unit time.

The code is a verbatim copy of the research code that trained the released
3D checkpoints; parameter and buffer names are unchanged so the original
``state_dict`` files load with ``strict=True``.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "wavenumber_grid_3d",
    "spectral_curl_3d",
    "FNOLayer3D",
    "FNO3D",
    "FNOHybridSpectralFlowMatcher3D",
]


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def wavenumber_grid_3d(D, H, W, device, dtype=torch.float32):
    """Wavenumbers (cycles/voxel) of an ``rfftn`` over the last three dims.

    Returns:
        kd: [1, 1, D, 1, 1]  (``fftfreq(D)``)
        kh: [1, 1, 1, H, 1]  (``fftfreq(H)``)
        kw: [1, 1, 1, 1, Wh] (``rfftfreq(W)``, ``Wh = W // 2 + 1``)
        k_sq: [1, 1, D, H, Wh] = kd^2 + kh^2 + kw^2
    """
    Wh = W // 2 + 1
    kd = torch.fft.fftfreq(D, d=1.0, device=device, dtype=dtype).view(1, 1, D, 1, 1)
    kh = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    kw = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype).view(1, 1, 1, 1, Wh)
    k_sq = kd ** 2 + kh ** 2 + kw ** 2
    return kd, kh, kw, k_sq


def spectral_curl_3d(A_hat: torch.Tensor, kd, kh, kw) -> torch.Tensor:
    """Curl of a vector potential in Fourier space.

        v_d = 2*pi*i * (kh * A_w - kw * A_h)
        v_h = 2*pi*i * (kw * A_d - kd * A_w)
        v_w = 2*pi*i * (kd * A_h - kh * A_d)

    Args:
        A_hat: [B, 3, D, H, Wh] complex coefficients (channels A_d, A_h, A_w)
        kd, kh, kw: wavenumber grids in cycles/voxel (see :func:`wavenumber_grid_3d`)

    Returns:
        v_hat: [B, 3, D, H, Wh] complex coefficients of the velocity
    """
    Ad = A_hat[:, 0:1]
    Ah = A_hat[:, 1:2]
    Aw = A_hat[:, 2:3]

    # 2*pi*i factor: fftfreq returns cycles/sample, curl needs radians
    twopi_i = 2.0 * math.pi * 1j

    vd_hat = twopi_i * (kh * Aw - kw * Ah)
    vh_hat = twopi_i * (kw * Ad - kd * Aw)
    vw_hat = twopi_i * (kd * Ah - kh * Ad)

    return torch.cat([vd_hat, vh_hat, vw_hat], dim=1)


# ---------------------------------------------------------------------------
# FNO layer
# ---------------------------------------------------------------------------

class FNOLayer3D(nn.Module):
    """3D Fourier layer with FiLM time conditioning.

        x -> [iFFT(R(k) * FFT(x)) + Conv1x1x1(x)] -> FiLM(t) -> GroupNorm -> GELU -> + x

    The spectral weight keeps the first ``n_modes`` positive and last
    ``n_modes`` negative frequencies along D and H and the first ``n_modes``
    (non-negative) frequencies along W of the ``rfftn`` output, i.e. four
    (kd, kh) quadrants x low kw:
        (+kd, +kh): [:m, :m, :m]     (-kd, +kh): [-m:, :m, :m]
        (+kd, -kh): [:m, -m:, :m]    (-kd, -kh): [-m:, -m:, :m]
    Weight shape: [C_in, C_out, 2m, 2m, m], stored as separate real and
    imaginary parameters.
    """

    def __init__(self, channels, n_modes, film_dim=None):
        """
        Args:
            channels: number of feature channels
            n_modes: modes kept per dimension (layer bandwidth)
            film_dim: size of the FiLM conditioning vector (None = no conditioning)
        """
        super().__init__()
        self.channels = channels
        self.n_modes = n_modes

        # Real/imag stored separately to avoid complex-parameter backward issues.
        scale = 1.0 / (channels * channels)
        self.spectral_weight_real = nn.Parameter(
            scale * torch.randn(channels, channels, n_modes * 2, n_modes * 2, n_modes))
        self.spectral_weight_imag = nn.Parameter(
            scale * torch.randn(channels, channels, n_modes * 2, n_modes * 2, n_modes))

        # Local branch: 1x1x1 conv (pointwise mixing)
        self.local_conv = nn.Conv3d(channels, channels, 1)

        # FiLM conditioning
        if film_dim is not None:
            self.film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(film_dim, 2 * channels),
            )
        else:
            self.film = None

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()

    def _spectral_conv(self, x):
        """Learned spectral filter. x: [B, C, D, H, W] -> [B, C, D, H, W]."""
        B, C, D, H, W = x.shape
        m = self.n_modes

        # FFT must run in float32 (complex half is not supported)
        x_float = x.float()
        x_hat = torch.fft.rfftn(x_float, dim=(-3, -2, -1), norm='ortho')  # [B, C, D, H, Wh]

        w = torch.complex(self.spectral_weight_real, self.spectral_weight_imag)  # [C, C, 2m, 2m, m]

        out_hat = torch.zeros_like(x_hat)

        # Quadrant 1: +kd, +kh, +kw -> indices [:m, :m, :m]
        out_hat[:, :, :m, :m, :m] = torch.einsum(
            'bcdhi,codhi->bodhi', x_hat[:, :, :m, :m, :m], w[:, :, :m, :m, :])
        # Quadrant 2: -kd, +kh, +kw -> indices [-m:, :m, :m]
        out_hat[:, :, -m:, :m, :m] = torch.einsum(
            'bcdhi,codhi->bodhi', x_hat[:, :, -m:, :m, :m], w[:, :, m:, :m, :])
        # Quadrant 3: +kd, -kh, +kw -> indices [:m, -m:, :m]
        out_hat[:, :, :m, -m:, :m] = torch.einsum(
            'bcdhi,codhi->bodhi', x_hat[:, :, :m, -m:, :m], w[:, :, :m, m:, :])
        # Quadrant 4: -kd, -kh, +kw -> indices [-m:, -m:, :m]
        out_hat[:, :, -m:, -m:, :m] = torch.einsum(
            'bcdhi,codhi->bodhi', x_hat[:, :, -m:, -m:, :m], w[:, :, m:, m:, :])

        result = torch.fft.irfftn(out_hat, s=(D, H, W), dim=(-3, -2, -1), norm='ortho')
        return result.to(x.dtype)  # back to the input dtype (fp16 under AMP)

    def forward(self, x, cond=None):
        """
        Args:
            x: [B, C, D, H, W]
            cond: [B, film_dim] time conditioning (optional)

        Returns:
            [B, C, D, H, W] (with residual connection)
        """
        residual = x

        h = self._spectral_conv(x) + self.local_conv(x)

        if self.film is not None and cond is not None:
            mod = self.film(cond)  # [B, 2*C]
            scale, shift = mod.chunk(2, dim=-1)
            h = (h * (1 + scale.view(-1, self.channels, 1, 1, 1))
                 + shift.view(-1, self.channels, 1, 1, 1))

        h = self.act(self.norm(h))
        return h + residual


# ---------------------------------------------------------------------------
# Full operator
# ---------------------------------------------------------------------------

class FNO3D(nn.Module):
    """FNO backbone -> vector potential A -> spectral truncation -> curl -> div-free v.

    Released checkpoints use ``max_res=128, k_max=0.25, width=32, n_modes=16,
    n_layers=6, time_dim=128`` (201,450,019 parameters).

    ``trunc_mask`` is a buffer for ``max_res``; other input resolutions build
    the mask on the fly, so a model can be run at any cubic resolution.
    Forward passes may run under ``torch.autocast``: the FFTs are always done
    in float32 and the velocity is cast back to the autocast dtype.
    """

    def __init__(self, max_res=32, k_max=0.25, width=64, n_modes=8,
                 n_layers=8, time_dim=128, verbose=False):
        """
        Args:
            max_res: training resolution (D = H = W = max_res); sets ``trunc_mask``
            k_max: truncation radius of the output vector potential in
                cycles/voxel (0.5 = Nyquist)
            width: feature channels of the FNO layers
            n_modes: modes per dimension in each FNO layer
            n_layers: number of FNO layers
            time_dim: sinusoidal time-embedding size
            verbose: print a one-line summary (kept output modes, parameter count)
        """
        super().__init__()
        self.max_res = max_res
        self.k_max = k_max
        self.width = width
        self.n_modes = n_modes
        self.n_layers = n_layers

        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Lift: 2 channels (rho_t, rho_target) -> width
        self.lift = nn.Conv3d(2, width, 1)

        self.fno_layers = nn.ModuleList([
            FNOLayer3D(width, n_modes, film_dim=time_dim)
            for _ in range(n_layers)
        ])

        # Project: width -> 3 (vector potential A)
        self.project = nn.Sequential(
            nn.Conv3d(width, width, 1),
            nn.GELU(),
            nn.Conv3d(width, 3, 1),
        )
        # Zero-init final layer for stable rollout training
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

        # Spectral truncation mask for max_res
        D = H = W = max_res
        Wh = W // 2 + 1
        kd_grid = torch.fft.fftfreq(D).view(D, 1, 1)
        kh_grid = torch.fft.fftfreq(H).view(1, H, 1)
        kw_grid = torch.fft.rfftfreq(W).view(1, 1, Wh)
        k_sq = kd_grid ** 2 + kh_grid ** 2 + kw_grid ** 2
        trunc_mask = (k_sq <= k_max ** 2).float()  # [D, H, Wh]
        self.register_buffer('trunc_mask', trunc_mask)

        if verbose:
            n_modes_active = (k_sq <= k_max ** 2).sum().item()
            n_params = sum(p.numel() for p in self.parameters())
            print(f"  {type(self).__name__}: width={width}, n_modes={n_modes}, "
                  f"layers={n_layers}, K={n_modes_active} output modes, "
                  f"max_res={max_res}, params={n_params:,}")

    def _time_sinusoidal(self, t, d):
        """Sinusoidal embedding of t: [B] -> [B, d]."""
        half = d // 2
        freqs = torch.exp(torch.arange(0, half, device=t.device, dtype=t.dtype)
                          * -(math.log(10000.0) / half))
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, d]

    def forward(self, rho_t, t, rho_target, output_res=None):
        """Density pair + time -> divergence-free velocity.

        Args:
            rho_t: [B, 1, D, H, W] current density (D = H = W)
            t: [B] time in [0, 1]
            rho_target: [B, 1, D, H, W] target density
            output_res: unused (kept for API compatibility)

        Returns:
            v: [B, 3, D, H, W] divergence-free velocity (channels D, H, W)
            A_hat: [B, 3, D, H, Wh] truncated spectrum of the vector potential
        """
        if output_res is None:
            output_res = rho_t.shape[2]
        B = rho_t.shape[0]
        D = H = W = rho_t.shape[2]
        Wh = W // 2 + 1

        t_emb = self._time_sinusoidal(t, self.time_dim)  # [B, time_dim]
        cond = self.time_mlp(t_emb)  # [B, time_dim]

        x = torch.cat([rho_t, rho_target], dim=1)  # [B, 2, D, H, W]
        x = self.lift(x)  # [B, width, D, H, W]

        for layer in self.fno_layers:
            x = layer(x, cond)

        A = self.project(x)  # [B, 3, D, H, W]

        # Spectral truncation + curl (FFT in float32 for complex support)
        A_hat = torch.fft.rfftn(A.float(), dim=(-3, -2, -1))  # [B, 3, D, H, Wh]

        if D == self.max_res:
            A_hat = A_hat * self.trunc_mask.unsqueeze(0).unsqueeze(0)
        else:
            kd = torch.fft.fftfreq(D, device=rho_t.device).view(D, 1, 1)
            kh = torch.fft.fftfreq(H, device=rho_t.device).view(1, H, 1)
            kw = torch.fft.rfftfreq(W, device=rho_t.device).view(1, 1, Wh)
            mask = ((kd ** 2 + kh ** 2 + kw ** 2) <= self.k_max ** 2).float()
            A_hat = A_hat * mask.unsqueeze(0).unsqueeze(0)

        # Wavenumber grids must be float32 (float16 loses precision at high modes)
        kd_wn, kh_wn, kw_wn, _ = wavenumber_grid_3d(D, H, W, rho_t.device, torch.float32)
        v_hat = spectral_curl_3d(A_hat, kd_wn, kh_wn, kw_wn)

        v = torch.fft.irfftn(v_hat, s=(D, H, W), dim=(-3, -2, -1))
        v = v.to(A.dtype)

        return v, A_hat

    def forward_velocity_only(self, rho_t, t, rho_target, output_res=None):
        """Same as :meth:`forward` but returns only ``v`` [B, 3, D, H, W]."""
        v, _ = self.forward(rho_t, t, rho_target, output_res)
        return v


# Name used by the research code and the original checkpoints' training logs.
FNOHybridSpectralFlowMatcher3D = FNO3D
