"""2D VIOT operator: an FNO that predicts a stream function.

The network maps the current density ``rho_t``, the target density
``rho_target`` and the time ``t`` to a velocity field that is divergence-free
by construction:

1. **Lift.** ``(rho_t, rho_target)`` are stacked as 2 channels and lifted to
   ``width`` feature channels with a 1x1 convolution.
2. **FNO trunk.** ``n_layers`` Fourier layers (:class:`FNOLayer2D`). Each
   applies a learned complex filter to the lowest ``n_modes`` Fourier modes
   (global receptive field), adds a pointwise 1x1 convolution, applies FiLM
   modulation from a sinusoidal time embedding, then GroupNorm + GELU and a
   residual connection.
3. **Project.** A 1x1 MLP maps the features to one scalar channel, the stream
   function ``psi``. The last layer is zero-initialised, so an untrained
   model predicts zero velocity.
4. **Spectral curl.** ``psi`` is transformed with ``rfft2`` and truncated to
   the disc ``|k| <= k_max`` (``k`` in cycles/pixel, so ``k_max = 0.25`` keeps
   modes up to a quarter of the sampling rate). The velocity is
   ``v_H = d psi / dW``, ``v_W = -d psi / dH`` computed in Fourier space
   (:func:`spectral_curl_2d`) and transformed back with ``irfft2``.

The output ``v`` has shape ``[B, 2, H, W]``: channel 0 is the velocity along
H (rows), channel 1 along W (columns), in pixels per unit time. Its discrete
spectral divergence is zero up to floating-point error, for any weights.

The class and parameter names match the original research code
(``FNOHybridSpectralFlowMatcher2D``), so released checkpoints load with
``strict=True``. A checkpoint is a bare ``state_dict`` that also contains the
``trunc_mask`` buffer; the hyper-parameters (``max_res``, ``k_max``,
``width``, ``n_modes``, ``n_layers``) are stored separately in the model's
``config.json``.
"""

import math

import torch
import torch.nn as nn

__all__ = ["spectral_curl_2d", "FNOLayer2D", "FNO2D", "FNOHybridSpectralFlowMatcher2D"]


# ---------------------------------------------------------------------------
# Spectral curl 2D
# ---------------------------------------------------------------------------

def spectral_curl_2d(psi_hat, kh, kw):
    """Spectral curl of a stream function.

    ``v_H = +i*2pi*kw*psi_hat``, ``v_W = -i*2pi*kh*psi_hat``.

    Args:
        psi_hat: ``[B, 1, H, W//2+1]`` complex ``rfft2`` of the stream function.
        kh, kw: wavenumbers in cycles/pixel, broadcastable to ``psi_hat``
            (``fftfreq(H)`` and ``rfftfreq(W)``).

    Returns:
        ``[B, 2, H, W//2+1]`` complex spectrum of ``(v_H, v_W)``.
    """
    c = 2.0 * math.pi * 1j
    return torch.cat([c * kw * psi_hat, -c * kh * psi_hat], dim=1)


# ---------------------------------------------------------------------------
# FNO layer
# ---------------------------------------------------------------------------

class FNOLayer2D(nn.Module):
    """Single Fourier Neural Operator layer with FiLM time conditioning.

    x -> [FFT -> R(k)·x_hat -> iFFT] + [W·x] -> FiLM(t) -> GroupNorm -> GELU -> + residual

    The spectral branch learns frequency-domain filters (global convolution)
    on the lowest ``n_modes`` modes along W and ``n_modes`` positive plus
    ``n_modes`` negative modes along H. The local branch is a 1x1 conv
    (pointwise mixing). FiLM conditioning modulates features based on time.
    """

    def __init__(self, channels, n_modes, film_dim=None):
        """
        Args:
            channels: number of feature channels
            n_modes: max frequency modes to keep per dimension (controls bandwidth)
            film_dim: dimension of FiLM conditioning (None = no conditioning)
        """
        super().__init__()
        self.channels = channels
        self.n_modes = n_modes

        # Spectral weights: stored as real [C_in, C_out, modes_h, modes_w]
        # (separate real + imag parts) to avoid complex tensor backward issues on CPU/MKL
        scale = 1.0 / (channels * channels)
        self.spectral_weight_real = nn.Parameter(
            scale * torch.randn(channels, channels, n_modes * 2, n_modes))
        self.spectral_weight_imag = nn.Parameter(
            scale * torch.randn(channels, channels, n_modes * 2, n_modes))

        # Local branch: 1x1 conv
        self.local_conv = nn.Conv2d(channels, channels, 1)

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
        """Apply learned spectral filter via rfft2."""
        B, C, H, W = x.shape
        m = self.n_modes

        x_hat = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, Wh]

        # Build complex weight from real/imag parts
        w = torch.complex(self.spectral_weight_real, self.spectral_weight_imag)

        # Multiply by spectral weights for low-frequency modes
        out_hat = torch.zeros_like(x_hat)

        # Positive kh, low kw
        out_hat[:, :, :m, :m] = torch.einsum(
            'bcij,cdij->bdij', x_hat[:, :, :m, :m], w[:, :, :m, :m])
        # Negative kh, low kw
        out_hat[:, :, -m:, :m] = torch.einsum(
            'bcij,cdij->bdij', x_hat[:, :, -m:, :m], w[:, :, m:, :m])

        return torch.fft.irfft2(out_hat, s=(H, W), norm='ortho')

    def forward(self, x, cond=None):
        """
        Args:
            x: [B, C, H, W]
            cond: [B, film_dim] time conditioning (optional)
        """
        residual = x

        # Spectral + local branches
        h = self._spectral_conv(x) + self.local_conv(x)

        # FiLM conditioning
        if self.film is not None and cond is not None:
            mod = self.film(cond)  # [B, 2*C]
            scale, shift = mod.chunk(2, dim=-1)
            h = h * (1 + scale.view(-1, self.channels, 1, 1)) + shift.view(-1, self.channels, 1, 1)

        h = self.act(self.norm(h))
        return h + residual


# ---------------------------------------------------------------------------
# FNO stream-function operator
# ---------------------------------------------------------------------------

class FNO2D(nn.Module):
    """FNO backbone -> pixel psi -> spectral truncation -> spectral curl -> div-free v.

    Architecture:
      1. Lift: conv2d (2 -> width) to embed (rho_t, rho_target) into feature channels
      2. N FNO layers with spectral convolutions + FiLM time conditioning
      3. Project: conv2d (width -> 1) to get scalar stream function psi
      4. FFT -> truncate at k_max -> spectral curl -> iFFT -> div-free v

    Call :meth:`forward_velocity_only` (``rho_t``, ``t``, ``rho_target``) to
    get ``v`` of shape ``[B, 2, H, W]``; :meth:`forward` additionally returns
    the truncated stream-function spectrum ``psi_hat``.

    Inputs at a resolution other than ``max_res`` are accepted (the FNO
    weights are defined per mode); the truncation mask is then rebuilt on the
    fly for that grid.
    """

    def __init__(self, max_res=64, k_max=0.25, width=64, n_modes=16,
                 n_layers=8, time_dim=128, verbose=False):
        """
        Args:
            max_res: spatial resolution (size of the stored truncation mask)
            k_max: spectral truncation radius (cycles/pixel) for the final div-free curl
            width: feature channel width
            n_modes: modes per dimension in FNO layers (controls layer bandwidth)
            n_layers: number of FNO layers
            time_dim: time embedding dimension
            verbose: print a one-line summary (modes kept, parameter count)
        """
        super().__init__()
        self.max_res = max_res
        self.k_max = k_max

        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Lift: 2 channels (rho_t, rho_target) -> width
        self.lift = nn.Conv2d(2, width, 1)

        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOLayer2D(width, n_modes, film_dim=time_dim)
            for _ in range(n_layers)
        ])

        # Project: width -> 1 (stream function)
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        # Zero-init final layer for stable rollout training
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

        # Spectral truncation mask
        H = W = max_res
        Wh = W // 2 + 1
        kh_grid = torch.fft.fftfreq(H).view(H, 1)
        kw_grid = torch.fft.rfftfreq(W).view(1, Wh)
        k_sq = kh_grid ** 2 + kw_grid ** 2
        trunc_mask = (k_sq <= k_max ** 2).float()
        self.register_buffer('trunc_mask', trunc_mask)

        if verbose:
            n_modes_active = (k_sq <= k_max ** 2).sum().item()
            n_params = sum(p.numel() for p in self.parameters())
            print(f"  FNOHybridSpectralFlowMatcher2D: width={width}, n_modes={n_modes}, "
                  f"layers={n_layers}, K={n_modes_active} output modes, params={n_params:,}")

    def _time_sinusoidal(self, t, d):
        half = d // 2
        freqs = torch.exp(torch.arange(0, half, device=t.device, dtype=t.dtype)
                          * -(math.log(10000.0) / half))
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, rho_t, t, rho_target, output_res=None):
        """
        Args:
            rho_t: [B, 1, H, W] current density
            t: [B] time in [0, 1)
            rho_target: [B, 1, H, W] target density
            output_res: unused (kept for API compatibility with the original code)

        Returns:
            (v, psi_hat): v [B, 2, H, W] divergence-free velocity,
            psi_hat [B, 1, H, W//2+1] truncated stream-function spectrum.
        """
        if output_res is None:
            output_res = rho_t.shape[2]
        B, _, H, W = rho_t.shape
        Wh = W // 2 + 1

        # Time embedding
        t_emb = self._time_sinusoidal(t, self.time_dim)
        cond = self.time_mlp(t_emb)  # [B, time_dim]

        # Lift
        x = torch.cat([rho_t, rho_target], dim=1)  # [B, 2, H, W]
        x = self.lift(x)  # [B, width, H, W]

        # FNO layers
        for layer in self.fno_layers:
            x = layer(x, cond)

        # Project to stream function
        psi = self.project(x)  # [B, 1, H, W]

        # Spectral truncation + curl
        psi_hat = torch.fft.rfft2(psi)
        if H == self.max_res:
            psi_hat = psi_hat * self.trunc_mask.unsqueeze(0).unsqueeze(0)
        else:
            kh = torch.fft.fftfreq(H, device=rho_t.device).view(H, 1)
            kw = torch.fft.rfftfreq(W, device=rho_t.device).view(1, Wh)
            mask = ((kh ** 2 + kw ** 2) <= self.k_max ** 2).float()
            psi_hat = psi_hat * mask.unsqueeze(0).unsqueeze(0)

        kh = torch.fft.fftfreq(H, device=rho_t.device).view(1, 1, H, 1)
        kw = torch.fft.rfftfreq(W, device=rho_t.device).view(1, 1, 1, Wh)
        v_hat = spectral_curl_2d(psi_hat, kh, kw)
        v = torch.fft.irfft2(v_hat, s=(H, W))

        return v, psi_hat

    def forward_velocity_only(self, rho_t, t, rho_target, output_res=None):
        """Return only the velocity ``v`` ([B, 2, H, W])."""
        v, _ = self.forward(rho_t, t, rho_target, output_res)
        return v


# Name used by the original research code and by old scripts.
FNOHybridSpectralFlowMatcher2D = FNO2D
