"""Small visualization helpers shared by the examples (matplotlib + Pillow)."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def _cmap(name="inferno"):
    import matplotlib
    return matplotlib.colormaps[name]


def to_rgb(img: np.ndarray, vmax: float | None = None, cmap: str = "inferno") -> np.ndarray:
    """[H, W] non-negative array -> [H, W, 3] uint8."""
    vmax = float(img.max()) + 1e-12 if vmax is None else vmax
    return (_cmap(cmap)(np.clip(img / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)


def save_gif(frames, path, vmax=None, duration_ms=50, hold_every=None, hold_ms=600, scale=1):
    """Save a list of [H, W] arrays as an animated GIF with one global colour scale."""
    from PIL import Image
    vmax = max(float(f.max()) for f in frames) if vmax is None else vmax
    imgs, durations = [], []
    for i, f in enumerate(frames):
        im = Image.fromarray(to_rgb(f, vmax))
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.BILINEAR)
        imgs.append(im)
        hold = hold_every is not None and i % hold_every == 0
        durations.append(hold_ms if hold else duration_ms)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations, loop=0)


def save_strip(frames, path, n=6, vmax=None):
    """Save ``n`` evenly spaced frames side by side as a PNG."""
    from PIL import Image
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    vmax = max(float(f.max()) for f in frames) if vmax is None else vmax
    Image.fromarray(np.concatenate([to_rgb(frames[i], vmax) for i in idx], axis=1)).save(path)


def mip_view(vol: torch.Tensor, yaw_deg: float = 45.0, pitch_deg: float = 20.0) -> np.ndarray:
    """Maximum-intensity projection of a [D, H, W] volume seen from (yaw, pitch)."""
    D = vol.shape[-1]
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    Ry = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = torch.tensor([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    R = (Ry @ Rx).to(vol.device, vol.dtype)
    lin = torch.linspace(-1, 1, D, device=vol.device, dtype=vol.dtype)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1) @ R
    rot = F.grid_sample(vol[None, None], pts[None], mode="bilinear", padding_mode="zeros", align_corners=True)
    return rot[0, 0].amax(0).cpu().numpy()
