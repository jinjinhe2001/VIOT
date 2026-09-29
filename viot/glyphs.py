"""Glyph and sketch densities that match the training data.

2D: ``glyph_density_2d`` renders a character with the recipe of the glyph
pools used for ``cjk_2d`` (and the held-out Latin pools): centred at 80% of the
grid, Gaussian blur (sigma = 2 * H / 64), a 1e-5 floor, mass normalization, and
participation-ratio area normalization to 0.197 * H * W. By default the area
normalization keeps the glyph within 75% of the grid
(``viot.data_2d.fit_area_in_frame``), which leaves room for the transport, and
thin glyphs (``I``, ``l``, ``j``, ``一``) reach that area by being rendered
bolder (``viot.data_2d.thicken_to_area``) instead of being enlarged past the
border. ``max_extent=None`` restores the exact pool recipe.

3D: ``glyph_volume_3d`` / ``mask_to_volume`` extrude a 2D mask into a 128^3
volume exactly like the ``font_3d`` training set (``scripts/data/voxelize_font_3d.py``):
extrusion depth 0.3 * max(H, W), scale to 70% of the grid, blur sigma 0.7,
then an isotropic resize so that the total mass is 8000 (at most 90% of the
grid), clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def default_font() -> str | None:
    """Path of DejaVu Sans (shipped with matplotlib), if available."""
    try:
        from matplotlib import font_manager
        return font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------
def render_char_2d(ch: str, font_path: str, resolution: int, pad_frac: float = 0.1):
    """Render one character centred with a ``pad_frac`` margin; [H, W] in [0, 1] or None."""
    from PIL import Image, ImageDraw, ImageFont
    H = W = resolution
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    interior = int(H * (1.0 - 2 * pad_frac))
    size = int(interior * 1.0)
    for _ in range(6):
        try:
            test_font = ImageFont.truetype(font_path, size)
        except Exception:
            return None
        try:
            l, t, r, b = test_font.getbbox(ch)
            bw, bh = r - l, b - t
        except Exception:
            bw, bh = size, size
        if max(bw, bh) == 0:
            return None
        scale = interior / max(bw, bh)
        size = max(8, int(size * scale))
        if abs(scale - 1.0) < 0.01:
            break
    try:
        final_font = ImageFont.truetype(font_path, size)
        l, t, r, b = final_font.getbbox(ch)
    except Exception:
        return None
    cx = (W - (r - l)) // 2 - l
    cy = (H - (b - t)) // 2 - t
    draw.text((cx, cy), ch, fill=255, font=final_font)
    return np.array(img, dtype=np.float32) / 255.0


def glyph_density_2d(ch: str, font_path: str, resolution: int = 256,
                     blur_sigma: float = 2.0, pr_frac: float | None = None,
                     max_extent: float | None = 0.75) -> torch.Tensor | None:
    """Training-pool density of one character, [1, 1, H, W] with unit mass (or None).

    ``max_extent`` (default 0.75, ``viot.data_2d.FRAME_MAX_EXTENT``) keeps the
    glyph within that fraction of the grid, rendering thin glyphs bolder until
    they reach the target area (``fit_area_in_frame``, ``thicken_to_area``);
    ``None`` uses the unbounded area normalization of the pools, which can
    push thin glyphs past the border.
    """
    import torchvision.transforms.functional as TF
    from .data_2d import (MNIST_TARGET_PR_FRAC, fit_area_in_frame, normalize_area_on_grid,
                          thicken_to_area)
    H = W = resolution
    arr = render_char_2d(ch, font_path, H)
    if arr is None or arr.sum() < 10:
        return None
    sigma = blur_sigma * (H / 64.0)
    k_size = max(int(sigma * 6) | 1, 3)
    target_pr = (MNIST_TARGET_PR_FRAC if pr_frac is None else pr_frac) * H * W

    def render(x):
        x = TF.gaussian_blur(x, kernel_size=k_size, sigma=sigma)
        x = x.clamp(min=0) + 1e-5
        x = x / (x.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        if max_extent is None:
            x[0] = normalize_area_on_grid(x[0], target_pr, H, W)
        else:
            x[0] = fit_area_in_frame(x[0], target_pr, H, W, max_extent=max_extent)
        x = x.clamp(min=0)
        return x / (x.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    x = torch.from_numpy(arr)[None, None]
    return render(x) if max_extent is None else thicken_to_area(x, render, target_pr)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------
def _gaussian_blur_3d(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return grid
    g = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
    k_size = max(int(sigma * 6) | 1, 3)
    d_range = torch.arange(k_size, dtype=torch.float32) - k_size // 2
    k1d = torch.exp(-0.5 * d_range ** 2 / (sigma ** 2))
    k1d = k1d / k1d.sum()
    for dim in range(3):
        shape = [1] * 5
        shape[dim + 2] = k_size
        k = k1d.view(*shape)
        pad = [0] * 6
        pad[2 * (2 - dim)] = k_size // 2
        pad[2 * (2 - dim) + 1] = k_size // 2
        g = F.pad(g, pad, mode="constant", value=0)
        g = F.conv3d(g, k)
    return g.squeeze().numpy()


def render_glyph_mask(font_path: str, ch: str, render_size: int = 256):
    """Tightly cropped binary mask of a glyph (as in voxelize_font_3d.py), or None."""
    from PIL import Image, ImageDraw, ImageFont
    target_px = int(render_size * 0.85)
    lo, hi = 8, 1024
    while lo < hi - 1:
        mid = (lo + hi) // 2
        try:
            f = ImageFont.truetype(font_path, mid)
            bbox = f.getbbox(ch)
            if bbox is None:
                lo = mid
                continue
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            return None
        if max(w, h) < target_px:
            lo = mid
        else:
            hi = mid
    try:
        f = ImageFont.truetype(font_path, lo)
    except Exception:
        return None
    img = Image.new("L", (render_size * 2, render_size * 2), color=0)
    d = ImageDraw.Draw(img)
    try:
        d.text((render_size * 0.2, render_size * 0.2), ch, fill=255, font=f)
    except Exception:
        return None
    arr = np.array(img)
    nz = np.argwhere(arr > 32)
    if len(nz) == 0:
        return None
    y0, x0 = nz.min(axis=0)
    y1, x1 = nz.max(axis=0)
    return (arr[y0:y1 + 1, x0:x1 + 1] > 32).astype(np.float32)


def mask_to_volume(mask: np.ndarray, res: int = 128, n_target: int = 8000, sigma: float = 0.7,
                   margin: float = 0.9, init_fill: float = 0.7, depth_frac: float = 0.3):
    """Extrude a cropped 2D mask into a [res]^3 volume with total mass ~n_target.

    Returns ``(volume, clipped)``; ``clipped`` is True when the margin limited the
    resize so the mass stays below ``n_target``. Returns ``(None, False)`` for empty input.
    """
    Hm, Wm = mask.shape
    Dm = max(int(round(max(Hm, Wm) * depth_frac)), 4)
    vol3d = np.broadcast_to(mask[None, :, :], (Dm, Hm, Wm)).copy().astype(np.float32)

    max_ext = max(Dm, Hm, Wm)
    scale_to_res = init_fill * res / max_ext
    new_d = max(int(round(Dm * scale_to_res)), 4)
    new_h = max(int(round(Hm * scale_to_res)), 4)
    new_w = max(int(round(Wm * scale_to_res)), 4)
    vt = torch.from_numpy(vol3d).unsqueeze(0).unsqueeze(0)
    vol3d = F.interpolate(vt, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False).squeeze().numpy()

    if new_d > res or new_h > res or new_w > res:
        cd = max(0, (new_d - res) // 2)
        ch_ = max(0, (new_h - res) // 2)
        cw = max(0, (new_w - res) // 2)
        vol3d = vol3d[cd:cd + min(res, new_d), ch_:ch_ + min(res, new_h), cw:cw + min(res, new_w)]
        new_d, new_h, new_w = vol3d.shape
    grid = np.zeros((res, res, res), dtype=np.float32)
    od, oh, ow = (res - new_d) // 2, (res - new_h) // 2, (res - new_w) // 2
    grid[od:od + new_d, oh:oh + new_h, ow:ow + new_w] = vol3d

    grid = _gaussian_blur_3d(grid, sigma)

    s_now = float(grid.sum())
    if s_now <= 0:
        return None, False
    k = (n_target / s_now) ** (1.0 / 3.0)
    nz = np.argwhere(grid > 0.05)
    if len(nz) == 0:
        return None, False
    cur_ext = max(int((nz.max(axis=0) - nz.min(axis=0) + 1).max()), 1)
    max_k = (margin * res) / cur_ext
    clipped = False
    if k > max_k:
        k = max_k
        clipped = True
    new_size = max(int(round(res * k)), 8)

    gt = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
    gt = F.interpolate(gt, size=(new_size, new_size, new_size), mode="trilinear", align_corners=False).squeeze().numpy()
    out = np.zeros((res, res, res), dtype=np.float32)
    if new_size <= res:
        off = (res - new_size) // 2
        out[off:off + new_size, off:off + new_size, off:off + new_size] = gt
    else:
        off = (new_size - res) // 2
        out = gt[off:off + res, off:off + res, off:off + res].copy()
    return np.clip(out, 0.0, 1.0), clipped


def glyph_volume_3d(ch: str, font_path: str, res: int = 128, **kwargs) -> np.ndarray | None:
    """``font_3d``-style volume of one character ([res]^3, peak ~1, mass ~8000), or None."""
    mask = render_glyph_mask(font_path, ch)
    if mask is None or mask.sum() < 100:
        return None
    vol, _ = mask_to_volume(mask, res=res, **kwargs)
    return vol
