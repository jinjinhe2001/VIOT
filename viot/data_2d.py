"""2D training-pair samplers and density normalisation.

Every sampler returns ``(rho_0, rho_1)``, two ``[B, 1, H, W]`` float32
densities that each sum to 1 (sum normalisation, not peak normalisation).
Source and target indices are drawn independently and uniformly with
replacement from the torch global RNG, so ``torch.manual_seed`` makes a
sequence of batches reproducible.

``mnist``
    Live torchvision MNIST (all 60k training images by default). Per image:
    bilinear upsample 28 -> H, Gaussian blur with sigma ~ U[0.5, 2.0], clamp
    and add a 1e-4 floor, mass-normalise, rescale to a participation ratio of
    ``MNIST_TARGET_PR_FRAC * H * W`` (:func:`normalize_area_on_grid`),
    mass-normalise again.
``pool``
    A pre-rendered pool: a ``.pt`` file holding a float tensor
    ``[N, 1, H, W]`` (glyphs, shapes, ...), used for the CJK, Latin-font and
    MPEG-7 models. Pairs are drawn from the pool and re-normalised; there is
    no augmentation. The original trainer called this sampler ``chinese``;
    that name is accepted as an alias.

:func:`strokes_to_density` turns a hand-drawn stroke image into a density
that matches the ``mnist`` training distribution (used by the paint GUI).
:func:`fit_area_in_frame` is the area normalisation of the examples, GUIs and
web demo: it keeps the shape within ``FRAME_MAX_EXTENT`` (75%) of the canvas,
and :func:`thicken_to_area` gives thin shapes the training area by drawing
them bolder instead of enlarging them past the border.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "MNIST_TARGET_PR_FRAC",
    "FRAME_MAX_EXTENT",
    "SAMPLERS",
    "normalize_area_on_grid",
    "fit_area_in_frame",
    "thicken_to_area",
    "sample_mnist_pairs",
    "sample_pool_pairs",
    "get_sampler",
    "strokes_to_density",
]

# Target participation ratio as fraction of H*W.
# Computed from 2000 MNIST digits at 64x64 after blur+floor+mass_normalize.
MNIST_TARGET_PR_FRAC = 0.197

# Largest size (fraction of the canvas) of sketches and glyphs prepared for the
# examples, GUIs and web demo: the operators can carry parts of a shape up to
# about 10% of the canvas past the extent of its target, so a target filling
# 90% of the frame still produces rollout frames that run off the canvas.
FRAME_MAX_EXTENT = 0.75

SAMPLERS = ("mnist", "pool")
_SAMPLER_ALIASES = {"chinese": "pool"}

# Lazily loaded datasets, keyed by (root, train) and by path.
_mnist_datasets = {}
_pools = {}


def _default_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Area (participation-ratio) normalisation
# ---------------------------------------------------------------------------

def normalize_area_on_grid(img: torch.Tensor, target_pr: float,
                           H: int, W: int, n_iter: int = 6) -> torch.Tensor:
    """Rescale an image on HxW grid so its participation ratio matches target.

    Participation ratio PR = (sum rho)^2 / (sum rho^2) measures effective
    area and is invariant under mass scaling, so subsequent mass
    renormalization won't change it.

    Args:
        img: [1, H, W] digit image already on the grid
        target_pr: desired participation ratio (effective area in pixels)
        H, W: grid dimensions
        n_iter: max refinement iterations

    Returns:
        [1, H, W] rescaled image
    """
    for _ in range(n_iter):
        d = img[0]  # [H, W]
        total = d.sum()
        if total < 1e-10:
            return img

        pr = (total ** 2) / ((d ** 2).sum() + 1e-20)

        # Converged?
        if abs(pr / target_pr - 1.0) < 0.005:
            break

        # Affine zoom: s > 1 shrinks (reduces PR), s < 1 enlarges
        s = (pr / (target_pr + 1e-6)) ** 0.5
        theta = torch.tensor([[s, 0, 0], [0, s, 0]],
                             dtype=img.dtype, device=img.device).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)[0]

    return img  # [1, H, W]


def _occupied_extent(d: torch.Tensor, rel_threshold: float) -> float:
    """Largest side of the bbox of ``d > rel_threshold * max``, as a fraction of the grid."""
    occ = d > rel_threshold * d.max()
    rows = torch.nonzero(occ.any(dim=1))
    cols = torch.nonzero(occ.any(dim=0))
    if rows.numel() == 0:
        return 0.0
    return max((rows[-1] - rows[0] + 1).item() / d.shape[0],
               (cols[-1] - cols[0] + 1).item() / d.shape[1])


def _zoom(img: torch.Tensor, s: float, H: int, W: int) -> torch.Tensor:
    """Isotropic zoom about the centre as in :func:`normalize_area_on_grid` (s > 1 shrinks)."""
    theta = torch.tensor([[s, 0, 0], [0, s, 0]],
                         dtype=img.dtype, device=img.device).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
    return F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)[0]


def fit_area_in_frame(img: torch.Tensor, target_pr: float, H: int, W: int,
                      max_extent: float = FRAME_MAX_EXTENT, n_iter: int = 6,
                      rel_threshold: float = 0.05) -> torch.Tensor:
    """:func:`normalize_area_on_grid` that keeps the shape inside the canvas.

    The zoom that brings the participation ratio to ``target_pr`` is
    isotropic, so it enlarges thin shapes (``1``, ``I``, ``l``, a straight
    stroke) until they run past the border, where ``grid_sample`` drops them.
    Here every zoom step is limited so that the occupied part of the image
    (pixels above ``rel_threshold`` times the peak) spans at most
    ``max_extent`` of the grid; a shape that is already larger is shrunk to
    that size. The zoom mechanics are those of :func:`normalize_area_on_grid`,
    and the result is identical to it whenever the limit never applies. When
    it does, the participation ratio stays below ``target_pr``;
    :func:`thicken_to_area` makes up the difference (``strokes_to_density``
    and ``viot.glyphs.glyph_density_2d`` use both).

    Args:
        img: [1, H, W] density on the grid
        target_pr: desired participation ratio (effective area in pixels)
        H, W: grid dimensions
        max_extent: largest allowed side of the occupied bbox, fraction of the grid
        n_iter: max refinement iterations (as in :func:`normalize_area_on_grid`)
        rel_threshold: occupancy threshold relative to the peak

    Returns:
        [1, H, W] rescaled image
    """
    # Zooms that hit the limit aim one pixel inside it so that resampling
    # cannot push the bbox over; up to two extra passes shrink a result that
    # still exceeds ``max_extent`` after the ``n_iter`` refinement steps.
    fit = max_extent - 1.0 / max(H, W)
    for it in range(n_iter + 2):
        d = img[0]  # [H, W]
        total = d.sum()
        if total < 1e-10:
            return img

        pr = (total ** 2) / ((d ** 2).sum() + 1e-20)
        ext = _occupied_extent(d, rel_threshold)

        if it >= n_iter or abs(pr / target_pr - 1.0) < 0.005:
            # Area converged (or out of refinement steps): only enforce the frame.
            if ext <= max_extent:
                break
            s = ext / fit
        else:
            # Affine zoom: s > 1 shrinks (reduces PR), s < 1 enlarges
            s = (pr / (target_pr + 1e-6)) ** 0.5
            s_fit = ext / fit
            if s_fit > s:
                # Enlarging by less than a pixel would only resample the image.
                if ext <= max_extent and (1.0 / s_fit - 1.0) * ext * max(H, W) < 1.0:
                    break
                s = s_fit
        img = _zoom(img, s, H, W)

    return img  # [1, H, W]


def _pr(d: torch.Tensor) -> float:
    total = d.sum()
    return ((total ** 2) / ((d ** 2).sum() + 1e-20)).item()


def _ink_distance(img: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean distance (px) from every pixel of ``img`` [H, W] to the
    nearest pixel at or above half its peak (brute-force separable transform)."""
    img = img.detach().cpu()
    H, W = img.shape
    big = float(H * H + W * W)            # exceeds every squared distance
    f = torch.where(img >= 0.5 * img.max(), 0.0, big).to(torch.float64)
    y = torch.arange(H, dtype=torch.float64)
    x = torch.arange(W, dtype=torch.float64)
    dy2 = (y[:, None] - y[None, :]) ** 2   # [H, H]
    dx2 = (x[:, None] - x[None, :]) ** 2   # [W, W]
    g = torch.empty(H, W, dtype=torch.float64)
    d2 = torch.empty(H, W, dtype=torch.float64)
    for i in range(0, H, 32):              # squared distance along the columns
        g[i:i + 32] = (dy2[i:i + 32, :, None] + f[None]).amin(dim=1)
    for i in range(0, H, 32):              # then along the rows
        d2[i:i + 32] = (g[i:i + 32, None, :] + dx2[None]).amin(dim=2)
    return d2.sqrt()


def thicken_to_area(img: torch.Tensor, render, target_pr: float,
                    tol: float = 0.005) -> torch.Tensor:
    """Draw the ink of ``img`` bolder until ``render`` reaches ``target_pr``.

    ``img`` is a stroke or glyph image (``[..., H, W]``, ink bright, before
    any blur) and ``render`` the rest of a density pipeline, ending in
    :func:`fit_area_in_frame`. When that frame-capped zoom leaves the
    participation ratio of ``render(img)`` below ``target_pr``, the ink is
    thickened, as if drawn with a wider pen: every pixel within ``r`` of the
    ink (pixels at or above half the peak; exact Euclidean distance) is raised
    to the peak, with a one-pixel linear edge so that the result is continuous
    in ``r``. The smallest ``r`` for which ``render`` reaches ``target_pr``
    (bisection to 1/32 px) is used. Unlike blurring, thickening keeps the flat
    strokes and soft edges of the training densities; an incompressible
    transport preserves the distribution of density values, so sketches
    prepared this way can be matched more closely than blurred ones.

    Returns ``render(img)`` unchanged when it already reaches the target
    (within ``tol``), else ``render`` of the thickened image.
    """
    out = render(img)
    if _pr(out) >= (1.0 - tol) * target_pr:
        return out
    H, W = img.shape[-2:]
    peak = img.max()
    dist = _ink_distance(img.reshape(H, W))
    lo, hi, best, r = 0.0, None, None, 1.0
    for _ in range(64):
        ink = (r + 0.5 - dist).clamp(0.0, 1.0).to(img).reshape(img.shape) * peak
        y = render(torch.maximum(img, ink))
        if _pr(y) < (1.0 - tol) * target_pr:
            lo = r
        else:
            hi, best = r, y
        if hi is None:
            if r >= max(H, W):
                best = y
                break
            r *= 2.0
        elif hi - lo < 1.0 / 32:
            break
        else:
            r = 0.5 * (lo + hi)
    return best


# ---------------------------------------------------------------------------
# MNIST sampler
# ---------------------------------------------------------------------------

def _get_mnist(root, train):
    key = (os.path.abspath(root), bool(train))
    if key not in _mnist_datasets:
        import torchvision
        _mnist_datasets[key] = torchvision.datasets.MNIST(
            root=root, download=True, train=train)
    return _mnist_datasets[key]


def sample_mnist_pairs(B: int, H: int = 64, W: int = 64, device=None,
                       root: str = ".data", train: bool = True) -> tuple:
    """Generate density pairs from MNIST digits.

    Each sample: source = random digit, target = independent random digit.
    Pipeline: resize -> blur -> floor -> mass normalize -> normalize area ->
    mass normalize. Area normalization happens AFTER blur so all digits have
    a consistent spatial extent regardless of blur sigma.

    Args:
        B: batch size
        H, W: output resolution
        device: output device (default: CUDA if available, else CPU). The
            resize/blur/area steps run on this device.
        root: torchvision MNIST root (downloaded there if missing)
        train: use the 60k training split (True) or the 10k test split

    Returns:
        (rho_0, rho_1): each [B, 1, H, W], mass-normalized
    """
    import torchvision

    device = _default_device(device)
    dataset = _get_mnist(root, train)

    target_pr = MNIST_TARGET_PR_FRAC * H * W

    densities = []
    for _ in range(2):  # source and target
        indices = torch.randint(0, len(dataset), (B,))
        digits = []
        for idx in indices:
            img, _ = dataset[idx.item()]
            digit = torchvision.transforms.functional.to_tensor(img)  # [1, 28, 28]
            digits.append(digit)
        digits = torch.stack(digits).to(device)  # [B, 1, 28, 28]

        # Resize to HxW
        digits = F.interpolate(digits, size=(H, W), mode='bilinear', align_corners=False)

        # Random Gaussian blur for variety (sigma 0.5-2.0)
        for i in range(B):
            sigma = torch.rand(1).item() * 1.5 + 0.5
            k_size = int(sigma * 6) | 1  # ensure odd
            k_size = max(k_size, 3)
            digits[i:i+1] = torchvision.transforms.functional.gaussian_blur(
                digits[i:i+1], kernel_size=k_size, sigma=sigma)

        # Add small floor to avoid degenerate zero-mass regions
        digits = digits.clamp(min=0) + 1e-4

        # Normalize to probability distribution
        digits = digits / (digits.sum(dim=(-2, -1), keepdim=True) + 1e-12)

        # Normalize area LAST — after floor + mass normalization so the metric
        # is computed on the exact final distribution, then re-normalize mass
        for i in range(B):
            digits[i] = normalize_area_on_grid(digits[i], target_pr, H, W)
        digits = digits.clamp(min=0)
        digits = digits / (digits.sum(dim=(-2, -1), keepdim=True) + 1e-12)

        densities.append(digits)

    return densities[0], densities[1]


# ---------------------------------------------------------------------------
# Pre-rendered pool sampler
# ---------------------------------------------------------------------------

def _get_pool(path):
    key = os.path.abspath(path)
    if key not in _pools:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Density pool not found at {path}.")
        print(f"Loading density pool from {path}...")
        pool = torch.load(path, map_location='cpu', weights_only=False)
        print(f"  Loaded {pool.shape[0]} samples, shape={tuple(pool.shape)}")
        _pools[key] = pool
    return _pools[key]


def sample_pool_pairs(B: int, H: int = 256, W: int = 256, device=None,
                      data_path: str = None) -> tuple:
    """Draw density pairs from a pre-rendered ``[N, 1, H, W]`` pool (``.pt``).

    Source and target indices are independent uniform draws with replacement.
    The pool is loaded once per path and kept in memory (CPU).

    Args:
        B: batch size
        H, W: resolution (must match the pool; kept for a uniform sampler API)
        device: output device (default: CUDA if available, else CPU)
        data_path: path to the ``.pt`` file (a float tensor ``[N, 1, H, W]``)

    Returns:
        (rho_0, rho_1): each [B, 1, H, W], mass-normalized
    """
    if data_path is None:
        raise ValueError("sample_pool_pairs needs data_path (a .pt tensor [N, 1, H, W]).")
    device = _default_device(device)
    pool = _get_pool(data_path)

    N = pool.shape[0]
    # Source and target from independent random draws
    idx_src = torch.randint(0, N, (B,))
    idx_tgt = torch.randint(0, N, (B,))
    rho_0 = pool[idx_src].to(device)
    rho_1 = pool[idx_tgt].to(device)

    # Re-normalize mass in case of any float drift
    rho_0 = rho_0 / (rho_0.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    rho_1 = rho_1 / (rho_1.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    return rho_0, rho_1


def get_sampler(name: str, resolution: int, data_path: str = None,
                device=None, mnist_root: str = ".data"):
    """Return ``sample(B) -> (rho_0, rho_1)`` for the named sampler.

    Args:
        name: ``"mnist"`` or ``"pool"`` (``"chinese"`` is an alias of ``"pool"``)
        resolution: grid size ``H = W``
        data_path: pool ``.pt`` file (required for ``"pool"``, ignored for ``"mnist"``)
        device: output device (default: CUDA if available, else CPU)
        mnist_root: torchvision MNIST root for ``"mnist"``
    """
    name = _SAMPLER_ALIASES.get(name, name)
    R = resolution
    device = _default_device(device)
    if name == "mnist":
        return lambda B: sample_mnist_pairs(B, R, R, device, root=mnist_root)
    if name == "pool":
        if data_path is None:
            raise ValueError("The 'pool' sampler needs data_path (--data-path X.pt).")
        return lambda B: sample_pool_pairs(B, R, R, device, data_path=data_path)
    raise ValueError(f"Unknown sampler {name!r}; expected one of {SAMPLERS} "
                     f"(or aliases {tuple(_SAMPLER_ALIASES)})")


# ---------------------------------------------------------------------------
# Hand-drawn strokes -> training-distribution density
# ---------------------------------------------------------------------------

def strokes_to_density(stroke_img, resolution: int = 256, device=None,
                       fill_frac: float = 0.70, blur_sigma: float = 2.0,
                       max_extent: float = None, pr_frac: float = None):
    """Convert a hand-drawn stroke image into an MNIST-like density.

    Pipeline (matches the ``mnist`` sampler so size and position of the
    strokes do not matter): bbox crop -> pad to square -> resize to
    ``fill_frac`` of the canvas -> Gaussian blur (``blur_sigma``) -> floor ->
    mass-normalize -> participation-ratio area-normalize -> mass-normalize.
    With ``max_extent`` the area normalization is :func:`fit_area_in_frame`,
    and strokes too thin to reach the area within the frame are drawn bolder
    (:func:`thicken_to_area`, applied to the resized strokes before the blur).

    Args:
        stroke_img: ``resolution x resolution`` grayscale strokes, either a
            PIL image (mode ``"L"``) or a 2D array with values in [0, 255]
            (white strokes on black).
        resolution: expected canvas size
        device: device of the returned tensor (default CPU)
        fill_frac, blur_sigma: GUI defaults 0.70 and 2.0
        max_extent: ``None`` (default) area-normalizes with
            :func:`normalize_area_on_grid`, exactly like the paper's GUI; thin
            sketches can then be enlarged past the border. A float keeps the
            sketch within ``max_extent`` of the canvas and thickens thin
            strokes to the training area (the GUIs and the web demo use
            ``FRAME_MAX_EXTENT`` = 0.75).
        pr_frac: participation ratio as a fraction of ``H * W`` (default
            ``MNIST_TARGET_PR_FRAC``)

    Returns:
        ``[1, 1, H, H]`` float32 density summing to 1, or ``None`` if the
        canvas is essentially empty.
    """
    import torchvision.transforms.functional as TF

    arr = np.asarray(stroke_img, dtype=np.float32) / 255.0
    if arr.sum() < 1.0:
        return None
    H, W = arr.shape
    if not (H == W == resolution):
        raise ValueError(f"expected {resolution}x{resolution} canvas, got {arr.shape}")

    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    crop = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h,
       (side - w) // 2:(side - w) // 2 + w] = crop

    fill = max(int(round(H * fill_frac)), 8)
    sq_t = torch.from_numpy(sq).unsqueeze(0).unsqueeze(0)
    sq_t = F.interpolate(sq_t, size=(fill, fill),
                         mode="bilinear", align_corners=False)
    canvas = torch.zeros(1, 1, H, W)
    pad = (H - fill) // 2
    canvas[:, :, pad:pad + fill, pad:pad + fill] = sq_t

    k_size = max(int(blur_sigma * 6) | 1, 3)
    target_pr = (MNIST_TARGET_PR_FRAC if pr_frac is None else pr_frac) * H * W

    def render(canvas):
        canvas = TF.gaussian_blur(canvas, kernel_size=k_size,
                                  sigma=blur_sigma)

        canvas = canvas.clamp(min=0) + 1e-5
        canvas = canvas / (canvas.sum(dim=(-2, -1), keepdim=True) + 1e-12)

        if max_extent is None:
            canvas[0] = normalize_area_on_grid(canvas[0], target_pr, H, W)
        else:
            canvas[0] = fit_area_in_frame(canvas[0], target_pr, H, W, max_extent=max_extent)

        canvas = canvas.clamp(min=0)
        return canvas / (canvas.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    if max_extent is None:
        canvas = render(canvas)
    else:
        canvas = thicken_to_area(canvas, render, target_pr)
    if device is not None:
        canvas = canvas.to(device)
    return canvas
