"""3D training data: voxel pools, pair sampling and augmentation.

A voxel pool is a ``.pt`` file holding one float tensor ``[N, 1, R, R, R]``
(or ``[N, R, R, R]``) of non-negative occupancies/densities, e.g. the
128^3 ShapeNet, HuMMan and 3D-font pools built by ``scripts/data``. The whole
pool is loaded into host memory (once per DDP rank).

The released 3D models were trained with the trainer flags
``--raw-sampler --peak-norm``, which is what :func:`sample_voxel_pairs`
implements: the pool values are used as-is (negatives clamped), every sample
is scaled to ``max(rho) = 1``, shapes whose mass reaches the 2-voxel border
are shrunk toward the centre (:func:`ensure_within_domain`), and the peak is
re-normalised to 1. The legacy participation-ratio resampling used without
``--raw-sampler`` is not part of this release.

Random numbers come from the global torch CPU generator, in the same order as
the research code (source indices, target indices; then per sample in
:func:`augment_batch_3d`: three flips and one rotation draw).
"""

import torch
import torch.nn.functional as F

__all__ = [
    "load_voxels",
    "ensure_within_domain",
    "sample_voxel_pairs",
    "augment_batch_3d",
]


def load_voxels(data_path: str) -> torch.Tensor:
    """Load a voxel pool saved with ``torch.save``.

    Args:
        data_path: ``.pt`` file with an ``[N, 1, D, H, W]`` or ``[N, D, H, W]`` tensor

    Returns:
        [N, 1, D, H, W] tensor (on the CPU)
    """
    data = torch.load(data_path, weights_only=True)
    if data.dim() == 4:
        data = data.unsqueeze(1)  # [N, D, H, W] -> [N, 1, D, H, W]
    return data


def ensure_within_domain(rho, threshold=0.99, shrink_factor=0.95, max_iter=5):
    """Shrink samples whose mass touches the grid border, then mass-normalise.

    For each sample, while less than ``threshold`` of its mass lies inside
    the interior (a 2-voxel border excluded on every side), zoom it out by
    ``1 / shrink_factor`` about the centre (trilinear, zero padding), at most
    ``max_iter`` times. Finally every sample is clamped to >= 0 and divided by
    its sum. ``rho`` is modified in place (and also returned).

    Args:
        rho: [B, 1, D, H, W] density batch

    Returns:
        [B, 1, D, H, W] samples with sum 1
    """
    B, _, D, H, W = rho.shape

    border = 2
    for i in range(B):
        for _ in range(max_iter):
            total_mass = rho[i].sum()
            if total_mass < 1e-10:
                break
            interior_mass = rho[i, :, border:D-border, border:H-border, border:W-border].sum()
            frac = (interior_mass / total_mass).item()
            if frac >= threshold:
                break

            # Shrink via affine zoom (s > 1 shrinks toward the centre)
            s = 1.0 / shrink_factor
            theta = torch.tensor([
                [s, 0, 0, 0],
                [0, s, 0, 0],
                [0, 0, s, 0],
            ], dtype=rho.dtype, device=rho.device).unsqueeze(0)  # [1, 3, 4]
            grid = F.affine_grid(theta, (1, 1, D, H, W), align_corners=False)
            img = rho[i:i+1]  # [1, 1, D, H, W]
            img = F.grid_sample(img, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
            rho[i] = img[0]

    # Re-normalize mass (the zoom with zero padding can lose mass)
    rho = rho.clamp(min=0)
    rho = rho / (rho.sum(dim=(-3, -2, -1), keepdim=True) + 1e-12)
    return rho


def sample_voxel_pairs(shapes_src, shapes_tgt, batch_size, device, peak_norm=True):
    """Draw a batch of (source, target) density pairs.

    Source indices are drawn uniformly from ``shapes_src`` and then target
    indices from ``shapes_tgt`` (independently, with replacement). Pass the
    same pool twice for self-pairs (HuMMan, 3D fonts).

    Args:
        shapes_src, shapes_tgt: [N, 1, D, H, W] pools (e.g. from :func:`load_voxels`)
        batch_size: number of pairs B
        device: device of the returned tensors
        peak_norm: ``True`` (as trained, ``--peak-norm``) returns samples with
            ``max(rho) = 1``; ``False`` returns them with ``sum(rho) = 1``.

    Returns:
        (rho_0, rho_1), each [B, 1, D, H, W]
    """
    B = batch_size
    densities = []
    for shapes in [shapes_src, shapes_tgt]:
        N = shapes.shape[0]
        idx = torch.randint(0, N, (B,))
        rho = shapes[idx].to(device).clone()
        rho = rho.clamp(min=0)

        if peak_norm:
            peak = rho.amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-12)
            rho = rho / peak

        densities.append(rho)

    rho_0, rho_1 = densities
    rho_0 = ensure_within_domain(rho_0)
    rho_1 = ensure_within_domain(rho_1)

    if peak_norm:
        # ensure_within_domain mass-normalizes at the end; re-apply the peak norm
        p0 = rho_0.amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-12)
        p1 = rho_1.amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-12)
        rho_0 = rho_0 / p0
        rho_1 = rho_1 / p1

    return rho_0, rho_1


def augment_batch_3d(rho_0: torch.Tensor, rho_1: torch.Tensor):
    """Random flips and a random rotation, applied identically to each pair.

    Per sample: flip W, H, D (each with probability 1/2, in that order), then
    rotate by k * 90 degrees in the H-W plane, k ~ U{0, 1, 2, 3}. The inputs
    are modified in place (and also returned).

    Args:
        rho_0, rho_1: [B, 1, D, H, W]

    Returns:
        (rho_0, rho_1)
    """
    B = rho_0.shape[0]

    for i in range(B):
        # Random flip on W (dim -1)
        if torch.rand(1).item() > 0.5:
            rho_0[i] = rho_0[i].flip(-1)
            rho_1[i] = rho_1[i].flip(-1)

        # Random flip on H (dim -2)
        if torch.rand(1).item() > 0.5:
            rho_0[i] = rho_0[i].flip(-2)
            rho_1[i] = rho_1[i].flip(-2)

        # Random flip on D (dim -3)
        if torch.rand(1).item() > 0.5:
            rho_0[i] = rho_0[i].flip(-3)
            rho_1[i] = rho_1[i].flip(-3)

        # Random 90-degree rotation about the D axis (H-W plane)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            rho_0[i] = torch.rot90(rho_0[i], k, dims=(-2, -1))
            rho_1[i] = torch.rot90(rho_1[i], k, dims=(-2, -1))

    return rho_0, rho_1
