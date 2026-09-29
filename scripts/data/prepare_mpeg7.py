r"""Prepare the MPEG-7 CE-Shape-1 silhouettes for VIOT 2D training (data/mpeg7_256_datamean_clean.pt).

Input: the 1400 binary GIF silhouettes of MPEG-7 CE-Shape-1 (70 classes x 20). Pipeline:
  1. skip the classes ``pencil`` and ``watch`` (40 files) -> 1360 shapes, in case-sensitive
     file-name order;
  2. flip vertically (row 0 = bottom, so ``imshow(origin="lower")`` shows the shape upright);
  3. tight crop -> centred square pad with a 10% margin per side -> resize to H x W (antialiased);
  4. Gaussian blur sigma = 0.008 * H (~2.05 at 256), +1e-5 floor, mass normalisation;
  5. three global passes of participation-ratio normalisation to the data-mean PR (computed over
     all 1360 shapes), each followed by ``_shrink_to_boundary`` (keeps <0.5% of the mass in the
     2-pixel border), then mass norm;
  6. drop samples whose PR is below --pr-min-frac x target, or, with --keep-list, keep exactly the
     listed files.
Output: a float32 tensor [N, 1, H, W] with unit mass per sample, plus ``<output>.meta.json``
(class and file name per sample, in tensor order).

Download and unzip the dataset (about 3.4 MB):
    mkdir -p data/raw/mpeg7 && cd data/raw/mpeg7
    wget https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip
    unzip MPEG7dataset.zip        # -> original/*.gif
Paper file (1294 samples, PR fraction ~0.244):
    python scripts/data/prepare_mpeg7.py --data-dir data/raw/mpeg7/original --resolution 256 \
        --keep-list scripts/data/splits/mpeg7_256_datamean_clean_files.txt \
        --output data/mpeg7_256_datamean_clean.pt
The paper file keeps 1294 of the 1360 shapes, in file-name order; the 66 removed ones are thin
shapes (16 Bone, 14 spring, 12 fork, 12 hammer, 8 spoon, 2 guitar, 2 sea_snake, 1 fish). Its
exact build command was not archived and the default PR filter removes none of them, so the
kept list was recovered by matching the paper tensor row by row against a rebuild; it is shipped
as mpeg7_256_datamean_clean_files.txt. With it the rows (and the split indices of
mpeg7_256_split_s0.json) correspond one to one to the paper file, and the values agree to float32
rounding (relative L2 difference ~1e-6 per sample, checked with torch 2.11 on CPU), not bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

try:
    import viot  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from viot.data_2d import normalize_area_on_grid as _normalize_area_on_grid  # noqa: E402

SKIP_FILES = {"confusions.gif", "shapedata.gif"}
SKIP_CLASSES = {"pencil", "watch"}

BORDER_K = 2          # pixels to consider as boundary
BORDER_THRESH = 0.005  # max allowed mass fraction in the border ring


def _shrink_to_boundary(img: torch.Tensor, H: int, W: int,
                        border_k: int = BORDER_K,
                        thresh: float = BORDER_THRESH,
                        n_iter: int = 8) -> torch.Tensor:
    """Iteratively shrink a [1, H, W] density until border mass is below thresh.

    Uses the same affine-zoom convention as _normalize_area_on_grid (s > 1
    shrinks content toward the centre).
    """
    border = torch.zeros(H, W, dtype=torch.bool, device=img.device)
    border[:border_k, :] = True
    border[-border_k:, :] = True
    border[:, :border_k] = True
    border[:, -border_k:] = True

    for _ in range(n_iter):
        d = img[0]
        total = d.sum()
        if total < 1e-10:
            return img
        bfrac = d[border].sum() / (total + 1e-12)
        if bfrac < thresh:
            break
        s = 1.0 + float(bfrac) * 2.0  # gentle shrink proportional to overflow
        theta = torch.tensor([[s, 0, 0], [0, s, 0]],
                             dtype=img.dtype, device=img.device).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)[0]
    return img


def load_silhouette(path: Path) -> np.ndarray:
    """Load a GIF silhouette as [H, W] float32 in [0, 1].

    MPEG-7 shapes are white (255) on black (0).  Flip vertically so
    that imshow(origin='lower') renders them right-side up.
    """
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.float32) / 255.0
    return np.flipud(arr).copy()


def tight_crop(arr: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Crop to the bounding box of non-background pixels."""
    mask = arr > threshold
    if not mask.any():
        return arr
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return arr[r0:r1 + 1, c0:c1 + 1]


def pad_square(arr: np.ndarray, pad_value: float = 0.0,
               margin_frac: float = 0.0) -> np.ndarray:
    """Pad [H, W] to a centered square with margin on each side.

    The output side length is max(h, w) / (1 - 2*margin_frac) so that the
    shape content occupies the central (1 - 2*margin_frac) fraction.
    """
    h, w = arr.shape
    content = max(h, w)
    if margin_frac > 0:
        total = int(np.ceil(content / (1.0 - 2.0 * margin_frac)))
    else:
        total = content
    out = np.full((total, total), pad_value, dtype=arr.dtype)
    y0 = (total - h) // 2
    x0 = (total - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def parse_class_name(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MPEG-7 CE-Shape-1 for VIOT 2D")
    parser.add_argument("--data-dir", type=str,
                        default="data/raw/mpeg7/original",
                        help="Directory with the MPEG-7 *.gif files")
    parser.add_argument("--resolution", type=int, default=64,
                        choices=[64, 128, 256])
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: data/mpeg7_<RES>.pt)")
    parser.add_argument("--blur-sigma-frac", type=float, default=0.008)
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Fractional margin on each side so shapes don't "
                             "touch the boundary (0.1 = 10%% padding per edge)")
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="Subset of classes to include (default: all)")
    parser.add_argument("--target-pr", type=float, default=None,
                        help="Fixed PR target (default: auto-compute from "
                             "data mean)")
    parser.add_argument("--pr-norm-iter", type=int, default=3,
                        help="Number of global PR-normalization passes "
                             "(default: 3)")
    parser.add_argument("--pr-min-frac", type=float, default=0.8,
                        help="Drop samples whose PR < this fraction of target "
                             "(0.8 = drop if PR < 80%% of target)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for PR normalization "
                             "(e.g. 'cuda', 'cuda:0', default: 'cpu')")
    parser.add_argument("--keep-list", type=str, default=None,
                        help="Keep exactly the GIF files listed in this file (one name per line), "
                             "in that order, instead of the --pr-min-frac filter. The paper file: "
                             "scripts/data/splits/mpeg7_256_datamean_clean_files.txt")
    parser.add_argument("--save-viz", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output = Path(args.output) if args.output else Path(
        f"data/mpeg7_{args.resolution}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    H = W = args.resolution
    blur_sigma = args.blur_sigma_frac * H
    k_size = int(blur_sigma * 6) | 1
    k_size = max(k_size, 3)

    # Case-sensitive file-name order ("Bone-1.gif" < "apple-1.gif"), i.e. the order of
    # sorted(PosixPath) on Linux, where the paper file was built. (Sorting WindowsPath
    # objects is case-insensitive and would give a different row order.)
    gif_files = sorted(
        (p for p in data_dir.glob("*.gif")
         if p.name not in SKIP_FILES),
        key=lambda p: p.name,
    )
    if args.classes:
        allowed = set(c.lower() for c in args.classes)
        gif_files = [p for p in gif_files
                     if parse_class_name(p.name).lower() in allowed]

    pr_mode = (f"{args.target_pr:.1f} (fixed)"
               if args.target_pr else "auto (data mean)")
    print("=" * 70)
    print(" MPEG-7 CE-Shape-1 -> VIOT 2D dataset")
    print("=" * 70)
    print(f"  data_dir       : {data_dir}")
    print(f"  n_files        : {len(gif_files)}")
    print(f"  resolution     : {H}x{W}")
    print(f"  margin         : {args.margin:.0%}")
    print(f"  blur sigma     : {blur_sigma:.2f}  kernel {k_size}")
    print(f"  target PR      : {pr_mode}")
    print(f"  PR norm iters  : {args.pr_norm_iter}")
    print(f"  output         : {output}")
    print("=" * 70)

    frames: list[np.ndarray] = []
    metadata: list[dict] = []
    class_counts: dict[str, int] = {}
    skipped = 0

    for p in gif_files:
        cls = parse_class_name(p.name)
        if cls.lower() in SKIP_CLASSES:
            skipped += 1
            continue
        sil = load_silhouette(p)
        if sil.sum() < 1.0:
            skipped += 1
            continue
        sil = tight_crop(sil)
        sil = pad_square(sil, margin_frac=args.margin)
        # Resize to target resolution now (sizes vary after pad_square)
        t = torch.from_numpy(sil).unsqueeze(0).unsqueeze(0).float()  # [1,1,S,S]
        t = TF.resize(t, [H, W], antialias=True)
        frames.append(t.squeeze(0))  # [1, H, W]
        metadata.append({"class": cls, "filename": p.name})
        class_counts[cls] = class_counts.get(cls, 0) + 1

    classes_sorted = sorted(class_counts.keys())
    print(f"\n  loaded shapes  : {len(frames)}  (skipped {skipped} blank)")
    print(f"  classes        : {len(classes_sorted)}")
    for c in classes_sorted:
        print(f"    {c:20s} : {class_counts[c]}")

    raw = torch.stack(frames)  # [N, 1, H, W]

    # --- blur + mass-normalize (once) ---
    print(f"\n[preprocess]  blur -> floor -> mass-norm", flush=True)
    data = raw.clone()
    BATCH = 256
    for start in range(0, data.shape[0], BATCH):
        data[start:start + BATCH] = TF.gaussian_blur(
            data[start:start + BATCH], kernel_size=k_size, sigma=blur_sigma)
    data = data.clamp(min=0) + 1e-5
    data = data / (data.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    # --- iterative PR normalization ---
    device = torch.device(args.device)
    if device.type != "cpu":
        print(f"\n[device]  moving data to {device}", flush=True)
        data = data.to(device)

    print(f"\n[PR-norm]  {args.pr_norm_iter} global iterations", flush=True)
    for g_iter in range(args.pr_norm_iter):
        mass_g = data.sum(dim=(-2, -1)).squeeze()
        pr_g = (mass_g ** 2) / (data.pow(2).sum(dim=(-2, -1)).squeeze()
                                + 1e-20)
        target_pr = (args.target_pr if args.target_pr is not None
                     else pr_g.mean().item())
        print(f"  [iter {g_iter}]  mean PR = {pr_g.mean():.1f}, "
              f"std = {pr_g.std():.1f}, target = {target_pr:.1f}",
              flush=True)

        for i in range(data.shape[0]):
            data[i] = _normalize_area_on_grid(data[i], target_pr, H, W)
            data[i] = _shrink_to_boundary(data[i], H, W)

        data = data.clamp(min=0)
        data = data / (data.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    if device.type != "cpu":
        data = data.cpu()

    mass = data.sum(dim=(-2, -1))
    pr = (mass ** 2) / (data.pow(2).sum(dim=(-2, -1)) + 1e-20)
    pr_flat = pr.reshape(-1)

    if args.keep_list:
        # Keep exactly the listed files, in list order (replaces the PR filter).
        with open(args.keep_list) as f:
            keep_names = f.read().split()
        row_of = {m["filename"]: i for i, m in enumerate(metadata)}
        absent = [n for n in keep_names if n not in row_of]
        if absent:
            raise SystemExit(f"{len(absent)} files of {args.keep_list} were not processed, e.g. {absent[:5]}")
        rows = [row_of[n] for n in keep_names]
        print(f"\n[keep-list]  keeping {len(rows)} of {len(metadata)} samples listed in {args.keep_list}")
        data = data[torch.tensor(rows)]
        metadata = [metadata[i] for i in rows]
        class_counts = {}
        for m in metadata:
            class_counts[m["class"]] = class_counts.get(m["class"], 0) + 1
        classes_sorted = sorted(class_counts.keys())
        mass = data.sum(dim=(-2, -1))
        pr = (mass ** 2) / (data.pow(2).sum(dim=(-2, -1)) + 1e-20)
        pr_flat = pr.reshape(-1)

    pr_cutoff = args.pr_min_frac * target_pr
    keep_mask = pr_flat >= pr_cutoff
    n_dropped = int((~keep_mask).sum())
    if args.keep_list:
        n_dropped = 0
    if n_dropped > 0:
        dropped_idx = (~keep_mask).nonzero(as_tuple=True)[0].tolist()
        print(f"\n[filter]  dropping {n_dropped} samples with PR < {pr_cutoff:.0f} "
              f"({args.pr_min_frac:.0%} of target):")
        for di in dropped_idx:
            m = metadata[di]
            print(f"    {m['class']:15s} ({m['filename']:25s})  PR={pr_flat[di]:.0f}")
        data = data[keep_mask]
        metadata = [metadata[i] for i in range(len(metadata)) if keep_mask[i]]
        class_counts = {}
        for m in metadata:
            class_counts[m["class"]] = class_counts.get(m["class"], 0) + 1
        classes_sorted = sorted(class_counts.keys())
        mass = data.sum(dim=(-2, -1))
        pr = (mass ** 2) / (data.pow(2).sum(dim=(-2, -1)) + 1e-20)

    print(f"\n  final dataset   : {tuple(data.shape)}")
    print(f"  mass mean/std   : {mass.mean():.6f} / {mass.std():.6f}")
    print(f"  PR   mean/std   : {pr.mean():.1f} / {pr.std():.1f}   "
          f"target={target_pr:.0f}")

    torch.save(data, output)
    size_mb = output.stat().st_size / 1e6
    print(f"\n  saved tensor    : {output}  ({size_mb:.1f} MB)")

    meta_path = output.with_suffix(output.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": "MPEG-7 CE-Shape-1",
            "resolution": args.resolution,
            "n_samples": data.shape[0],
            "n_classes": len(classes_sorted),
            "classes": classes_sorted,
            "class_counts": class_counts,
            "margin": args.margin,
            "target_pr": target_pr,
            "target_pr_source": ("fixed" if args.target_pr is not None
                                 else "data_mean"),
            "pr_norm_iter": args.pr_norm_iter,
            "blur_sigma": blur_sigma,
            "samples": metadata,
        }, f, indent=2)
    print(f"  saved metadata  : {meta_path}")

    if args.save_viz:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(36, data.shape[0])
        idxs = np.linspace(0, data.shape[0] - 1, n_show).astype(int)
        ncols = 6
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.5 * ncols, 2.5 * nrows))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, idx in zip(axes.ravel(), idxs):
            m = metadata[idx]
            ax.imshow(data[idx, 0].numpy(), cmap="inferno", origin="lower")
            ax.set_title(f"{m['class']}", fontsize=9)
        fig.suptitle(
            f"MPEG-7 -> VIOT  (N={data.shape[0]}, {H}x{W}, "
            f"PR={pr.mean():.0f}+/-{pr.std():.1f})", fontsize=12)
        viz = output.with_suffix(".viz.png")
        fig.tight_layout()
        fig.savefig(viz, dpi=110, bbox_inches="tight")
        print(f"  saved viz       : {viz}")


if __name__ == "__main__":
    main()
