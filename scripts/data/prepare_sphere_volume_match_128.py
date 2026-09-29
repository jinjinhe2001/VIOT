r"""Build a soft sphere whose total mass matches a target voxel pool (the `sphere2airplane_3d` source).

Pipeline:
  1. Build a sigmoid soft-ball sigmoid((radius - r) / softness), peak-normalised to 1, at the
     canvas centre.
  2. Binary-search the radius until the sum is within 0.5 of --target-sum.
  3. Stack --n-copies identical copies (the pair sampler draws source indices from this pool).

Output: float32 [n_copies, 1, canvas, canvas, canvas].

Paper file data/sphere_match_airplanes_s02_128.pt (8 copies, sum 7966.98, max 1; the target is the
mean mass 7967 of the airplane pool). These flags reproduce it bit for bit (radius 12.3803):
  python scripts/data/prepare_sphere_volume_match_128.py --softness 0.2 --target-sum 7967 \
      --n-copies 8 --output data/sphere_match_airplanes_s02_128.pt
Note the script defaults (--softness 1.5 --target-sum 8000) do NOT give the paper file.
The held-out split retrain uses a sphere matched to the PC15k train-set mean instead; it is written
by split_airplanes_wtt.py.
"""
import argparse
import os

import torch


def make_sphere(radius: float, canvas: int = 128, softness: float = 1.5) -> torch.Tensor:
    z = torch.arange(canvas).float() - (canvas - 1) / 2.0
    grid = torch.stack(torch.meshgrid(z, z, z, indexing="ij"), dim=0)
    r = grid.pow(2).sum(0).sqrt()
    sphere = torch.sigmoid((radius - r) / softness)
    sphere = sphere / sphere.max().clamp(min=1e-12)  # peak = 1
    return sphere


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-sum", type=float, default=8000.0)
    p.add_argument("--canvas", type=int, default=128)
    p.add_argument("--softness", type=float, default=1.5)
    p.add_argument("--n-copies", type=int, default=8)
    p.add_argument("--output", type=str,
                   default="data/sphere_volresize_n8000_128.pt")
    args = p.parse_args()

    # Binary search on radius until sum hits target.
    # 4/3 pi r^3 = 8000  -> r ~= 12.4, but soft-ball + peak-norm makes effective sum
    # smaller, so we search in a wider range.
    lo, hi = 1.0, 60.0
    target = args.target_sum
    best = None
    for it in range(60):
        mid = 0.5 * (lo + hi)
        s = make_sphere(mid, args.canvas, args.softness)
        cur = s.sum().item()
        if abs(cur - target) < 0.5:
            best = (mid, cur, s)
            break
        if cur < target:
            lo = mid
        else:
            hi = mid
        best = (mid, cur, s)
    radius, cur_sum, sphere = best
    print(f"Found radius={radius:.4f}, sum={cur_sum:.2f} (target={target})")
    print(f"  peak={sphere.max().item():.4f}, "
          f"voxels>0.05 = {(sphere > 0.05).sum().item()}")

    spheres = sphere.unsqueeze(0).unsqueeze(0).expand(
        args.n_copies, 1, args.canvas, args.canvas, args.canvas).contiguous().clone()
    print(f"Output: {tuple(spheres.shape)} {spheres.dtype}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(spheres, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
