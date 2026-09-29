r"""Voxelize watertight ShapeNet airplanes for `sphere2airplane_3d` (data/shapenet_airplanes_wtt_volresize_n8000_128.pt).

Pipeline per mesh (``<mesh-root>/<model id>/models/model_decimated.obj``):
1. Center the mesh and scale it uniformly so its max bbox extent is init_fill (0.7) of the grid.
2. Voxelize at pitch 1/res with trimesh (``voxelized(1/res).fill()``) -> binary occupancy.
3. Small Gaussian blur (sigma 0.7) -> density in [0, 1] with a thin transition.
4. Trilinear-resize the volume by k = (n_target / current_sum)^(1/3):
   - if k <= max_k: resize, then pad to res^3 (sum ~ n_target);
   - if k >  max_k: cap k so the shape spans at most margin * res (sum < n_target), flagged "clipped".
Output: float32 [N, 1, res, res, res] with values in [0, 1], ``<output>_ids.txt`` (model id per row)
and ``<output>_clipped.txt``.

Meshes are processed by a process pool in completion order, so the row order of a fresh build is
not deterministic. Pass ``--match-ids`` with the shipped id list to put the rows in the paper's order.

Paper file (4045 airplanes, 0 failed, 79 clipped, sums min/mean/max 2217/7967/8221), from the
watertight, normalised, decimated ShapeNetCore airplanes archive model_wtt_normed_dcmed_outwardN.zip
(see scripts/data/README.md for provenance and licence):
  unzip model_wtt_normed_dcmed_outwardN.zip -d data/raw
  python scripts/data/voxelize_airplanes_wtt_volresize.py \
      --mesh-root data/raw/model_wtt_normed_dcmed_outwardN \
      --resolution 128 --sigma 0.7 --n-target 8000 --margin 0.9 --init-fill 0.7 --workers 24 \
      --output data/shapenet_airplanes_wtt_volresize_n8000_128.pt \
      --match-ids scripts/data/splits/shapenet_airplanes_wtt_volresize_n8000_128_ids.txt
"""

import argparse
import multiprocessing as mp
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def _gaussian_blur_3d(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return grid
    g = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
    k_size = int(sigma * 6) | 1
    k_size = max(k_size, 3)
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
        g = torch.nn.functional.pad(g, pad, mode="constant", value=0)
        g = torch.nn.functional.conv3d(g, k)
    return g.squeeze().numpy()


def list_models(root: str):
    out = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name, "models", "model_decimated.obj")
        if os.path.isfile(sub):
            out.append((name, sub))
    return out


def _voxelize_one(item, n_target: int, res: int, sigma: float, margin: float, init_fill: float):
    model_id, path = item
    try:
        m = trimesh.load(path, force="mesh")
        if len(m.vertices) == 0:
            return None
        # Step 1: center mesh, scale so max extent fills `init_fill` of the grid.
        bounds = m.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        m.apply_translation(-center)
        pre_ext = float((m.bounds[1] - m.bounds[0]).max())
        if pre_ext <= 0:
            return None
        m.apply_scale(init_fill / pre_ext)

        # Step 2: voxelize binary
        pitch = 1.0 / res
        vg = m.voxelized(pitch).fill()
        dense = vg.matrix.astype(np.float32)
        d, h, w = dense.shape
        # Crop if oversized (shouldn't happen with init_fill < 1)
        if d > res or h > res or w > res:
            cd = max(0, (d - res) // 2)
            ch = max(0, (h - res) // 2)
            cw = max(0, (w - res) // 2)
            dense = dense[cd:cd + min(res, d), ch:ch + min(res, h), cw:cw + min(res, w)]
            d, h, w = dense.shape
        grid = np.zeros((res, res, res), dtype=np.float32)
        od = (res - d) // 2
        oh = (res - h) // 2
        ow = (res - w) // 2
        grid[od:od + d, oh:oh + h, ow:ow + w] = dense

        # Step 3: small blur (preserves [0,1])
        grid = _gaussian_blur_3d(grid, sigma)

        # Step 4: spatial resize to hit target sum
        s_now = float(grid.sum())
        if s_now <= 0:
            return None
        k = (n_target / s_now) ** (1.0 / 3.0)

        # Find current bbox extent to determine max allowed k
        nz = np.argwhere(grid > 0.05)
        if len(nz) == 0:
            return None
        bbox = nz.max(axis=0) - nz.min(axis=0) + 1
        cur_ext = max(int(bbox.max()), 1)
        max_k = (margin * res) / cur_ext

        clipped = False
        if k > max_k:
            k = max_k
            clipped = True
        new_size = max(int(round(res * k)), 8)

        gt = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
        gt = F.interpolate(gt, size=(new_size, new_size, new_size),
                           mode="trilinear", align_corners=False)
        gt = gt.squeeze().numpy()

        out = np.zeros((res, res, res), dtype=np.float32)
        if new_size <= res:
            off = (res - new_size) // 2
            out[off:off+new_size, off:off+new_size, off:off+new_size] = gt
        else:
            off = (new_size - res) // 2
            out = gt[off:off+res, off:off+res, off:off+res].copy()

        out = np.clip(out, 0.0, 1.0)
        return (model_id, out, clipped, float(out.sum()), float(out.max()))
    except Exception:
        return None


def reorder_to_ids(ids, items, ref_path):
    """Reorder parallel lists ``items`` (list of lists) to the id order in ``ref_path``."""
    with open(ref_path) as f:
        ref = f.read().split()
    pos = {k: i for i, k in enumerate(ids)}
    missing = [k for k in ref if k not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} ids of {ref_path} were not generated, e.g. {missing[:5]}")
    extra = len(ids) - len(ref)
    if extra:
        print(f"  --match-ids: dropping {extra} generated shapes that are not in {ref_path}")
    order = [pos[k] for k in ref]
    return ref, [[lst[i] for i in order] for lst in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--n-target", type=int, default=8000)
    ap.add_argument("--margin", type=float, default=0.9,
                    help="Max bbox extent (fraction of res) after spatial resize.")
    ap.add_argument("--init-fill", type=float, default=0.7,
                    help="Per-shape pre-resize fill: max bbox extent set to this fraction of grid.")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--match-ids", default=None,
                    help="Reorder (and subset) the output rows to this id list, e.g. "
                         "scripts/data/splits/shapenet_airplanes_wtt_volresize_n8000_128_ids.txt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    items = list_models(args.mesh_root)
    if args.max_shapes > 0:
        items = items[:args.max_shapes]
    print(f"Found {len(items)} models | n_target={args.n_target} | res={args.resolution} "
          f"| sigma={args.sigma} | init_fill={args.init_fill} | margin={args.margin}")

    fn = partial(_voxelize_one, n_target=args.n_target, res=args.resolution,
                 sigma=args.sigma, margin=args.margin, init_fill=args.init_fill)
    grids, ids, sums, peaks, clipped = [], [], [], [], []
    n_fail = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, items, chunksize=8)):
            if r is None:
                n_fail += 1
                continue
            ids.append(r[0]); grids.append(r[1])
            clipped.append(r[2]); sums.append(r[3]); peaks.append(r[4])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(items)} | failed={n_fail} | clipped={sum(clipped)} | {time.time()-t0:.1f}s")
    print(f"Done: {len(grids)}/{len(items)} kept, {n_fail} failed in {time.time()-t0:.1f}s")

    if args.match_ids:
        ids, (grids, sums, peaks, clipped) = reorder_to_ids(
            ids, [grids, sums, peaks, clipped], args.match_ids)

    sums_a = np.array(sums); peaks_a = np.array(peaks); clipped_a = np.array(clipped)
    print(f"Sum  | min={sums_a.min():.1f} mean={sums_a.mean():.1f} max={sums_a.max():.1f}")
    print(f"Peak | min={peaks_a.min():.4f} mean={peaks_a.mean():.4f} max={peaks_a.max():.4f}")
    print(f"Clipped (sum < target due to grid bound): {int(clipped_a.sum())}/{len(clipped_a)} "
          f"({100*clipped_a.mean():.1f}%)")
    nc = sums_a[~clipped_a]
    if len(nc) > 0:
        print(f"Non-clipped sum: min={nc.min():.1f} max={nc.max():.1f} (target={args.n_target})")

    data = torch.from_numpy(np.stack(grids).astype(np.float32)).unsqueeze(1).contiguous()
    print(f"Output: {tuple(data.shape)} {data.dtype}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved {args.output}")
    with open(args.output.replace(".pt", "_ids.txt"), "w") as f:
        f.write("\n".join(ids))
    with open(args.output.replace(".pt", "_clipped.txt"), "w") as f:
        for mid, c in zip(ids, clipped):
            if c:
                f.write(mid + "\n")


if __name__ == "__main__":
    main()
