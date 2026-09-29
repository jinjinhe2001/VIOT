r"""Voxelize HuMMan SMPL bodies for `humman_3d` (data/humman_volresize_n14000_128.pt).

Mirrors ``voxelize_airplanes_wtt_volresize.py`` but reads SMPL body meshes from a HuMMan
``.npz`` (keys verts [N, 6890, 3], faces [13776, 3], global_orient [N, 3], transl [N, 3])
instead of OBJ files.

Pipeline per frame:
1. Canonicalize: undo global_orient + transl so all bodies share a canonical frame.
2. Center + uniformly scale so max bbox extent fills `init_fill` of the grid.
3. Voxelize at pitch=1/res (binary occupancy after fill).
4. Small Gaussian blur (sigma small, e.g. 0.7) -> density in [0,1].
5. Trilinear-resize the volume by k = (n_target / current_sum)^(1/3).
   - If k <= max_k: resize then pad to res^3 (sum ~ exact target).
   - If k > max_k:  cap k so resized volume still fits (sum < target). Flagged "clipped".
Rows are written in npz order (deterministic); ``<output>_ids.txt`` lists the npz row of each
sample and ``<output>_clipped.txt`` the clipped ones.

Paper training pool (5000 poses = all rows of humman_sub_all179_5k.npz, 0 clipped; "n14000"
is the target mass per pose):
  python scripts/data/voxelize_humman_volresize.py --input data/raw/humman_sub_all179_5k.npz \
      --n-target 14000 --sigma 0.7 --init-fill 0.7 --margin 0.9 \
      --output data/humman_volresize_n14000_128.pt
Held-out interpolation pool (1181 poses): build data/humman_interp_heldout.npz with
make_humman_heldout_interp.py, then run the same command with
  --input data/humman_interp_heldout.npz --output data/humman_interp_heldout_n14000_128.pt
The builder of humman_sub_all179_5k.npz from the raw HuMMan release is not included (see
scripts/data/README.md).
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


def _axis_angle_to_matrix(aa):
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32)
            + np.sin(theta) * K
            + (1 - np.cos(theta)) * (K @ K))


def _canonicalize(verts, global_orient, transl):
    R = _axis_angle_to_matrix(global_orient)
    return (verts - transl) @ R


# Worker globals (set by initializer to share read-only large arrays via fork)
_VERTS = None
_FACES = None
_ORIENT = None
_TRANSL = None


def _init_worker(verts, faces, orient, transl):
    global _VERTS, _FACES, _ORIENT, _TRANSL
    _VERTS = verts
    _FACES = faces
    _ORIENT = orient
    _TRANSL = transl


def _voxelize_one(idx, n_target: int, res: int, sigma: float, margin: float, init_fill: float):
    try:
        v_can = _canonicalize(_VERTS[idx], _ORIENT[idx], _TRANSL[idx])
        m = trimesh.Trimesh(vertices=v_can, faces=_FACES, process=False)
        if len(m.vertices) == 0:
            return None
        # Step 1: center, scale so max extent fills init_fill of grid
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

        # Step 3: small blur
        grid = _gaussian_blur_3d(grid, sigma)

        # Step 4: spatial resize to hit target sum
        s_now = float(grid.sum())
        if s_now <= 0:
            return None
        k = (n_target / s_now) ** (1.0 / 3.0)

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
        return (idx, out, clipped, float(out.sum()), float(out.max()))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True,
                    help="HuMMan .npz with verts/faces/global_orient/transl")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--n-target", type=int, default=8000)
    ap.add_argument("--margin", type=float, default=0.9,
                    help="Max bbox extent (fraction of res) after spatial resize.")
    ap.add_argument("--init-fill", type=float, default=0.7,
                    help="Per-shape pre-resize fill: max bbox extent set to this fraction of grid.")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    d = np.load(args.input, allow_pickle=True)
    verts = d["verts"].astype(np.float32)
    faces = d["faces"].astype(np.int64)
    orient = d["global_orient"].astype(np.float32)
    transl = d["transl"].astype(np.float32)
    N = verts.shape[0]
    if args.max_shapes > 0:
        N = min(N, args.max_shapes)
    indices = list(range(N))
    print(f"Found {N} frames | verts {verts.shape} faces {faces.shape} "
          f"| n_target={args.n_target} | res={args.resolution} | sigma={args.sigma} "
          f"| init_fill={args.init_fill} | margin={args.margin}")

    fn = partial(_voxelize_one, n_target=args.n_target, res=args.resolution,
                 sigma=args.sigma, margin=args.margin, init_fill=args.init_fill)
    grids, ids, sums, peaks, clipped = [], [], [], [], []
    n_fail = 0
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(verts, faces, orient, transl)) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, indices, chunksize=8)):
            if r is None:
                n_fail += 1
                continue
            ids.append(r[0]); grids.append(r[1])
            clipped.append(r[2]); sums.append(r[3]); peaks.append(r[4])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{N} | failed={n_fail} | clipped={sum(clipped)} | {time.time()-t0:.1f}s")
    print(f"Done: {len(grids)}/{N} kept, {n_fail} failed in {time.time()-t0:.1f}s")

    sums_a = np.array(sums); peaks_a = np.array(peaks); clipped_a = np.array(clipped)
    print(f"Sum  | min={sums_a.min():.1f} mean={sums_a.mean():.1f} max={sums_a.max():.1f}")
    print(f"Peak | min={peaks_a.min():.4f} mean={peaks_a.mean():.4f} max={peaks_a.max():.4f}")
    print(f"Clipped (sum < target due to grid bound): {int(clipped_a.sum())}/{len(clipped_a)} "
          f"({100*clipped_a.mean():.1f}%)")
    nc = sums_a[~clipped_a]
    if len(nc) > 0:
        print(f"Non-clipped sum: min={nc.min():.1f} max={nc.max():.1f} (target={args.n_target})")

    # Re-order by original index for reproducibility
    order = np.argsort(ids)
    grids_arr = np.stack([grids[i] for i in order]).astype(np.float32)
    ids_ord = [int(ids[i]) for i in order]
    clipped_ord = [bool(clipped[i]) for i in order]

    data = torch.from_numpy(grids_arr).unsqueeze(1).contiguous()
    print(f"Output: {tuple(data.shape)} {data.dtype}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved {args.output}")
    with open(args.output.replace(".pt", "_ids.txt"), "w") as f:
        f.write("\n".join(str(i) for i in ids_ord))
    with open(args.output.replace(".pt", "_clipped.txt"), "w") as f:
        for mid, c in zip(ids_ord, clipped_ord):
            if c:
                f.write(str(mid) + "\n")


if __name__ == "__main__":
    main()
