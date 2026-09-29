r"""Synthesize NEW held-out HuMMan poses (never voxelized / trained on) without an SMPL model.

The training set (humman_sub_all179_5k.npz) is 5000 frames sub-sampled from 2657 (subject, action)
motion clips, 1-11 frames per clip, temporally spaced. SMPL meshes share topology, so for two
consecutive sampled frames of the SAME clip we take the vertex-wise midpoint of the CANONICALIZED
meshes (global orientation / translation removed exactly as the voxelizer does) -> a plausible
in-between pose that is a genuinely new body configuration. We keep only frame pairs whose RMS
vertex displacement lies in [d_min, d_max] (small enough for linear blending to stay plausible,
large enough that the new pose is not a duplicate).

Output npz has verts/faces/global_orient(=0)/transl(=0)/pid/seq/frame like the source, so
scripts/data/voxelize_humman_volresize.py runs on it unchanged.

Paper held-out pool (1181 midpoint poses; the defaults below accept 1181 < --max-samples pairs):
  python scripts/data/make_humman_heldout_interp.py --input data/raw/humman_sub_all179_5k.npz \
      --output data/humman_interp_heldout.npz
  python scripts/data/voxelize_humman_volresize.py --input data/humman_interp_heldout.npz \
      --n-target 14000 --sigma 0.7 --init-fill 0.7 --margin 0.9 \
      --output data/humman_interp_heldout_n14000_128.pt
"""
import argparse
import collections

import numpy as np


def axis_angle_to_matrix(aa):
    th = np.linalg.norm(aa) + 1e-12
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def canonicalize(verts, go, tr):
    R = axis_angle_to_matrix(go)
    return (verts - tr) @ R  # R^T applied on the right


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--d-min', type=float, default=0.03, help='min RMS vertex displacement (canonical units)')
    ap.add_argument('--d-max', type=float, default=0.25)
    ap.add_argument('--max-samples', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    d = np.load(a.input, allow_pickle=True)
    verts, faces = d['verts'], d['faces']
    go, tr = d['global_orient'], d['transl']
    pid, seq, fr = d['pid'], d['seq'], d['frame']
    groups = collections.defaultdict(list)
    for i, (p, s) in enumerate(zip(pid.tolist(), seq.tolist())):
        groups[(p, s)].append(i)
    cands = []
    for key, idx in groups.items():
        idx = sorted(idx, key=lambda i: int(fr[i]))
        for i0, i1 in zip(idx[:-1], idx[1:]):
            v0 = canonicalize(verts[i0], go[i0], tr[i0])
            v1 = canonicalize(verts[i1], go[i1], tr[i1])
            rms = float(np.sqrt(((v1 - v0) ** 2).sum(-1).mean()))
            if a.d_min <= rms <= a.d_max:
                cands.append((i0, i1, rms))
    print(f'clips={len(groups)} consecutive pairs total={sum(max(len(v) - 1, 0) for v in groups.values())} '
          f'accepted in [{a.d_min},{a.d_max}]: {len(cands)}', flush=True)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(cands)
    cands = cands[:a.max_samples]
    new_v, meta = [], []
    for i0, i1, rms in cands:
        v0 = canonicalize(verts[i0], go[i0], tr[i0])
        v1 = canonicalize(verts[i1], go[i1], tr[i1])
        new_v.append(0.5 * (v0 + v1))
        meta.append((pid[i0], seq[i0], int(fr[i0]), int(fr[i1]), rms))
    new_v = np.stack(new_v).astype(np.float32)
    n = len(new_v)
    rms_all = np.array([m[4] for m in meta])
    print(f'synthesized {n} midpoint poses | rms disp min/median/max {rms_all.min():.3f}/{np.median(rms_all):.3f}/{rms_all.max():.3f}', flush=True)
    np.savez(a.output, verts=new_v, faces=faces,
             global_orient=np.zeros((n, 3), np.float32), transl=np.zeros((n, 3), np.float32),
             pid=np.array([m[0] for m in meta]), seq=np.array([m[1] for m in meta]),
             frame=np.array([m[2] for m in meta]), frame_b=np.array([m[3] for m in meta]), rms=rms_all.astype(np.float32),
             source=np.array([a.input]))
    print('saved', a.output)


if __name__ == '__main__':
    main()
