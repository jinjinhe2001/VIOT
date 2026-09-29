r"""Split the airplane pool by the official ShapeNetCore.v2.PC15k split and build the matching sphere.

Used for the held-out retrain of sphere -> airplane (paper run exp91).

Input : data/shapenet_airplanes_wtt_volresize_n8000_128.pt (+ _ids.txt from voxelize_airplanes_wtt_volresize.py)
Output: <out-dir>/airplanes_wtt_train_128.pt (PC15k train+val, 3237), <out-dir>/airplanes_wtt_test_128.pt
        (PC15k test, 808), their *_ids.txt, airplanes_wtt_split_meta.json, and
        sphere_match_airplanes_wtt_s02_128.pt (softness 0.2, 8 copies, sum matched to the train-set
        mean mass, like sphere_match_airplanes_s02_128.pt for the full-pool model).

The split comes either from the shipped id lists (default in scripts/train/*_heldout_split.sh):
  python scripts/data/split_airplanes_wtt.py --data data/shapenet_airplanes_wtt_volresize_n8000_128.pt \
      --train-ids scripts/data/splits/airplanes_wtt_train_128_ids.txt \
      --test-ids scripts/data/splits/airplanes_wtt_test_128_ids.txt --out-dir data
or from a copy of the PC15k point-cloud release (its airplane folder 02691156/{train,val,test}/<id>.npy):
  python scripts/data/split_airplanes_wtt.py --pc15k-root <ShapeNetCore.v2.PC15k>/02691156 --out-dir data
Rows keep the order of the id lists (train: PC15k train then val, each sorted; test: sorted), so the
outputs do not depend on the row order of the input pool. The paper run gave a sphere radius of
12.3789 (train mean mass 7964.17).
"""
import argparse
import json
import os

import torch


def make_sphere(radius, canvas, softness):
    z = torch.arange(canvas).float() - (canvas - 1) / 2.0
    r = torch.stack(torch.meshgrid(z, z, z, indexing='ij')).pow(2).sum(0).sqrt()
    s = torch.sigmoid((radius - r) / softness)
    return s / s.max()


def _read_ids(path):
    with open(path) as f:
        return f.read().split()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default='data/shapenet_airplanes_wtt_volresize_n8000_128.pt')
    ap.add_argument('--train-ids', default=None, help='id list of the training split (PC15k train+val)')
    ap.add_argument('--test-ids', default=None, help='id list of the held-out split (PC15k test)')
    ap.add_argument('--pc15k-root', default=None,
                    help='alternative to --train-ids/--test-ids: .../ShapeNetCore.v2.PC15k/02691156')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--tag', default='airplanes_wtt')
    ap.add_argument('--sphere-softness', type=float, default=0.2)
    ap.add_argument('--sphere-copies', type=int, default=8)
    a = ap.parse_args()

    X = torch.load(a.data, map_location='cpu', weights_only=False)
    ids = open(a.data.replace('.pt', '_ids.txt')).read().split()
    assert X.shape[0] == len(ids), (X.shape, len(ids))
    print('loaded', tuple(X.shape), X.dtype, 'ids', len(ids), flush=True)
    pos = {i: k for k, i in enumerate(ids)}
    if a.train_ids and a.test_ids:
        want_train, want_test = _read_ids(a.train_ids), _read_ids(a.test_ids)
        absent = [i for i in want_train + want_test if i not in pos]
        if absent:
            raise SystemExit(f'{len(absent)} split ids are not in {a.data}, e.g. {absent[:5]}')
    elif a.pc15k_root:
        split = {s: sorted(f[:-4] for f in os.listdir(os.path.join(a.pc15k_root, s)))
                 for s in ('train', 'val', 'test')}
        want_train, want_test = split['train'] + split['val'], split['test']
    else:
        ap.error('give --train-ids and --test-ids, or --pc15k-root')
    train_ids = [i for i in want_train if i in pos]
    test_ids = [i for i in want_test if i in pos]
    missing = [i for i in ids if i not in set(train_ids) | set(test_ids)]
    print(f'train+val {len(train_ids)} | test {len(test_ids)} | not in the split lists {len(missing)}', flush=True)
    sums = X.sum(dim=(1, 2, 3, 4)).numpy()
    mean_sum = None
    os.makedirs(a.out_dir, exist_ok=True)
    for name, sel in (('train', train_ids), ('test', test_ids)):
        idx = torch.tensor([pos[i] for i in sel])
        Y = X[idx].contiguous()
        out = os.path.join(a.out_dir, f'{a.tag}_{name}_128.pt')
        torch.save(Y, out)
        with open(out.replace('.pt', '_ids.txt'), 'w') as f:
            f.write('\n'.join(sel))
        s = sums[idx.numpy()]
        print(f'{name}: {tuple(Y.shape)} -> {out} | sum min/mean/max {s.min():.0f}/{s.mean():.0f}/{s.max():.0f} | '
              f'peak mean {Y.amax(dim=(1, 2, 3, 4)).mean():.3f}', flush=True)
        if name == 'train':
            mean_sum = float(s.mean())
    lo, hi = 1.0, 60.0
    mid = cur = None
    s_ = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s_ = make_sphere(mid, 128, a.sphere_softness)
        cur = s_.sum().item()
        if abs(cur - mean_sum) < 0.5:
            break
        if cur < mean_sum:
            lo = mid
        else:
            hi = mid
    sp = s_[None, None].expand(a.sphere_copies, 1, 128, 128, 128).contiguous().clone()
    outs = os.path.join(a.out_dir, f'sphere_match_{a.tag}_s{str(a.sphere_softness).replace(".", "")}_128.pt')
    torch.save(sp, outs)
    print(f'sphere: radius={mid:.3f} sum={cur:.1f} (target = train mean {mean_sum:.1f}) -> {outs}', flush=True)
    meta = {'source': os.path.basename(a.data), 'split': 'PC15k official: train+val -> train, test -> held-out',
            'n_train': len(train_ids), 'n_test': len(test_ids), 'n_missing': len(missing), 'train_mean_sum': mean_sum,
            'sphere_radius': mid, 'sphere_softness': a.sphere_softness}
    with open(os.path.join(a.out_dir, f'{a.tag}_split_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
