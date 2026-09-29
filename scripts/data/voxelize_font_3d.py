r"""Generate the extruded 3D glyph dataset used to train `font_3d` (data/font_3d_volresize_n8000_128.pt).

For each (font, char) pair (``viot.glyphs``, identical to the original research code):
  1. render the glyph to a tightly cropped binary mask at 256 px (``render_glyph_mask``);
  2. extrude along z by depth_frac * max(H, W) (0.3);
  3. trilinear-resize so the largest extent is init_fill * res (0.7), centre in res^3;
  4. Gaussian blur, sigma 0.7;
  5. isotropic trilinear resize so the total mass is n_target (8000), capped so the shape spans at
     most margin * res (0.9; "clipped" shapes stay below n_target); clip to [0, 1]
     (steps 2-5: ``mask_to_volume``).
Output: float32 [N, 1, res, res, res] (peak ~1, sum ~n_target) and ``<output>_ids.txt``
(``<font file stem>_<char>`` per sample, in tensor order).

Shapes are produced by a process pool in completion order, so the row order of a fresh build is
not deterministic. Pass ``--match-ids`` with a shipped id list to put the rows in the paper's order.

Paper file (1364 = 62 glyphs x 22 DejaVu faces, 0 clipped, sums 7728-8277). The faces are the 22
TrueType files of DejaVu 2.37 (DejaVuSans{,-Bold,-BoldOblique,-ExtraLight,-Oblique},
DejaVuSansCondensed{,-Bold,-BoldOblique,-Oblique}, DejaVuSansMono{,-Bold,-BoldOblique,-Oblique},
DejaVuSerif{,-Bold,-BoldItalic,-Italic}, DejaVuSerifCondensed{,-Bold,-BoldItalic,-Italic},
DejaVuMathTeXGyre): /usr/share/fonts/truetype/dejavu with the Debian/Ubuntu packages
fonts-dejavu-core, fonts-dejavu-extra and fonts-dejavu-mono 2.37 (as used for the paper), or the
dejavu-fonts-ttf-2.37 release archive:
  python scripts/data/voxelize_font_3d.py --font-paths /usr/share/fonts/truetype/dejavu \
      --output data/font_3d_volresize_n8000_128.pt \
      --match-ids scripts/data/splits/font_3d_volresize_n8000_128_ids.txt
Held-out faces of the paper's held-out table (496 = 62 x 8): FreeMono, FreeMonoBold, FreeSans,
FreeSansBold, FreeSerif, FreeSerifBold (.ttf, GNU FreeFont 20211204, Debian package
fonts-freefont-ttf) and RobotoSlab-Regular, RobotoSlab-Bold (.otf, package fonts-roboto-slab):
  F=/usr/share/fonts/truetype/freefont; R=/usr/share/fonts/opentype/roboto/slab
  python scripts/data/voxelize_font_3d.py --font-paths $F/FreeMono.ttf $F/FreeMonoBold.ttf \
      $F/FreeSans.ttf $F/FreeSansBold.ttf $F/FreeSerif.ttf $F/FreeSerifBold.ttf \
      $R/RobotoSlab-Regular.otf $R/RobotoSlab-Bold.otf \
      --output data/font3d_heldout_fonts_n8000_128.pt \
      --match-ids scripts/data/splits/font3d_heldout_fonts_n8000_128_ids.txt
(the exact command of the held-out file was not archived; defaults otherwise).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

try:
    import viot  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from viot.glyphs import mask_to_volume, render_glyph_mask  # noqa: E402


def _gen_one(item, n_target: int, res: int, sigma: float, margin: float,
             init_fill: float, depth_frac: float, render_size: int):
    font_path, ch = item
    try:
        mask = render_glyph_mask(font_path, ch, render_size=render_size)
        if mask is None or mask.sum() < 100:  # skip tiny / missing glyphs
            return None
        out, clipped = mask_to_volume(mask, res=res, n_target=n_target, sigma=sigma, margin=margin,
                                      init_fill=init_fill, depth_frac=depth_frac)
        if out is None:
            return None
        font_id = os.path.splitext(os.path.basename(font_path))[0]
        return (f"{font_id}_{ch}", out, clipped, float(out.sum()), float(out.max()))
    except Exception:
        return None


def collect_fonts(font_paths_or_dirs, exts=(".ttf", ".otf")):
    out = []
    for p in font_paths_or_dirs:
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(exts):
                        out.append(os.path.join(root, f))
    return sorted(set(out))


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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--font-paths", nargs="+", required=True,
                    help="One or more .ttf/.otf files or directories (searched recursively)")
    ap.add_argument("--charset", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--render-size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--n-target", type=int, default=8000)
    ap.add_argument("--margin", type=float, default=0.9)
    ap.add_argument("--init-fill", type=float, default=0.7)
    ap.add_argument("--depth-frac", type=float, default=0.3,
                    help="Z extrusion depth as fraction of max(H,W)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--match-ids", default=None,
                    help="Reorder (and subset) the output rows to this id list, e.g. "
                         "scripts/data/splits/font_3d_volresize_n8000_128_ids.txt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fonts = collect_fonts(args.font_paths)
    print(f"Fonts found: {len(fonts)}")

    # Pre-filter fonts: each must successfully render at least one char
    valid_fonts = []
    for fp in fonts:
        ok = render_glyph_mask(fp, "A", render_size=args.render_size)
        if ok is not None and ok.sum() >= 100:
            valid_fonts.append(fp)
    print(f"Fonts that render 'A': {len(valid_fonts)}/{len(fonts)}")

    items = [(fp, ch) for fp in valid_fonts for ch in args.charset]
    print(f"Glyphs to generate: {len(items)} ({len(args.charset)} chars x {len(valid_fonts)} fonts) "
          f"| n_target={args.n_target} | depth_frac={args.depth_frac}")

    fn = partial(_gen_one, n_target=args.n_target, res=args.resolution,
                 sigma=args.sigma, margin=args.margin, init_fill=args.init_fill,
                 depth_frac=args.depth_frac, render_size=args.render_size)
    grids, ids, sums, peaks, clipped = [], [], [], [], []
    n_fail = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, items, chunksize=4)):
            if r is None:
                n_fail += 1
                continue
            ids.append(r[0]); grids.append(r[1])
            clipped.append(r[2]); sums.append(r[3]); peaks.append(r[4])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(items)} | failed={n_fail} | clipped={sum(clipped)} | "
                      f"{time.time() - t0:.1f}s")
    print(f"Done: {len(grids)}/{len(items)} kept, {n_fail} failed in {time.time() - t0:.1f}s")

    if not grids:
        print("No valid glyphs!")
        return

    if args.match_ids:
        ids, (grids, sums, peaks, clipped) = reorder_to_ids(
            ids, [grids, sums, peaks, clipped], args.match_ids)

    sums_a = np.array(sums); peaks_a = np.array(peaks); clipped_a = np.array(clipped)
    print(f"Sum  | min={sums_a.min():.1f} mean={sums_a.mean():.1f} max={sums_a.max():.1f}")
    print(f"Peak | min={peaks_a.min():.4f} mean={peaks_a.mean():.4f} max={peaks_a.max():.4f}")
    print(f"Clipped: {int(clipped_a.sum())}/{len(clipped_a)} ({100 * clipped_a.mean():.1f}%)")
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


if __name__ == "__main__":
    main()
