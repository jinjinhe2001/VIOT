r"""Render a (held-out) glyph pool with exactly the pipeline of generate_chinese_chars.py.

render (``viot.glyphs.render_char_2d``) -> Gaussian blur sigma = 2.0 * res / 64 -> +1e-5 floor ->
mass norm -> participation-ratio normalisation to MNIST_TARGET_PR_FRAC -> mass norm.
Writes ``<output>.pt`` ([N, 1, H, W] float32, unit mass) and ``<output>_meta.txt``
(one ``font<TAB>char`` line per sample, in tensor order).

Pools of the paper's held-out table (fonts from a Windows installation; they are not
redistributable, so supply them with --font-dir or as paths):
  # CJK, 300 unseen characters (U+4E00 + 3000 ... 3299), the 3 training fonts   -> 900 samples
  python scripts/data/make_glyph_pool.py --cjk-start 3000 --cjk-n 300 \
      --font-dir <fonts> --fonts simhei.ttf msyh.ttc simsun.ttc --output data/cjk_heldout_chars_256.pt
  # CJK, the first 300 training characters in 2 unseen typefaces (KaiTi, FangSong) -> 600 samples
  python scripts/data/make_glyph_pool.py --cjk-start 0 --cjk-n 300 \
      --font-dir <fonts> --fonts simkai.ttf simfang.ttf --output data/cjk_heldout_fonts_256.pt
  # Latin letters + digits (62 glyphs) in 8 unseen font families                -> 496 samples
  python scripts/data/make_glyph_pool.py --latin --font-dir <fonts> \
      --fonts arial.ttf times.ttf georgia.ttf calibri.ttf consola.ttf segoeui.ttf tahoma.ttf cour.ttf \
      --output data/font_heldout_fonts_256.pt
The expected per-sample (font, char) lists are in scripts/data/splits/*_meta.txt.
"""
import argparse
import os
import string
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

try:
    import viot  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from viot.data_2d import MNIST_TARGET_PR_FRAC, normalize_area_on_grid  # noqa: E402
from viot.glyphs import render_char_2d  # noqa: E402


def process(chunk, blur_sigma, k_size, target_pr, H, W):
    chunk = TF.gaussian_blur(chunk, kernel_size=k_size, sigma=blur_sigma)
    chunk = chunk.clamp(min=0) + 1e-5
    chunk = chunk / (chunk.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    for i in range(chunk.shape[0]):
        chunk[i] = normalize_area_on_grid(chunk[i], target_pr, H, W)
    chunk = chunk.clamp(min=0)
    return chunk / (chunk.sum(dim=(-2, -1), keepdim=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--blur-sigma', type=float, default=2.0, help='scaled by res/64 like the training generator')
    ap.add_argument('--cjk-start', type=int, default=None, help='offset from U+4E00 of the first character')
    ap.add_argument('--cjk-n', type=int, default=300)
    ap.add_argument('--latin', action='store_true', help='A-Z a-z 0-9 (62 glyphs)')
    ap.add_argument('--fonts', nargs='+', required=True,
                    help='font files: paths, or names looked up in --font-dir (output is font-major)')
    ap.add_argument('--font-dir', default=None, help='directory used to resolve bare font file names')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    H = W = a.resolution
    if a.latin:
        chars = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    else:
        if a.cjk_start is None:
            ap.error('give --latin or --cjk-start')
        chars = [chr(0x4E00 + i) for i in range(a.cjk_start, a.cjk_start + a.cjk_n)]
    fonts = []
    for f in a.fonts:
        p = f if os.path.isfile(f) or a.font_dir is None else os.path.join(a.font_dir, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f'font not found: {f!r} (font dir: {a.font_dir!r})')
        fonts.append(p)
    target_pr = MNIST_TARGET_PR_FRAC * H * W
    blur_sigma = a.blur_sigma * (H / 64.0)
    k_size = max(int(blur_sigma * 6) | 1, 3)
    print(f'chars={len(chars)} fonts={len(fonts)} blur_sigma={blur_sigma} k={k_size} target_pr={target_pr:.0f}', flush=True)
    out, meta, skipped = [], [], 0
    for font in fonts:
        batch, bmeta = [], []
        for ch in chars:
            arr = render_char_2d(ch, font, H)
            if arr is None or arr.sum() < 10:
                skipped += 1
                continue
            batch.append(torch.from_numpy(arr).unsqueeze(0))
            bmeta.append((os.path.basename(font), ch))
        if batch:
            out.append(process(torch.stack(batch), blur_sigma, k_size, target_pr, H, W))
            meta += bmeta
        print(f'  {os.path.basename(font)}: {len(batch)} glyphs', flush=True)
    X = torch.cat(out).contiguous()
    s = X.sum((1, 2, 3))
    pr = (s ** 2) / (X.pow(2).sum((1, 2, 3)) + 1e-12) / (H * W)
    print(f'pool {tuple(X.shape)} skipped={skipped} | sum {s.mean():.4f} | max {X.amax((1, 2, 3)).mean():.6f} | '
          f'PRfrac {pr.mean():.4f}+-{pr.std():.4f}', flush=True)
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    torch.save(X, a.output)
    with open(a.output.replace('.pt', '_meta.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(f'{fn}\t{ch}' for fn, ch in meta))
    print('saved', a.output)


if __name__ == '__main__':
    main()
