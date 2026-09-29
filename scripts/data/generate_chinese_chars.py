r"""Render the CJK glyph pool used to train `cjk_2d` (data/chinese_chars_256.pt).

Each of the first ``--n-chars`` code points of the CJK Unified Ideographs block (U+4E00 onward;
not a frequency list) is rendered with every font in ``--fonts`` (font-major order), then
  1. centred at 80% of the canvas (``viot.glyphs.render_char_2d``),
  2. Gaussian blur, sigma = blur_sigma * H / 64 (8.0 at 256^2, kernel 49),
  3. +1e-5 floor and mass normalisation,
  4. participation-ratio (area) normalisation to MNIST_TARGET_PR_FRAC * H * W (0.197),
  5. mass normalisation.
Glyphs missing from a font (blank renders) are skipped.

Output: a float32 tensor [N, 1, H, W], each sample summing to 1.

Paper file (9000 = 3000 characters x 3 fonts, in this font order):
  python scripts/data/generate_chinese_chars.py --resolution 256 --n-chars 3000 \
      --font-dir <dir with the fonts> --fonts simhei.ttf msyh.ttc simsun.ttc \
      --output data/chinese_chars_256.pt
The three fonts are SimHei (simhei.ttf), Microsoft YaHei (msyh.ttc, first face) and SimSun
(simsun.ttc, first face), as bundled with Windows. They are commercial fonts and are not
redistributable, so you must supply them. Any other CJK fonts (e.g. Noto Sans/Serif CJK,
SIL OFL) work, but give a different pool than the paper's.
"""

import argparse
import os
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


def common_chinese_chars(n=3500):
    """The first ``n`` code points of the CJK Unified Ideographs block (U+4E00 ...)."""
    start = 0x4E00
    return [chr(start + i) for i in range(n)]


def resolve_fonts(fonts, font_dir=None):
    """Return font paths: entries that are existing files are kept, others are joined to font_dir."""
    out = []
    for f in fonts:
        if os.path.isfile(f):
            out.append(f)
        elif font_dir is not None and os.path.isfile(os.path.join(font_dir, f)):
            out.append(os.path.join(font_dir, f))
        else:
            raise FileNotFoundError(f"font not found: {f!r} (font dir: {font_dir!r})")
    return out


def _process_chunk(batch, k_size, blur_sigma, target_pr, H, W):
    chunk = torch.stack(batch)  # [B, 1, H, W]
    chunk = TF.gaussian_blur(chunk, kernel_size=k_size, sigma=blur_sigma)
    chunk = chunk.clamp(min=0) + 1e-5
    chunk = chunk / (chunk.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    for i in range(chunk.shape[0]):
        chunk[i] = normalize_area_on_grid(chunk[i], target_pr, H, W)
    chunk = chunk.clamp(min=0)
    return chunk / (chunk.sum(dim=(-2, -1), keepdim=True) + 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--n-chars", type=int, default=3000,
                        help="Number of distinct characters (first N code points from U+4E00)")
    parser.add_argument("--blur-sigma", type=float, default=2.0,
                        help="Gaussian blur sigma at 64^2; scaled by resolution/64")
    parser.add_argument("--fonts", type=str, nargs="+", required=True,
                        help="Font files (paths, or names looked up in --font-dir). Order matters: "
                             "the output is font-major. Paper: simhei.ttf msyh.ttc simsun.ttc")
    parser.add_argument("--font-dir", type=str, default=None,
                        help="Directory used to resolve --fonts entries that are bare file names")
    parser.add_argument("--output", type=str, default="data/chinese_chars_256.pt")
    args = parser.parse_args()

    H = W = args.resolution
    fonts = resolve_fonts(args.fonts, args.font_dir)
    print(f"Using fonts: {[os.path.basename(f) for f in fonts]}")
    print(f"Resolution: {H}x{W}, chars: {args.n_chars}")

    chars = common_chinese_chars(args.n_chars)
    print(f"Generating {len(chars)} chars x {len(fonts)} fonts = {len(chars) * len(fonts)} samples")

    target_pr = MNIST_TARGET_PR_FRAC * H * W
    print(f"Target participation ratio: {target_pr:.0f} pixels ({MNIST_TARGET_PR_FRAC:.3f} of {H * W})")

    blur_sigma = args.blur_sigma * (H / 64.0)
    k_size = int(blur_sigma * 6) | 1
    k_size = max(k_size, 3)
    print(f"Gaussian blur: sigma={blur_sigma:.2f}, kernel={k_size}")

    all_samples = []
    skipped = 0
    batch = []
    for fi, font in enumerate(fonts):
        print(f"\nFont {fi + 1}/{len(fonts)}: {os.path.basename(font)}")
        batch = []
        batch_idx = 0
        for ci, ch in enumerate(chars):
            arr = render_char_2d(ch, font, H)
            if arr is None or arr.sum() < 10:
                skipped += 1
                continue
            batch.append(torch.from_numpy(arr).unsqueeze(0))  # [1, H, W]

            # Process in chunks of 500 to bound memory
            if len(batch) >= 500 or ci == len(chars) - 1:
                all_samples.append(_process_chunk(batch, k_size, blur_sigma, target_pr, H, W))
                batch = []
                batch_idx += 1
                print(f"  chunk {batch_idx}: processed {ci + 1}/{len(chars)} chars")

    if batch:
        all_samples.append(_process_chunk(batch, k_size, blur_sigma, target_pr, H, W))

    data = torch.cat(all_samples, dim=0)  # [N, 1, H, W]
    print(f"\nFinal dataset: {data.shape}, dtype={data.dtype}")
    print(f"Skipped (missing glyph or blank): {skipped}")
    print(f"Mass: mean={data.sum(dim=(-2, -1)).mean():.6f}, std={data.sum(dim=(-2, -1)).std():.6f}")
    print(f"PR: {((data.sum(dim=(-2, -1)) ** 2) / (data.pow(2).sum(dim=(-2, -1)) + 1e-12)).mean():.1f} pixels")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved: {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
