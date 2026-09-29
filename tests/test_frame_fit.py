"""Tests for the frame-aware area normalization of sketches and glyphs
(``viot.data_2d.fit_area_in_frame`` and ``thicken_to_area``).

Unlike the parity tests these need no reference code: they check that
``fit_area_in_frame`` is bit-for-bit ``normalize_area_on_grid`` whenever its
size limit does not apply and keeps thin shapes inside the canvas, that
``thicken_to_area`` then brings them to the target area without leaving the
frame, and that ``strokes_to_density`` without ``max_extent`` is unchanged.

Run with ``pytest tests/test_frame_fit.py -v``.
"""

import os
import sys
from pathlib import Path
from unittest import SkipTest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viot import data_2d  # noqa: E402
from viot.data_2d import (FRAME_MAX_EXTENT, MNIST_TARGET_PR_FRAC,  # noqa: E402
                          fit_area_in_frame, normalize_area_on_grid,
                          strokes_to_density, thicken_to_area)

N = 256
TARGET = MNIST_TARGET_PR_FRAC * N * N


def extent(d):
    """Largest side of the bbox of ``d > 0.05 * max`` (fraction of the grid) and
    whether that bbox reaches within 1 px of the border."""
    d = np.asarray(d, dtype=np.float64).squeeze()
    occ = d > 0.05 * d.max()
    r = np.where(occ.any(1))[0]
    c = np.where(occ.any(0))[0]
    touching = bool(r[0] <= 1 or c[0] <= 1 or r[-1] >= d.shape[0] - 2 or c[-1] >= d.shape[1] - 2)
    return max(r[-1] - r[0] + 1, c[-1] - c[0] + 1) / d.shape[0], touching


def border_mass(d, w=3):
    """Fraction of the mass within ``w`` px of the border."""
    d = np.asarray(d, dtype=np.float64).squeeze()
    return 1.0 - d[w:-w, w:-w].sum() / d.sum()


def pr(d):
    d = d.double()
    return (d.sum() ** 2 / (d ** 2).sum()).item()


def density(mask, blur=2.0, floor=1e-5):
    """[1, H, W] blurred, floored, mass-normalized density of a [H, W] mask."""
    import torchvision.transforms.functional as TF
    x = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    k = max(int(blur * 6) | 1, 3)
    x = TF.gaussian_blur(x, kernel_size=k, sigma=blur).clamp(min=0) + floor
    return (x / x.sum())[0]


def disk(cx, cy, r, n=N):
    yy, xx = np.mgrid[:n, :n]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)


def ring(cx, cy, r, w, n=N):
    return disk(cx, cy, r, n) - disk(cx, cy, r - w, n)


def bar(y0, y1, x0, x1, n=N):
    m = np.zeros((n, n), np.float32)
    m[y0:y1, x0:x1] = 1
    return m


def stroke_image(polylines, width):
    from PIL import Image, ImageDraw
    img = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(img)
    r = width // 2
    for pts in polylines:
        d.line(pts, fill=255, width=2 * r)
        for x, y in pts:
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return img


def strokes_to_density_original(stroke_img, resolution=256, fill_frac=0.70, blur_sigma=2.0):
    """``strokes_to_density`` as released before ``max_extent`` was added (verbatim)."""
    import torchvision.transforms.functional as TF
    arr = np.asarray(stroke_img, dtype=np.float32) / 255.0
    if arr.sum() < 1.0:
        return None
    H, W = arr.shape
    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    crop = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h,
       (side - w) // 2:(side - w) // 2 + w] = crop
    fill = max(int(round(H * fill_frac)), 8)
    sq_t = torch.from_numpy(sq).unsqueeze(0).unsqueeze(0)
    sq_t = F.interpolate(sq_t, size=(fill, fill), mode="bilinear", align_corners=False)
    canvas = torch.zeros(1, 1, H, W)
    pad = (H - fill) // 2
    canvas[:, :, pad:pad + fill, pad:pad + fill] = sq_t
    k_size = max(int(blur_sigma * 6) | 1, 3)
    canvas = TF.gaussian_blur(canvas, kernel_size=k_size, sigma=blur_sigma)
    canvas = canvas.clamp(min=0) + 1e-5
    canvas = canvas / (canvas.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    target_pr = data_2d.MNIST_TARGET_PR_FRAC * H * W
    canvas[0] = normalize_area_on_grid(canvas[0], target_pr, H, W)
    canvas = canvas.clamp(min=0)
    canvas = canvas / (canvas.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    return canvas


# Shapes that area normalization leaves well inside the frame (the limit does not apply).
ROUND_SHAPES = {
    "small_disk": disk(128, 128, 40),          # enlarged
    "big_disk": disk(128, 128, 110),           # shrunk
    "ring": ring(120, 136, 70, 16),            # enlarged, off-centre
    "two_blobs": disk(90, 100, 30) + disk(170, 160, 25),
}
# Thin shapes that the unbounded zoom pushes past the border.
THIN_SHAPES = {
    "vbar": bar(40, 216, 124, 132),            # a '1' / 'l'
    "hbar": bar(124, 130, 30, 226),            # a dash / '一'
    "T": bar(40, 48, 60, 196) + bar(48, 216, 125, 131),
}


def render_fn(max_extent, blur=2.0):
    """The tail of a density pipeline: blur, floor, mass-normalize, frame-capped zoom."""
    import torchvision.transforms.functional as TF

    def render(img):
        k = max(int(blur * 6) | 1, 3)
        x = TF.gaussian_blur(img, kernel_size=k, sigma=blur).clamp(min=0) + 1e-5
        x = x / x.sum()
        x[0] = fit_area_in_frame(x[0], TARGET, N, N, max_extent=max_extent)
        x = x.clamp(min=0)
        return x / x.sum()
    return render


def test_equal_when_limit_does_not_apply():
    for name, mask in ROUND_SHAPES.items():
        x = density(mask)
        ref = normalize_area_on_grid(x.clone(), TARGET, N, N)
        assert extent(ref)[0] < 0.88, name   # precondition: the limit is not active
        out = fit_area_in_frame(x.clone(), TARGET, N, N, max_extent=0.9)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        assert torch.equal(out, ref), f"{name}: max |diff| {(out - ref).abs().max().item():.3e}"
        assert abs(pr(out) / TARGET - 1) < 0.005, name
        if extent(ref)[0] < FRAME_MAX_EXTENT - 0.02:   # also below the default limit
            assert torch.equal(fit_area_in_frame(x.clone(), TARGET, N, N), ref), name


def test_thin_shapes_stay_in_frame():
    for name, mask in THIN_SHAPES.items():
        x = density(mask)
        ref = normalize_area_on_grid(x.clone(), TARGET, N, N)
        assert extent(ref)[1], f"{name}: expected the unbounded zoom to touch the border"
        for max_extent in (0.9, 0.75):
            out = fit_area_in_frame(x.clone(), TARGET, N, N, max_extent=max_extent)
            ext, touching = extent(out)
            assert ext <= max_extent and not touching, f"{name} @ {max_extent}: extent {ext:.3f}"
            assert ext >= max_extent - 3.0 / N, f"{name} @ {max_extent}: zoom stopped early ({ext:.3f})"
            assert pr(out) < TARGET, name
            if extent(x)[0] < max_extent - 0.02:
                assert pr(out) > pr(x), name   # enlarged, as far as the frame allows
            assert torch.isfinite(out).all() and (out >= 0).all()


def test_shape_larger_than_limit_is_shrunk():
    # PR already on target: normalize_area_on_grid returns the input unchanged,
    # fit_area_in_frame shrinks it into the frame.
    x = density(ring(128, 128, 125, 30))
    target = pr(x[0])
    assert extent(x)[0] > 0.95
    assert torch.equal(normalize_area_on_grid(x.clone(), target, N, N), x)
    out = fit_area_in_frame(x.clone(), target, N, N, max_extent=0.9)
    ext, touching = extent(out)
    assert ext <= 0.9 and not touching


def test_empty_image_is_returned_unchanged():
    z = torch.zeros(1, N, N)
    assert torch.equal(fit_area_in_frame(z.clone(), TARGET, N, N), z)


def test_ink_distance_is_exact():
    g = torch.Generator().manual_seed(0)
    img = (torch.rand(24, 24, generator=g) > 0.97).float() * 0.8
    img[3, 5] = 1.0
    d = data_2d._ink_distance(img)
    ys, xs = torch.nonzero(img >= 0.5, as_tuple=True)
    yy, xx = torch.meshgrid(torch.arange(24.), torch.arange(24.), indexing="ij")
    ref = ((yy[..., None] - ys) ** 2 + (xx[..., None] - xs) ** 2).amin(-1).double().sqrt()
    assert torch.equal(d, ref)


def test_thicken_to_area():
    for max_extent in (0.9, FRAME_MAX_EXTENT):
        render = render_fn(max_extent)
        # a shape that reaches the area within the frame is returned as rendered
        blob = torch.from_numpy(disk(128, 128, 40))[None, None]
        assert torch.equal(thicken_to_area(blob, render, TARGET), render(blob))
        for name, mask in THIN_SHAPES.items():
            img = torch.from_numpy(mask)[None, None]
            plain = render(img)
            out = thicken_to_area(img, render, TARGET)
            assert pr(plain[0, 0]) < 0.99 * TARGET, name          # the frame keeps it short ...
            assert abs(pr(out[0, 0]) / TARGET - 1) < 0.006, name   # ... thickening reaches the area
            ext, touching = extent(out)
            assert max_extent - 4.0 / N <= ext <= max_extent and not touching, f"{name}: {ext:.3f}"
            assert border_mass(out) < 1e-4, name
            # bolder, not blurrier: the ink keeps a flat top as wide as the bar
            o = out[0, 0].double()
            assert (o > 0.9 * o.max()).sum() > 4 * (plain[0, 0] > 0.9 * plain[0, 0].max()).sum(), name


def test_strokes_to_density_default_unchanged():
    sketches = [
        ([[(60, 40), (120, 200), (190, 90)]], 18),                    # scribble
        ([[(5, 240), (80, 250)]], 6),                                 # small, off-centre, wide bbox
        ([[(128, 51), (128, 205)]], 22),                              # vertical line '1'
        ([[(64, 51), (192, 51)], [(128, 51), (128, 215)]], 22),       # 'T'
    ]
    for i, (polylines, width) in enumerate(sketches):
        img = stroke_image(polylines, width)
        ref = strokes_to_density_original(img)
        assert torch.equal(strokes_to_density(img, resolution=N), ref), f"sketch {i}"
        assert torch.equal(strokes_to_density(img, resolution=N, max_extent=None), ref), f"sketch {i}"
        assert torch.equal(strokes_to_density(img, resolution=N, pr_frac=MNIST_TARGET_PR_FRAC), ref)
    assert strokes_to_density(stroke_image([], 4), resolution=N, max_extent=FRAME_MAX_EXTENT) is None


def test_strokes_to_density_max_extent():
    # Sketches that fit: identical to the uncapped result.
    scribble = stroke_image([[(70, 60), (120, 40), (175, 60), (180, 105), (130, 128), (185, 150),
                              (175, 205), (110, 220), (65, 195)]], 22)
    assert torch.equal(strokes_to_density(scribble, resolution=N, max_extent=0.9),
                       strokes_to_density(scribble, resolution=N))
    circle = stroke_image([[(128 + 30 * np.cos(a), 128 + 30 * np.sin(a))
                            for a in np.linspace(0, 2 * np.pi, 41)]], 16)
    assert torch.equal(strokes_to_density(circle, resolution=N, max_extent=FRAME_MAX_EXTENT),
                       strokes_to_density(circle, resolution=N))
    # Thin sketches: the uncapped result runs past the border, the capped one does
    # not, and it keeps the training area.
    for polylines, width in (([[(128, 51), (128, 205)]], 22),
                             ([[(128, 51), (128, 205)]], 4),
                             ([[(64, 51), (192, 51)], [(128, 51), (128, 215)]], 22),
                             ([[(51, 205), (205, 51)]], 22),
                             ([[(40, 60), (41, 60)], [(215, 200), (216, 200)]], 6)):
        img = stroke_image(polylines, width)
        unc = strokes_to_density(img, resolution=N)
        # (two dots: the uncapped zoom pushes one of them off the canvas)
        assert extent(unc)[1] or extent(unc)[0] > 0.92 or extent(unc)[0] < 0.5
        for max_extent in (0.9, FRAME_MAX_EXTENT):
            cap = strokes_to_density(img, resolution=N, max_extent=max_extent)
            ext, touching = extent(cap)
            assert ext <= max_extent and not touching, f"extent {ext:.3f}"
            assert border_mass(cap) < 1e-4, f"border mass {border_mass(cap):.1e}"
            assert abs(pr(cap[0, 0]) / TARGET - 1) < 0.006, f"PR {pr(cap[0, 0]) / TARGET:.4f}"
            assert abs(cap.sum().item() - 1) < 1e-5
    # another target area (the GUI passes the model's, e.g. 0.244 for mpeg7_2d)
    img = stroke_image([[(128, 51), (128, 205)]], 22)
    cap = strokes_to_density(img, resolution=N, max_extent=FRAME_MAX_EXTENT, pr_frac=0.244)
    assert abs(pr(cap[0, 0]) / (0.244 * N * N) - 1) < 0.006
    assert extent(cap)[0] <= FRAME_MAX_EXTENT


def test_glyph_density_2d_in_frame():
    from viot.glyphs import default_font, glyph_density_2d
    font = default_font()
    if font is None:
        raise SkipTest("DejaVu Sans (matplotlib) not available")
    for ch in "IJLfijlt1":
        d = glyph_density_2d(ch, font, N)
        ext, touching = extent(d)
        assert ext <= FRAME_MAX_EXTENT and not touching, f"{ch}: extent {ext:.3f}"
        assert border_mass(d) < 1e-4, f"{ch}: border mass {border_mass(d):.1e}"
        assert abs(pr(d[0, 0]) / TARGET - 1) < 0.006, f"{ch}: PR {pr(d[0, 0]) / TARGET:.4f}"
    assert extent(glyph_density_2d("j", font, N, max_extent=None))[1]   # the pool recipe clips 'j'
    for ch in "O8":   # the limit does not apply: exactly the pool recipe
        assert torch.equal(glyph_density_2d(ch, font, N), glyph_density_2d(ch, font, N, max_extent=None))
