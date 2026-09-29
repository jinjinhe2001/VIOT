// Hand-drawn strokes -> MNIST-distribution density, a line-by-line port of
// viot.data_2d.strokes_to_density():
//   bbox crop -> pad to square -> bilinear resize to 70% fill -> Gaussian blur
//   (sigma 2, reflect) -> floor -> mass-normalize -> participation-ratio area
//   normalization -> mass-normalize.
// The area normalization keeps the sketch within 75% of the canvas
// (fit_area_in_frame, max_extent=0.75), which leaves room for the transport;
// strokes too thin to reach the training area within it are drawn bolder
// (thicken_to_area) instead of being enlarged past the border.
// {maxExtent: null} gives the uncapped version of the paper's GUI.

const N = 256;
const MNIST_TARGET_PR_FRAC = 0.197;
const FILL_FRAC = 0.70;
const BLUR_SIGMA = 2.0;

// torch.nn.functional.interpolate(mode='bilinear', align_corners=False), square.
function resizeBilinear(src, inSize, outSize) {
  const scale = inSize / outSize;
  const idx0 = new Int32Array(outSize), idx1 = new Int32Array(outSize), lam = new Float64Array(outSize);
  for (let o = 0; o < outSize; o++) {
    let s = scale * (o + 0.5) - 0.5;
    if (s < 0) s = 0;
    const i0 = Math.floor(s);
    idx0[o] = i0;
    idx1[o] = i0 < inSize - 1 ? i0 + 1 : i0;
    lam[o] = s - i0;
  }
  const out = new Float64Array(outSize * outSize);
  for (let y = 0; y < outSize; y++) {
    const ry0 = idx0[y] * inSize, ry1 = idx1[y] * inSize, ly = lam[y];
    for (let x = 0; x < outSize; x++) {
      const c0 = idx0[x], c1 = idx1[x], lx = lam[x];
      const top = src[ry0 + c0] * (1 - lx) + src[ry0 + c1] * lx;
      const bot = src[ry1 + c0] * (1 - lx) + src[ry1 + c1] * lx;
      out[y * outSize + x] = top * (1 - ly) + bot * ly;
    }
  }
  return out;
}

// torchvision.transforms.functional.gaussian_blur (reflect padding), separable.
function gaussianBlur(img, ksize, sigma) {
  const half = (ksize - 1) / 2;
  const k = new Float64Array(ksize);
  let ks = 0;
  for (let i = 0; i < ksize; i++) {
    const x = -half + i;
    k[i] = Math.exp(-0.5 * (x / sigma) ** 2);
    ks += k[i];
  }
  for (let i = 0; i < ksize; i++) k[i] /= ks;
  const refl = (i) => (i < 0 ? -i : i >= N ? 2 * (N - 1) - i : i);
  const tmp = new Float64Array(N * N), out = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let s = 0;
      for (let t = 0; t < ksize; t++) s += k[t] * img[y * N + refl(x + t - half)];
      tmp[y * N + x] = s;
    }
  }
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let s = 0;
      for (let t = 0; t < ksize; t++) s += k[t] * tmp[refl(y + t - half) * N + x];
      out[y * N + x] = s;
    }
  }
  return out;
}

// affine_grid + grid_sample(bilinear, zeros, align_corners=False) with theta = diag(s, s).
function zoom(img, s) {
  const out = new Float64Array(N * N);
  const pos = new Float64Array(N);
  for (let j = 0; j < N; j++) pos[j] = ((s * ((2 * j + 1) / N - 1) + 1) * N - 1) / 2;
  for (let i = 0; i < N; i++) {
    const iy = pos[i], y0 = Math.floor(iy), fy = iy - y0;
    for (let j = 0; j < N; j++) {
      const ix = pos[j], x0 = Math.floor(ix), fx = ix - x0;
      let v = 0;
      for (let dy = 0; dy < 2; dy++) {
        const yy = y0 + dy;
        if (yy < 0 || yy >= N) continue;
        const wy = dy ? fy : 1 - fy;
        for (let dx = 0; dx < 2; dx++) {
          const xx = x0 + dx;
          if (xx < 0 || xx >= N) continue;
          v += img[yy * N + xx] * wy * (dx ? fx : 1 - fx);
        }
      }
      out[i * N + j] = v;
    }
  }
  return out;
}

function normalizeArea(img, targetPr, nIter = 6) {
  for (let it = 0; it < nIter; it++) {
    let total = 0, sq = 0;
    for (let i = 0; i < img.length; i++) { total += img[i]; sq += img[i] * img[i]; }
    if (total < 1e-10) return img;
    const pr = (total * total) / (sq + 1e-20);
    if (Math.abs(pr / targetPr - 1) < 0.005) break;
    img = zoom(img, Math.sqrt(pr / (targetPr + 1e-6)));
  }
  return img;
}

// Largest side of the bbox of the pixels above relThreshold * max, as a fraction of N.
function occupiedExtent(img, relThreshold) {
  let mx = 0;
  for (let i = 0; i < img.length; i++) if (img[i] > mx) mx = img[i];
  const thr = relThreshold * mx;
  let r0 = N, r1 = -1, c0 = N, c1 = -1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (img[y * N + x] > thr) {
        if (y < r0) r0 = y;
        if (y > r1) r1 = y;
        if (x < c0) c0 = x;
        if (x > c1) c1 = x;
      }
    }
  }
  return r1 < 0 ? 0 : Math.max(r1 - r0 + 1, c1 - c0 + 1) / N;
}

// fit_area_in_frame(): normalizeArea() whose zoom steps keep the occupied bbox
// within maxExtent of the canvas (identical to normalizeArea() when that limit
// never applies).
function fitAreaInFrame(img, targetPr, maxExtent, nIter = 6, relThreshold = 0.05) {
  const fit = maxExtent - 1 / N;
  for (let it = 0; it < nIter + 2; it++) {
    let total = 0, sq = 0;
    for (let i = 0; i < img.length; i++) { total += img[i]; sq += img[i] * img[i]; }
    if (total < 1e-10) return img;
    const pr = (total * total) / (sq + 1e-20);
    const ext = occupiedExtent(img, relThreshold);
    let s;
    if (it >= nIter || Math.abs(pr / targetPr - 1) < 0.005) {
      if (ext <= maxExtent) break;
      s = ext / fit;
    } else {
      s = Math.sqrt(pr / (targetPr + 1e-6));
      const sFit = ext / fit;
      if (sFit > s) {
        if (ext <= maxExtent && (1 / sFit - 1) * ext * N < 1) break;
        s = sFit;
      }
    }
    img = zoom(img, s);
  }
  return img;
}

function participationRatio(img) {
  let total = 0, sq = 0;
  for (let i = 0; i < img.length; i++) { total += img[i]; sq += img[i] * img[i]; }
  return (total * total) / (sq + 1e-20);
}

// Exact Euclidean distance from every pixel to the nearest pixel at or above
// half the peak (viot.data_2d._ink_distance): 1D distances along the columns,
// then a brute-force minimum along the rows. The ink test is made in float32,
// like the reference: resized binary strokes contain pixels of exactly half the
// peak, which float64 round-off would otherwise put on either side.
function inkDistance(img) {
  let mx = 0;
  for (let i = 0; i < img.length; i++) if (img[i] > mx) mx = img[i];
  const thr = Math.fround(0.5 * Math.fround(mx));
  const isInk = (v) => Math.fround(v) >= thr;
  const big = 2 * N * N;
  const g = new Float64Array(N * N);
  for (let x = 0; x < N; x++) {
    let last = -1;
    for (let y = 0; y < N; y++) {
      if (isInk(img[y * N + x])) last = y;
      g[y * N + x] = last < 0 ? Infinity : y - last;
    }
    last = -1;
    for (let y = N - 1; y >= 0; y--) {
      if (isInk(img[y * N + x])) last = y;
      if (last >= 0 && last - y < g[y * N + x]) g[y * N + x] = last - y;
    }
    for (let y = 0; y < N; y++) {
      const d = g[y * N + x];
      g[y * N + x] = d === Infinity ? big : d * d;
    }
  }
  const dist = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let m = Infinity;
      for (let x2 = 0; x2 < N; x2++) {
        const v = g[y * N + x2] + (x - x2) * (x - x2);
        if (v < m) m = v;
      }
      dist[y * N + x] = Math.sqrt(m);
    }
  }
  return dist;
}

// thicken_to_area(): if render(img) falls short of the target area, raise every
// pixel within r of the ink to the peak (one-pixel linear edge) with the smallest
// r (bisection to 1/32 px) for which render() reaches it.
function thickenToArea(img, render, targetPr, tol = 0.005) {
  const out = render(img);
  if (participationRatio(out) >= (1 - tol) * targetPr) return out;
  let peak = 0;
  for (let i = 0; i < img.length; i++) if (img[i] > peak) peak = img[i];
  const dist = inkDistance(img);
  let lo = 0, hi = null, best = null, r = 1;
  for (let it = 0; it < 64; it++) {
    const t = new Float64Array(img.length);
    for (let i = 0; i < img.length; i++) {
      const ink = Math.min(Math.max(r + 0.5 - dist[i], 0), 1) * peak;
      t[i] = img[i] > ink ? img[i] : ink;
    }
    const y = render(t);
    if (participationRatio(y) < (1 - tol) * targetPr) lo = r;
    else { hi = r; best = y; }
    if (hi === null) {
      if (r >= N) { best = y; break; }
      r *= 2;
    } else if (hi - lo < 1 / 32) {
      break;
    } else {
      r = 0.5 * (lo + hi);
    }
  }
  return best;
}

function massNormalize(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  s += 1e-12;
  for (let i = 0; i < a.length; i++) a[i] /= s;
  return a;
}

/**
 * arr: Float32Array(256*256) stroke intensities in [0, 1]. Returns Float32Array or null.
 * maxExtent: largest size of the sketch as a fraction of the canvas; null for the
 * uncapped area normalization.
 */
export function strokesToDensity(arr, { maxExtent = 0.75 } = {}) {
  let total = 0;
  for (let i = 0; i < arr.length; i++) total += arr[i];
  if (total < 1.0) return null;
  let r0 = N, r1 = -1, c0 = N, c1 = -1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (arr[y * N + x] > 0) {
        if (y < r0) r0 = y;
        if (y > r1) r1 = y;
        if (x < c0) c0 = x;
        if (x > c1) c1 = x;
      }
    }
  }
  const h = r1 - r0 + 1, w = c1 - c0 + 1, side = Math.max(h, w);
  const sq = new Float64Array(side * side);
  const oy = Math.floor((side - h) / 2), ox = Math.floor((side - w) / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) sq[(oy + y) * side + ox + x] = arr[(r0 + y) * N + c0 + x];

  const fill = Math.max(Math.round(N * FILL_FRAC), 8);
  const res = resizeBilinear(sq, side, fill);
  let canvas = new Float64Array(N * N);
  const pad = Math.floor((N - fill) / 2);
  for (let y = 0; y < fill; y++) for (let x = 0; x < fill; x++) canvas[(pad + y) * N + pad + x] = res[y * fill + x];

  const ksize = Math.max((Math.trunc(BLUR_SIGMA * 6)) | 1, 3);
  const targetPr = MNIST_TARGET_PR_FRAC * N * N;
  const capped = maxExtent > 0;
  const render = (c) => {
    c = gaussianBlur(c, ksize, BLUR_SIGMA);
    for (let i = 0; i < c.length; i++) c[i] = Math.max(c[i], 0) + 1e-5;
    massNormalize(c);
    c = capped ? fitAreaInFrame(c, targetPr, maxExtent) : normalizeArea(c, targetPr);
    for (let i = 0; i < c.length; i++) c[i] = Math.max(c[i], 0);
    return massNormalize(c);
  };
  canvas = capped ? thickenToArea(canvas, render, targetPr) : render(canvas);
  return Float32Array.from(canvas);
}
