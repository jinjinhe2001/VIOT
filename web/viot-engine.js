// VIOT in the browser: WebGPU inference for the 2D MNIST operator (256^2).
//
// The network is the paper's FNO stream-function operator (width 64, 32 modes,
// 8 layers). Spectral convolutions are evaluated as truncated DFT products,
// the lift is folded into the first spectral layer, and the spectral weights of
// layers 1-7 are int4/int8 with per-(input channel, mode) f16 scales. Rollouts
// use MacCormack advection with mass renormalization, exactly as in the paper.

const N = 256, NN = N * N, C = 64, KW = 32, KH = 64, MODES = KH * KW, NL = 8;
const KR = 64, PKH = 2 * KR + 1, PKW = KR + 1;   // final psi band: kh -64..64, kw 0..64
const MAX_STEPS = 400;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
export async function fetchWeights(url, onProgress = () => {}, { cacheName = 'viot-weights-v1' } = {}) {
  let cache = null;
  try { cache = await caches.open(cacheName); } catch { /* Cache API unavailable (e.g. file://) */ }
  let resp = cache ? await cache.match(url) : null;
  const fromCache = !!resp;
  if (!resp) {
    resp = await fetch(url, { mode: 'cors' });
    if (!resp.ok) throw new Error(`Failed to download weights (${resp.status})`);
  }
  const total = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.clone().body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total, fromCache);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache && !fromCache) {
    try { await cache.put(url, new Response(buf, { headers: resp.headers })); } catch { /* quota */ }
  }
  return buf.buffer;
}

export function parseSafetensors(buffer) {
  const dv = new DataView(buffer);
  const hlen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, hlen)));
  const base = 8 + hlen;
  const T = {};
  const ctor = { F32: Float32Array, F16: Uint16Array, U32: Uint32Array, I8: Int8Array, U8: Uint8Array };
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [b, e] = info.data_offsets;
    const Ctor = ctor[info.dtype];
    if (!Ctor) throw new Error(`unsupported dtype ${info.dtype}`);
    T[name] = { shape: info.shape, dtype: info.dtype, data: new Ctor(buffer, base + b, (e - b) / Ctor.BYTES_PER_ELEMENT) };
  }
  return { tensors: T, meta: header.__metadata__ || {} };
}

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------
const PRELUDE = /* wgsl */`
fn erf_(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
          t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, x >= 0.0);
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
`;

// C[M x Nc] = A[M x K] * B[K x Nc], 64x64 tiles, 4x4 outputs per thread.
function gemmWGSL(epi) {
  const bindings = {
    none: '',
    local: `@group(0) @binding(4) var<storage, read> Y: array<f32>;
            @group(0) @binding(5) var<storage, read> SV: array<f32>;
            @group(0) @binding(6) var<storage, read> FILM: array<f32>;
            @group(0) @binding(7) var<uniform> S: vec4u;`,
    gelu: `@group(0) @binding(4) var<storage, read> BIAS: array<f32>;`,
  }[epi];
  const body = {
    none: `Cm[idx] = v;`,
    local: `let fo = (S.x * ${NL}u + P.layer) * 128u;
            let h = v + SV[P.layer * 64u + r] + Y[idx];
            Cm[idx] = h * (1.0 + FILM[fo + r]) + FILM[fo + 64u + r];`,
    gelu: `Cm[idx] = gelu(v + BIAS[r]);`,
  }[epi];
  return PRELUDE + /* wgsl */`
struct GP { M: u32, Nc: u32, K: u32, lda: u32, ldb: u32, ldc: u32, layer: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Cm: array<f32>;
@group(0) @binding(3) var<uniform> P: GP;
${bindings}
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row0 = wg.y * 64u;
  let col0 = wg.x * 64u;
  let tid = lid.y * 16u + lid.x;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < P.K; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      let r = e >> 4u; let kk = e & 15u;
      let gr = row0 + r; let gk = k0 + kk;
      var a = 0.0;
      if (gr < P.M && gk < P.K) { a = A[gr * P.lda + gk]; }
      As[kk * 64u + r] = a;
      let kb = e >> 6u; let c = e & 63u;
      let gkb = k0 + kb; let gc = col0 + c;
      var b = 0.0;
      if (gkb < P.K && gc < P.Nc) { b = B[gkb * P.ldb + gc]; }
      Bs[kb * 64u + c] = b;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let a0 = As[kk * 64u + lid.y]; let a1 = As[kk * 64u + lid.y + 16u];
      let a2 = As[kk * 64u + lid.y + 32u]; let a3 = As[kk * 64u + lid.y + 48u];
      let b0 = Bs[kk * 64u + lid.x]; let b1 = Bs[kk * 64u + lid.x + 16u];
      let b2 = Bs[kk * 64u + lid.x + 32u]; let b3 = Bs[kk * 64u + lid.x + 48u];
      acc[0] = fma(a0, b0, acc[0]);   acc[1] = fma(a0, b1, acc[1]);
      acc[2] = fma(a0, b2, acc[2]);   acc[3] = fma(a0, b3, acc[3]);
      acc[4] = fma(a1, b0, acc[4]);   acc[5] = fma(a1, b1, acc[5]);
      acc[6] = fma(a1, b2, acc[6]);   acc[7] = fma(a1, b3, acc[7]);
      acc[8] = fma(a2, b0, acc[8]);   acc[9] = fma(a2, b1, acc[9]);
      acc[10] = fma(a2, b2, acc[10]); acc[11] = fma(a2, b3, acc[11]);
      acc[12] = fma(a3, b0, acc[12]); acc[13] = fma(a3, b1, acc[13]);
      acc[14] = fma(a3, b2, acc[14]); acc[15] = fma(a3, b3, acc[15]);
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = row0 + lid.y + 16u * i;
    if (r >= P.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let c = col0 + lid.x + 16u * j;
      if (c >= P.Nc) { continue; }
      let v = acc[i * 4u + j];
      let idx = r * P.ldc + c;
      ${body}
    }
  }
}`;
}

// OUT[b][P x Q] = F[P x R] * IN[b][R x Q]  (complex, interleaved)
const CMUL_WGSL = PRELUDE + /* wgsl */`
struct CP { P: u32, Q: u32, R: u32, sIn: u32, sOut: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> F: array<vec2f>;
@group(0) @binding(1) var<storage, read> IN: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> OUT: array<vec2f>;
@group(0) @binding(3) var<uniform> U: CP;
var<workgroup> Fs: array<vec2f, 256>;
var<workgroup> Is: array<vec2f, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let q = wg.x * 16u + lid.x;
  let p = wg.y * 16u + lid.y;
  let ib = wg.z * U.sIn;
  let ob = wg.z * U.sOut;
  var acc = vec2f(0.0);
  for (var r0 = 0u; r0 < U.R; r0 += 16u) {
    let fr = r0 + lid.x;
    var f = vec2f(0.0);
    if (p < U.P && fr < U.R) { f = F[p * U.R + fr]; }
    Fs[lid.y * 16u + lid.x] = f;
    let ir = r0 + lid.y;
    var x = vec2f(0.0);
    if (ir < U.R && q < U.Q) { x = IN[ib + ir * U.Q + q]; }
    Is[lid.y * 16u + lid.x] = x;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      acc += cmul(Fs[lid.y * 16u + k], Is[k * 16u + lid.x]);
    }
    workgroupBarrier();
  }
  if (p < U.P && q < U.Q) { OUT[ob + p * U.Q + q] = acc; }
}`;

function modeMixWGSL(bits) {
  const wpc = 64 * bits / 32;           // u32 words per (mode, c) row of 64 outputs
  const per = 32 / bits;                // values per word
  return PRELUDE + /* wgsl */`
@group(0) @binding(0) var<storage, read> XH: array<vec2f>;
@group(0) @binding(1) var<storage, read> WR: array<u32>;
@group(0) @binding(2) var<storage, read> WI: array<u32>;
@group(0) @binding(3) var<storage, read> SR: array<u32>;
@group(0) @binding(4) var<storage, read> SI: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec2f>;
var<workgroup> xs: array<vec2f, 64>;
var<workgroup> sr: array<f32, 64>;
var<workgroup> si: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let mode = wg.x;
  let d = lid.x;
  xs[d] = XH[d * ${MODES}u + mode];
  let s = mode * 64u + d;
  sr[d] = unpack2x16float(SR[s >> 1u])[s & 1u];
  si[d] = unpack2x16float(SI[s >> 1u])[s & 1u];
  workgroupBarrier();
  let wb = mode * ${64 * wpc}u + d / ${per}u;
  let sh = ${32 - bits}u - ${bits}u * (d % ${per}u);
  var acc = vec2f(0.0);
  for (var c = 0u; c < 64u; c++) {
    let qr = f32(bitcast<i32>(WR[wb + c * ${wpc}u] << sh) >> ${32 - bits}u);
    let qi = f32(bitcast<i32>(WI[wb + c * ${wpc}u] << sh) >> ${32 - bits}u);
    acc += cmul(xs[c], vec2f(qr * sr[c], qi * si[c]));
  }
  Y[d * ${MODES}u + mode] = acc;
}`;
}

const MODEMIX0_WGSL = PRELUDE + /* wgsl */`
@group(0) @binding(0) var<storage, read> XH: array<vec2f>;
@group(0) @binding(1) var<storage, read> W0: array<vec2f>;
@group(0) @binding(2) var<storage, read> DC: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec2f>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let mode = wg.x;
  let d = lid.x;
  var acc = cmul(XH[mode], W0[(mode * 2u) * 64u + d]) + cmul(XH[${MODES}u + mode], W0[(mode * 2u + 1u) * 64u + d]);
  if (mode == 0u) { acc += DC[d]; }
  Y[d * ${MODES}u + mode] = acc;
}`;

const LIFT_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> R2: array<f32>;
@group(0) @binding(1) var<storage, read> LW: array<f32>;   // [64][2] then bias [64]
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.x + g.y * 4194304u;
  let c = idx / ${NN}u;
  let p = idx % ${NN}u;
  X[idx] = LW[c * 2u] * R2[p] + LW[c * 2u + 1u] * R2[${NN}u + p] + LW[128u + c];
}`;

const GN_STATS1_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> H: array<f32>;
@group(0) @binding(1) var<storage, read_write> PART: array<vec2f>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let base = wg.y * ${8 * NN}u + wg.x * 2048u;
  var v: array<f32, 8>;
  var s = 0.0;
  for (var i = 0u; i < 8u; i++) { v[i] = H[base + lid.x + i * 256u]; s += v[i]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
    workgroupBarrier();
  }
  let mean = red[0] / 2048.0;
  workgroupBarrier();
  var m2 = 0.0;
  for (var i = 0u; i < 8u; i++) { let d = v[i] - mean; m2 += d * d; }
  red[lid.x] = m2;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { PART[wg.y * 256u + wg.x] = vec2f(mean, red[0]); }
}`;

const GN_STATS2_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> PART: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> ST: array<vec2f>;
var<workgroup> mu: array<f32, 256>;
var<workgroup> m2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let p = PART[wg.x * 256u + lid.x];
  mu[lid.x] = p.x; m2[lid.x] = p.y;
  workgroupBarrier();
  var n = 2048.0;
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid.x < k) {
      let d = mu[lid.x + k] - mu[lid.x];
      mu[lid.x] = mu[lid.x] + 0.5 * d;
      m2[lid.x] = m2[lid.x] + m2[lid.x + k] + d * d * n * 0.5;
    }
    n = n * 2.0;
    workgroupBarrier();
  }
  if (lid.x == 0u) { ST[wg.x] = vec2f(mu[0], inverseSqrt(m2[0] / ${8 * NN}.0 + 1e-5)); }
}`;

const GN_APPLY_WGSL = PRELUDE + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> H: array<f32>;
@group(0) @binding(2) var<storage, read> ST: array<vec2f>;
@group(0) @binding(3) var<storage, read> GB: array<f32>;
@group(0) @binding(4) var<uniform> L: vec4u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.x + g.y * 4194304u;
  let c = idx / ${NN}u;
  let st = ST[c / 8u];
  let xn = (H[idx] - st.x) * st.y * GB[L.x * 64u + c] + GB[${NL * 64}u + L.x * 64u + c];
  X[idx] = X[idx] + gelu(xn);
}`;

const PSI_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> HID: array<f32>;
@group(0) @binding(1) var<storage, read> W2: array<f32>;   // [64] then bias
@group(0) @binding(2) var<storage, read_write> PSI: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let p = g.x;
  var s = W2[64];
  for (var d = 0u; d < 64u; d++) { s = fma(W2[d], HID[d * ${NN}u + p], s); }
  PSI[p] = s;
}`;

const CURL_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> PH: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> VS: array<vec2f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= ${PKH * PKW}u) { return; }
  let kh = i32(i / ${PKW}u) - ${KR};
  let kw = i32(i % ${PKW}u);
  var vh = vec2f(0.0);
  var vw = vec2f(0.0);
  if (kh * kh + kw * kw <= ${KR * KR}) {
    let p = PH[i];
    let a = 6.283185307179586 * f32(kw) / ${N}.0;
    let b = 6.283185307179586 * f32(kh) / ${N}.0;
    vh = vec2f(-a * p.y, a * p.x);
    vw = vec2f(b * p.y, -b * p.x);
  }
  VS[i] = vh;
  VS[${PKH * PKW}u + i] = vw;
}`;

// Semi-Lagrangian midpoint pass (grid_sample bilinear, border padding).
const ADVECT_WGSL = /* wgsl */`
struct AP { dt: f32, pass2: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> V: array<f32>;
@group(0) @binding(1) var<storage, read> SRC: array<f32>;    // field to transport
@group(0) @binding(2) var<storage, read> RHO: array<f32>;    // rho^n (pass 2 only)
@group(0) @binding(3) var<storage, read> RHAT: array<f32>;   // rho_hat (pass 2 only)
@group(0) @binding(4) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(5) var<uniform> U: AP;
struct Bil { i00: u32, i01: u32, i10: u32, i11: u32, fx: f32, fy: f32 };
fn bil(x: f32, y: f32) -> Bil {
  let xc = clamp(x, 0.0, ${N - 1}.0);
  let yc = clamp(y, 0.0, ${N - 1}.0);
  let x0 = floor(xc); let y0 = floor(yc);
  let ix0 = u32(x0); let iy0 = u32(y0);
  let ix1 = min(ix0 + 1u, ${N - 1}u); let iy1 = min(iy0 + 1u, ${N - 1}u);
  return Bil(iy0 * ${N}u + ix0, iy0 * ${N}u + ix1, iy1 * ${N}u + ix0, iy1 * ${N}u + ix1, xc - x0, yc - y0);
}
fn lerp4(a: f32, b: f32, c: f32, d: f32, w: Bil) -> f32 {
  return (a * (1.0 - w.fx) + b * w.fx) * (1.0 - w.fy) + (c * (1.0 - w.fx) + d * w.fx) * w.fy;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let p = g.x;
  let i = f32(p / ${N}u);
  let j = f32(p % ${N}u);
  let dt = U.dt;
  let vy = V[p]; let vx = V[${NN}u + p];
  let m = bil(j - vx * 0.5 * dt, i - vy * 0.5 * dt);
  let vmy = lerp4(V[m.i00], V[m.i01], V[m.i10], V[m.i11], m);
  let vmx = lerp4(V[${NN}u + m.i00], V[${NN}u + m.i01], V[${NN}u + m.i10], V[${NN}u + m.i11], m);
  let b = bil(j - vmx * dt, i - vmy * dt);
  let s = lerp4(SRC[b.i00], SRC[b.i01], SRC[b.i10], SRC[b.i11], b);
  if (U.pass2 == 0u) {
    OUT[p] = s;
  } else {
    OUT[p] = max(RHAT[p] + 0.5 * (RHO[p] - s), 0.0);
  }
}`;

const SUM1_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> SRC: array<f32>;
@group(0) @binding(1) var<storage, read_write> PART: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  red[lid.x] = SRC[wg.x * 256u + lid.x];
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { PART[wg.x] = red[0]; }
}`;

const SUM2_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> PART: array<f32>;
@group(0) @binding(1) var<storage, read_write> MASS: array<f32>;
@group(0) @binding(2) var<uniform> U: vec4u;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  red[lid.x] = PART[lid.x];
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid.x < k) { red[lid.x] += red[lid.x + k]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { MASS[U.x] = red[0]; }
}`;

const NORMALIZE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> SRC: array<f32>;
@group(0) @binding(1) var<storage, read> MASS: array<f32>;
@group(0) @binding(2) var<storage, read_write> R2: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  R2[g.x] = SRC[g.x] * MASS[0] / (MASS[1] + 1e-12);
}`;

// ---------------------------------------------------------------------------
// DFT matrices (float64 construction, exact angle reduction)
// ---------------------------------------------------------------------------
function twiddle(k, n, sign) {
  const a = 2 * Math.PI * (((k * n) % N + N) % N) / N;
  return [Math.cos(a), sign * Math.sin(a)];
}
function buildMatrices() {
  const khList = Array.from({ length: KH }, (_, j) => (j < KW ? j : j - KH));
  const Fw = new Float32Array(N * KW * 2);                // [w][2k..] cos, -sin; scaled 1/N
  for (let w = 0; w < N; w++) for (let k = 0; k < KW; k++) {
    const [c, s] = twiddle(k, w, -1);
    Fw[w * 2 * KW + 2 * k] = c / N; Fw[w * 2 * KW + 2 * k + 1] = s / N;
  }
  const FHf = new Float32Array(KH * N * 2);               // [j][h] e^{-i}
  const FHi = new Float32Array(N * KH * 2);               // [h][j] e^{+i} / N
  for (let j = 0; j < KH; j++) for (let h = 0; h < N; h++) {
    const [c, s] = twiddle(khList[j], h, -1);
    FHf[(j * N + h) * 2] = c; FHf[(j * N + h) * 2 + 1] = s;
    FHi[(h * KH + j) * 2] = c / N; FHi[(h * KH + j) * 2 + 1] = -s / N;
  }
  const Gw = new Float32Array(2 * KW * N);                // [2k / 2k+1][w]: c_k cos, -c_k sin
  for (let k = 0; k < KW; k++) for (let w = 0; w < N; w++) {
    const ck = k === 0 ? 1 : 2;
    const [c, s] = twiddle(k, w, 1);
    Gw[(2 * k) * N + w] = ck * c; Gw[(2 * k + 1) * N + w] = -ck * s;
  }
  // final psi -> velocity band
  const PFw = new Float32Array(N * PKW * 2);
  for (let w = 0; w < N; w++) for (let k = 0; k < PKW; k++) {
    const [c, s] = twiddle(k, w, -1);
    PFw[w * 2 * PKW + 2 * k] = c; PFw[w * 2 * PKW + 2 * k + 1] = s;
  }
  const PFHf = new Float32Array(PKH * N * 2);
  const PFHi = new Float32Array(N * PKH * 2);
  for (let j = 0; j < PKH; j++) for (let h = 0; h < N; h++) {
    const [c, s] = twiddle(j - KR, h, -1);
    PFHf[(j * N + h) * 2] = c; PFHf[(j * N + h) * 2 + 1] = s;
    PFHi[(h * PKH + j) * 2] = c; PFHi[(h * PKH + j) * 2 + 1] = -s;
  }
  const PGw = new Float32Array(2 * PKW * N);
  for (let k = 0; k < PKW; k++) for (let w = 0; w < N; w++) {
    const ck = k === 0 ? 1 : 2;
    const [c, s] = twiddle(k, w, 1);
    PGw[(2 * k) * N + w] = ck * c / NN; PGw[(2 * k + 1) * N + w] = -ck * s / NN;
  }
  return { Fw, FHf, FHi, Gw, PFw, PFHf, PFHi, PGw };
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
export class VIOTEngine {
  static async create(buffer, { powerPreference = 'high-performance' } = {}) {
    if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage) },
    });
    const eng = new VIOTEngine();
    eng.adapterInfo = adapter.info || {};
    eng.device = device;
    eng.lost = device.lost;
    eng._init(parseSafetensors(buffer));
    return eng;
  }

  _buf(data, usage = GPUBufferUsage.STORAGE) {
    const size = Math.max(16, Math.ceil(data.byteLength / 4) * 4);
    const b = this.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    b.unmap();
    return b;
  }
  _empty(bytes, usage = GPUBufferUsage.STORAGE) {
    return this.device.createBuffer({ size: Math.max(16, bytes), usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  }
  _uniform(u32s) { return this._buf(new Uint32Array(u32s), GPUBufferUsage.UNIFORM); }
  _pipe(code) {
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  _bg(pipe, bufs) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: bufs.map((b, i) => (b.buffer ? { binding: i, resource: b } : { binding: i, resource: { buffer: b } })),
    });
  }

  _init({ tensors: T, meta }) {
    this.meta = meta;
    this.T = T;
    const f32 = (name) => T[name].data;
    const bits = T['l1.real.q4'] ? 4 : 8;
    this.bits = bits;

    // --- constant matrices
    const M = buildMatrices();
    const B = {};
    for (const [k, v] of Object.entries(M)) B[k] = this._buf(v);

    // --- small parameters
    const lw = new Float32Array(64 * 2 + 64);
    lw.set(f32('lift.weight'), 0); lw.set(f32('lift.bias'), 128);
    B.lift = this._buf(lw);
    const locW = new Float32Array(NL * 64 * 64), locB = new Float32Array(NL * 64), gb = new Float32Array(2 * NL * 64);
    for (let l = 0; l < NL; l++) {
      locW.set(f32(`fno_layers.${l}.local_conv.weight`), l * 4096);
      locB.set(f32(`fno_layers.${l}.local_conv.bias`), l * 64);
      gb.set(f32(`fno_layers.${l}.norm.weight`), l * 64);
      gb.set(f32(`fno_layers.${l}.norm.bias`), NL * 64 + l * 64);
    }
    B.locW = this._buf(locW); B.locB = this._buf(locB); B.gb = this._buf(gb);
    B.p1w = this._buf(f32('project.0.weight')); B.p1b = this._buf(f32('project.0.bias'));
    const p2 = new Float32Array(65); p2.set(f32('project.2.weight'), 0); p2[64] = f32('project.2.bias')[0];
    B.p2 = this._buf(p2);
    B.w0 = this._buf(f32('l0.w')); B.dc = this._buf(f32('l0.dc'));
    B.wr = [], B.wi = [], B.sr = [], B.si = [];
    const q = bits === 4 ? 'q4' : 'q8';
    for (let l = 1; l < NL; l++) {
      B.wr[l] = this._buf(T[`l${l}.real.${q}`].data); B.wi[l] = this._buf(T[`l${l}.imag.${q}`].data);
      B.sr[l] = this._buf(T[`l${l}.real.scale`].data); B.si[l] = this._buf(T[`l${l}.imag.scale`].data);
    }

    // --- activations
    B.R2 = this._empty(2 * NN * 4);
    B.XW = this._empty(C * N * KW * 8);
    B.XH = this._empty(C * MODES * 8);
    B.YHAT = this._empty(C * MODES * 8);
    B.YH = this._empty(C * N * KW * 8);
    B.Y = this._empty(C * NN * 4);
    B.X = this._empty(C * NN * 4);
    B.H = this._empty(C * NN * 4);
    B.GNP = this._empty(8 * 256 * 8);
    B.GNS = this._empty(8 * 8);
    B.PSI = this._empty(NN * 4);
    B.PW = this._empty(N * PKW * 8);
    B.PH = this._empty(PKH * PKW * 8);
    B.VS = this._empty(2 * PKH * PKW * 8);
    B.VZ = this._empty(2 * N * PKW * 8);
    B.V = this._empty(2 * NN * 4);
    B.RA = this._empty(NN * 4);
    B.RB = this._empty(NN * 4);
    B.SUMP = this._empty(256 * 4);
    B.MASS = this._empty(16);
    B.FILM = this._empty(MAX_STEPS * NL * 128 * 4);
    B.step = this._empty(16, GPUBufferUsage.UNIFORM);
    B.adv = this._empty(16, GPUBufferUsage.UNIFORM);
    B.adv2 = this._empty(16, GPUBufferUsage.UNIFORM);
    this.B = B;

    // --- pipelines
    const P = this.P = {
      gemm: this._pipe(gemmWGSL('none')),
      gemmLocal: this._pipe(gemmWGSL('local')),
      gemmGelu: this._pipe(gemmWGSL('gelu')),
      cmul: this._pipe(CMUL_WGSL),
      mix: this._pipe(modeMixWGSL(bits)),
      mix0: this._pipe(MODEMIX0_WGSL),
      lift: this._pipe(LIFT_WGSL),
      gn1: this._pipe(GN_STATS1_WGSL),
      gn2: this._pipe(GN_STATS2_WGSL),
      gnApply: this._pipe(GN_APPLY_WGSL),
      psi: this._pipe(PSI_WGSL),
      curl: this._pipe(CURL_WGSL),
      advect: this._pipe(ADVECT_WGSL),
      sum1: this._pipe(SUM1_WGSL),
      sum2: this._pipe(SUM2_WGSL),
      norm: this._pipe(NORMALIZE_WGSL),
    };

    // --- uniforms (GEMM: M, Nc, K, lda, ldb, ldc, layer)
    const U = {
      fwdW0: this._uniform([2 * N, 2 * KW, N, N, 2 * KW, 2 * KW, 0, 0]),
      fwdW: this._uniform([C * N, 2 * KW, N, N, 2 * KW, 2 * KW, 0, 0]),
      invW: this._uniform([C * N, N, 2 * KW, 2 * KW, N, N, 0, 0]),
      local: Array.from({ length: NL }, (_, l) => this._uniform([C, NN, C, C, NN, NN, l, 0])),
      proj: this._uniform([C, NN, C, C, NN, NN, 0, 0]),
      pFwdW: this._uniform([N, 2 * PKW, N, N, 2 * PKW, 2 * PKW, 0, 0]),
      pInvW: this._uniform([2 * N, N, 2 * PKW, 2 * PKW, N, N, 0, 0]),
      // CMUL: P, Q, R, strideIn, strideOut
      fwdH: this._uniform([KH, KW, N, N * KW, KH * KW, 0, 0, 0]),
      invH: this._uniform([N, KW, KH, KH * KW, N * KW, 0, 0, 0]),
      pFwdH: this._uniform([PKH, PKW, N, 0, 0, 0, 0, 0]),
      pInvH: this._uniform([N, PKW, PKH, PKH * PKW, N * PKW, 0, 0, 0]),
      layer: Array.from({ length: NL }, (_, l) => this._uniform([l, 0, 0, 0])),
      slot0: this._uniform([0, 0, 0, 0]),
      slot1: this._uniform([1, 0, 0, 0]),
    };
    this.U = U;

    // --- bind groups
    const G = this.G = {};
    const r2c0 = { buffer: B.R2, offset: 0, size: NN * 4 };
    G.fwdW0 = this._bg(P.gemm, [B.R2, B.Fw, B.XW, U.fwdW0]);
    G.fwdH0 = this._bg(P.cmul, [B.FHf, B.XW, B.XH, U.fwdH]);
    G.mix0 = this._bg(P.mix0, [B.XH, B.w0, B.dc, B.YHAT]);
    G.fwdW = this._bg(P.gemm, [B.X, B.Fw, B.XW, U.fwdW]);
    G.fwdH = G.fwdH0;
    G.mix = [];
    for (let l = 1; l < NL; l++) G.mix[l] = this._bg(P.mix, [B.XH, B.wr[l], B.wi[l], B.sr[l], B.si[l], B.YHAT]);
    G.invH = this._bg(P.cmul, [B.FHi, B.YHAT, B.YH, U.invH]);
    G.invW = this._bg(P.gemm, [B.YH, B.Gw, B.Y, U.invW]);
    G.lift = this._bg(P.lift, [B.R2, B.lift, B.X]);
    G.local = Array.from({ length: NL }, (_, l) => this._bg(P.gemmLocal, [
      { buffer: B.locW, offset: l * 16384, size: 16384 }, B.X, B.H, U.local[l], B.Y, B.locB, B.FILM, B.step]));
    G.gn1 = this._bg(P.gn1, [B.H, B.GNP]);
    G.gn2 = this._bg(P.gn2, [B.GNP, B.GNS]);
    G.gnApply = Array.from({ length: NL }, (_, l) => this._bg(P.gnApply, [B.X, B.H, B.GNS, B.gb, U.layer[l]]));
    G.proj = this._bg(P.gemmGelu, [B.p1w, B.X, B.H, U.proj, B.p1b]);
    G.psi = this._bg(P.psi, [B.H, B.p2, B.PSI]);
    G.pFwdW = this._bg(P.gemm, [B.PSI, B.PFw, B.PW, U.pFwdW]);
    G.pFwdH = this._bg(P.cmul, [B.PFHf, B.PW, B.PH, U.pFwdH]);
    G.curl = this._bg(P.curl, [B.PH, B.VS]);
    G.pInvH = this._bg(P.cmul, [B.PFHi, B.VS, B.VZ, U.pInvH]);
    G.pInvW = this._bg(P.gemm, [B.VZ, B.PGw, B.V, U.pInvW]);
    // pass 1 ignores RHO/RHAT, so those slots get harmless read-only aliases
    G.adv1 = this._bg(P.advect, [B.V, r2c0, r2c0, B.V, B.RA, B.adv]);
    G.adv2 = this._bg(P.advect, [B.V, B.RA, r2c0, B.RA, B.RB, B.adv2]);
    G.sumRB = this._bg(P.sum1, [B.RB, B.SUMP]);
    G.sumR0 = this._bg(P.sum1, [r2c0, B.SUMP]);
    G.sum2m0 = this._bg(P.sum2, [B.SUMP, B.MASS, U.slot0]);
    G.sum2tot = this._bg(P.sum2, [B.SUMP, B.MASS, U.slot1]);
    G.norm = this._bg(P.norm, [B.RB, B.MASS, r2c0]);
    this._staging = [];
  }

  // Time conditioning for every step, computed on the CPU (tiny MLPs).
  _film(nSteps) {
    const W = (k) => this.T[k].data;
    const w0 = W('time_mlp.0.weight'), b0 = W('time_mlp.0.bias'), w2 = W('time_mlp.2.weight'), b2 = W('time_mlp.2.bias');
    const out = new Float32Array(nSteps * NL * 128);
    const silu = (x) => x / (1 + Math.exp(-x));
    for (let s = 0; s < nSteps; s++) {
      const t = Math.fround(s / nSteps);
      const temb = new Float64Array(128);
      for (let i = 0; i < 64; i++) {
        const f = Math.fround(Math.exp(i * -(Math.log(10000) / 64)));
        const a = Math.fround(t * f);
        temb[i] = Math.sin(a); temb[64 + i] = Math.cos(a);
      }
      const h = new Float64Array(256);
      for (let o = 0; o < 256; o++) {
        let acc = b0[o];
        for (let i = 0; i < 128; i++) acc += w0[o * 128 + i] * temb[i];
        h[o] = silu(acc);
      }
      const cs = new Float64Array(128);
      for (let o = 0; o < 128; o++) {
        let acc = b2[o];
        for (let i = 0; i < 256; i++) acc += w2[o * 256 + i] * h[i];
        cs[o] = silu(acc);
      }
      for (let l = 0; l < NL; l++) {
        const fw = W(`fno_layers.${l}.film.1.weight`), fb = W(`fno_layers.${l}.film.1.bias`);
        for (let o = 0; o < 128; o++) {
          let acc = fb[o];
          for (let i = 0; i < 128; i++) acc += fw[o * 128 + i] * cs[i];
          out[(s * NL + l) * 128 + o] = acc;
        }
      }
    }
    return out;
  }

  // One network evaluation (velocity into B.V), encoded into an open compute pass.
  _encodeForward(pass, stopAt = null) {
    const { P, G } = this;
    const run = (pipe, bg, x, y = 1, z = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(x, y, z); };
    // layer 0: folded lift + spectral conv on the two input channels
    run(P.gemm, G.fwdW0, 1, (2 * N) / 64);
    run(P.cmul, G.fwdH0, KW / 16, KH / 16, 2);
    run(P.mix0, G.mix0, MODES);
    for (let l = 0; l < NL; l++) {
      if (l > 0) {
        run(P.gemm, G.fwdW, 1, (C * N) / 64);
        run(P.cmul, G.fwdH, KW / 16, KH / 16, C);
        run(P.mix, G.mix[l], MODES);
      }
      run(P.cmul, G.invH, KW / 16, N / 16, C);
      run(P.gemm, G.invW, N / 64, (C * N) / 64);
      if (l === 0) run(P.lift, G.lift, 16384);
      if (stopAt === 'spec0') return;
      run(P.gemmLocal, G.local[l], NN / 64, 1);
      if (stopAt === 'h0') return;
      run(P.gn1, G.gn1, 256, 8);
      run(P.gn2, G.gn2, 8);
      run(P.gnApply, G.gnApply[l], 16384);
      if (stopAt === `x${l}`) return;
    }
    run(P.gemmGelu, G.proj, NN / 64, 1);
    run(P.psi, G.psi, NN / 256);
    if (stopAt === 'psi') return;
    run(P.gemm, G.pFwdW, Math.ceil((2 * PKW) / 64), N / 64);
    run(P.cmul, G.pFwdH, Math.ceil(PKW / 16), Math.ceil(PKH / 16), 1);
    run(P.curl, G.curl, Math.ceil((PKH * PKW) / 256));
    run(P.cmul, G.pInvH, Math.ceil(PKW / 16), N / 16, 2);
    run(P.gemm, G.pInvW, N / 64, (2 * N) / 64);
  }

  _encodeAdvect(pass) {
    const { P, G } = this;
    const run = (pipe, bg, x) => { pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(x); };
    run(P.advect, G.adv1, NN / 256);
    run(P.advect, G.adv2, NN / 256);
    run(P.sum1, G.sumRB, NN / 256);
    run(P.sum2, G.sum2tot, 1);
    run(P.norm, G.norm, NN / 256);
  }

  _setInputs(rho0, rho1, nSteps) {
    const q = this.device.queue;
    q.writeBuffer(this.B.R2, 0, rho0);
    q.writeBuffer(this.B.R2, NN * 4, rho1);
    if (this._filmSteps !== nSteps) {
      q.writeBuffer(this.B.FILM, 0, this._film(nSteps));
      this._filmSteps = nSteps;
    }
    const dt = 1 / nSteps;
    q.writeBuffer(this.B.adv, 0, new Float32Array([dt, 0, 0, 0]));
    const a2 = new ArrayBuffer(16);
    new Float32Array(a2, 0, 1)[0] = -dt;
    new Uint32Array(a2, 4, 1)[0] = 1;
    q.writeBuffer(this.B.adv2, 0, a2);
  }

  _getStaging(i) {
    while (this._staging.length <= i) {
      this._staging.push(this.device.createBuffer({ size: NN * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    }
    return this._staging[i];
  }

  /**
   * Roll out nSteps of transport from rho0 to rho1 (Float32Array(65536) each,
   * mass-normalized). onFrame(k, Float32Array) fires in order as frames finish
   * (k = 0 is rho0). Resolves to the list of frames.
   */
  async rollout(rho0, rho1, nSteps = 50, onFrame = () => {}, { signal } = {}) {
    if (nSteps > MAX_STEPS) throw new Error('too many steps');
    const dev = this.device, q = dev.queue, B = this.B;
    this._setInputs(rho0, rho1, nSteps);
    // initial mass
    {
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.P.sum1); pass.setBindGroup(0, this.G.sumR0); pass.dispatchWorkgroups(NN / 256);
      pass.setPipeline(this.P.sum2); pass.setBindGroup(0, this.G.sum2m0); pass.dispatchWorkgroups(1);
      pass.end();
      q.submit([enc.finish()]);
    }
    const frames = [Float32Array.from(rho0)];
    onFrame(0, frames[0]);
    const pending = [];
    // Keep a bounded number of steps in flight so a cancel takes effect quickly.
    const inflight = 4;
    for (let s = 0; s < nSteps; s++) {
      if (signal?.aborted) break;
      q.writeBuffer(B.step, 0, new Uint32Array([s, 0, 0, 0]));
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      this._encodeForward(pass);
      this._encodeAdvect(pass);
      pass.end();
      const st = this._getStaging(s);
      enc.copyBufferToBuffer(B.R2, 0, st, 0, NN * 4);
      q.submit([enc.finish()]);
      pending.push(st.mapAsync(GPUMapMode.READ).then(() => {
        const f = new Float32Array(st.getMappedRange().slice(0));
        st.unmap();
        return f;
      }));
      if (pending.length >= inflight) {
        const f = await pending.shift();
        frames.push(f);
        onFrame(frames.length - 1, f);
      }
    }
    while (pending.length) {
      const f = await pending.shift();
      if (signal?.aborted) continue;
      frames.push(f);
      onFrame(frames.length - 1, f);
    }
    return frames;
  }

  // Debug: single forward at t = 0, returning an intermediate buffer.
  async debugForward(rho0, rho1, stopAt) {
    const dev = this.device, B = this.B;
    this._setInputs(rho0, rho1, 50);
    dev.queue.writeBuffer(B.step, 0, new Uint32Array([0, 0, 0, 0]));
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    this._encodeForward(pass, stopAt === 'v' ? null : stopAt);
    pass.end();
    const src = { spec0: B.Y, h0: B.H, psi: B.PSI, v: B.V }[stopAt] || B.X;
    const size = src.size;
    const st = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(src, 0, st, 0, size);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }

  destroy() { this.device.destroy(); }
}

export const GRID = N;
