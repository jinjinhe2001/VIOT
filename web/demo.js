// Interactive VIOT demo: draw a source and a target, transport in the browser.
import { VIOTEngine, fetchWeights, GRID } from './viot-engine.js';
import { strokesToDensity } from './viot-preprocess.js';
import { encodeGIF } from './gif.js';
import { SHAPES, EXAMPLES } from './presets.js';

const N = GRID;
const STEPS = 50;
const FRAME_MS = 45;                 // playback pace (the desktop GUI uses 50 ms)
const DRAW_SCALE = 2;                // drawing canvases are 512 px for crisp strokes
const STROKE = '#fcfea4';            // top of inferno, as in the paper's GUI

// matplotlib 'inferno', 256 x RGB
const INFERNO_HEX = '00000300000400000601000701010901010b02010e02021003021204031404031605041806041b07051d08061f0906210a07230b07260d08280e082a0f092d10092f120a32130a34140b36160b39170b3b190b3e1a0b401c0c431d0c451f0c47200c4a220b4c240b4e260b50270b52290b542b0a562d0a582e0a5a300a5c32095d34095f3509603709613909623b09643c09653e0966400966410967430a68450a69460a69480b6a4a0b6a4b0c6b4d0c6b4f0d6c500d6c520e6c530e6d550f6d570f6d58106d5a116d5b116e5d126e5f126e60136e62146e63146e65156e66156e68166e6a176e6b176e6d186e6e186e70196e72196d731a6d751b6d761b6d781c6d7a1c6d7b1d6c7d1d6c7e1e6c801f6b811f6b83206b85206a86216a88216a8922698b22698d23698e24689024689125679325679526669626669827659928649b28649c29639e2963a02a62a12b61a32b61a42c60a62c5fa72d5fa92e5eab2e5dac2f5cae305baf315bb1315ab23259b43358b53357b73456b83556ba3655bb3754bd3753be3852bf3951c13a50c23b4fc43c4ec53d4dc73e4cc83e4bc93f4acb4049cc4148cd4247cf4446d04544d14643d24742d44841d54940d64a3fd74b3ed94d3dda4e3bdb4f3adc5039dd5238de5337df5436e05634e25733e35832e45a31e55b30e65c2ee65e2de75f2ce8612be9622aea6428eb6527ec6726ed6825ed6a23ee6c22ef6d21f06f1ff0701ef1721df2741cf2751af37719f37918f47a16f57c15f57e14f68012f68111f78310f7850ef8870df8880cf88a0bf98c09f98e08f99008fa9107fa9306fa9506fa9706fb9906fb9b06fb9d06fb9e07fba007fba208fba40afba60bfba80dfbaa0efbac10fbae12fbb014fbb116fbb318fbb51afbb71cfbb91efabb21fabd23fabf25fac128f9c32af9c52cf9c72ff8c931f8cb34f8cd37f7cf3af7d13cf6d33ff6d542f5d745f5d948f4db4bf4dc4ff3de52f3e056f3e259f2e45df2e660f1e864f1e968f1eb6cf1ed70f1ee74f1f079f1f27df2f381f2f485f3f689f4f78df5f891f6fa95f7fb99f9fc9dfafda0fcfea4';
function hexToBytes(h) { const a = new Uint8Array(h.length / 2); for (let i = 0; i < a.length; i++) a[i] = parseInt(h.substr(2 * i, 2), 16); return a; }

const root = document.getElementById('viot-demo');
const $ = (id) => document.getElementById(id);
const els = {
  src: $('demo-src'), tgt: $('demo-tgt'), out: $('demo-out'),
  overlay: $('demo-overlay'), load: $('demo-load'), bar: $('demo-progress-bar'), progress: $('demo-progress'),
  overlayText: $('demo-overlay-text'), step: $('demo-step'), scrub: $('demo-scrub'),
  run: $('demo-run'), cont: $('demo-continue'), replay: $('demo-replay'), brush: $('demo-brush'),
  brushVal: $('demo-brush-value'), clearSrc: $('demo-clear-src'), clearTgt: $('demo-clear-tgt'),
  reset: $('demo-reset'), gif: $('demo-gif'), examples: $('demo-examples'), history: $('demo-history'),
  status: $('demo-status'), srcHint: $('demo-src-hint'),
};
const LUT = hexToBytes(INFERNO_HEX);
// data-weights: one URL or a comma-separated list tried in order (e.g. a local copy, then the Hub)
const weightUrls = root.dataset.weights.split(',').map((u) => new URL(u.trim(), document.baseURI).href);

const state = {
  engine: null, loading: null, running: false,
  srcDensity: null,          // locked source (previous final frame) when chaining
  segment: [], segVmax: 1, chainFrames: [], keyframes: [],
  anim: null, busyExample: false,
};

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------
const outCtx = els.out.getContext('2d');
const outImg = outCtx.createImageData(N, N);

function paintDensity(ctx, img, rho, vmax) {
  const d = img.data;
  const inv = 1 / (vmax + 1e-12);
  for (let i = 0; i < rho.length; i++) {
    let v = rho[i] * inv;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    const k = Math.floor(v * 255) * 3;
    d[4 * i] = LUT[k]; d[4 * i + 1] = LUT[k + 1]; d[4 * i + 2] = LUT[k + 2]; d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function maxOf(a) { let m = 0; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }

function showFrame(k) {
  const f = state.segment[k];
  if (!f) return;
  paintDensity(outCtx, outImg, f, state.segVmax);
  els.step.textContent = `step ${k} / ${state.segment.length - 1 < STEPS ? STEPS : state.segment.length - 1}`;
  els.scrub.value = String(k);
}

function setStatus(msg) { els.status.textContent = msg; }

// ---------------------------------------------------------------------------
// Drawing canvases
// ---------------------------------------------------------------------------
class Sketch {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.placeholder = canvas.parentElement.querySelector('.demo-placeholder');
    this.locked = false;
    this.last = null;
    this.clear();
    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    for (const t of ['pointerup', 'pointercancel', 'pointerleave']) canvas.addEventListener(t, () => { this.last = null; });
  }
  setInk(on) { if (this.placeholder) this.placeholder.hidden = on; }
  clear() {
    this.locked = false;
    this.setInk(false);
    this.c.classList.remove('is-locked');
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.c.width, this.c.height);
  }
  pos(e) {
    const r = this.c.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * this.c.width, ((e.clientY - r.top) / r.height) * this.c.height];
  }
  width() { return Number(els.brush.value) * DRAW_SCALE; }
  down(e) {
    if (this.locked || state.running) return;
    e.preventDefault();
    this.c.setPointerCapture(e.pointerId);
    this.last = this.pos(e);
    this.dot(this.last);
  }
  move(e) {
    if (!this.last || this.locked) return;
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evs.length ? evs : [e]) {
      const p = this.pos(ev);
      this.line(this.last, p);
      this.last = p;
    }
  }
  dot([x, y]) {
    this.setInk(true);
    this.ctx.fillStyle = STROKE;
    this.ctx.beginPath();
    this.ctx.arc(x, y, this.width() / 2, 0, 2 * Math.PI);
    this.ctx.fill();
  }
  line(a, b) {
    const ctx = this.ctx;
    ctx.strokeStyle = STROKE;
    ctx.lineWidth = this.width();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  drawShape(polylines) {
    this.clear();
    const s = this.c.width;
    for (const pl of polylines) {
      const pts = pl.map(([x, y]) => [x * s, y * s]);
      this.dot(pts[0]);
      for (let i = 1; i < pts.length; i++) this.line(pts[i - 1], pts[i]);
      this.dot(pts[pts.length - 1]);
    }
  }
  // 256x256 stroke intensities in [0, 1] (2x2 box average of the 512 px canvas)
  strokes() {
    const px = this.ctx.getImageData(0, 0, this.c.width, this.c.height).data;
    const W = this.c.width;
    const out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        let s = 0;
        for (let dy = 0; dy < DRAW_SCALE; dy++) {
          for (let dx = 0; dx < DRAW_SCALE; dx++) s += px[4 * ((y * DRAW_SCALE + dy) * W + x * DRAW_SCALE + dx) + 1];
        }
        out[y * N + x] = Math.min(1, s / (DRAW_SCALE * DRAW_SCALE * 254));
      }
    }
    return out;
  }
  showDensity(rho) {
    this.locked = true;
    this.setInk(true);
    this.c.classList.add('is-locked');
    const tmp = document.createElement('canvas');
    tmp.width = N; tmp.height = N;
    const tctx = tmp.getContext('2d');
    paintDensity(tctx, tctx.createImageData(N, N), rho, maxOf(rho));
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(tmp, 0, 0, this.c.width, this.c.height);
  }
}

const src = new Sketch(els.src);
const tgt = new Sketch(els.tgt);

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
function hasWebGPU() { return 'gpu' in navigator; }

async function isCached() {
  try {
    const cache = await caches.open('viot-weights-v1');
    for (const u of weightUrls) if (await cache.match(u)) return true;
  } catch { /* no Cache API */ }
  return false;
}

async function downloadWeights(onProgress) {
  let lastErr;
  for (const u of weightUrls) {
    try { return await fetchWeights(u, onProgress); } catch (err) { lastErr = err; }
  }
  throw lastErr;
}

function ensureEngine() {
  if (state.engine) return Promise.resolve(state.engine);
  if (state.loading) return state.loading;
  els.load.hidden = true;
  els.progress.hidden = false;
  els.overlayText.textContent = 'Downloading model…';
  state.loading = (async () => {
    const buf = await downloadWeights((got, total, cached) => {
      const frac = total ? got / total : 0;
      els.bar.style.width = `${(100 * frac).toFixed(1)}%`;
      els.overlayText.textContent = cached ? 'Loading model from cache…'
        : `Downloading model… ${(got / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ''} MB`;
    });
    els.overlayText.textContent = 'Compiling GPU kernels…';
    const eng = await VIOTEngine.create(buf);
    eng.lost.then((info) => {
      state.engine = null; state.loading = null;
      showOverlay(`The GPU device was lost (${info.message || info.reason}). Reload the page to try again.`);
    });
    // warm-up pass so the first real rollout is not paying for pipeline creation
    const z = new Float32Array(N * N).fill(1 / (N * N));
    await eng.rollout(z, z, 2);
    state.engine = eng;
    els.overlay.hidden = true;
    const gpu = [eng.adapterInfo.vendor, eng.adapterInfo.architecture].filter(Boolean).join(' ');
    setStatus(`Model ready${gpu ? ` on ${gpu} GPU` : ''}. Draw ρ₀ and ρ₁, then press Run.`);
    return eng;
  })().catch((err) => {
    state.loading = null;
    showOverlay(`Could not start the model: ${err.message}`, true);
    throw err;
  });
  return state.loading;
}

function showOverlay(text, retry = false) {
  els.overlay.hidden = false;
  els.progress.hidden = true;
  els.overlayText.textContent = text;
  els.load.hidden = !retry;
  if (retry) els.load.textContent = 'Try again';
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------
function setRunning(on) {
  state.running = on;
  root.classList.toggle('is-running', on);
  for (const b of [els.run, els.cont, els.replay, els.gif, els.clearSrc, els.clearTgt, els.reset]) b.disabled = on;
  for (const b of els.examples.querySelectorAll('button')) b.disabled = on;
  els.scrub.disabled = on || state.segment.length < 2;
  if (!on) {
    const done = state.segment.length > 1;
    els.cont.disabled = !done;
    els.replay.disabled = !done;
    els.gif.disabled = state.chainFrames.length < 2;
  }
}

function play(fromStart = true) {
  cancelAnimationFrame(state.anim);
  const t0 = performance.now();
  const tick = (t) => {
    const k = Math.min(Math.floor((t - t0) / FRAME_MS), state.segment.length - 1);
    showFrame(k);
    if (k < STEPS && (state.running || k < state.segment.length - 1)) state.anim = requestAnimationFrame(tick);
  };
  if (fromStart) showFrame(0);
  state.anim = requestAnimationFrame(tick);
}

async function run() {
  if (state.running) return;
  const rho0 = state.srcDensity || strokesToDensity(src.strokes());
  if (!rho0) { setStatus('Draw a source density ρ₀ first.'); return; }
  const rho1 = strokesToDensity(tgt.strokes());
  if (!rho1) { setStatus('Draw a target density ρ₁.'); return; }
  let eng;
  try { eng = await ensureEngine(); } catch { return; }
  setRunning(true);
  setStatus('Transporting…');
  state.segment = [];
  state.segVmax = Math.max(maxOf(rho0), maxOf(rho1));
  play();
  const t0 = performance.now();
  let frames;
  try {
    frames = await eng.rollout(rho0, rho1, STEPS, (k, f) => { state.segment[k] = f; });
  } catch (err) {
    setRunning(false);
    setStatus(`Rollout failed: ${err.message}`);
    return;
  }
  const secs = (performance.now() - t0) / 1000;
  state.segment = frames;
  if (!state.keyframes.length) state.keyframes.push(frames[0]);
  state.keyframes.push(frames[frames.length - 1]);
  state.chainFrames.push(...(state.chainFrames.length ? frames.slice(1) : frames));
  renderHistory();
  setRunning(false);
  els.scrub.max = String(frames.length - 1);
  if (!state.anim) showFrame(frames.length - 1);
  const segs = state.keyframes.length - 1;
  setStatus(`${STEPS} steps in ${secs.toFixed(2)} s (${((1000 * secs) / STEPS).toFixed(1)} ms per step)` +
    ` · chain: ${segs} segment${segs > 1 ? 's' : ''}. Continue to transport this result into a new target.`);
}

function continueChain() {
  if (state.segment.length < 2) return;
  cancelAnimationFrame(state.anim);
  state.srcDensity = state.segment[state.segment.length - 1];
  src.showDensity(state.srcDensity);
  els.srcHint.textContent = 'previous result';
  tgt.clear();
  els.cont.disabled = true;
  setStatus('ρ₀ is now the previous result. Draw the next target and press Run.');
}

function clearSource() {
  state.srcDensity = null;
  src.clear();
  els.srcHint.textContent = 'draw here';
}

function resetAll() {
  cancelAnimationFrame(state.anim);
  clearSource();
  tgt.clear();
  state.segment = []; state.chainFrames = []; state.keyframes = [];
  outCtx.fillStyle = '#000'; outCtx.fillRect(0, 0, N, N);
  els.step.textContent = `step 0 / ${STEPS}`;
  els.scrub.value = '0';
  renderHistory();
  setRunning(false);
  setStatus(state.engine ? 'Chain reset. Draw ρ₀ and ρ₁, then press Run.' : '');
}

function renderHistory() {
  els.history.replaceChildren();
  els.history.hidden = state.keyframes.length === 0;
  state.keyframes.forEach((f, i) => {
    if (i > 0) {
      const arrow = document.createElement('span');
      arrow.className = 'demo-arrow';
      arrow.textContent = '→';
      arrow.setAttribute('aria-hidden', 'true');
      els.history.append(arrow);
    }
    const c = document.createElement('canvas');
    c.width = N; c.height = N;
    c.setAttribute('role', 'img');
    c.setAttribute('aria-label', `Keyframe ${i}`);
    const ctx = c.getContext('2d');
    paintDensity(ctx, ctx.createImageData(N, N), f, maxOf(f));
    els.history.append(c);
  });
  els.history.scrollLeft = els.history.scrollWidth;
}

function saveGIF() {
  const frames = state.chainFrames;
  if (frames.length < 2) return;
  let vmax = 0;
  for (const f of frames) vmax = Math.max(vmax, maxOf(f));
  const inv = 1 / (vmax + 1e-12);
  const gifFrames = frames.map((f, i) => {
    const idx = new Uint8Array(N * N);
    for (let p = 0; p < f.length; p++) idx[p] = Math.floor(Math.min(1, Math.max(0, f[p] * inv)) * 255);
    const isKey = i % STEPS === 0;
    return { indices: idx, delay: isKey ? 60 : 5 };
  });
  const bytes = encodeGIF(gifFrames, N, N, LUT);
  const url = URL.createObjectURL(new Blob([bytes], { type: 'image/gif' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'viot_transport.gif';
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

async function runExample(ex) {
  if (state.running) return;
  resetAll();
  const [first, ...rest] = ex.chain;
  src.drawShape(SHAPES[first]);
  for (let i = 0; i < rest.length; i++) {
    if (i > 0) {
      continueChain();
      await new Promise((r) => setTimeout(r, 350));
    }
    tgt.drawShape(SHAPES[rest[i]]);
    await run();
    if (!state.engine) return;
    await new Promise((r) => setTimeout(r, STEPS * FRAME_MS + 500));
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
els.run.addEventListener('click', run);
els.cont.addEventListener('click', continueChain);
els.replay.addEventListener('click', () => play());
els.clearSrc.addEventListener('click', clearSource);
els.clearTgt.addEventListener('click', () => tgt.clear());
els.reset.addEventListener('click', resetAll);
els.gif.addEventListener('click', saveGIF);
els.load.addEventListener('click', () => ensureEngine().catch(() => {}));
els.brush.addEventListener('input', () => { els.brushVal.textContent = `${els.brush.value} px`; });
els.scrub.addEventListener('input', () => { cancelAnimationFrame(state.anim); showFrame(Number(els.scrub.value)); });
for (const ex of EXAMPLES) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'demo-chip';
  b.textContent = ex.label;
  b.addEventListener('click', () => runExample(ex));
  els.examples.append(b);
}
root.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); run(); }
});

outCtx.fillStyle = '#000';
outCtx.fillRect(0, 0, N, N);
renderHistory();
els.brushVal.textContent = `${els.brush.value} px`;

if (!hasWebGPU()) {
  showOverlay('This demo needs WebGPU: use a recent Chrome or Edge, Safari 26+, or Firefox 141+ on Windows. The figure below shows example results.');
  els.run.disabled = true;
  for (const b of els.examples.querySelectorAll('button')) b.disabled = true;
} else {
  // If the weights are already cached, start the engine as soon as the demo is near the viewport.
  isCached().then((cached) => {
    if (!cached) return;
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); ensureEngine().catch(() => {}); }
    }, { rootMargin: '300px' });
    io.observe(root);
  });
}
