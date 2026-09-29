# VIOT in the browser (WebGPU)

`index.html` is a self-contained interactive demo: draw a source and a target
density and the released `mnist_2d` operator transports one into the other in
50 steps, on the visitor's GPU. There is no server-side computation.

```
cd web
python export_web.py --model mnist_2d --bits 4     # writes viot_mnist256_q4.safetensors (65 MB)
python -m http.server 8000                         # open http://localhost:8000
```

If the local weight file is missing, the page falls back to the copy on the
Hugging Face Hub (`data-weights` in `index.html` lists the URLs tried in order).
WebGPU is available in current Chrome and Edge, Safari 26+, and Firefox 141+ on
Windows.

## Files

| file | purpose |
|---|---|
| `viot-engine.js` | WebGPU inference: FNO forward pass, spectral curl, MacCormack advection, mass renormalization |
| `viot-preprocess.js` | sketch to density (port of `viot.data_2d.strokes_to_density` with `max_extent=0.75`) |
| `demo.js`, `demo.css`, `presets.js`, `gif.js` | user interface, example sketches, GIF export |
| `export_web.py` | converts a released checkpoint to the compact weight file |

## How the engine works

The operator is the paper's 2D FNO (width 64, 32 modes, 8 layers, 256^2 grid,
k_max = 0.25). Each spectral convolution is evaluated as a truncated DFT
product: a real-to-complex product along the width keeps 32 frequencies, a
complex product along the height keeps 64, the per-mode 64x64 complex channel
mixing follows, and the inverse transforms mirror the forward ones. The final
stream function is band-limited to |k| <= 0.25, turned into a divergence-free
velocity by the spectral curl, and the density is advanced with the same
MacCormack scheme and mass renormalization used in the paper. One step is 85
compute dispatches (tiled matrix products, GroupNorm reductions, bilinear
advection), all in fp32.

`export_web.py` folds the 2-to-64 lift into the first spectral layer, which is
exact, and stores the spectral weights of layers 1-7 in 4 bits (or 8 bits with
`--bits 8`) with an f16 scale per input channel and mode.

## Accuracy

Measured on the MNIST test split (32 pairs, 50 steps, relative to a strict-fp32
PyTorch rollout of the full-precision checkpoint):

| weights | change of the rollout | terminal L2 |
|---|---|---|
| fp32, PyTorch default (TF32 convolutions) | 2.9 % | 0.00109 |
| int8 | 0.4 % | 0.00111 |
| int4 (default) | 4.7 % | 0.00110 |
| fp32, strict | - | 0.00111 |

The engine itself matches PyTorch to fp32 round-off. For identical quantized
weights, the velocity agrees to 4.5e-5 relative error and a 50-step rollout to
0.4 %. On an RTX 5080 a 50-step rollout takes about 0.13 s.
