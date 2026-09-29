"""Export a released 2D operator to the compact format used by the WebGPU demo.

The browser engine (``viot-engine.js``) evaluates every spectral convolution as
a truncated DFT product. This script prepares the weights for it:

* the 2 -> 64 lift is folded exactly into the first spectral layer (its input
  then has two channels, and the lift bias only reaches the DC mode);
* the spectral weights of layers 1-7 are quantized to int4 or int8 with one f16
  scale per (input channel, mode);
* everything else stays fp32.

The result is a plain safetensors file (65 MB for int4, 124 MB for int8).

    python web/export_web.py --model mnist_2d --bits 4 --out web/viot_mnist256_q4.safetensors
    python web/export_web.py --model mnist_2d --bits 4 --check

``--check`` re-implements the forward pass with the same DFT products in float64
and compares the velocity with the reference ``FNO2D`` (float64) on random
densities, once with fp32 weights (expect ~1e-8) and once quantized.
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from viot.pretrained import build_model, load_config, load_state_dict, _resolve_dir  # noqa: E402

N, M, KR, WIDTH, LAYERS = 256, 32, 64, 64, 8     # the engine is specialised to this architecture
KH_LIST = list(range(M)) + list(range(-M, 0))


def check_arch(cfg):
    a = cfg["arch"]
    want = dict(max_res=N, width=WIDTH, n_modes=M, n_layers=LAYERS, k_max=KR / N)
    got = {k: a.get(k) for k in want}
    if got != want or a.get("class") != "FNO2D":
        raise SystemExit(f"the WebGPU engine supports {want}, got {got}")


def build_params(sd, bits):
    """Fold the lift into layer 0 and quantize layers 1-7 (float64 bookkeeping)."""
    p = {}
    L = sd["lift.weight"][:, :, 0, 0].double()
    bL = sd["lift.bias"].double()
    w0 = torch.complex(sd["fno_layers.0.spectral_weight_real"].double(),
                       sd["fno_layers.0.spectral_weight_imag"].double())      # [C, D, 2M, M]
    p["l0_w"] = torch.einsum("cj,cdhk->jdhk", L.to(w0.dtype), w0)               # [2, D, 2M, M]
    # an ortho rfft2 of the constant lift bias b is b * N at the DC mode only
    p["l0_dc"] = torch.einsum("c,cd->d", (bL * N).to(w0.dtype), w0[:, :, 0, 0])
    for l in range(1, LAYERS):
        for part in ("real", "imag"):
            v = sd[f"fno_layers.{l}.spectral_weight_{part}"].double()
            if bits >= 32:
                p[f"l{l}_{part}"] = v
                continue
            qmax = 2 ** (bits - 1) - 1
            s = (v.abs().amax(dim=1, keepdim=True) / qmax).float().half().double()
            s = torch.where(s == 0, torch.ones_like(s), s)
            q = torch.round(v / s).clamp(-qmax, qmax)
            p[f"l{l}_{part}_q"] = q.to(torch.int8)
            p[f"l{l}_{part}_s"] = s[:, 0]
            p[f"l{l}_{part}"] = q * s
    return p


# ---------------------------------------------------------------------------
# float64 re-implementation with explicit DFT products (mirrors the shaders)
# ---------------------------------------------------------------------------
def _dft_mats(device):
    dt = torch.float64
    n = torch.arange(N, dtype=dt, device=device)
    kw = torch.arange(M, dtype=dt, device=device)
    kh = torch.tensor(KH_LIST, dtype=dt, device=device)
    aw = 2 * math.pi * n[:, None] * kw[None, :] / N
    Ew = torch.complex(torch.cos(aw), -torch.sin(aw)) / N
    ah = 2 * math.pi * kh[:, None] * n[None, :] / N
    Eh = torch.complex(torch.cos(ah), -torch.sin(ah))
    Gh = torch.complex(torch.cos(ah.T), torch.sin(ah.T)) / N
    c = torch.full((M,), 2.0, dtype=dt, device=device)
    c[0] = 1.0
    ang = 2 * math.pi * kw[:, None] * n[None, :] / N
    return Ew, Eh, Gh, c[:, None] * torch.cos(ang), -c[:, None] * torch.sin(ang)


def _spectral_conv(x, w, mats):
    Ew, Eh, Gh, Cre, Cim = mats
    XH = torch.einsum("jh,bchk->bcjk", Eh, torch.einsum("bchw,wk->bchk", x.to(Ew.dtype), Ew))
    YH = torch.einsum("hj,bdjk->bdhk", Gh, torch.einsum("bcjk,cdjk->bdjk", XH, w.to(XH.dtype)))
    return torch.einsum("bdhk,kw->bdhw", YH.real, Cre) + torch.einsum("bdhk,kw->bdhw", YH.imag, Cim)


def _psi_to_velocity(psi):
    dt = torch.float64
    dev = psi.device
    n = torch.arange(N, dtype=dt, device=dev)
    kh = torch.arange(-KR, KR + 1, dtype=dt, device=dev)
    kw = torch.arange(0, KR + 1, dtype=dt, device=dev)
    aw = 2 * math.pi * n[:, None] * kw[None, :] / N
    ah = 2 * math.pi * kh[:, None] * n[None, :] / N
    Ew = torch.complex(torch.cos(aw), -torch.sin(aw))
    Eh = torch.complex(torch.cos(ah), -torch.sin(ah))
    P = torch.einsum("jh,bhk->bjk", Eh, torch.einsum("bhw,wk->bhk", psi[:, 0].to(Ew.dtype), Ew))
    mask = ((kh[:, None] ** 2 + kw[None, :] ** 2) <= KR * KR).to(dt)
    Gh = torch.complex(torch.cos(ah.T), torch.sin(ah.T))
    c = torch.full((KR + 1,), 2.0, dtype=dt, device=dev)
    c[0] = 1.0
    ang = 2 * math.pi * kw[:, None] * n[None, :] / N
    Cre, Cim = c[:, None] * torch.cos(ang), -c[:, None] * torch.sin(ang)
    out = []
    for V in ((2j * math.pi) * (kw[None, :] / N) * P * mask, (-2j * math.pi) * (kh[:, None] / N) * P * mask):
        Z = torch.einsum("hj,bjk->bhk", Gh, V)
        out.append((torch.einsum("bhk,kw->bhw", Z.real, Cre) + torch.einsum("bhk,kw->bhw", Z.imag, Cim)) / (N * N))
    return torch.stack(out, 1)


@torch.no_grad()
def forward_dft(sd, p, rho_t, t, rho1):
    dt = torch.float64
    dev = rho_t.device
    mats = _dft_mats(dev)
    g = lambda k: sd[k].to(dt).to(dev)
    freqs = torch.exp(torch.arange(64, dtype=dt, device=dev) * -(math.log(10000.0) / 64))
    ang = t.to(dt)[:, None] * freqs[None]
    temb = torch.cat([torch.sin(ang), torch.cos(ang)], 1)
    cond = F.silu(temb @ g("time_mlp.0.weight").T + g("time_mlp.0.bias")) @ g("time_mlp.2.weight").T + g("time_mlp.2.bias")
    r2 = torch.cat([rho_t, rho1], 1).to(dt)
    x = torch.einsum("dj,bjhw->bdhw", g("lift.weight")[:, :, 0, 0], r2) + g("lift.bias")[None, :, None, None]
    for l in range(LAYERS):
        if l == 0:
            spec = _spectral_conv(r2, p["l0_w"].to(dev), mats) + (p["l0_dc"].real.to(dev) / N)[None, :, None, None]
        else:
            spec = _spectral_conv(x, torch.complex(p[f"l{l}_real"], p[f"l{l}_imag"]).to(dev), mats)
        pre = f"fno_layers.{l}."
        h = spec + torch.einsum("dc,bchw->bdhw", g(pre + "local_conv.weight")[:, :, 0, 0], x) + g(pre + "local_conv.bias")[None, :, None, None]
        sc, sh = (F.silu(cond) @ g(pre + "film.1.weight").T + g(pre + "film.1.bias")).chunk(2, -1)
        h = h * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        x = F.gelu(F.group_norm(h, 8, g(pre + "norm.weight"), g(pre + "norm.bias"), eps=1e-5)) + x
    y = F.gelu(torch.einsum("dc,bchw->bdhw", g("project.0.weight")[:, :, 0, 0], x) + g("project.0.bias")[None, :, None, None])
    psi = torch.einsum("dc,bchw->bdhw", g("project.2.weight")[:, :, 0, 0], y) + g("project.2.bias")[None, :, None, None]
    return _psi_to_velocity(psi)


def check(cfg, sd, bits, device):
    model = build_model(cfg)
    model.load_state_dict(sd)
    model = model.double().to(device).eval()
    # realistic inputs: glyph densities built like the training pools ("3"->"8", "2"->"7")
    from viot.glyphs import default_font, glyph_density_2d
    font = default_font()
    if font is None:
        raise SystemExit("--check needs a font (install matplotlib, which ships DejaVu Sans)")
    rho = torch.cat([torch.cat([glyph_density_2d(a, font, N), glyph_density_2d(b, font, N)], 1)
                     for a, b in (("3", "8"), ("2", "7"))]).double().to(device)
    t = torch.tensor([0.0, 0.62], dtype=torch.float64, device=device)
    with torch.no_grad():
        v_ref = model.forward_velocity_only(rho[:, :1], t, rho[:, 1:])
    for b in (32, bits):
        v = forward_dft(sd, build_params(sd, b), rho[:, :1], t, rho[:, 1:])
        tag = "fp32 weights" if b == 32 else f"int{b} weights"
        print(f"{tag}: relative velocity error vs FNO2D (float64) = {((v - v_ref).norm() / v_ref.norm()).item():.3e}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def save_safetensors(path, tensors, meta):
    header, blobs, off = {}, [], 0
    for name, (dtype, shape, arr) in tensors.items():
        b = arr.tobytes()
        assert len(b) % 4 == 0, name   # keeps every typed-array view aligned without padding
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {k: str(v) for k, v in meta.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def export(cfg, sd, bits, out):
    p = build_params(sd, bits)
    f32 = lambda a: ("F32", tuple(a.shape), np.ascontiguousarray(a.detach().cpu().numpy().astype("<f4")))
    f16 = lambda a: ("F16", tuple(a.shape), np.ascontiguousarray(a.detach().cpu().numpy().astype("<f2")))
    T = {}
    for k in ["time_mlp.0.weight", "time_mlp.0.bias", "time_mlp.2.weight", "time_mlp.2.bias",
              "lift.weight", "lift.bias", "project.0.weight", "project.0.bias", "project.2.weight", "project.2.bias"]:
        T[k] = f32(sd[k].reshape(sd[k].shape[0], -1) if sd[k].dim() > 1 else sd[k])
    for l in range(LAYERS):
        pre = f"fno_layers.{l}."
        T[pre + "local_conv.weight"] = f32(sd[pre + "local_conv.weight"][:, :, 0, 0])
        for k in ["local_conv.bias", "film.1.weight", "film.1.bias", "norm.weight", "norm.bias"]:
            T[pre + k] = f32(sd[pre + k])
    w0 = p["l0_w"].permute(2, 3, 0, 1).reshape(2 * M * M, 2, WIDTH)               # [mode][j][d]
    T["l0.w"] = f32(torch.stack([w0.real, w0.imag], -1))
    T["l0.dc"] = f32(torch.stack([p["l0_dc"].real, p["l0_dc"].imag], -1))
    for l in range(1, LAYERS):
        for part in ("real", "imag"):
            q = p[f"l{l}_{part}_q"].permute(2, 3, 0, 1).reshape(2 * M * M, WIDTH, WIDTH)  # [mode][c][d]
            if bits == 4:
                u = (q.to(torch.int64) & 0xF).reshape(2 * M * M, WIDTH, WIDTH // 8, 8)
                word = torch.zeros(u.shape[:3], dtype=torch.int64)
                for i in range(8):
                    word |= u[..., i] << (4 * i)                                        # little-endian nibbles along d
                T[f"l{l}.{part}.q4"] = ("U32", tuple(word.shape), np.ascontiguousarray(word.numpy().astype("<u4")))
            else:
                T[f"l{l}.{part}.q8"] = ("I8", tuple(q.shape), np.ascontiguousarray(q.numpy().astype("i1")))
            T[f"l{l}.{part}.scale"] = f16(p[f"l{l}_{part}_s"].permute(1, 2, 0).reshape(2 * M * M, WIDTH))
    meta = dict(model=cfg["name"], res=N, k_max=KR / N, width=WIDTH, modes=M, layers=LAYERS,
                n_steps=cfg["rollout"]["n_steps"], bits=bits,
                quant="symmetric per (input channel, mode), f16 scales",
                layout="spectral: [mode=kh_idx*32+kw][c_in][d_out]; kh_idx 0..31 -> kh, 32..63 -> kh-64")
    save_safetensors(out, T, meta)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mnist_2d", help="released name or checkpoint folder")
    ap.add_argument("--local-dir", default=None, help="local copy of the checkpoint repository")
    ap.add_argument("--bits", type=int, choices=[4, 8], default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.model, a.local_dir)
    check_arch(cfg)
    sd = load_state_dict(_resolve_dir(a.model, a.local_dir) / "model.safetensors")
    if a.check:
        check(cfg, sd, a.bits, a.device)
    else:
        name = cfg["name"].replace("_2d", "")
        export(cfg, sd, a.bits, a.out or f"viot_{name}256_q{a.bits}.safetensors")


if __name__ == "__main__":
    main()
