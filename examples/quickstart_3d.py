"""Chain 3D transports through extruded glyphs with the released ``font_3d`` operator.

    python examples/quickstart_3d.py --text VIOT
    python examples/quickstart_3d.py --text 2026 --save-npy

Glyph volumes follow the training recipe (``viot.glyphs.glyph_volume_3d``,
DejaVu Sans Bold by default) and are peak-normalized like the training
sampler. The rollout uses the advection the model was trained with
(first-order semi-Lagrangian) and mass renormalization, as in the paper's
evaluation. A 128^3 transport needs a CUDA GPU with about 8 GB of memory.
Writes a GIF of a 45-degree maximum-intensity projection and, optionally, the
final volume of every segment as .npy.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from viot import load_pretrained                 # noqa: E402
from viot.glyphs import glyph_volume_3d           # noqa: E402
from viot.ops_3d import rollout_3d                # noqa: E402
from _viz import mip_view, save_gif               # noqa: E402


def dejavu_bold():
    try:
        from matplotlib import font_manager
        return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
                                     fallback_to_default=False)
    except Exception:
        return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # non-UTF-8 consoles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="font_3d")
    ap.add_argument("--local-dir", default=None, help="local copy of the checkpoint repository")
    ap.add_argument("--text", default="VIOT")
    ap.add_argument("--font", default=None, help="font file (default: DejaVu Sans Bold)")
    ap.add_argument("--advection", default=None, help="override the scheme the model was trained with")
    ap.add_argument("--save-npy", action="store_true")
    ap.add_argument("--out", default="font3d_chain.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    font = a.font or dejavu_bold()
    if font is None:
        sys.exit("no font found: pass --font")
    model, cfg = load_pretrained(a.model, device=a.device, local_dir=a.local_dir)
    res, steps = cfg["arch"]["max_res"], cfg["rollout"]["n_steps"]
    scheme = a.advection or cfg["rollout"]["advection"]

    chars = [c for c in a.text if not c.isspace()]
    vols = []
    for c in chars:
        v = glyph_volume_3d(c, font, res=res)
        if v is None:
            sys.exit(f"'{c}' is not available in {font}")
        v = torch.from_numpy(v)[None, None].to(a.device)
        vols.append(v / v.amax())

    rho = vols[0]
    views = [mip_view(rho[0, 0])]
    for i, (c, tgt) in enumerate(zip(chars[1:], vols[1:]), start=1):
        out = rollout_3d(model, rho, tgt, n_steps=steps, scheme=scheme,
                         return_frames=True, store_device="cpu")
        views += [mip_view(f[0, 0].to(a.device)) for f in out["frames"][1:]]
        rho = out["final"]
        l2 = (rho - tgt).norm().item()
        print(f"  -> {c}: terminal L2 {l2:.2f} (relative {l2 / tgt.norm().item():.3f})")
        if a.save_npy:
            np.save(f"{os.path.splitext(a.out)[0]}_seg{i}_{c}.npy", rho[0, 0].cpu().numpy())

    save_gif(views, a.out, hold_every=steps, scale=2)
    print(f"wrote {a.out} ({len(views)} frames, {scheme})")


if __name__ == "__main__":
    main()
