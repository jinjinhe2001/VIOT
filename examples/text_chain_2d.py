"""Chain transports through the characters of a string (2D).

Every character is rendered with the glyph-pool recipe of the training data
(``viot.glyphs.glyph_density_2d``); the final density of each transport is the
source of the next one, with no reset in between. The area normalization of
that recipe would enlarge thin glyphs (I, l, j, 一) past the border, so it is
capped to keep every glyph within 75% of the frame, which also leaves room for
the transport; thin glyphs are rendered bolder until they have the training
area instead (``--max-extent``; 0 gives the exact pool recipe).

    python examples/text_chain_2d.py --model latin_font_2d --text VIOT
    python examples/text_chain_2d.py --model cjk_2d --text 春江潮水连海平 --font /path/to/NotoSansCJK-Bold.otf
    python examples/text_chain_2d.py --model mnist_2d --text 2026

The CJK model was trained on SimHei, Microsoft YaHei and SimSun renderings; any
CJK font works, the closer to those the better. The default font is DejaVu Sans
(shipped with matplotlib), which only covers Latin characters and digits.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from viot import load_pretrained                       # noqa: E402
from viot.data_2d import FRAME_MAX_EXTENT               # noqa: E402
from viot.glyphs import default_font, glyph_density_2d  # noqa: E402
from viot.ops_2d import rollout_2d                      # noqa: E402
from _viz import save_gif, save_strip                   # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # non-UTF-8 consoles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="latin_font_2d")
    ap.add_argument("--local-dir", default=None, help="local copy of the checkpoint repository")
    ap.add_argument("--text", default="VIOT")
    ap.add_argument("--font", default=None, help="TrueType/OpenType font file (default: DejaVu Sans)")
    ap.add_argument("--advection", default=None, help="override the scheme the model was trained with")
    ap.add_argument("--max-extent", type=float, default=FRAME_MAX_EXTENT,
                    help="largest glyph size as a fraction of the frame (0 or negative: no cap, as in the pools)")
    ap.add_argument("--out", default="text_chain.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    font = a.font or default_font()
    if font is None:
        sys.exit("no font found: pass --font")
    model, cfg = load_pretrained(a.model, device=a.device, local_dir=a.local_dir)
    res, steps = cfg["arch"]["max_res"], cfg["rollout"]["n_steps"]
    scheme = a.advection or cfg["rollout"]["advection"]

    chars = [c for c in a.text if not c.isspace()]
    max_extent = a.max_extent if a.max_extent > 0 else None
    targets = []
    for c in chars:
        d = glyph_density_2d(c, font, res, max_extent=max_extent)
        if d is None:
            sys.exit(f"'{c}' is not available in {font}")
        targets.append(d.to(a.device))

    rho = targets[0]
    frames = [rho[0, 0].cpu().numpy()]
    for c, tgt in zip(chars[1:], targets[1:]):
        out = rollout_2d(model, rho, tgt, n_steps=steps, scheme=scheme, return_frames=True)
        frames += [f[0, 0].cpu().numpy() for f in out["frames"][1:]]
        rho = out["final"]
        rel = (rho - tgt).norm().item() / tgt.norm().item()
        print(f"  -> {c}: relative terminal L2 {rel:.3f}")

    save_gif(frames, a.out, hold_every=steps)
    save_strip(frames, os.path.splitext(a.out)[0] + "_strip.png", n=min(len(frames), 4 * len(chars)))
    print(f"wrote {a.out} ({len(frames)} frames, {scheme})")


if __name__ == "__main__":
    main()
