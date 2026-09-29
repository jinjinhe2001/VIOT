"""Transport one MNIST test digit into another with a released 2D operator.

    python examples/quickstart_2d.py                     # mnist_2d, random test pair
    python examples/quickstart_2d.py --seed 3 --out pair3.gif

Writes a GIF of the 50-step rollout and a PNG strip, and prints the terminal
error, the mass error and the (spectral) divergence of the predicted velocity.

The pair comes from the training sampler on the test split. That sampler scales
every digit to the same effective area, which pushes about a third of them
(most 1s and 7s) past the border of the canvas; pairs with such a digit are
redrawn, unless ``--keep-clipped`` is given.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from viot import load_pretrained                       # noqa: E402
from viot.data_2d import sample_mnist_pairs            # noqa: E402
from viot.ops_2d import rollout_2d                      # noqa: E402
from _viz import save_gif, save_strip                   # noqa: E402


def spectral_divergence(v: torch.Tensor) -> float:
    """Mean |div v| evaluated spectrally (exactly zero up to round-off for VIOT)."""
    H, W = v.shape[-2:]
    kh = torch.fft.fftfreq(H, device=v.device).view(H, 1)
    kw = torch.fft.rfftfreq(W, device=v.device).view(1, -1)
    div_hat = 2j * torch.pi * (kh * torch.fft.rfft2(v[:, 0]) + kw * torch.fft.rfft2(v[:, 1]))
    return torch.fft.irfft2(div_hat, s=(H, W)).abs().mean().item()


def in_frame(rho: torch.Tensor) -> bool:
    """True if the digit stays inside the canvas: the pixels above 5% of the peak
    span at most 92% of it and keep 2 px from the border, and less than 1e-4 of
    the mass lies within 3 px of the border."""
    d = rho[0, 0]
    H, W = d.shape
    ys, xs = torch.nonzero(d > 0.05 * d.max(), as_tuple=True)
    border = 1.0 - (d[3:H - 3, 3:W - 3].sum() / d.sum()).item()
    return bool(ys.min() >= 2 and xs.min() >= 2 and ys.max() <= H - 3 and xs.max() <= W - 3
                and max(ys.max() - ys.min(), xs.max() - xs.min()) + 1 <= 0.92 * H
                and border < 1e-4)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # non-UTF-8 consoles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mnist_2d")
    ap.add_argument("--local-dir", default=None, help="local copy of the checkpoint repository")
    ap.add_argument("--data-root", default=".data", help="where torchvision stores MNIST")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-clipped", action="store_true",
                    help="keep the first pair even if a digit runs past the border")
    ap.add_argument("--out", default="mnist_transport.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, cfg = load_pretrained(a.model, device=a.device, local_dir=a.local_dir)
    res, steps, scheme = cfg["arch"]["max_res"], cfg["rollout"]["n_steps"], cfg["rollout"]["advection"]

    torch.manual_seed(a.seed)
    for redraws in range(100):
        rho_0, rho_1 = sample_mnist_pairs(1, res, res, device=a.device, root=a.data_root, train=False)
        if a.keep_clipped or (in_frame(rho_0) and in_frame(rho_1)):
            break
    out = rollout_2d(model, rho_0, rho_1, n_steps=steps, scheme=scheme,
                     return_frames=True, return_velocities=True)
    final = out["final"]

    l2 = (final - rho_1).norm().item()
    print(f"{cfg['name']}: {steps} steps ({scheme})")
    if redraws:
        print(f"  (skipped {redraws} pair{'s' if redraws > 1 else ''} with a digit cut off at the border)")
    print(f"  terminal L2 {l2:.5f}   relative {l2 / rho_1.norm().item():.3f}")
    print(f"  mass error {abs(final.sum().item() - rho_0.sum().item()):.2e}")
    print(f"  mean |div v| (spectral) {max(spectral_divergence(v) for v in out['velocities']):.2e}")

    frames = [f[0, 0].cpu().numpy() for f in out["frames"]] + [rho_1[0, 0].cpu().numpy()]
    save_gif(frames, a.out)
    save_strip(frames[:-1], os.path.splitext(a.out)[0] + "_strip.png")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
