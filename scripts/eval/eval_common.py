"""Shared model loading for the evaluation scripts.

A model is given either as ``--model <released name or checkpoint folder>`` (loaded with
:func:`viot.pretrained.load_pretrained`, architecture from its ``config.json``) or as
``--ckpt <state_dict .pt/.safetensors>`` plus architecture flags (defaults: the 2D
256^2 / width 64 / 32 modes / 8 layers / k_max 0.25 models, or the 3D 128^3 / width 32 /
16 modes / 6 layers / k_max 0.25 models).
"""
import sys
from pathlib import Path

import torch

try:
    import viot  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARCH_DEFAULTS = {
    2: dict(resolution=256, k_max=0.25, fno_width=64, fno_modes=32, fno_layers=8),
    3: dict(resolution=128, k_max=0.25, fno_width=32, fno_modes=16, fno_layers=6),
}


def add_model_args(ap, dim):
    """Add ``--model/--checkpoint-dir`` and ``--ckpt`` + architecture flags to a parser."""
    d = ARCH_DEFAULTS[dim]
    g = ap.add_argument_group('model (give --model or --ckpt)')
    g.add_argument('--model', default=None,
                   help='released model name (e.g. mnist_2d) or a folder with config.json + model.safetensors')
    g.add_argument('--checkpoint-dir', default=None,
                   help='local copy of the checkpoint repository (default: $VIOT_CHECKPOINT_DIR, else download '
                        'from the Hugging Face Hub)')
    g.add_argument('--ckpt', default=None, help='state_dict file (.pt or .safetensors); architecture from the flags below')
    g.add_argument('--resolution', type=int, default=d['resolution'],
                   help=f"grid size of the model (--ckpt only; default {d['resolution']})")
    g.add_argument('--k-max', type=float, default=d['k_max'])
    g.add_argument('--fno-width', type=int, default=d['fno_width'])
    g.add_argument('--fno-modes', type=int, default=d['fno_modes'])
    g.add_argument('--fno-layers', type=int, default=d['fno_layers'])


def read_state_dict(path):
    """Bare state_dict from ``.safetensors`` / ``.pt`` (also accepts {'model': sd} or {'model_state_dict': sd})."""
    path = str(path)
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('model_state_dict', 'model'):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}


def load_model(a, dim, device):
    """Return ``(model, info)``; ``info`` records what was loaded (for the output JSON)."""
    if (a.model is None) == (a.ckpt is None):
        raise SystemExit('give exactly one of --model and --ckpt')
    if a.model is not None:
        from viot.pretrained import load_pretrained
        model, cfg = load_pretrained(a.model, device=device, local_dir=a.checkpoint_dir)
        if cfg.get('dim', dim) != dim:
            raise SystemExit(f'{a.model} is a {cfg.get("dim")}D model; this script needs a {dim}D model')
        arch = {k: v for k, v in cfg['arch'].items() if k != 'class'}
        print(f'loaded released model {a.model} | arch {arch}', flush=True)
        return model, {'model': a.model, 'ckpt': a.model, 'arch': arch, 'config': cfg}
    if dim == 2:
        from viot.model_2d import FNO2D as Model
    else:
        from viot.model_3d import FNO3D as Model
    arch = dict(max_res=a.resolution, k_max=a.k_max, width=a.fno_width, n_modes=a.fno_modes, n_layers=a.fno_layers)
    model = Model(**arch)
    missing, unexpected = model.load_state_dict(read_state_dict(a.ckpt), strict=False)
    print(f'loaded {a.ckpt} | missing={list(missing)} unexpected={list(unexpected)}', flush=True)
    return model.to(device).eval(), {'model': None, 'ckpt': a.ckpt, 'arch': arch, 'config': None}
