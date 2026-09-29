"""Released VIOT checkpoints.

Each released model is a folder with ``model.safetensors`` (fp32 weights) and
``config.json`` (architecture, rollout settings, training recipe, provenance).
Folders are fetched from the Hugging Face Hub, or read from a local copy of the
checkpoint repository.

    from viot import load_pretrained
    model, cfg = load_pretrained("mnist_2d", device="cuda")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

HF_REPO = os.environ.get("VIOT_HF_REPO", "jinjinhe2001/VIOT")

MODELS = {
    "mnist_2d": "2D MNIST digits, 256^2 (interactive demo model)",
    "cjk_2d": "2D Chinese characters, 256^2",
    "mpeg7_2d": "2D MPEG-7 silhouettes, 256^2",
    "latin_font_2d": "2D Latin glyphs (A-Z, a-z, 0-9), 256^2",
    "sphere2airplane_3d": "3D sphere to ShapeNet airplanes, 128^3",
    "humman_3d": "3D HuMMan human poses, 128^3",
    "font_3d": "3D extruded DejaVu glyphs, 128^3",
    "humman_3d_figures": "3D HuMMan continuation used for the paper's HuMMan figures, 128^3",
}


def _resolve_dir(name_or_dir: str, local_dir: str | os.PathLike | None) -> Path:
    p = Path(name_or_dir)
    if p.is_dir() and (p / "config.json").exists():
        return p
    if local_dir is None:
        local_dir = os.environ.get("VIOT_CHECKPOINT_DIR")
    if local_dir is not None:
        q = Path(local_dir) / name_or_dir
        if (q / "config.json").exists():
            return q
        raise FileNotFoundError(f"{q} does not contain config.json")
    if name_or_dir not in MODELS:
        raise KeyError(f"unknown model '{name_or_dir}'; available: {', '.join(MODELS)}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub, or pass local_dir= a local copy "
                          "of the checkpoint repository") from e
    root = snapshot_download(HF_REPO, allow_patterns=[f"{name_or_dir}/*"])
    return Path(root) / name_or_dir


def load_config(name_or_dir: str, local_dir: str | os.PathLike | None = None) -> dict:
    """Return the ``config.json`` of a released model."""
    with open(_resolve_dir(name_or_dir, local_dir) / "config.json", encoding="utf-8") as f:
        return json.load(f)


def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the (untrained) operator described by ``config["arch"]``."""
    arch = dict(config["arch"])
    cls = arch.pop("class")
    if cls == "FNO2D":
        from .model_2d import FNO2D as Model
    elif cls == "FNO3D":
        from .model_3d import FNO3D as Model
    else:
        raise ValueError(f"unknown architecture {cls}")
    return Model(**arch)


def load_state_dict(path: str | os.PathLike) -> dict:
    """Load weights from ``.safetensors`` or an original ``.pt`` state_dict."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    return {k.removeprefix("module."): v for k, v in sd.items()}


def load_pretrained(name_or_dir: str, device: str | torch.device = "cpu",
                    local_dir: str | os.PathLike | None = None):
    """Load a released model.

    Args:
        name_or_dir: a name from ``MODELS`` (downloaded from the Hub, or looked up in
            ``local_dir`` / ``$VIOT_CHECKPOINT_DIR``), or a folder containing
            ``model.safetensors`` and ``config.json``.
        device: target device.
        local_dir: optional local copy of the checkpoint repository.

    Returns:
        ``(model, config)`` with the model in eval mode.
    """
    folder = _resolve_dir(name_or_dir, local_dir)
    with open(folder / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    model = build_model(config)
    model.load_state_dict(load_state_dict(folder / "model.safetensors"), strict=True)
    return model.to(device).eval(), config
