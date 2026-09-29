"""VIOT: a variational optimal transport operator on incompressible flow.

Modules
-------
model_2d / model_3d   FNO operators that predict a stream function (2D) or a
                      vector potential (3D); velocities are divergence-free by
                      construction.
ops_2d / ops_3d       advection schemes, rollouts and flow diagnostics.
data_2d / data_3d     training-pair samplers and density normalization.
train_2d / train_3d   training entry points (``python -m viot.train_2d``).
pretrained            released checkpoints (``load_pretrained("mnist_2d")``).
"""

__version__ = "1.0.0"


def load_pretrained(*args, **kwargs):
    """Shortcut for :func:`viot.pretrained.load_pretrained`."""
    from .pretrained import load_pretrained as _load
    return _load(*args, **kwargs)
